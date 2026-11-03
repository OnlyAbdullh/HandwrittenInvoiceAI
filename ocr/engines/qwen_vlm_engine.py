"""Qwen VLM as an OCR engine, over one of three transports.

* ``subprocess`` (the default in every shipped config) -- a long-lived worker
  (`tools/qwen_worker.py`) in a second virtualenv, because Qwen3.5 needs
  transformers>=5 and surya-ocr pins <5.
* ``transformers`` -- the same :class:`tools.qwen_worker.QwenRuntime`, loaded in
  this process. For a venv that does not also carry surya.
* ``http`` -- an Ollama-compatible server. No token scores, so no confidence.

The subprocess transport is shared by every request thread the HTTP service
runs, so each request/response exchange holds a lock and carries an id: without
both, two concurrent invoices could each receive the other's cell texts. Every
read has a timeout; a worker that stops answering is killed and restarted
rather than holding a request thread forever.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult, Recognition
from core.exceptions import RecognitionTimeoutError
from ocr.registry import engine_registry

logger = logging.getLogger(__name__)

PROMPTS = {
    "number": (
        "هذه خانة من فاتورة، تحتوي رقماً مكتوباً بخط اليد بالأرقام الهندية "
        "(٠١٢٣٤٥٦٧٨٩) وقد يحتوي فاصلة عشرية. اكتب الرقم فقط بالأرقام "
        "الإنجليزية 0-9. لا تكتب أي حرف أبداً. إن كانت الخانة فارغة أو غير "
        "مقروءة فاكتب: EMPTY"
    ),
    # An identifier, which may mix digits with Latin letters and dashes. Kept
    # apart from "number" so that "A-1903" is not forced down to "1903".
    "code": (
        "هذه خانة رقم الفاتورة. اكتب رقم الفاتورة فقط كما يظهر، بالأرقام "
        "الإنجليزية 0-9 والأحرف اللاتينية والشرطة إن وجدت، دون أي كلمة أخرى. "
        "إن كانت الخانة فارغة أو غير مقروءة فاكتب: EMPTY"
    ),
    "arabic_text": (
        "هذه خانة من فاتورة، تحتوي وصف منتج مكتوباً بخط اليد بالعربية. "
        "اكتب النص كما هو تماماً. قد تظهر أسماء ماركات أجنبية. "
        "إن كانت الخانة فارغة أو غير مقروءة فاكتب: EMPTY"
    ),
    # A date is NOT a number. The "number" prompt forbids every non-digit
    # character, which strips the separators out of "٢٠٢٦/٨/١٥" and leaves an
    # eight-digit run nothing downstream can parse back into a date.
    "date": (
        "هذه خانة من فاتورة، تحتوي تاريخاً مكتوباً بخط اليد. "
        "اكتب التاريخ بالأرقام الإنجليزية 0-9 مع الفواصل كما تظهر "
        "(مثال: 2026/8/15). لا تكتب أي حرف. "
        "إن كانت الخانة فارغة أو غير مقروءة فاكتب: EMPTY"
    ),
    "any": "اقرأ النص في هذه الصورة كاملاً من البداية إلى النهاية.",
}

# The pipeline root, for anchoring the worker paths below. Three parents up:
# ocr/engines/qwen_vlm_engine.py -> ocr/engines -> ocr -> the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_BACKENDS = ("subprocess", "transformers", "http")


def _resolve_local(path: str) -> str:
    """An OS-native absolute path for a config value written project-relative.

    Two separate problems, one fix. On Windows **CreateProcess refuses a
    relative path written with forward slashes** -- ``os.path.exists`` says the
    file is there and ``Popen`` still raises ``FileNotFoundError [WinError 2]``.
    And the service is started from wherever the desktop app happens to launch
    it, so a relative path would be resolved against the wrong directory anyway.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate)


def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _recognition(result: dict) -> Recognition:
    if "error" in result:
        raise RuntimeError(result["error"])
    return Recognition(text=(result.get("text") or "").strip(), confidence=result.get("confidence"))


class _WorkerProcess:
    """One worker subprocess: start, exchange one request, restart, close.

    Thread-safe. A daemon thread drains stdout into a queue so a read can time
    out (Windows pipes do not support select()). Each request carries an id and
    any line answering a different id -- the late reply to a request that
    already timed out -- is discarded rather than handed to the next caller.
    """

    def __init__(self, cmd: list[str], startup_timeout_s: float, request_timeout_s: float):
        self._cmd = cmd
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        # Set by close(). A dead worker is restarted on the next request; a
        # closed one is not -- its owner unloaded it on purpose.
        self._closed = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen worker was closed.")
            self._start_locked()

    def _start_locked(self) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        logger.info("Starting Qwen worker: %s", " ".join(self._cmd))
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # the worker's own diagnostics reach the service console
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        threading.Thread(
            target=self._drain, args=(self._proc, self._lines), daemon=True
        ).start()

        noise: list[str] = []
        while True:
            line = self._next_line(self._startup_timeout_s, during="startup")
            if line is None:
                tail = "\n".join(noise[-8:]) or "(no output)"
                raise RuntimeError(f"Qwen worker exited before becoming ready. Last output:\n{tail}")
            if line == "READY":
                logger.info("Qwen worker ready.")
                return
            if line:
                noise.append(line)
                logger.info("[qwen worker] %s", line)

    @staticmethod
    def _drain(proc: subprocess.Popen, lines: "queue.Queue[str | None]") -> None:
        for raw in proc.stdout:
            lines.put(raw.strip())
        lines.put(None)  # EOF: the process is gone

    def _next_line(self, timeout_s: float, during: str) -> str | None:
        try:
            return self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self._kill_locked()
            raise RecognitionTimeoutError(
                f"Qwen worker gave no answer within {timeout_s:.0f}s during {during}; "
                "it was stopped and will be restarted on the next request."
            ) from None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._kill_locked(graceful=True)

    def _kill_locked(self, graceful: bool = False) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if graceful:
                proc.stdin.close()
                proc.wait(timeout=10)
                return
        except Exception:  # noqa: BLE001
            pass
        proc.kill()

    # -- one exchange -----------------------------------------------------

    def request(self, items: list[dict]) -> list[dict]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen worker was closed; the model is no longer loaded.")
            if not self.is_alive():
                self._start_locked()

            request_id = next(self._ids)
            payload = json.dumps({"id": request_id, "items": items}, ensure_ascii=False)
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._kill_locked()
                raise RuntimeError(f"Qwen worker pipe closed: {exc}") from exc

            while True:
                line = self._next_line(self._request_timeout_s, during="a read")
                if line is None:
                    self._proc = None
                    raise RuntimeError("Qwen worker exited unexpectedly.")
                if not line:
                    continue
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    logger.info("[qwen worker] %s", line)
                    continue
                if response.get("id") != request_id:
                    logger.warning("Discarding a stale Qwen worker reply (id %s).", response.get("id"))
                    continue
                break

        if "error" in response:
            raise RuntimeError(response["error"])
        results = response.get("results") or []
        if len(results) != len(items):
            raise RuntimeError(
                f"Qwen worker answered {len(results)} result(s) for {len(items)} item(s)."
            )
        return results


@engine_registry.register("qwen_vlm")
class QwenVLMEngine:
    """Reads crops with a Qwen VLM. Also usable as a whole-page `OCREngine`."""

    name = "qwen_vlm"

    def __init__(
        self,
        model_path: str = "OCR_Models/Arabic-Qwen3.5-OCR-v4",
        backend: str = "subprocess",
        host: str = "http://localhost:11434",
        model_name: str = "arabic-ocr",
        kind: str = "any",
        max_new_tokens: int = 64,
        worker_python: str = "qwen_venv/Scripts/python.exe",
        worker_script: str = "tools/qwen_worker.py",
        arch: str | None = None,
        timeout_s: float = 120.0,
        startup_timeout_s: float = 600.0,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {_BACKENDS}.")

        self.model_path = _resolve_local(model_path)
        self.backend = backend
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.kind = kind
        self.max_new_tokens = max_new_tokens
        self.arch = arch
        self.timeout_s = timeout_s

        self._worker: _WorkerProcess | None = None
        self._runtime = None
        self._runtime_lock = threading.Lock()

        if backend == "subprocess":
            python = _resolve_local(worker_python)
            script = _resolve_local(worker_script)
            if not os.path.exists(python):
                raise FileNotFoundError(
                    f"worker_python not found: {python}. Point it at the python.exe of a "
                    "venv that has transformers>=5.0 (see requirements-vlm.txt), or set "
                    "INVOICE_AI_VLM_PYTHON."
                )
            if not os.path.exists(script):
                raise FileNotFoundError(f"worker_script not found: {script}")
            if not os.path.isdir(self.model_path):
                raise FileNotFoundError(
                    f"model_path not found: {self.model_path}. Set INVOICE_AI_MODEL_DIR."
                )

            cmd = [python, script, "--model", self.model_path,
                   "--max-new-tokens", str(max_new_tokens)]
            if arch:
                cmd += ["--arch", arch]
            self._worker = _WorkerProcess(cmd, startup_timeout_s, timeout_s)
            # Started now so the model load (~30-60s) happens at service startup
            # (warmup), not inside the first invoice's request.
            self._worker.start()
        elif backend == "transformers":
            from tools.qwen_worker import QwenRuntime

            self._runtime = QwenRuntime(self.model_path, arch=arch, max_new_tokens=max_new_tokens)

    # -- readiness / lifecycle -------------------------------------------

    def is_ready(self) -> bool:
        if self._worker is not None:
            return self._worker.is_alive()
        return True

    def warmup(self) -> None:
        if self._worker is not None and not self._worker.is_alive():
            self._worker.start()

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    # -- reading ----------------------------------------------------------

    def read_batch(self, items: Sequence[tuple[Image.Image, str]]) -> list[Recognition]:
        """Read (image, kind) pairs; one Recognition each, in order.

        Images are sent as-is: sizing is architecture-specific and lives in one
        place, `tools.qwen_worker.fit`.
        """
        if not items:
            return []

        prompts = [PROMPTS.get(kind, PROMPTS["any"]) for _, kind in items]

        if self.backend == "subprocess":
            payload = [
                {"image": _png_b64(img), "prompt": prompt, "kind": kind}
                for (img, kind), prompt in zip(items, prompts)
            ]
            return [_recognition(r) for r in self._worker.request(payload)]

        if self.backend == "transformers":
            batch = [
                {"image": img, "prompt": prompt, "kind": kind}
                for (img, kind), prompt in zip(items, prompts)
            ]
            with self._runtime_lock:
                return [_recognition(r) for r in self._runtime.read_batch(batch)]

        return [self._read_http(img, prompt) for (img, _), prompt in zip(items, prompts)]

    def read(self, img: Image.Image, kind: str | None = None) -> Recognition:
        return self.read_batch([(img, kind or self.kind)])[0]

    def _read_http(self, img: Image.Image, prompt: str) -> Recognition:
        import requests

        r = requests.post(f"{self.host}/api/generate", json={
            "model": self.model_name,
            "prompt": prompt,
            "images": [_png_b64(img)],
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens},
        }, timeout=self.timeout_s)
        r.raise_for_status()
        # The generate endpoint exposes no token scores; None, not a guess.
        return Recognition(r.json().get("response", "").strip(), confidence=None)

    def recognize(self, image: ImagePayload) -> OCRResult:
        """Whole-page reading, for the `single_engine` flow."""
        reading = self.read(to_pil(image.image))
        h, w = image.image.shape[:2]
        return OCRResult(
            fragments=[OCRFragment(
                text=reading.text,
                bbox=BoundingBox(x=0, y=0, w=w, h=h),
                confidence=reading.confidence,
            )],
            engine_name="qwen_vlm",
            raw_engine_output=reading.text,
        )
