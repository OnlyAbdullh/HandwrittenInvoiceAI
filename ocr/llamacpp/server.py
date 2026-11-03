"""A private ``llama-server`` process: started, health-checked, restarted, stopped.

The service owns the server the way it owns the Qwen worker process: nobody
starts it by hand, it is loaded once and reused for every request, and it is
stopped when the service stops. What is particular to an HTTP server rather
than a pipe:

* **Nothing outside this process can use it.** It binds to loopback only, on a
  port picked fresh each launch, and requires an API key generated per launch.
  The key travels in the child's environment (``LLAMA_API_KEY``), not on its
  command line, where any local user could read it from the process list.
* **"Started" is not "ready".** The port opens long before the weights are
  loaded; ``/health`` answers 503 until they are. Readiness is polled, with a
  deadline, and the process is watched while waiting so a crash is reported
  immediately with the server's own last log lines rather than as a timeout.
* **Ready is not "can see".** A server whose ``--mmproj`` failed to load still
  serves text. ``/props`` is checked for vision support, because a text-only
  server would answer every image request with an invented reading.
* **It must not outlive the service.** Stopped on ``close()``; when the object
  owning it is dropped or the interpreter exits (a ``weakref.finalize``); and --
  on Windows, where a desktop app may kill the service outright -- by a Job
  Object that takes the server down with its parent.

Requests are serialized: the server runs a single slot (``--parallel 1``, the
memory-frugal setting for a 4 GB GPU), so a second concurrent request would only
queue inside the server, where its timeout would count time spent waiting.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from core.exceptions import InferenceServerError, RecognitionTimeoutError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LOOPBACK = "127.0.0.1"

# The first llama.cpp release that can load the Qwen3.5 architecture
# ("models : support qwen3.5 series", ggml-org/llama.cpp#19468).
MIN_QWEN35_BUILD = 7990

# Flags this module sets itself. Passing them through `extra_args` would undo
# the loopback-only, key-protected, single-model guarantees above.
RESERVED_ARGS = frozenset(
    {
        "--host", "--port", "--api-key", "--api-key-file",
        "-m", "--model", "-mu", "--model-url", "-hf", "-hfr", "--hf-repo",
        "-mm", "--mmproj", "-mmu", "--mmproj-url", "--webui", "--ssl-key-file", "--ssl-cert-file",
    }
)

_HEALTH_POLL_S = 0.25
_LOG_TAIL_LINES = 40

# Windows: start the server without opening a console window. The service is
# typically launched hidden by the desktop app, and a console popping up per
# model load would be both ugly and closable by the user.
_CREATE_NO_WINDOW = 0x08000000


def _winget_install(name: str) -> str | None:
    """llama-server where `winget install llama.cpp` put it, if PATH does not say.

    winget unpacks the release into its own package folder and links it onto
    PATH -- but a PowerShell window opened before the install, or a desktop
    app started before it, does not see that PATH change, and on some machines
    the link is never created. The package folder is predictable, so it is
    looked in directly rather than making the user find it.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    winget = Path(local) / "Microsoft" / "WinGet"
    executable = name if name.lower().endswith(".exe") else f"{name}.exe"
    candidates = [winget / "Links" / executable]
    candidates += sorted((winget / "Packages").glob(f"ggml.llamacpp_*/{executable}"))
    candidates += sorted((winget / "Packages").glob(f"ggml.llamacpp_*/*/{executable}"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_executable(server_path: str) -> str:
    """The llama-server binary `server_path` names: a path, or a name on PATH.

    A relative path is resolved against the project root, like every other
    path in the configuration. A directory is accepted and searched for the
    binary, because "the folder I unzipped llama.cpp into" is what people
    reach for.
    """
    text = (server_path or "").strip()
    if not text:
        raise FileNotFoundError("llama-server path is empty. Set server_path or INVOICE_AI_LLAMA_SERVER.")

    candidate = Path(text).expanduser()
    looks_like_path = candidate.is_absolute() or "/" in text or "\\" in text

    if not looks_like_path and not candidate.exists():
        found = shutil.which(text) or _winget_install(text)
        if found is None:
            raise FileNotFoundError(
                f"'{text}' was not found on PATH. Install llama.cpp (see docs/llamacpp-nuqta-setup.md) "
                "or set INVOICE_AI_LLAMA_SERVER to the full path of llama-server(.exe)."
            )
        return found

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_dir():
        for name in ("llama-server.exe", "llama-server"):
            if (candidate / name).is_file():
                return str(candidate / name)
        raise FileNotFoundError(f"No llama-server executable inside the folder '{candidate}'.")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"llama-server not found at '{candidate}'. Check server_path / INVOICE_AI_LLAMA_SERVER."
        )
    return str(candidate)


def parse_build_number(version_output: str) -> int | None:
    """The build number from `llama-server --version` output, if it states one.

    Two formats exist in the wild: "version: 7990 (abc123)" and, on later
    builds, "version: 0.8.0-dev (build 12092, commit abc123)".
    """
    match = re.search(r"build (\d+)", version_output) or re.search(
        r"version:\s*(\d+)\s*\(", version_output
    )
    return int(match.group(1)) if match else None


def diagnose(log_tail: str) -> str:
    """A plain-language hint for the failures people actually hit."""
    text = log_tail.lower()
    if "unknown model architecture" in text:
        return (
            "This llama.cpp build does not know the model's architecture. Qwen3.5-based models "
            f"such as Nuqta need build b{MIN_QWEN35_BUILD} or newer; install a current release."
        )
    if "out of memory" in text or "cudamalloc failed" in text or "failed to allocate" in text:
        return (
            "The GPU ran out of memory. Lower gpu_layers (INVOICE_AI_LLAMA_GPU_LAYERS), set "
            "mmproj_offload: false, or set gpu_layers: 0 to run on the CPU only."
        )
    if ("failed to load" in text and "mmproj" in text) or ("clip" in text and "failed" in text):
        return "The vision projector (mmproj) failed to load. Check that it belongs to this model."
    if "error while loading shared libraries" in text or "cannot open shared object" in text:
        return "llama-server is missing a library it was built against. Reinstall llama.cpp."
    if "invalid argument" in text or "unknown argument" in text or "error: invalid" in text:
        return "llama-server rejected a command-line argument; the build may be too old. Install a current release."
    return ""


@dataclass(frozen=True)
class LlamaServerSettings:
    """How to launch the server. Everything machine-specific is in here."""

    launcher: tuple[str, ...]  # the executable, plus any interpreter in front of it
    model_path: Path
    mmproj_path: Path
    ctx_size: int = 4096
    # None: leave it to llama.cpp, whose default is to fit as many layers as
    # the free VRAM holds and keep the rest on the CPU.
    gpu_layers: str | None = None
    mmproj_offload: bool = True
    threads: int | None = None
    # Smallest number of image tokens a crop is resized up to. None: the
    # projector's own setting, else llama.cpp's Qwen-VL floor of 8.
    image_min_tokens: int | None = None
    port: int = 0
    extra_args: tuple[str, ...] = ()
    startup_timeout_s: float = 600.0

    def command(self, port: int) -> list[str]:
        cmd = [
            *self.launcher,
            "--model", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--host", LOOPBACK,
            "--port", str(port),
            # A fixed context: 0 would mean the model's full trained context,
            # whose KV cache does not fit a laptop.
            "--ctx-size", str(self.ctx_size),
            # One slot. More would multiply the KV cache, and requests are
            # serialized here anyway.
            "--parallel", "1",
            # No browser UI on a port nobody is meant to open, and no network.
            # `--no-webui` rather than its newer name `--no-ui`: current builds
            # still accept it, and b7990 -- the oldest that loads Qwen3.5 --
            # knows only this spelling.
            "--no-webui",
            "--offline",
        ]
        if self.gpu_layers is not None:
            cmd += ["--n-gpu-layers", self.gpu_layers]
        if not self.mmproj_offload:
            cmd.append("--no-mmproj-offload")
        if self.threads is not None:
            cmd += ["--threads", str(self.threads)]
        if self.image_min_tokens is not None:
            cmd += ["--image-min-tokens", str(self.image_min_tokens)]
        cmd += list(self.extra_args)
        return cmd


def check_extra_args(extra_args: Sequence[str]) -> tuple[str, ...]:
    reserved = sorted({arg.split("=", 1)[0] for arg in extra_args} & RESERVED_ARGS)
    if reserved:
        raise ValueError(
            f"extra_args may not set {reserved}: the service chooses the host, port, key and "
            "model files itself so the server stays private to it."
        )
    return tuple(str(arg) for arg in extra_args)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


class _ChildLifetime:
    """Ties a child process's lifetime to this one, where the OS allows it.

    On Windows the child is put in a Job Object flagged
    KILL_ON_JOB_CLOSE: when this process ends -- however it ends, including
    being killed from Task Manager or by the desktop app -- the handle closes
    and Windows terminates the server with it. Elsewhere the finalizer and
    `close()` are what stop it.
    """

    def __init__(self) -> None:
        self._job = None

    def bind(self, proc: subprocess.Popen) -> None:
        if sys.platform != "win32":
            return
        try:
            self._job = _windows_kill_on_close_job(proc)
        except OSError as exc:
            # Not fatal: close() and the finalizer still stop the server. It is only
            # a hard kill of the service that would leave it running.
            logger.warning(
                "Could not tie llama-server to the service's lifetime (%s). It is still stopped "
                "on shutdown, but would survive the service being force-killed.",
                exc,
            )

    def release(self) -> None:
        if self._job is not None:
            import ctypes

            # Pointer-sized: a bare int would be converted as a C int and could
            # truncate a 64-bit handle. Closing the job ends any process left
            # in it -- by now the server has already been stopped.
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._job))
            self._job = None


def _windows_kill_on_close_job(proc: subprocess.Popen):
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    if not kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error
    return job


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a server process: politely, then for certain."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


class LlamaServer:
    """One llama-server process, owned by this one."""

    # How often a dead server may be relaunched from the readiness probe. The
    # desktop app polls /health every few seconds; a model that cannot load
    # must not be reloaded on every poll.
    RESTART_BACKOFF_S = 30.0

    def __init__(self, settings: LlamaServerSettings, request_timeout_s: float = 120.0):
        self.settings = settings
        self.request_timeout_s = request_timeout_s

        self._lifecycle_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._api_key = ""
        self._log: deque[str] = deque(maxlen=400)
        self._lifetime = _ChildLifetime()
        self._closed = False
        self._restart_thread: threading.Thread | None = None
        self._last_failed_start = 0.0
        self.build_info: str | None = None

        # Stops the process if this object is dropped without close() -- the
        # CLI's orchestrator simply goes out of scope -- and at interpreter
        # exit. A finalizer rather than an atexit list of servers: the server
        # object is usually garbage-collected *before* exit, and a weak list
        # would have forgotten it while its process kept running.
        self._reaper: weakref.finalize | None = None

        # Loopback traffic must never be routed through a proxy configured for
        # the machine's internet access.
        self._session = requests.Session()
        self._session.trust_env = False

    # -- state ------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self._port}"

    @property
    def closed(self) -> bool:
        return self._closed

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def log_tail(self, lines: int = _LOG_TAIL_LINES) -> str:
        return "\n".join(list(self._log)[-lines:])

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Launch the server and block until it can read images. Idempotent."""
        with self._lifecycle_lock:
            if self._closed:
                raise InferenceServerError("llama-server was closed; the model is no longer loaded.")
            if self.is_alive():
                return
            try:
                self._start_locked()
            except Exception:
                self._last_failed_start = time.monotonic()
                raise

    def _start_locked(self) -> None:
        # An auto-picked port can be taken between picking and binding. Only
        # then is a retry worth it -- any other startup failure would fail the
        # same way again, after loading the whole model again.
        attempts = 1 if self.settings.port else 3
        for attempt in range(1, attempts + 1):
            port = self.settings.port or _free_port()
            try:
                self._launch(port)
                self._wait_until_ready()
                self._check_vision()
                logger.info(
                    "llama-server ready on %s:%d (pid %d, %s).",
                    LOOPBACK, port, self._proc.pid, self.build_info or "build unknown",
                )
                return
            except InferenceServerError as exc:
                self._stop_locked()
                bind_failed = "bind" in exc.log_tail.lower() and "port" in exc.log_tail.lower()
                if bind_failed and attempt < attempts:
                    logger.warning("llama-server could not bind port %d; retrying on another.", port)
                    continue
                raise

    def _launch(self, port: int) -> None:
        # A relaunch after a crash: drop the dead process's finalizer and Job
        # Object handle before taking on the new one.
        self._stop_locked()
        self._port = port
        self._api_key = secrets.token_urlsafe(32)
        self._log.clear()
        self.build_info = None

        command = self.settings.command(port)
        env = dict(os.environ, LLAMA_API_KEY=self._api_key)
        # A machine-wide LLAMA_ARG_* setting must not move the server off
        # loopback or swap its model behind the configuration's back. Command
        # line flags win over these anyway; dropping them makes it explicit.
        for key in ("LLAMA_ARG_HOST", "LLAMA_ARG_PORT", "LLAMA_ARG_MODEL", "LLAMA_ARG_MMPROJ", "LLAMA_API_KEY_FILE"):
            env.pop(key, None)

        logger.info("Starting llama-server: %s", " ".join(command))
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except OSError as exc:
            raise InferenceServerError(f"Could not launch llama-server ({command[0]}): {exc}") from exc

        # Holds the process, never `self`, so it cannot keep the server alive.
        self._reaper = weakref.finalize(self, _terminate, self._proc)
        self._lifetime.bind(self._proc)
        threading.Thread(
            target=self._drain, args=(self._proc, self._log), name="llama-server-log", daemon=True
        ).start()

    @staticmethod
    def _drain(proc: subprocess.Popen, log: deque) -> None:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log.append(line)
                logger.debug("[llama-server] %s", line)

    def _failure(self, message: str) -> InferenceServerError:
        # Give the drain thread a moment to collect the last lines a dying
        # process wrote: they are the reason it died.
        time.sleep(0.2)
        tail = self.log_tail()
        hint = diagnose(tail)
        return InferenceServerError(f"{message} {hint}".strip(), log_tail=tail)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_s
        url = f"{self.base_url}/health"

        while True:
            code = self._proc.poll()
            if code is not None:
                raise self._failure(f"llama-server exited during startup (exit code {code}).")

            try:
                response = self._session.get(url, headers=self._headers(), timeout=2)
                if response.status_code == 200:
                    return
                # 503 while the model loads; anything else is reported if it
                # persists past the deadline.
            except requests.RequestException:
                pass  # not listening yet

            if time.monotonic() > deadline:
                raise self._failure(
                    f"llama-server did not become ready within {self.settings.startup_timeout_s:.0f}s."
                )
            time.sleep(_HEALTH_POLL_S)

    def _check_vision(self) -> None:
        try:
            response = self._session.get(f"{self.base_url}/props", headers=self._headers(), timeout=10)
            response.raise_for_status()
            props = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise self._failure(f"llama-server is up but /props could not be read: {exc}") from exc

        self.build_info = props.get("build_info")
        modalities = props.get("modalities")
        if modalities is None:
            logger.warning(
                "llama-server (%s) does not report its modalities; cannot confirm the vision "
                "projector loaded. Builds older than b%d cannot load Qwen3.5 models anyway.",
                self.build_info or "unknown build", MIN_QWEN35_BUILD,
            )
            return
        if not modalities.get("vision"):
            raise self._failure(
                "llama-server started without image support: the mmproj file did not load as a "
                "vision projector, so the model would answer without seeing the invoice."
            )

    def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        if self._reaper is not None:
            self._reaper.detach()
            self._reaper = None
        if proc is not None:
            _terminate(proc)
        self._lifetime.release()

    def close(self) -> None:
        """Stop the server for good. A closed server is never restarted."""
        with self._lifecycle_lock:
            self._closed = True
            self._stop_locked()
        self._session.close()

    # -- readiness, for /health --------------------------------------------

    def is_ready(self) -> bool:
        """Whether a request sent now would be answered.

        Never blocks on a model load. If the server died, a relaunch is
        started in the background (at most once per RESTART_BACKOFF_S) and
        this answers False until it is up -- so a crash heals itself instead of
        leaving the service reporting "not ready" for ever.
        """
        if self._closed:
            return False
        if not self.is_alive():
            self._restart_in_background()
            return False
        try:
            response = self._session.get(f"{self.base_url}/health", headers=self._headers(), timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _restart_in_background(self) -> None:
        if self._restart_thread is not None and self._restart_thread.is_alive():
            return
        if time.monotonic() - self._last_failed_start < self.RESTART_BACKOFF_S:
            return

        def restart() -> None:
            try:
                self.start()
            except Exception as exc:  # noqa: BLE001 - reported, retried after the backoff
                logger.error("Relaunching llama-server failed: %s", exc)

        logger.warning("llama-server is not running; relaunching it in the background.")
        self._restart_thread = threading.Thread(target=restart, name="llama-server-restart", daemon=True)
        self._restart_thread.start()

    # -- requests ---------------------------------------------------------

    def post_json(self, path: str, payload: dict, timeout_s: float | None = None) -> dict:
        """POST `payload`, return the JSON answer. Starts the server if needed."""
        timeout_s = timeout_s or self.request_timeout_s
        with self._request_lock:
            if not self.is_alive():
                if self._proc is not None or self._port is not None:
                    logger.warning("llama-server had stopped; relaunching it before this request.")
                self.start()

            url = f"{self.base_url}{path}"
            try:
                response = self._session.post(url, json=payload, headers=self._headers(), timeout=(5, timeout_s))
            except requests.Timeout as exc:
                self._after_timeout()
                raise RecognitionTimeoutError(
                    f"llama-server gave no answer within {timeout_s:.0f}s."
                ) from exc
            except requests.ConnectionError as exc:
                if not self.is_alive():
                    raise self._failure("llama-server exited while reading.") from exc
                raise InferenceServerError(f"Could not reach llama-server: {exc}") from exc

        if response.status_code != 200:
            raise InferenceServerError(
                f"llama-server answered HTTP {response.status_code}: {_error_message(response)}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise InferenceServerError(f"llama-server answered with invalid JSON: {exc}") from exc

    def _after_timeout(self) -> None:
        """A slow answer is not a dead server -- but a server that no longer
        answers /health is, and is stopped so the next request relaunches it."""
        try:
            healthy = self._session.get(f"{self.base_url}/health", headers=self._headers(), timeout=5).ok
        except requests.RequestException:
            healthy = False
        if not healthy:
            logger.warning("llama-server stopped responding; it will be relaunched on the next request.")
            with self._lifecycle_lock:
                self._stop_locked()


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body)[:500]
