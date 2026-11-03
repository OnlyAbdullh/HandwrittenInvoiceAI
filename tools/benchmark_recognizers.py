"""Compare every OCR model under OCR_Models/ on the same cell crops, to pick the best one.

Two steps, because they need different Python environments:

1. **Cut the crops** (the service's venv -- it has OpenCV and the pipeline)::

       python tools/export_crops.py --photo C:\\invoices\\my_invoice.jpg --out datasets/my_invoice

   Every cell the pipeline would read, cut exactly as the recognizer receives it.
   Or, for the labelled invoices in evaluation/ground_truth/:
   ``python tools/export_crops.py --out datasets/crops``.

2. **Run every model on them** (the VLM venv -- torch and transformers)::

       qwen_venv\\Scripts\\python.exe tools/benchmark_recognizers.py --crops datasets/my_invoice
       qwen_venv\\Scripts\\python.exe tools/benchmark_recognizers.py --crops datasets/my_invoice --mode pipeline

   Then open ``evaluation/reports/model_benchmark.html``: every crop beside every
   model's reading, plus a ranked summary.

**Labelled or not.** Crops exported from your own photo have no labels, so there is
no error rate to rank by: the report shows each model's reading next to the crop
for you to judge, and the summary ranks by how often a reading passes the
pipeline's own output validation (a number cell that comes back as words fails
it) -- a necessary condition for a good reading, not proof of one. To get real
scores, type the correct text into the ``text`` fields of the crops'
``labels.jsonl`` (or export with ``--truth-file``) and run again: models are then
ranked by character error rate (CER).

**Two modes.**

* ``--mode generic`` (default) -- one plain prompt for every model and no decoding
  constraints. The fair comparison of the models themselves, since the
  pipeline's prompts were tuned on one of them.
* ``--mode pipeline`` -- each model read the way the service would read it: the
  per-kind prompts, and for the two models the service can actually run, the
  production code itself -- ``tools/qwen_worker.py``'s runtime (Qwen3.5 /
  Qwen2.5-VL checkpoints: per-kind decoding, digit constraints, resizing) and
  the llama.cpp backend (GGUF: grammar, image-token floor). The answer to "what
  would my invoices get".

**What is run.** Each folder under OCR_Models/ is classified from its own files:

* chat VLMs (Qwen2/2.5/3-VL, Qwen3.5, GLM-OCR) and Florence-2 -- transformers;
* encoder-decoder models (TrOCR, Nougat) -- transformers;
* GGUF folders -- every ``*.gguf`` that is not an ``mmproj`` is a separate entry
  (``Q4_K_M`` and ``Q8_0`` are compared with each other), run by llama.cpp's
  ``llama-server`` with the folder's mmproj. Needs ``llama-server`` on PATH or
  ``--llama-server``.

Anything else is reported as skipped with the reason, never silently dropped.
Models are loaded one at a time and unloaded before the next, so a 4 GB GPU can
compare them all; a model that does not fit on the GPU is retried on the CPU.

Read the numbers for what they are: a few dozen crops from one or two invoices
rule a model out when it is clearly worse; they do not certify a close winner.
"""

from __future__ import annotations

import argparse
import base64
import gc
import html
import io
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "OCR_Models"
sys.path.insert(0, str(PROJECT_ROOT))

# A generic instruction, the same for every model, deliberately not the
# per-kind prompts tuned for one specific model (see module docstring).
PROMPT = "اقرأ النص المكتوب في هذه الصورة بدقة. اكتب النص فقط، دون أي شرح أو علامات إضافية."

MAX_NEW_TOKENS = 48


# --------------------------------------------------------------------------
# Scoring -- copied rather than imported, so the scoring has no dependency on
# the service's venv. Same rules as string_matching.normalization /
# evaluation.metrics: NFKC, Arabic-Indic digits folded, diacritics and
# punctuation stripped, case folded.
# --------------------------------------------------------------------------

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {
    ord(ch): str(i) for group in (_ARABIC_INDIC, _EXTENDED_ARABIC_INDIC) for i, ch in enumerate(group)
}
_DIACRITICS = re.compile("[ً-ْـ]")
_LETTER_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
    "ی": "ي", "ې": "ي", "ک": "ك", "ۀ": "ه", "ھ": "ه", "ہ": "ه",
})
_PUNCTUATION = re.compile(r"[.,;:!؟?،؛(){}\[\]<>\"'`«»/\\|_\-+*=&^%#@~$]")
_WHITESPACE = re.compile(r"\s+")


def strip_caption(text: str) -> str:
    """Drop a printed caption from the front of a reading, same rule as
    ocr/recognizers/output_validation.py: several header boxes on these forms
    print the caption *inside* the same ruled box as the value
    ("رقم الفاتورة : 00010"), so a model reading both is reading the box
    correctly -- the caption just isn't part of what the label says.
    """
    for separator in (":", "："):
        head, found, tail = text.partition(separator)
        if found and tail.strip():
            return tail.strip()
    return text


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = strip_caption(text)
    result = unicodedata.normalize("NFKC", text).translate(_DIGIT_MAP)
    result = _DIACRITICS.sub("", result).translate(_LETTER_FOLD)
    result = _PUNCTUATION.sub(" ", result)
    return _WHITESPACE.sub(" ", result).strip().casefold()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(predicted: str, expected: str | None) -> float | None:
    """Character error rate against the label; None when the crop has no label."""
    if expected is None:
        return None
    truth = normalize_text(expected)
    if not truth:
        return None
    return round(levenshtein(normalize_text(predicted), truth) / len(truth), 3)


def passes_validation(text: str, kind: str) -> bool | None:
    """Whether the pipeline's own output validation would keep this reading.

    None when the validator cannot be imported (it needs only the standard
    library, numpy and Pillow from this project, so that should be rare).
    """
    try:
        from core.domain.roles import ContentKind
        from ocr.recognizers.output_validation import clean
    except Exception:  # noqa: BLE001
        return None
    try:
        return clean(text, ContentKind(kind)) is not None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# What's in OCR_Models/
# --------------------------------------------------------------------------

_CHAT_VLM_ARCHES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "GlmOcrForConditionalGeneration",
}
_ENCODER_DECODER_ARCHES = {"VisionEncoderDecoderModel"}
_FLORENCE_ARCHES = {"Florence2ForConditionalGeneration"}

# The architectures the service's transformers worker runs in production
# (tools/qwen_worker.py ARCH_CLASSES). Only these get its exact decoding in
# --mode pipeline.
_PRODUCTION_ARCHES = {"qwen3_5", "qwen2_5_vl"}


@dataclass
class Candidate:
    name: str
    path: Path               # the model folder, or the .gguf file
    family: str = ""
    mmproj: Path | None = None
    model_type: str = ""
    skipped_reason: str | None = None


def classify(model_dir: Path) -> tuple[str, str | None, str]:
    """(family, skip_reason, model_type). family is "" when skip_reason is set."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        files = sorted(p.name for p in model_dir.iterdir())[:8] if model_dir.is_dir() else []
        return "", (f"no config.json (not a transformers model, or an incomplete download); "
                    f"contains: {', '.join(files) or 'nothing'}"), ""

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "", f"config.json unreadable: {exc}", ""

    archs = set(config.get("architectures") or [])
    model_type = config.get("model_type") or ""
    if archs & _CHAT_VLM_ARCHES:
        return "chat_vlm", None, model_type
    if archs & _FLORENCE_ARCHES or model_type.startswith("florence"):
        return "florence2", None, model_type
    if archs & _ENCODER_DECODER_ARCHES:
        return "encoder_decoder", None, model_type
    return "", f"unrecognized architecture(s) {sorted(archs) or '(none)'}", model_type


def model_root(folder: Path) -> Path:
    """Where the model's config.json actually is.

    Downloads are not always flat: a repository cloned into a folder of the same
    name, or a checkpoint kept in a subfolder, puts config.json one or two
    levels down. One unambiguous match is used; several (or none) leave the
    folder as it is, to be reported.
    """
    if (folder / "config.json").exists():
        return folder
    found = sorted({p.parent for depth in ("*/config.json", "*/*/config.json") for p in folder.glob(depth)})
    return found[0] if len(found) == 1 else folder


def discover(models_dir: Path, only: list[str] | None) -> list[Candidate]:
    """Every model to compare. A GGUF folder becomes one entry per quantization."""
    candidates: list[Candidate] = []
    for folder in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        ggufs = sorted(folder.glob("*.gguf"))
        if ggufs:
            projectors = [g for g in ggufs if "mmproj" in g.name.lower()]
            models = [g for g in ggufs if g not in projectors]
            if not projectors:
                candidates.append(Candidate(
                    folder.name, folder, skipped_reason="GGUF folder without an *mmproj*.gguf: "
                    "a text-only model cannot read images"))
                continue
            for model in models:
                candidates.append(Candidate(f"{folder.name} [{model.stem}]", model, "gguf", mmproj=projectors[0]))
            continue

        root = model_root(folder)
        family, reason, model_type = classify(root)
        candidates.append(Candidate(folder.name, root, family, model_type=model_type, skipped_reason=reason))

    if only:
        wanted = [w.lower() for w in only]
        candidates = [c for c in candidates if any(w in c.name.lower() for w in wanted)]
    return candidates


# --------------------------------------------------------------------------
# Readers. Each loads one model, reads (image, kind), and unloads.
# --------------------------------------------------------------------------


class Reader:
    device = "cpu"

    def read(self, image, kind: str) -> tuple[str, float | None]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _free_torch(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


class ChatVLMReader(Reader):
    """Any chat-template VLM through transformers, with a plain generate()."""

    def __init__(self, model_dir: Path, device: str, prompts: dict[str, str] | None):
        import torch
        import transformers
        from transformers import AutoProcessor

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        arch_name = config["architectures"][0]
        cls = getattr(transformers, arch_name, None) or transformers.AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if device == "cuda":
            kwargs["dtype"] = torch.bfloat16
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = torch.float32
        self.model = cls.from_pretrained(str(model_dir), **kwargs).eval()
        self.device = device
        self.prompts = prompts

    def read(self, image, kind):
        import torch

        prompt = self.prompts.get(kind, self.prompts["any"]) if self.prompts else PROMPT
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            images = process_vision_info(messages)[0]
        except Exception:  # noqa: BLE001 - not every chat model needs/has this helper
            images = [image]

        inputs = self.processor(text=[text_input], images=images, padding=True, return_tensors="pt").to(self.model.device)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        with torch.inference_mode():
            # Mild anti-repetition -- stops a small VLM looping a single token on a
            # crop that defeats it -- PLUS explicit eos/pad: without them some
            # models never stop and fill every one of max_new_tokens with
            # increasingly random continuation text past a correct answer.
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Short-field answers: the first line is the answer, anything after is
        # the model continuing to talk once it had already finished.
        return text.strip().split("\n\n")[0].split("\n")[0].strip(), None

    def close(self):
        del self.model, self.processor
        _free_torch(self.device)


class ProductionQwenReader(Reader):
    """The service's own transformers runtime -- tools/qwen_worker.QwenRuntime."""

    def __init__(self, model_dir: Path, device: str, prompts: dict[str, str]):
        from tools.qwen_worker import QwenRuntime

        self.runtime = QwenRuntime(str(model_dir))
        self.device = "cuda" if str(self.runtime.model.device).startswith("cuda") else "cpu"
        self.prompts = prompts

    def read(self, image, kind):
        result = self.runtime.read_batch([{"image": image, "prompt": self.prompts.get(kind, self.prompts["any"]), "kind": kind}])[0]
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("text") or "", result.get("confidence")

    def close(self):
        del self.runtime
        _free_torch(self.device)


class EncoderDecoderReader(Reader):
    """TrOCR, Nougat and other VisionEncoderDecoder checkpoints. No prompt."""

    def __init__(self, model_dir: Path, device: str, prompts=None):
        import torch
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
        self.device = device

    def read(self, image, kind):
        import torch

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip(), None

    def close(self):
        del self.model, self.processor
        _free_torch(self.device)


class FlorenceReader(Reader):
    """Florence-2 OCR fine-tunes: a task token instead of a prompt."""

    TASK = "<OCR>"

    def __init__(self, model_dir: Path, device: str, prompts=None):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        cls = getattr(transformers, "Florence2ForConditionalGeneration", None) or transformers.AutoModelForCausalLM
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = cls.from_pretrained(str(model_dir), dtype=dtype, trust_remote_code=True).to(device).eval()
        self.device = device

    def read(self, image, kind):
        import torch

        inputs = self.processor(text=self.TASK, images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(self.model.dtype),
                max_new_tokens=MAX_NEW_TOKENS, num_beams=3, do_sample=False,
            )
        raw = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        try:
            parsed = self.processor.post_process_generation(raw, task=self.TASK, image_size=image.size)
            text = parsed.get(self.TASK, raw) if isinstance(parsed, dict) else str(parsed)
        except Exception:  # noqa: BLE001 - older processors lack the post-processor
            text = re.sub(r"</?s>|<pad>", "", raw)
        return str(text).strip(), None

    def close(self):
        del self.model, self.processor
        _free_torch(self.device)


class GGUFReader(Reader):
    """A GGUF model + mmproj through the service's own llama.cpp backend."""

    def __init__(self, candidate: Candidate, prompts: dict[str, str] | None, server_path: str, gpu_layers: str | None):
        from ocr.llamacpp.client import LlamaCppVisionBackend
        from ocr.llamacpp.gguf import check_vision_model_pair
        from ocr.llamacpp.server import LlamaServer, LlamaServerSettings, resolve_executable

        check_vision_model_pair(candidate.path, candidate.mmproj)
        pipeline = prompts is not None
        self.server = LlamaServer(
            LlamaServerSettings(
                launcher=(resolve_executable(server_path),),
                model_path=candidate.path,
                mmproj_path=candidate.mmproj,
                gpu_layers=gpu_layers,
                # Same floor as the service in pipeline mode; llama.cpp's own
                # default in generic mode, like every other model's defaults.
                image_min_tokens=64 if pipeline else None,
            ),
            request_timeout_s=600,
        )
        self.server.start()
        self.backend = LlamaCppVisionBackend(
            self.server,
            prompts=prompts or {"any": PROMPT},
            max_new_tokens=64 if pipeline else MAX_NEW_TOKENS,
            constrained_decoding=pipeline,
            timeout_s=600,
        )
        self.device = f"llama.cpp (gpu_layers={gpu_layers or 'auto'})"

    def read(self, image, kind):
        reading = self.backend.read(image, kind if self.backend.constrained_decoding else "any")
        return reading.text, reading.confidence

    def close(self):
        self.server.close()


def build_reader(candidate: Candidate, mode: str, server_path: str, gpu_layers: str | None) -> Reader:
    prompts = None
    if mode == "pipeline":
        from ocr.engines.qwen_vlm_engine import PROMPTS

        prompts = dict(PROMPTS)

    if candidate.family == "gguf":
        return GGUFReader(candidate, prompts, server_path, gpu_layers)

    import torch

    def on_device(make):
        if not torch.cuda.is_available():
            return make("cpu")
        try:
            return make("cuda")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    CUDA out of memory; retrying {candidate.name} on CPU (will be slow).")
            return make("cpu")

    if candidate.family == "chat_vlm":
        if mode == "pipeline" and candidate.model_type in _PRODUCTION_ARCHES:
            return ProductionQwenReader(candidate.path, "auto", prompts)
        return on_device(lambda device: ChatVLMReader(candidate.path, device, prompts))
    if candidate.family == "florence2":
        return on_device(lambda device: FlorenceReader(candidate.path, device))
    return on_device(lambda device: EncoderDecoderReader(candidate.path, device))


# --------------------------------------------------------------------------
# Running the bake-off
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
    family: str = ""
    skipped_reason: str | None = None
    device: str | None = None
    load_s: float = 0.0
    per_crop: list[dict] = field(default_factory=list)

    def _mean(self, key: str, rows=None):
        values = [c[key] for c in (rows if rows is not None else self.per_crop) if c.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    @property
    def mean_cer(self) -> float | None:
        return self._mean("cer")

    @property
    def exact_match_rate(self) -> float | None:
        scored = [c for c in self.per_crop if c.get("cer") is not None]
        return round(sum(c["cer"] == 0.0 for c in scored) / len(scored), 3) if scored else None

    @property
    def valid_rate(self) -> float | None:
        checked = [c for c in self.per_crop if c.get("valid") is not None]
        return round(sum(bool(c["valid"]) for c in checked) / len(checked), 3) if checked else None

    @property
    def mean_seconds(self) -> float | None:
        return self._mean("seconds")

    @property
    def mean_confidence(self) -> float | None:
        return self._mean("confidence")

    def summary(self) -> dict:
        return {
            "name": self.name, "family": self.family, "device": self.device, "load_s": self.load_s,
            "skipped_reason": self.skipped_reason, "mean_cer": self.mean_cer,
            "exact_match_rate": self.exact_match_rate, "valid_rate": self.valid_rate,
            "mean_seconds": self.mean_seconds, "mean_confidence": self.mean_confidence,
        }


def load_crops(crops_dir: Path) -> list[dict]:
    from PIL import Image

    rows = []
    for line in (crops_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["pil"] = Image.open(crops_dir / row["image"]).convert("RGB")
        rows.append(row)
    return rows


def run_model(candidate: Candidate, crops: list[dict], max_crops: int | None, mode: str,
              server_path: str, gpu_layers: str | None) -> ModelResult:
    result = ModelResult(name=candidate.name, family=candidate.family)
    if candidate.skipped_reason:
        result.skipped_reason = candidate.skipped_reason
        return result

    started = time.perf_counter()
    try:
        reader = build_reader(candidate, mode, server_path, gpu_layers)
    except Exception as exc:  # noqa: BLE001 - one bad model must not end the bake-off
        result.skipped_reason = f"failed to load: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
        return result
    result.device = reader.device
    result.load_s = round(time.perf_counter() - started, 1)

    try:
        for row in (crops[:max_crops] if max_crops else crops):
            started = time.perf_counter()
            confidence = None
            try:
                text, confidence = reader.read(row["pil"], row.get("kind") or "any")
            except Exception as exc:  # noqa: BLE001
                text = f"<error: {type(exc).__name__}: {exc}>"
            seconds = round(time.perf_counter() - started, 2)
            errored = text.startswith("<error:")
            result.per_crop.append({
                "image": row["image"], "cell": row.get("cell"), "kind": row.get("kind"), "role": row.get("role"),
                "expected": row.get("text"), "predicted": text,
                "cer": None if errored else cer(text, row.get("text")),
                "valid": False if errored else passes_validation(text, row.get("kind") or "any"),
                "confidence": confidence, "seconds": seconds,
            })
    finally:
        reader.close()
    return result


def rank(results: list[ModelResult]) -> tuple[list[ModelResult], bool]:
    """Ran models best first, and whether the ranking is by error rate."""
    ran = [r for r in results if not r.skipped_reason and r.per_crop]
    labelled = any(r.mean_cer is not None for r in ran)
    if labelled:
        key = lambda r: (r.mean_cer if r.mean_cer is not None else 9e9, -(r.exact_match_rate or 0), r.mean_seconds or 9e9)
    else:
        key = lambda r: (-(r.valid_rate or 0), r.mean_seconds or 9e9)
    return sorted(ran, key=key), labelled


def to_markdown(results: list[ModelResult], mode: str, crops_dir: str) -> str:
    ranked, labelled = rank(results)
    skipped = [r for r in results if r.skipped_reason]

    lines = [
        "# Recognizer bake-off",
        "",
        f"Crops: `{crops_dir}`. Mode: **{mode}**. {len(results)} model(s); {len(ranked)} ran, {len(skipped)} skipped.",
        "",
        ("Ranked by character error rate against the labels." if labelled else
         "**The crops have no labels**, so models are ranked by how often their reading passes the "
         "pipeline's output validation -- a sanity check, not accuracy. Judge the readings in the HTML "
         "report, or fill in `text` in labels.jsonl and run again for real scores."),
        "",
        "| Rank | Model | Family | Device | Mean CER | Exact match | Passes validation | Mean confidence | Mean s/crop | Load s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.name} | {r.family} | {r.device} | {r.mean_cer} | {r.exact_match_rate} | "
            f"{r.valid_rate} | {r.mean_confidence} | {r.mean_seconds} | {r.load_s} |"
        )
    if skipped:
        lines += ["", "## Skipped", "", "| Model | Reason |", "|---|---|"]
        for r in skipped:
            lines.append(f"| {r.name} | {r.skipped_reason} |")
    for r in ranked:
        lines += [f"\n## {r.name}\n", "| Cell | Kind | Expected | Predicted | CER | Valid |", "|---|---|---|---|---|---|"]
        for c in r.per_crop:
            lines.append(f"| {c['cell']} | {c['kind']} | {c['expected']} | {c['predicted']} | {c['cer']} | {c['valid']} |")
    return "\n".join(lines)


def _thumbnail_uri(image) -> str:
    thumb = image.copy()
    thumb.thumbnail((360, 140))
    buffer = io.BytesIO()
    thumb.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def to_html(results: list[ModelResult], crops: list[dict], mode: str, crops_dir: str) -> str:
    """Every crop beside every model's reading -- the report for judging by eye."""
    ranked, labelled = rank(results)
    esc = html.escape

    summary_rows = "".join(
        f"<tr><td>{i}</td><td>{esc(r.name)}</td><td>{esc(r.family)}</td><td>{esc(str(r.device))}</td>"
        f"<td>{r.mean_cer if r.mean_cer is not None else '–'}</td>"
        f"<td>{r.exact_match_rate if r.exact_match_rate is not None else '–'}</td>"
        f"<td>{r.valid_rate if r.valid_rate is not None else '–'}</td>"
        f"<td>{r.mean_confidence if r.mean_confidence is not None else '–'}</td>"
        f"<td>{r.mean_seconds}</td><td>{r.load_s}</td></tr>"
        for i, r in enumerate(ranked, 1)
    )
    skipped_rows = "".join(
        f"<tr><td>{esc(r.name)}</td><td>{esc(r.skipped_reason or '')}</td></tr>" for r in results if r.skipped_reason
    )

    by_model = {r.name: {c["image"]: c for c in r.per_crop} for r in ranked}
    header = "".join(f"<th>{esc(r.name)}</th>" for r in ranked)
    body = []
    for row in crops:
        cells = []
        for r in ranked:
            c = by_model[r.name].get(row["image"])
            if c is None:
                cells.append("<td></td>")
                continue
            css = "rej" if c["valid"] is False else ""
            if c["cer"] is not None:
                css += " ok" if c["cer"] == 0 else (" near" if c["cer"] <= 0.34 else " bad")
            meta = f"{c['seconds']}s" + (f" · conf {c['confidence']:.3f}" if c.get("confidence") is not None else "")
            if c["cer"] is not None:
                meta += f" · CER {c['cer']}"
            cells.append(f'<td class="{css.strip()}"><div dir="auto">{esc(c["predicted"]) or "<i>(empty)</i>"}</div>'
                         f'<small>{meta}</small></td>')
        expected = esc(row["text"]) if row.get("text") is not None else "<i>no label</i>"
        body.append(
            f'<tr><td class="crop"><img src="{_thumbnail_uri(row["pil"])}"><br><small>{esc(str(row.get("cell")))} · '
            f'{esc(str(row.get("role")))} · {esc(str(row.get("kind")))}</small><div dir="auto">{expected}</div></td>'
            + "".join(cells) + "</tr>"
        )

    note = ("Ranked by character error rate against the labels." if labelled else
            "These crops have no labels: ranked by how often a reading passes the pipeline's output "
            "validation, which is a sanity check, not accuracy. Judge the readings below, or type the "
            "correct text into labels.jsonl and run again.")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>OCR model bake-off</title>
<style>
body{{font-family:system-ui,'Segoe UI',Tahoma,sans-serif;margin:16px;color:#222}}
table{{border-collapse:collapse;margin-bottom:24px}} td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}
th{{background:#f3f3f3;position:sticky;top:0}} .crop img{{max-width:360px;border:1px solid #999}}
td div{{font-size:18px}} small{{color:#666}} .ok{{background:#e3f6e3}} .near{{background:#fff6d6}}
.bad{{background:#fde3e3}} .rej div{{text-decoration:line-through;color:#a33}} .scroll{{overflow-x:auto}}
</style></head><body>
<h1>OCR model bake-off</h1>
<p>Crops: <code>{esc(crops_dir)}</code> · mode <b>{esc(mode)}</b> · {len(crops)} crop(s). {esc(note)}
Struck-through readings are ones the pipeline's output validation would reject.</p>
<h2>Summary</h2>
<table><tr><th>Rank</th><th>Model</th><th>Family</th><th>Device</th><th>Mean CER</th><th>Exact match</th>
<th>Passes validation</th><th>Mean confidence</th><th>Mean s/crop</th><th>Load s</th></tr>{summary_rows}</table>
{"<h2>Skipped</h2><table><tr><th>Model</th><th>Reason</th></tr>" + skipped_rows + "</table>" if skipped_rows else ""}
<h2>Every reading</h2>
<div class="scroll"><table><tr><th>Crop</th>{header}</tr>{"".join(body)}</table></div>
</body></html>"""


def describe_exit(code: int) -> str:
    """A crashed model's exit code, in words where it is a known one."""
    known = {
        0xC0000005: "access violation (a native crash, typically in CUDA or bitsandbytes)",
        0xC0000409: "stack buffer overrun (a native crash)",
        0xC00000FD: "stack overflow",
        0xC0000017: "out of memory",
        -9: "killed (SIGKILL, typically the out-of-memory killer)",
        -11: "segmentation fault (a native crash)",
    }
    # POSIX reports a signal as a small negative number; Windows reports an
    # NTSTATUS, unsigned (3221225477) or signed (-1073741819) depending on the caller.
    key = code if -64 < code < 0 else code & 0xFFFFFFFF
    return known.get(key, f"exit code {code}")


def run_isolated(candidate: Candidate, command: list[str], result_file: Path) -> ModelResult:
    """Run one model in its own process, so a native crash costs that model only.

    A model that crashes inside CUDA or bitsandbytes takes the whole Python
    process with it -- no exception, no traceback, no report. In a child
    process it becomes one row reading "crashed", and the GPU memory it held is
    returned to the next model in full.
    """
    import subprocess
    import threading
    from collections import deque

    tail: deque[str] = deque(maxlen=12)
    proc = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    def echo() -> None:
        for line in proc.stderr:
            sys.stderr.write(line)
            if line.strip():
                tail.append(line.rstrip())

    reader = threading.Thread(target=echo, daemon=True)
    reader.start()
    code = proc.wait()
    reader.join(timeout=5)

    if code == 0 and result_file.exists():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return ModelResult(
            name=data["name"], family=data.get("family", ""), skipped_reason=data.get("skipped_reason"),
            device=data.get("device"), load_s=data.get("load_s", 0.0), per_crop=data.get("per_crop", []),
        )
    last = " | ".join(list(tail)[-3:])
    return ModelResult(
        name=candidate.name, family=candidate.family,
        skipped_reason=f"crashed: {describe_exit(code)}" + (f" -- last output: {last[:400]}" if last else ""),
    )


def write_reports(out: Path, results: list[ModelResult], crops: list[dict], mode: str, crops_label: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(to_markdown(results, mode, crops_label), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps([{**r.summary(), "per_crop": r.per_crop} for r in results], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out.with_suffix(".html").write_text(to_html(results, crops, mode, crops_label), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", default="datasets/crops")
    ap.add_argument("--models", default=str(MODELS_DIR))
    ap.add_argument("--only", nargs="*", help="run only models whose name contains one of these")
    ap.add_argument("--max-crops", type=int, default=None, help="cap crops per model, for a quick pass")
    ap.add_argument("--mode", choices=("generic", "pipeline"), default="generic",
                    help="generic: one plain prompt for all; pipeline: read the way the service would")
    ap.add_argument("--llama-server", default=os.environ.get("INVOICE_AI_LLAMA_SERVER", "llama-server"),
                    help="llama-server executable for GGUF models (default: INVOICE_AI_LLAMA_SERVER or PATH)")
    ap.add_argument("--gpu-layers", default=os.environ.get("INVOICE_AI_LLAMA_GPU_LAYERS"),
                    help="GGUF only: 'all', '0' (CPU) or a count; default lets llama.cpp fit the free VRAM")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run every model in this process (faster to start; one native crash ends the run)")
    ap.add_argument("--out", default="evaluation/reports/model_benchmark")
    # Internal: the child-process side of isolation.
    ap.add_argument("--exact", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    crops_path = Path(args.crops)
    if not crops_path.is_absolute():
        crops_path = PROJECT_ROOT / crops_path
    if not (crops_path / "labels.jsonl").exists():
        print(f"No crops at {crops_path}. Make them first, in the service's venv:\n"
              f"  python tools/export_crops.py --photo <your invoice photo> --out {args.crops}", file=sys.stderr)
        return 2
    crops = load_crops(crops_path)
    gpu_layers = None if (args.gpu_layers or "auto").lower() == "auto" else args.gpu_layers

    if args.result_file:
        # Child process: run exactly one model and hand the result back.
        candidate = next(c for c in discover(Path(args.models), None) if c.name == args.exact)
        result = run_model(candidate, crops, args.max_crops, args.mode, args.llama_server, gpu_layers)
        Path(args.result_file).write_text(
            json.dumps({**result.summary(), "per_crop": result.per_crop}, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0

    labelled = sum(row.get("text") is not None for row in crops)
    print(f"{len(crops)} crop(s) from {crops_path} ({labelled} labelled). Mode: {args.mode}.")
    if labelled and labelled < 5:
        print(f"Only {labelled} crop(s) carry a correct answer, so error rates rest on very little. "
              f"Type the right text into the other `text` fields of {crops_path / 'labels.jsonl'} for a real comparison.")

    candidates = discover(Path(args.models), args.only)
    if not candidates:
        print(f"No models found under {args.models}.", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    import tempfile

    results: list[ModelResult] = []
    with tempfile.TemporaryDirectory(prefix="ocr-bench-") as scratch:
        for index, candidate in enumerate(candidates):
            print(f"\n== {candidate.name}", flush=True)
            if candidate.skipped_reason or args.no_isolate:
                result = run_model(candidate, crops, args.max_crops, args.mode, args.llama_server, gpu_layers)
            else:
                result_file = Path(scratch) / f"{index}.json"
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--crops", str(crops_path), "--models", str(args.models), "--mode", args.mode,
                    "--llama-server", args.llama_server, "--exact", candidate.name, "--result-file", str(result_file),
                ]
                if args.max_crops:
                    command += ["--max-crops", str(args.max_crops)]
                if args.gpu_layers:
                    command += ["--gpu-layers", args.gpu_layers]
                result = run_isolated(candidate, command, result_file)

            if result.skipped_reason:
                print(f"   skipped: {result.skipped_reason}")
            else:
                print(f"   {result.family} on {result.device}: CER {result.mean_cer}, exact {result.exact_match_rate}, "
                      f"passes validation {result.valid_rate}, {result.mean_seconds}s/crop (load {result.load_s}s)")
            results.append(result)
            # After every model: a run interrupted later still leaves a report.
            write_reports(out, results, crops, args.mode, args.crops)

    ranked, by_cer = rank(results)
    print("\nRanking (" + ("lowest CER first" if by_cer else "no labels: most readings passing validation first") + "):")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r.name}  CER={r.mean_cer}  valid={r.valid_rate}  {r.mean_seconds}s/crop")
    print(f"\nReport: {out.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
