"""Compare every model under OCR_Models/ on the same labelled crops.

    D:/OCR/.venv_vlm/Scripts/python.exe tools/benchmark_recognizers.py

Run in the VLM virtualenv (torch + transformers>=5), not the service's own venv --
same reason `tools/qwen_worker.py` does. Needs crops exported first:

    python tools/export_crops.py --out datasets/crops

For each model directory, this loads it, reads every crop, scores the reading
against the label with character error rate (CER) and exact match, times it,
then unloads it before moving to the next -- so models can be compared on a
small GPU one at a time rather than needing all of them loaded together.

Two things this bake-off deliberately does NOT do, so read the numbers as a
starting point, not a verdict:

* **One generic prompt for every model**, not the per-kind, per-quirk prompts
  tuned for Arabic-Qwen3.5-OCR-v4 in ocr/engines/qwen_vlm_engine.py. Reusing
  those would bias the comparison toward the model they were measured on.
* **20 crops from one invoice.** Enough to rule a model out when it is badly
  worse, not enough to certify a winner. Label more invoices
  (evaluation/ground_truth/, then tools/export_crops.py) before trusting a
  close result.

A model that does not load (missing remote code, wrong processor, out of
memory on both GPU and CPU) is reported as skipped with the reason, not
silently dropped.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "OCR_Models"

# A generic instruction, the same for every model, deliberately not the
# per-kind prompts tuned for one specific model (see module docstring).
PROMPT = "اقرأ النص المكتوب في هذه الصورة بدقة. اكتب النص فقط، دون أي شرح أو علامات إضافية."

MAX_NEW_TOKENS = 48


# --------------------------------------------------------------------------
# Scoring -- copied rather than imported, so this script has no dependency on
# the service's own venv. Same rules as string_matching.normalization /
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


def cer(predicted: str, expected: str) -> float:
    truth = normalize_text(expected)
    if not truth:
        return 0.0
    return levenshtein(normalize_text(predicted), truth) / len(truth)


# --------------------------------------------------------------------------
# What's in OCR_Models/, and how to load it
# --------------------------------------------------------------------------

# Architectures this script knows how to drive, and which family. Anything
# else -- GGUF, a folder with no config.json, a known-broken download -- is
# reported as skipped rather than guessed at.
_CHAT_VLM_ARCHES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "GlmOcrForConditionalGeneration",
}
_ENCODER_DECODER_ARCHES = {"VisionEncoderDecoderModel"}


@dataclass
class ModelResult:
    name: str
    family: str = ""
    skipped_reason: str | None = None
    device: str | None = None
    load_s: float = 0.0
    per_crop: list[dict] = field(default_factory=list)

    @property
    def mean_cer(self) -> float | None:
        scored = [c["cer"] for c in self.per_crop if c["cer"] is not None]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def exact_match_rate(self) -> float | None:
        scored = [c for c in self.per_crop if c["cer"] is not None]
        return round(sum(c["cer"] == 0.0 for c in scored) / len(scored), 3) if scored else None

    @property
    def mean_seconds(self) -> float | None:
        timed = [c["seconds"] for c in self.per_crop if c.get("seconds") is not None]
        return round(sum(timed) / len(timed), 2) if timed else None


def classify(model_dir: Path) -> tuple[str, str | None]:
    """(family, skip_reason). family is "" when skip_reason is set."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return "", "no config.json (not a transformers model, or an incomplete download)"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "", f"config.json unreadable: {exc}"

    archs = set(config.get("architectures") or [])
    if archs & _CHAT_VLM_ARCHES:
        return "chat_vlm", None
    if archs & _ENCODER_DECODER_ARCHES:
        return "encoder_decoder", None
    return "", f"unrecognized architecture(s) {sorted(archs) or '(none)'}"


def load_chat_vlm(model_dir: Path, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    arch_name = config["architectures"][0]
    cls = getattr(transformers, arch_name, None) or transformers.AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32
    model = cls.from_pretrained(str(model_dir), **kwargs).eval()
    return model, processor


def read_chat_vlm(model, processor, image) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        images = process_vision_info(messages)[0]
    except Exception:  # noqa: BLE001 - not every chat model needs/has this helper
        images = [image]

    inputs = processor(text=[text_input], images=images, padding=True, return_tensors="pt").to(model.device)
    tokenizer = getattr(processor, "tokenizer", processor)
    with torch.inference_mode():
        # Mild anti-repetition -- stops a small VLM looping a single token on a
        # crop that defeats it -- PLUS explicit eos/pad: without them some
        # models never stop and fill every one of max_new_tokens with
        # increasingly random continuation text past a correct answer.
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=1.15, no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    # Belt and suspenders beyond eos: these are short-field answers, so the
    # first line is the answer and anything after a blank line is the model
    # continuing to talk once it had already finished.
    return text.split("\n\n")[0].split("\n")[0].strip()


def load_encoder_decoder(model_dir: Path, device: str):
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel

    processor = AutoProcessor.from_pretrained(str(model_dir))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
    return model, processor


def read_encoder_decoder(model, processor, image) -> str:
    import torch

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


ADAPTERS = {
    "chat_vlm": (load_chat_vlm, read_chat_vlm),
    "encoder_decoder": (load_encoder_decoder, read_encoder_decoder),
}


# --------------------------------------------------------------------------
# Running the bake-off
# --------------------------------------------------------------------------


def load_crops(crops_dir: Path) -> list[dict]:
    from PIL import Image

    rows = []
    for line in (crops_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["pil"] = Image.open(crops_dir / row["image"]).convert("RGB")
        rows.append(row)
    return rows


def run_model(model_dir: Path, crops: list[dict], max_crops: int | None) -> ModelResult:
    import torch

    result = ModelResult(name=model_dir.name)
    family, skip_reason = classify(model_dir)
    if skip_reason:
        result.skipped_reason = skip_reason
        return result
    result.family = family
    load_fn, read_fn = ADAPTERS[family]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        t0 = time.perf_counter()
        try:
            model, processor = load_fn(model_dir, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    CUDA out of memory; retrying {model_dir.name} on CPU (will be slow).")
            device = "cpu"
            model, processor = load_fn(model_dir, device)
        result.device = device
        result.load_s = round(time.perf_counter() - t0, 1)
    except Exception as exc:  # noqa: BLE001 - one bad model must not end the bake-off
        result.skipped_reason = f"failed to load: {type(exc).__name__}: {exc}"
        return result

    sample = crops[:max_crops] if max_crops else crops
    for row in sample:
        started = time.perf_counter()
        try:
            text = read_fn(model, processor, row["pil"])
            score = cer(text, row["text"])
        except Exception as exc:  # noqa: BLE001
            text, score = f"<error: {exc}>", None
        result.per_crop.append({
            "cell": row["cell"], "kind": row["kind"], "expected": row["text"],
            "predicted": text, "cer": score, "seconds": round(time.perf_counter() - started, 2),
        })

    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def to_markdown(results: list[ModelResult]) -> str:
    ranked = sorted((r for r in results if r.mean_cer is not None), key=lambda r: r.mean_cer)
    skipped = [r for r in results if r.skipped_reason]

    lines = [
        "# Recognizer bake-off",
        "",
        f"{len(results)} model(s) found under OCR_Models/; {len(ranked)} ran, {len(skipped)} skipped.",
        "",
        "One generic prompt for every model, 20 crops from one invoice. A useful first",
        "filter, not a verdict -- see the module docstring in tools/benchmark_recognizers.py.",
        "",
        "| Rank | Model | Family | Device | Mean CER | Exact match | Mean s/crop | Load s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.name} | {r.family} | {r.device} | {r.mean_cer} | "
            f"{r.exact_match_rate} | {r.mean_seconds} | {r.load_s} |"
        )

    if skipped:
        lines += ["", "## Skipped", "", "| Model | Reason |", "|---|---|"]
        for r in skipped:
            lines.append(f"| {r.name} | {r.skipped_reason} |")

    for r in ranked:
        lines += [f"\n## {r.name}\n", "| Cell | Kind | Expected | Predicted | CER |", "|---|---|---|---|---|"]
        for c in r.per_crop:
            lines.append(f"| {c['cell']} | {c['kind']} | {c['expected']} | {c['predicted']} | {c['cer']} |")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", default="datasets/crops")
    ap.add_argument("--models", default=str(MODELS_DIR))
    ap.add_argument("--only", nargs="*", help="model folder name(s) to run; default is every folder")
    ap.add_argument("--max-crops", type=int, default=None, help="cap crops per model, for a quick pass")
    ap.add_argument("--out", default="evaluation/reports/model_benchmark")
    args = ap.parse_args()

    crops_path = PROJECT_ROOT / args.crops
    if not (crops_path / "labels.jsonl").exists():
        print(f"No crops at {crops_path}. Run: python tools/export_crops.py --out {args.crops}", file=sys.stderr)
        return 2
    crops = load_crops(crops_path)
    print(f"{len(crops)} labelled crop(s) from {crops_path}.")

    models_dir = Path(args.models)
    names = args.only or sorted(p.name for p in models_dir.iterdir() if p.is_dir())

    results: list[ModelResult] = []
    for name in names:
        print(f"\n== {name}")
        result = run_model(models_dir / name, crops, args.max_crops)
        if result.skipped_reason:
            print(f"   skipped: {result.skipped_reason}")
        else:
            print(f"   {result.family} on {result.device}: mean CER {result.mean_cer}, "
                  f"exact match {result.exact_match_rate}, {result.mean_seconds}s/crop")
        results.append(result)

    report = to_markdown(results)
    out_dir = (PROJECT_ROOT / args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(PROJECT_ROOT / (args.out + ".md")).write_text(report, encoding="utf-8")
    Path(PROJECT_ROOT / (args.out + ".json")).write_text(
        json.dumps([
            {**r.__dict__, "mean_cer": r.mean_cer, "exact_match_rate": r.exact_match_rate,
             "mean_seconds": r.mean_seconds, "per_crop": r.per_crop}
            for r in results
        ], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nReport: {args.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
