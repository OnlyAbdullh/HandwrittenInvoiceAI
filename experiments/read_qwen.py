"""Stage 3 (VLM venv): read crop variants with a transformers Qwen model.

Uses the service's own runtime (`tools/qwen_worker.QwenRuntime`): same prompts,
same per-kind constrained decoding, same image fitting. One crop per call, so a
reading never depends on what else was in its batch and the cache stays valid.

Every reading is cached by (crop bytes, kind, prompt, settings); a variant whose
crops were already read costs nothing.

    D:\\OCR\\.venv_vlm\\Scripts\\python.exe experiments\\read_qwen.py --variants base pad20
    ... --model OCR_Models/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3 --tag qwen25
    ... --variants page --max-side 1536 --suffix s1536
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import CROPS, PRED, ROOT, ReadCache, manifest, pipeline_prompts, stop_requested, utf8_console, write_jsonl

# Whole rows and whole pages are not content kinds of the service; they get a
# transcription prompt and a token budget sized for what they hold.
EXTRA_PROMPTS = {
    "row": "هذه صورة سطر واحد من جدول فاتورة مكتوب بخط اليد. اكتب كل ما في السطر من اليمين إلى اليسار: "
           "اسم المنتج ثم العدد ثم السعر الإفرادي ثم السعر الإجمالي، واكتب الأرقام بالأرقام الإنجليزية 0-9.",
    "page": "اقرأ النص في هذه الصورة كاملاً من البداية إلى النهاية.",
}
EXTRA_GENERATION = {
    "row": {"max_new_tokens": 96, "repetition_penalty": 1.05},
    "page": {"max_new_tokens": 700, "repetition_penalty": 1.05},
}


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OCR_Models/Arabic-Qwen3.5-OCR-v4")
    ap.add_argument("--tag", default="qwen35")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--max-side", type=int, default=None, help="longest side sent to the model (service: 768)")
    ap.add_argument("--suffix", default="", help="appended to the variant name in the predictions folder")
    ap.add_argument("--unconstrained", action="store_true", help="no character-set constraint on numbers/dates")
    ap.add_argument("--kind-override", default=None, help="read every crop with this kind's prompt and decoding")
    ap.add_argument("--prompts-file", default=None, help="JSON {kind: prompt} overriding the service's prompts")
    args = ap.parse_args()

    import tools.qwen_worker as worker
    from PIL import Image

    if args.max_side:
        worker.MAX_SIDE = args.max_side
    worker.GENERATION.update(EXTRA_GENERATION)
    if args.unconstrained:
        for kind in ("number", "date"):
            worker.GENERATION[kind] = {k: v for k, v in worker.GENERATION[kind].items() if k != "charset"}
    prompts = {**pipeline_prompts(), **EXTRA_PROMPTS}
    if args.prompts_file:
        import json

        prompts.update(json.loads(Path(args.prompts_file).read_text(encoding="utf-8")))
    settings = f"max_side={worker.MAX_SIDE}|constrained={not args.unconstrained}|bs=1"

    cache = ReadCache(args.tag)
    pending = []
    for variant in args.variants:
        for row in manifest(variant):
            if row["path"] is None:
                continue
            kind = args.kind_override or row["kind"]
            row["_key"] = cache.key(row["sha1"], kind, prompts.get(kind, prompts["any"]), settings)
            if cache.get(row["_key"]) is None:
                pending.append((row, kind))

    runtime = None
    if pending and not stop_requested():
        started = time.perf_counter()
        runtime = worker.QwenRuntime(str(ROOT / args.model), max_new_tokens=1024)
        print(f"model loaded in {time.perf_counter() - started:.1f}s; {len(pending)} crop(s) to read")

    for index, (row, kind) in enumerate(pending):
        if stop_requested():
            print("experiments/STOP found: stopping; everything read so far is cached.")
            break
        if cache.get(row["_key"]) is not None:
            continue  # the same crop under another variant name
        image = Image.open(CROPS / row["path"]).convert("RGB")
        started = time.perf_counter()
        result = runtime.read_batch([{"image": image, "prompt": prompts.get(kind, prompts["any"]), "kind": kind}])[0]
        seconds = round(time.perf_counter() - started, 3)
        cache.put(row["_key"], {
            "text": result.get("text"), "confidence": result.get("confidence"), "min_prob": result.get("min_prob"),
            "error": result.get("error"), "seconds": seconds,
        })
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(pending)}")

    for variant in args.variants:
        rows = []
        for row in manifest(variant):
            kind = args.kind_override or row["kind"]
            reading = {"text": None, "confidence": None, "min_prob": None, "error": "no crop", "seconds": 0.0}
            if row["path"] is not None:
                key = cache.key(row["sha1"], kind, prompts.get(kind, prompts["any"]), settings)
                if cache.get(key) is None:
                    rows = None  # interrupted before this variant was fully read
                    break
                reading = {k: v for k, v in cache.get(key).items() if k != "key"}
            row.pop("_key", None)
            rows.append({**row, "read_kind": kind, "model": args.tag, "settings": settings, **{f"pred_{k}" if k == "text" else k: v for k, v in reading.items()}})
        out = PRED / args.tag / f"{variant}{('_' + args.suffix) if args.suffix else ''}.jsonl"
        if rows is None:
            print(f"{out.relative_to(ROOT)}: incomplete, not written (run the same command again to finish)")
            continue
        write_jsonl(out, rows)
        print(f"{out.relative_to(ROOT)}: {len(rows)} reading(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
