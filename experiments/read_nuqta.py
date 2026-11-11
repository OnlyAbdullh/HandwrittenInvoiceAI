"""Stage 3 (service venv): read crop variants with Nuqta through llama.cpp.

Builds the service's own `llamacpp` recognizer from config/nuqta_config.yaml --
the model card's prompt, the projector's image floor, grammar-constrained
numbers -- so a reading here is the reading the service would produce.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\read_nuqta.py --variants base
    ... --quant Q8_0 --tag nuqta_q8
"""

from __future__ import annotations

import argparse
import json
import time

from PIL import Image

from common import CROPS, PRED, ROOT, ReadCache, manifest, stop_requested, utf8_console, write_jsonl

from config.loader import load_config  # noqa: E402
from ocr.recognizers.llamacpp_recognizer import LlamaCppRecognizer  # noqa: E402


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--tag", default="nuqta_q4")
    ap.add_argument("--kind-override", default=None)
    args = ap.parse_args()

    params = dict(load_config(str(ROOT / "config" / "nuqta_config.yaml")).ocr.recognizer.params)
    params["model_path"] = f"OCR_Models/Nuqta-Arabic-OCR-4B-GGUF/nuqta-ocr-4b-{args.quant}.gguf"
    params["timeout_s"] = 600
    reading_settings = {k: v for k, v in params.items() if k not in ("model_path", "mmproj_path", "server_path", "gpu_layers", "timeout_s", "startup_timeout_s")}
    settings = json.dumps(reading_settings, sort_keys=True, ensure_ascii=False)
    prompts = params.get("prompts") or {}

    cache = ReadCache(args.tag)
    pending = []
    for variant in args.variants:
        for row in manifest(variant):
            if row["path"] is None:
                continue
            kind = args.kind_override or row["kind"]
            kind = kind if kind in ("number", "date", "code", "arabic_text") else "any"
            row["_key"] = cache.key(row["sha1"], kind, prompts.get(kind, ""), settings)
            if cache.get(row["_key"]) is None:
                pending.append((row, kind))

    if pending and not stop_requested():
        started = time.perf_counter()
        recognizer = LlamaCppRecognizer(**params)
        print(f"llama-server ready in {time.perf_counter() - started:.1f}s; {len(pending)} crop(s) to read")
        try:
            for index, (row, kind) in enumerate(pending):
                if stop_requested():
                    print("experiments/STOP found: stopping; everything read so far is cached.")
                    break
                if cache.get(row["_key"]) is not None:
                    continue
                image = Image.open(CROPS / row["path"]).convert("RGB")
                started = time.perf_counter()
                try:
                    reading = recognizer._backend.read(image, kind)
                    result = {"text": reading.text, "confidence": reading.confidence, "min_prob": None, "error": None}
                except Exception as exc:  # noqa: BLE001 - one crop must not end the run
                    result = {"text": None, "confidence": None, "min_prob": None, "error": f"{type(exc).__name__}: {exc}"}
                cache.put(row["_key"], {**result, "seconds": round(time.perf_counter() - started, 3)})
                if (index + 1) % 20 == 0:
                    print(f"  {index + 1}/{len(pending)}", flush=True)
        finally:
            recognizer.close()

    for variant in args.variants:
        rows = []
        for row in manifest(variant):
            kind = args.kind_override or row["kind"]
            kind = kind if kind in ("number", "date", "code", "arabic_text") else "any"
            reading = {"text": None, "confidence": None, "min_prob": None, "error": "no crop", "seconds": 0.0}
            if row["path"] is not None:
                cached = cache.get(cache.key(row["sha1"], kind, prompts.get(kind, ""), settings))
                if cached is None:
                    rows = None  # interrupted before this variant was fully read
                    break
                reading = {k: v for k, v in cached.items() if k != "key"}
            row.pop("_key", None)
            rows.append({**row, "read_kind": kind, "model": args.tag, "settings": settings,
                         **{f"pred_{k}" if k == "text" else k: v for k, v in reading.items()}})
        out = PRED / args.tag / f"{variant}.jsonl"
        if rows is None:
            print(f"{out.relative_to(ROOT)}: incomplete, not written (run the same command again to finish)")
            continue
        write_jsonl(out, rows)
        print(f"{out.relative_to(ROOT)}: {len(rows)} reading(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
