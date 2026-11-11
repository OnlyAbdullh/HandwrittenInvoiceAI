"""Stage 6 (service venv): the whole pipeline on cached readings, no model.

For every cell-level predictions file, each labelled cell is "read" as the
cached reading of that configuration and the real pipeline does the rest --
mapping, parsing, catalog matching, arithmetic reconciliation -- and the wire
response is scored exactly as tools/evaluate.py scores it. This answers what a
per-cell difference is worth once the arithmetic checks have had their say.

Cells the benchmark has no label for (unclassified boxes) are read as empty,
with one exception: form B's first item row has two boxes labelled line total,
and the benchmark labels the narrow fragment (r2c4). The real pipeline also
reads the full cell (r2c3); here it gets the configuration's reading of the
widened `colfix` crop when one was read, else the Qwen3.5 one.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\e2e_replay.py [--one-cell-per-column] [model/variant ...]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

from common import EXP, PRED, ROOT, read_jsonl, utf8_console

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from core.domain.ocr import Recognition  # noqa: E402
from core.domain.roles import ContentKind  # noqa: E402
from evaluation.metrics import score_invoice, summarize  # noqa: E402
from evaluation.schema import InvoiceTruth  # noqa: E402
from ocr.recognizers import output_validation  # noqa: E402
from ocr.recognizers.registry import recognizer_registry  # noqa: E402
from orchestration.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402
from tools.evaluate import item_raw_readings, load_catalogs, raw_readings  # noqa: E402

DUPLICATE_OF = {("onlyone", "r2c3"): "r2c4"}  # full cell -> the fragment the benchmark labelled


class Replay:
    """Reads each crop as a cached reading, validated as the service validates."""

    name = "replay"
    readings: dict[str, tuple[str, float | None]] = {}

    def __init__(self, **_):
        pass

    def read(self, crop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops) -> list[Recognition]:
        out = []
        for crop in crops:
            text, confidence = Replay.readings.get((crop.region.source_id or "").split(":")[-1], ("", None))
            kind = crop.content_kind or ContentKind.ANY
            cleaned = output_validation.clean(text or "", kind)
            if cleaned is None:
                out.append(Recognition("", confidence=0.0))
            elif not cleaned:
                out.append(Recognition(""))
            else:
                out.append(Recognition(cleaned, confidence=confidence))
        return out


def cached(model: str, variant: str) -> dict[tuple[str, str], tuple[str, float | None]]:
    return {(r["invoice"], r["cell"]): (r.get("pred_text") or "", r.get("confidence"))
            for r in read_jsonl(PRED / model / f"{variant}.jsonl")}


def main() -> int:
    utf8_console()
    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="model/variant; default: every cell-level predictions file")
    ap.add_argument("--one-cell-per-column", action="store_true", help="run with the layout's duplicate-cell rule on")
    ap.add_argument("--out", default="e2e_replay.csv")
    ap.add_argument("--details", action="store_true", help="print every wrong field")
    args = ap.parse_args()

    targets = [tuple(t.split("/", 1)) for t in args.targets] or [
        (p.parent.name, p.stem) for p in sorted(PRED.glob("*/*.jsonl"))
        if p.parent.name != "cache" and not p.stem.startswith(("row", "page", "ORACLE"))
    ]

    config = load_config(str(ROOT / "config" / "default_config.yaml"))
    config.output.formatter, config.output.formatter_params, config.debug_dir = "invoice_json", {}, None
    if args.one_cell_per_column:
        params = config.table_extraction.classifier_params
        params["one_cell_per_column"] = True
        for profile in params.get("profiles", []):
            profile["one_cell_per_column"] = True
    recognizer_registry.register(Replay.name)(Replay)
    config.ocr.recognizer.name, config.ocr.recognizer.params = Replay.name, {}
    orchestrator = PipelineOrchestrator(config)
    catalogs = load_catalogs(ROOT / "evaluation" / "catalog.json")

    truths = [InvoiceTruth.model_validate_json(p.read_text(encoding="utf-8"))
              for p in sorted((ROOT / "evaluation" / "ground_truth").glob("*.json"))]
    images = {t.image: cv2.imread(str(ROOT / "bills" / t.image)) for t in truths}
    widened_default = cached("qwen35", "colfix")

    table = []
    for model, variant in targets:
        readings = cached(model, variant)
        if not readings:
            continue
        widened = cached(model, "colfix") or widened_default
        scores = []
        for truth in truths:
            invoice = Path(truth.image).parent.name
            Replay.readings = {cell: value for (inv, cell), value in readings.items() if inv == invoice}
            for (inv, full), fragment in DUPLICATE_OF.items():
                if inv == invoice and (inv, fragment) in widened:
                    Replay.readings[full] = widened[(inv, fragment)]
            run = orchestrator.run(ImagePayload(image=images[truth.image]), catalogs=catalogs,
                                   write_debug_images=False, persist=False)
            scores.append(score_invoice(run.output, truth, seconds=0.0, model_calls=run.timings.get("crop_count"),
                                        readings=raw_readings(run), item_readings=item_raw_readings(run)))
        summary = {k: v for k, v in summarize(scores).items()
                   if k not in ("by_field", "mean_seconds", "max_seconds", "failed", "invoices")}
        wrong = []
        for score in scores:
            name = Path(score.image).parent.name
            wrong += [f"{name}:{f.name}" for f in score.header if not f.correct]
            wrong += [f"{name}:row{m.row}.{m.column}" for m in score.numeric_misses]
        table.append({"model": model, "variant": variant, "layout_rule": "one_cell_per_column" if args.one_cell_per_column else "baseline",
                      **summary, "wrong_fields": " ".join(wrong)})
        if args.details:
            print(model, variant, wrong)

    if not table:
        print("nothing to replay")
        return 1
    keys = list(table[0])
    out = EXP / args.out
    with out.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(table)} configuration(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
