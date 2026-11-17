"""Measure extraction accuracy against hand-labelled invoices.

    python tools/evaluate.py --config config/qwen_config.yaml
    python tools/evaluate.py --config config/qwen_config.yaml --images bills \
        --truth evaluation/ground_truth --label blur-off

Runs the same PipelineOrchestrator the service runs, with the request catalogs
from `evaluation/catalog.json`, and scores the **wire response** against each
truth file (see evaluation/metrics.py for what is counted). Writes a Markdown
and a JSON report under `evaluation/reports/`.

Run it before and after every change meant to improve accuracy. A change that
was not measured here has not been shown to help -- with three labelled invoices
it has barely been shown at all, so label more (README: "Labelling invoices").

    python tools/evaluate.py --perfect-readings datasets/sahror datasets/onlyone

runs the same pipeline with every cell "read" as its label from those
`tools/export_crops.py` folders, instead of by a model. What is still wrong
then is the layout, the parser or the arithmetic -- not recognition. No model
is loaded, so it runs anywhere.

    python tools/evaluate.py --config config/default_config.yaml --reading-cache evaluation/reading_cache

keeps every reading the model makes on disk and reads only crops it has not
read before (`ocr/recognizers/cached_recognizer.py`). After a change to the
parser or the arithmetic nothing is read again and the model is not loaded;
after a change to the layout only the crops that moved are. A run that was
stopped resumes from what it had read. Delete the folder after changing a
prompt or the decoding policy.

The JSON report lists every cell's raw reading and confidence under each
invoice's `readings`.

    python tools/evaluate.py --config config/default_config.yaml --held-out datasets/finetune --label base
    INVOICE_AI_MODEL_DIR=OCR_Models/invoice-ft-001 python tools/evaluate.py \\
        --config config/default_config.yaml --held-out datasets/finetune --label ft-001

scores only the invoices of a fine-tuning dataset's test split, which is the
only fair comparison of a fine-tuned model with the base one: any other
invoice is one it was trained or selected on (docs/fine-tuning.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from core.domain.ocr import Recognition  # noqa: E402
from core.domain.catalog import Catalogs, NamedEntry  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from evaluation.metrics import InvoiceScore, score_invoice, summarize, to_markdown  # noqa: E402
from evaluation.resources import process_tree_memory  # noqa: E402
from evaluation.schema import EvaluationCatalog, InvoiceTruth  # noqa: E402
from ocr.recognizers.registry import recognizer_registry  # noqa: E402
from orchestration.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402

PERFECT_READINGS = "perfect_readings"


class PerfectReadings:
    """Reads each crop as its label in an export folder: what the invoice
    would be if recognition were perfect. Only this tool registers it; the
    service can never select it."""

    name = PERFECT_READINGS
    labels: dict[str, str | None] = {}  # region id suffix -> label, for the invoice being run
    unlabelled: list[str] = []

    def __init__(self, **_):
        pass

    def read(self, crop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops) -> list[Recognition]:
        out = []
        for crop in crops:
            key = (crop.region.source_id or "").split(":")[-1]
            text = PerfectReadings.labels.get(key)
            if text is None:
                PerfectReadings.unlabelled.append(key)
            out.append(Recognition(text or "", confidence=1.0))
        return out


def load_export_labels(folder: Path) -> dict[str, str | None]:
    """Region id suffix (r3c1, u0, ...) -> label, from an export's labels.jsonl."""
    labels: dict[str, str | None] = {}
    for line in (folder / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[Path(row["image"]).stem.rsplit("_", 1)[-1]] = row.get("text")
    return labels


def load_catalogs(path: Path) -> Catalogs:
    catalog = EvaluationCatalog.model_validate_json(path.read_text(encoding="utf-8"))

    def entries(items):
        return [NamedEntry(name=i.name, entry_id=i.id, aliases=list(i.aliases)) for i in items]

    return Catalogs(
        merchants=entries(catalog.merchants),
        products=entries(catalog.products),
        cities=entries(catalog.cities),
    )


def raw_readings(run) -> dict[str, str | None]:
    """What was read for the header fields whose wire value is normalized."""
    invoice = getattr(run.result, "invoice", None)
    if invoice is None:
        return {}

    def text(value) -> str | None:
        if value.raw is not None:
            return value.raw
        return None if value.value is None else str(value.value)

    header = invoice.header
    return {
        "invoice_number": text(header.invoice_number),
        "date": text(header.invoice_date),
        "total": text(header.total_amount),
    }


def item_raw_readings(run) -> list[dict[str, str | None]]:
    """What was read for each line item's numbers, in wire order."""
    invoice = getattr(run.result, "invoice", None)
    if invoice is None:
        return []

    def text(value) -> str | None:
        if value.raw is not None:
            return value.raw
        return None if value.value is None else str(value.value)

    return [
        {"quantity": text(item.quantity), "unit_price": text(item.unit_price), "total_price": text(item.total_price)}
        for item in invoice.line_items
    ]


def all_readings(run) -> list[dict]:
    """Every box that was read: its cell, role, raw text and confidence."""
    out = []
    for matched in getattr(run.result, "elements", []) or []:
        element = matched.element
        if not element.fragments:
            continue
        confidences = [f.confidence for f in element.fragments if f.confidence is not None]
        out.append({
            "cell": element.id.split(":")[-1],
            "role": element.role.value,
            "text": element.merged_text,
            "confidence": round(min(confidences), 3) if confidences else None,
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/qwen_config.yaml")
    p.add_argument("--images", default="bills", help="directory the truth files' `image` paths are relative to")
    p.add_argument("--truth", default="evaluation/ground_truth")
    p.add_argument("--catalog", default="evaluation/catalog.json")
    p.add_argument("--out", default="evaluation/reports")
    p.add_argument("--label", default=None, help="name for this run in the report file name")
    p.add_argument("--perfect-readings", nargs="+", type=Path, metavar="EXPORT_DIR",
                   help="read every cell as its label from these export_crops folders (one per invoice, "
                        "named like the invoice's folder under bills/) instead of with the model")
    p.add_argument("--reading-cache", type=Path, metavar="DIR",
                   help="keep the model's readings in DIR and read only crops not read before")
    p.add_argument("--held-out", type=Path, metavar="DATASET_DIR",
                   help="score only the invoices in the test split of this fine-tuning dataset "
                        "(tools/build_finetune_dataset.py): the ones no fine-tuned model has seen")
    return p.parse_args()


def held_out_images(dataset_dir: Path) -> set[str]:
    """The truth `image` paths in a fine-tuning dataset's test split."""
    card = json.loads((dataset_dir / "dataset_card.json").read_text(encoding="utf-8"))
    return set(card["splits"]["test"]["sources"])


def utf8_console() -> None:
    """Print Arabic and symbols like "≥" even where the console or pipe is cp1252."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    utf8_console()
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    truths = [
        InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(Path(args.truth).glob("*.json"))
    ]
    if not truths:
        print(f"No truth files under {args.truth}", file=sys.stderr)
        return 2
    if args.held_out:
        kept = held_out_images(args.held_out)
        truths = [truth for truth in truths if truth.image in kept]
        if not truths:
            print(f"The test split of {args.held_out} holds no invoice with a truth file under {args.truth}. "
                  "With few invoices the split can be empty: label more, then rebuild the dataset.", file=sys.stderr)
            return 2
        print(f"Held-out invoices only: {len(truths)} of the test split of {args.held_out}.")

    config = load_config(args.config)
    # Score what the desktop app receives, whatever the config prints.
    config.output.formatter = "invoice_json"
    config.output.formatter_params = {}
    config.debug_dir = None

    catalogs = load_catalogs(Path(args.catalog))

    exports: dict[str, Path] = {}
    if args.perfect_readings:
        recognizer_registry.register(PERFECT_READINGS)(PerfectReadings)
        config.ocr.recognizer.name = PERFECT_READINGS
        config.ocr.recognizer.params = {}
        exports = {folder.name: folder for folder in args.perfect_readings}
        args.label = args.label or "perfect-readings"
    elif args.reading_cache:
        if config.ocr.recognizer is None:
            print("--reading-cache needs a flow that reads with `ocr.recognizer`.", file=sys.stderr)
            return 2
        wrapped = {"name": config.ocr.recognizer.name, "params": dict(config.ocr.recognizer.params)}
        config.ocr.recognizer.name = "cached"
        config.ocr.recognizer.params = {"recognizer": wrapped, "cache_dir": str(args.reading_cache)}

    started = time.perf_counter()
    orchestrator = PipelineOrchestrator(config)
    orchestrator.warmup()  # load the model now, so load time and per-invoice time are not mixed
    load_seconds = round(time.perf_counter() - started, 1)
    print(f"Pipeline built in {load_seconds:.1f}s ({orchestrator.engine_name}).")

    scores: list[InvoiceScore] = []
    readings_by_image: dict[str, list[dict]] = {}
    for truth in truths:
        image_path = Path(args.images) / truth.image
        print(f"- {truth.image} ...", end="", flush=True)

        image = cv2.imread(str(image_path))
        if image is None:
            scores.append(InvoiceScore(image=truth.image, form=truth.form, error=f"cannot read {image_path}"))
            print(" missing")
            continue

        if args.perfect_readings:
            folder = exports.get(Path(truth.image).parent.name)
            if folder is None:
                scores.append(InvoiceScore(image=truth.image, form=truth.form,
                                           error=f"no export folder named {Path(truth.image).parent.name!r}"))
                print(" no export folder")
                continue
            PerfectReadings.labels = load_export_labels(folder)
            PerfectReadings.unlabelled = []

        t0 = time.perf_counter()
        try:
            run = orchestrator.run(
                ImagePayload(image=image), catalogs=catalogs, write_debug_images=False, persist=False
            )
        except Exception as exc:  # noqa: BLE001 - one failure must not end the evaluation
            scores.append(InvoiceScore(image=truth.image, form=truth.form, error=f"{type(exc).__name__}: {exc}"))
            print(f" FAILED ({exc})")
            continue
        seconds = time.perf_counter() - t0

        score = score_invoice(
            run.output, truth, seconds=seconds, model_calls=run.timings.get("crop_count"),
            readings=raw_readings(run), item_readings=item_raw_readings(run),
        )
        scores.append(score)
        readings_by_image[truth.image] = all_readings(run)
        print(f" {seconds:.1f}s")
        if args.perfect_readings and PerfectReadings.unlabelled:
            print(f"    read as empty (no label in the export): {', '.join(PerfectReadings.unlabelled)}")

    # Measured while the model (and any llama-server or worker it started) is
    # still alive: a child's peak cannot be read once it has exited.
    run_info = {"load_seconds": load_seconds, "engine": orchestrator.engine_name, **process_tree_memory()}
    recognizer = getattr(orchestrator.flow, "recognizer", None)
    if args.reading_cache and hasattr(recognizer, "misses"):
        # Seconds per invoice mean nothing for a run answered from the cache.
        run_info["readings_from_cache"] = recognizer.hits
        run_info["readings_made"] = recognizer.misses
        print(f"Readings: {recognizer.misses} made by the model, {recognizer.hits} from {args.reading_cache}.")

    config_name = Path(args.config).name
    report = to_markdown(scores, config_name, run_info)

    # Written before anything is printed: a console that cannot show a
    # character must never cost the measurement (a Windows pipe is cp1252).
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{Path(args.config).stem}" + (f"-{args.label}" if args.label else "")
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "config": config_name,
                "run": run_info,
                "summary": summarize(scores),
                "invoices": [
                    {**dataclasses.asdict(s), "complete": s.complete, "complete_without_review": s.complete_without_review,
                     "readings": readings_by_image.get(s.image, [])}
                    for s in scores
                ],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print()
    print(report)
    print(f"\nReport: {out_dir / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
