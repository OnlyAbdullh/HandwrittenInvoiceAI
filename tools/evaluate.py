"""Measure extraction accuracy against hand-labelled invoices.

    python tools/evaluate.py --config config/qwen_config.yaml
    python tools/evaluate.py --config config/qwen_config.yaml --images bills \
        --truth evaluation/ground_truth --label blur-off

Runs the same PipelineOrchestrator the service runs, with the request catalogs
from `evaluation/catalog.json`, and scores the **wire response** against each
truth file (see evaluation/metrics.py for what is counted). Writes a Markdown
and a JSON report under `evaluation/reports/`.

Run it before and after every change meant to improve accuracy. A change that
was not measured here has not been shown to help -- with two labelled invoices
it has barely been shown at all, so label more (README: "Labelling invoices").
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
from core.domain.catalog import Catalogs, NamedEntry  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from evaluation.metrics import InvoiceScore, score_invoice, summarize, to_markdown  # noqa: E402
from evaluation.schema import EvaluationCatalog, InvoiceTruth  # noqa: E402
from orchestration.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402


def load_catalogs(path: Path) -> Catalogs:
    catalog = EvaluationCatalog.model_validate_json(path.read_text(encoding="utf-8"))

    def entries(items):
        return [NamedEntry(name=i.name, entry_id=i.id, aliases=list(i.aliases)) for i in items]

    return Catalogs(
        merchants=entries(catalog.merchants),
        products=entries(catalog.products),
        cities=entries(catalog.cities),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/qwen_config.yaml")
    p.add_argument("--images", default="bills", help="directory the truth files' `image` paths are relative to")
    p.add_argument("--truth", default="evaluation/ground_truth")
    p.add_argument("--catalog", default="evaluation/catalog.json")
    p.add_argument("--out", default="evaluation/reports")
    p.add_argument("--label", default=None, help="name for this run in the report file name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    truths = [
        InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(Path(args.truth).glob("*.json"))
    ]
    if not truths:
        print(f"No truth files under {args.truth}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    # Score what the desktop app receives, whatever the config prints.
    config.output.formatter = "invoice_json"
    config.output.formatter_params = {}
    config.debug_dir = None

    catalogs = load_catalogs(Path(args.catalog))

    started = time.perf_counter()
    orchestrator = PipelineOrchestrator(config)
    print(f"Pipeline built in {time.perf_counter() - started:.1f}s ({orchestrator.engine_name}).")

    scores: list[InvoiceScore] = []
    for truth in truths:
        image_path = Path(args.images) / truth.image
        print(f"- {truth.image} ...", end="", flush=True)

        image = cv2.imread(str(image_path))
        if image is None:
            scores.append(InvoiceScore(image=truth.image, form=truth.form, error=f"cannot read {image_path}"))
            print(" missing")
            continue

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

        score = score_invoice(run.output, truth, seconds=seconds, model_calls=run.timings.get("crop_count"))
        scores.append(score)
        print(f" {seconds:.1f}s")

    config_name = Path(args.config).name
    report = to_markdown(scores, config_name)
    print()
    print(report)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{Path(args.config).stem}" + (f"-{args.label}" if args.label else "")
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {"config": config_name, "summary": summarize(scores), "invoices": [dataclasses.asdict(s) for s in scores]},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {out_dir / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
