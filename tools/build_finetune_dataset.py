"""Build the recognizer's training set from every labelled invoice.

    python tools/build_finetune_dataset.py
    python tools/build_finetune_dataset.py --config config/default_config.yaml --out datasets/finetune

For every truth file under `evaluation/ground_truth/`, the photo goes through
the configured geometric correction, grid detection and classification -- the
service's own -- and each cell is cut exactly as the recognizer receives it
(`tools/export_crops.py`). Each labelled crop becomes one example:

    {"image": "images/abd_r3c0.png", "prompt": "<the service's prompt for a number>",
     "text": "٣,٤٠", "kind": "number", "role": "unit_price", "source": "abd/photo.jpg",
     "cell": "r3c0", "group": "invoice:abd", "split": "train"}

(`text` is in the digits the writer used; the truth file is typed in 0-9.)

written to `<out>/train.jsonl`, `val.jsonl` and `test.jsonl`, with the crops in
`<out>/images/` and a summary in `<out>/dataset_card.json`. Cells below the
last item that hold no writing but would still be sent to the recognizer are
included with the answer `EMPTY`.

An invoice is never divided between splits, nor are a writer's invoices (the
truth file's `writer`); a group keeps its split when more invoices are added.
`evaluation/finetune_dataset.py` has the rules and the reasons.

No model runs. Re-run it whenever a truth file is added or the layout or
cropper configuration changes: the crops are a product of that configuration,
and a model trained on one framing is tested on whatever the service cuts.

Then train: `tools/finetune_recognizer.py` (docs/fine-tuning.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_crops  # noqa: E402
from config.loader import load_config  # noqa: E402
from evaluation.finetune_dataset import SPLITS, SplitFractions, describe, group_of, split_of, to_record  # noqa: E402
from evaluation.schema import InvoiceTruth  # noqa: E402
from ocr.engines.qwen_vlm_engine import PROMPTS  # noqa: E402


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def prompt_for(kind: str) -> str:
    return PROMPTS.get(kind, PROMPTS["any"])


def examples_of(truth: InvoiceTruth, config, images_dir: Path, out_dir: Path) -> tuple[list[dict], str | None]:
    """One invoice's exported rows (labelled and empty), or why it gave none."""
    photo = images_dir / truth.image
    data = photo.read_bytes() if photo.is_file() else b""
    # imdecode rather than imread: imread cannot open a non-ASCII path on Windows.
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        return [], f"cannot read {photo}"

    stem = Path(truth.image).parent.name or Path(truth.image).stem
    page = export_crops.prepare_page(image, config)
    rows = export_crops.export_labelled(page, truth, truth.image, stem, out_dir)
    rows += export_crops.export_empty(page, truth, truth.image, stem, out_dir)
    if not any(row.get("text") or row.get("empty") for row in rows):
        return [], "no cell could be labelled (the layout found no field the truth has a value for)"
    return rows, None


def build(config_path: str, truth_dir: Path, images_dir: Path, out_dir: Path,
          fractions: SplitFractions, seed: str) -> dict:
    config = load_config(config_path)

    # Replaced whole: a crop left over from an earlier layout, or from a truth
    # file since deleted, would sit in images/ looking like data.
    if (out_dir / "images").is_dir():
        shutil.rmtree(out_dir / "images")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped: dict[str, str] = {}
    unlabelled = 0
    for path in sorted(truth_dir.glob("*.json")):
        truth = InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
        print(f"- {truth.image}")
        rows, problem = examples_of(truth, config, images_dir, out_dir)
        if problem:
            skipped[truth.image] = problem
            print(f"  skipped: {problem}")
            continue

        invoice = Path(truth.image).parent.name or Path(truth.image).stem
        group = group_of(truth.writer, invoice)
        split = split_of(group, fractions, seed)
        made = [to_record(row, prompt_for(row["kind"]), group, split, truth.digits) for row in rows]
        unlabelled += sum(record is None for record in made)
        kept = [record for record in made if record is not None]
        records.extend(kept)
        empty = sum(bool(row.get("empty")) for row in rows)
        print(f"  {len(kept)} example(s) -> {split} ({group})"
              + (f"; {empty} of them cells below the last item, answered EMPTY -- "
                 "they hold writing only if the truth is missing an item" if empty else ""))

    for split in SPLITS:
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as sink:
            for record in records:
                if record["split"] == split:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    card = describe(records, skipped)
    card["config"] = Path(config_path).name
    card["unlabelled_crops_left_out"] = unlabelled
    (out_dir / "dataset_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default_config.yaml",
                    help="the configuration the service runs with: it decides how every crop is cut")
    ap.add_argument("--images", default="bills", help="directory the truth files' `image` paths are relative to")
    ap.add_argument("--truth", default="evaluation/ground_truth")
    ap.add_argument("--out", default="datasets/finetune")
    ap.add_argument("--val", type=float, default=0.1, help="share of groups held out for validation")
    ap.add_argument("--test", type=float, default=0.1, help="share of groups held out for the final test")
    ap.add_argument("--seed", default="", help="changes which groups are held out; keep it fixed between runs")
    args = ap.parse_args(argv)

    fractions = SplitFractions(train=1.0 - args.val - args.test, val=args.val, test=args.test)
    card = build(args.config, Path(args.truth), Path(args.images), Path(args.out), fractions, args.seed)

    print(f"\n{card['cells']} example(s) from {card['groups']} group(s) written to {args.out}")
    for split in SPLITS:
        part = card["splits"][split]
        print(f"  {split:5s} {part['cells']:5d} cell(s), {len(part['groups'])} group(s), {part['by_kind']}")
    for warning in card["warnings"]:
        print(f"  ! {warning}")
    return 0 if card["cells"] else 1


if __name__ == "__main__":
    sys.exit(main())
