"""Export labelled cell crops: the dataset for fine-tuning the recognizer.

    python tools/export_crops.py --out datasets/crops

For every truth file under `evaluation/ground_truth/`, the page is run through
the same geometric correction, grid detection and classification the service
uses. Each non-blank cell is cropped exactly as the recognizer would see it
(same padding, upscale and grayscale image) and paired with the text the paper
says there, taken from the truth file. Writes `<out>/images/*.png` and
`<out>/labels.jsonl`, one JSON object per line:

    {"image": "images/sahror_r3c1.png", "text": "212000", "kind": "number",
     "role": "line_total", "source": "sahror/img.png", "cell": "r3c1"}

Dates are not exported: the truth stores the ISO date, but the paper says
"7/8/2026", and a crop labelled with text it does not show teaches the model to
emit a format it cannot see. Add a `date_as_written` field to the truth first.

Items are assigned to the non-blank item rows in order, top to bottom, which is
how a handwritten invoice is filled in. A row count that disagrees with the
truth's item count is reported and that invoice skipped, rather than exporting
labels pasted onto the wrong cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from core.domain.ocr import TextRegion  # noqa: E402
from core.domain.roles import CellRole  # noqa: E402
from evaluation.schema import InvoiceTruth  # noqa: E402
from ocr.cropping.padded_cropper import PaddedRegionCropper  # noqa: E402
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder  # noqa: E402
from table_extraction.classifiers.factory import build_layout_classifier  # noqa: E402
from table_extraction.factory import build_table_extractor  # noqa: E402

ITEM_ROLES = {
    CellRole.PRODUCT_NAME: "product_name",
    CellRole.QUANTITY: "quantity",
    CellRole.UNIT_PRICE: "unit_price",
    CellRole.LINE_TOTAL: "total_price",
}


def _number(value: float | None) -> str | None:
    if value is None:
        return None
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def header_labels(truth: InvoiceTruth) -> dict[CellRole, str | None]:
    return {
        CellRole.INVOICE_NUMBER: truth.invoice_number,
        CellRole.CITY: truth.city,
        CellRole.MERCHANT_NAME: truth.customer_name,
        CellRole.TOTAL_AMOUNT: _number(truth.total),
        CellRole.TOTAL_IN_FIGURES: _number(truth.total),
    }


def export_one(truth: InvoiceTruth, config, images_dir: Path, out_dir: Path) -> list[dict]:
    image = cv2.imread(str(images_dir / truth.image))
    if image is None:
        print(f"  skipped: cannot read {images_dir / truth.image}")
        return []

    display = PreprocessingPipelineBuilder.build(config.preprocessing.geometric_steps).run(
        ImagePayload(image=image)
    )
    ocr_image = PreprocessingPipelineBuilder.build(config.preprocessing.ocr_photometric_steps).run(display)
    table_image = PreprocessingPipelineBuilder.build(config.preprocessing.table_photometric_steps).run(display)

    height, width = display.image.shape[:2]
    layout = build_layout_classifier(config.table_extraction).classify(
        build_table_extractor(config.table_extraction).extract(table_image), (width, height)
    )

    cropper_params = config.ocr.cropper.params if config.ocr.cropper else {}
    cropper = PaddedRegionCropper(**cropper_params)

    def is_blank(region) -> bool:
        return cropper.min_ink_ratio > 0 and cropper._is_blank(ocr_image.image, region.bbox)

    # Item rows that hold writing, top to bottom.
    rows: dict[int, dict[CellRole, object]] = {}
    for region in layout.regions:
        if region.role in ITEM_ROLES and region.row is not None:
            rows.setdefault(region.row, {})[region.role] = region
    filled_rows = [
        row for _, row in sorted(rows.items())
        if any(not is_blank(region) for region in row.values())
    ]

    if len(filled_rows) != len(truth.items):
        print(
            f"  skipped: {len(filled_rows)} filled item row(s) detected but the truth has "
            f"{len(truth.items)}; fix the layout or the truth before exporting."
        )
        return []

    labelled: list[tuple[object, str, str]] = []
    for region in layout.regions:
        text = header_labels(truth).get(region.role)
        if text:
            labelled.append((region, text, region.role.value))
    for row, item in zip(filled_rows, truth.items):
        for role, field_name in ITEM_ROLES.items():
            region = row.get(role)
            value = item.product_name if field_name == "product_name" else _number(getattr(item, field_name))
            if region is not None and value:
                labelled.append((region, value, role.value))

    rows_out: list[dict] = []
    stem = Path(truth.image).parent.name or Path(truth.image).stem
    for region, text, role in labelled:
        if is_blank(region):
            continue  # a label for a cell with no ink would teach the model to hallucinate
        crops = cropper.crop(ocr_image, [TextRegion(bbox=region.bbox, content_kind=region.content_kind)])
        if not crops:
            continue

        cell = region.id.split(":")[-1]
        name = f"images/{stem}_{cell}.png"
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        Image.fromarray(crops[0].image).save(out_dir / name)
        rows_out.append({
            "image": name,
            "text": text,
            "kind": region.content_kind.value,
            "role": role,
            "source": truth.image,
            "cell": cell,
        })
    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/qwen_config.yaml")
    ap.add_argument("--images", default="bills")
    ap.add_argument("--truth", default="evaluation/ground_truth")
    ap.add_argument("--out", default="datasets/crops")
    args = ap.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with (out_dir / "labels.jsonl").open("w", encoding="utf-8") as sink:
        for path in sorted(Path(args.truth).glob("*.json")):
            truth = InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
            print(f"- {truth.image}")
            for row in export_one(truth, config, Path(args.images), out_dir):
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n{total} labelled crop(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
