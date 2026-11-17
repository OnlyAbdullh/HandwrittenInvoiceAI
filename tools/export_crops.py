"""Export cell crops exactly as the recognizer sees them.

Two uses:

**A labelled dataset** -- for fine-tuning, and for scoring models by error rate::

    python tools/export_crops.py --out datasets/crops

For every truth file under `evaluation/ground_truth/`, the page is run through
the same geometric correction, grid detection and classification the service
uses. Each non-blank cell is cropped exactly as the recognizer would see it
(same padding, upscale and grayscale image) and paired with the text the paper
says there, taken from the truth file.

**One photo of your own** -- to compare models on it::

    python tools/export_crops.py --photo C:\\invoices\\my_invoice.jpg --out datasets/my_invoice
    python tools/export_crops.py --photo my_invoice.jpg --truth-file my_invoice.json --out datasets/my_invoice

Every cell the pipeline would read is exported. Without a truth file each label's
``text`` is null: `tools/benchmark_recognizers.py` then shows every model's
reading side by side for you to judge, and you can type the right answers into
``labels.jsonl`` to have them scored. With a truth file (the format of
`evaluation/ground_truth/*.json`; its ``image`` field is ignored) the crops are
labelled as in the dataset mode.

Both write `<out>/images/*.png` and `<out>/labels.jsonl`, one JSON object per line:

    {"image": "images/sahror_r3c1.png", "text": "212000", "kind": "number",
     "role": "line_total", "source": "sahror/img.png", "cell": "r3c1"}

A label is what the crop shows, not the normalized value. A number typed in the
truth file as a string ("5,70") is labelled exactly so; one given as a JSON
number gets its shortest spelling ("5.7"), which is wrong for a cell showing
٥,٧٠ -- see `evaluation/schema.py`. A date is labelled day/month/year
("7/8/2026"), how these forms are filled in, unless the truth gives
`date_written`.

`tools/build_finetune_dataset.py` turns these exports into a training set.

Items are assigned to the non-blank item rows in order, top to bottom, which is
how a handwritten invoice is filled in. A row count that disagrees with the
truth's item count is reported and that invoice skipped, rather than exporting
labels pasted onto the wrong cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from core.domain.layout import InvoiceLayout, LayoutRegion  # noqa: E402
from core.domain.ocr import TextRegion  # noqa: E402
from core.domain.roles import CellRole  # noqa: E402
from evaluation.schema import InvoiceTruth, plain_number  # noqa: E402
from ocr.cropping.padded_cropper import PaddedRegionCropper  # noqa: E402
from orchestration.flows.layout_driven import SKIPPED_ROLES  # noqa: E402
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
    """A truth amount's shortest spelling (see `evaluation.schema.plain_number`).

    A label is this only when the truth does not say how the number is written;
    a truth file gives the written form by typing the number as a string.
    """
    return plain_number(value)


def _written_date(iso: str | None) -> str | None:
    """A truth date (ISO) as these forms carry it: day/month/year."""
    return InvoiceTruth(image="", date=iso).date_text if iso else None


def header_labels(truth: InvoiceTruth) -> dict[CellRole, str | None]:
    return {
        CellRole.INVOICE_NUMBER: truth.invoice_number,
        CellRole.INVOICE_DATE: truth.date_text,
        CellRole.CITY: truth.city,
        CellRole.MERCHANT_NAME: truth.customer_name,
        CellRole.TOTAL_AMOUNT: truth.total_text,
        CellRole.TOTAL_IN_FIGURES: truth.total_text,
    }


@dataclass
class PreparedPage:
    """One photo after the pipeline's geometry, layout and cropping setup."""

    layout: InvoiceLayout
    ocr_image: ImagePayload
    cropper: PaddedRegionCropper
    scaled: dict | None = None

    def is_blank(self, region: LayoutRegion) -> bool:
        return self.cropper.min_ink_ratio > 0 and self.cropper._is_blank(self.ocr_image.image, region.bbox)

    def write_crop(self, region: LayoutRegion, stem: str, out_dir: Path) -> str | None:
        """Save the region's crop as the recognizer would get it; its relative path, or None."""
        crops = self.cropper.crop(self.ocr_image, [TextRegion(bbox=region.bbox, content_kind=region.content_kind)])
        if not crops:
            return None
        name = f"images/{stem}_{region.id.split(':')[-1]}.png"
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        Image.fromarray(crops[0].image).save(out_dir / name)
        return name


def prepare_page(image, config) -> PreparedPage:
    display = PreprocessingPipelineBuilder.build(config.preprocessing.pipeline_geometric_steps()).run(
        ImagePayload(image=image)
    )
    ocr_image = PreprocessingPipelineBuilder.build(config.preprocessing.ocr_photometric_steps).run(display)
    table_image = PreprocessingPipelineBuilder.build(config.preprocessing.table_photometric_steps).run(display)

    height, width = display.image.shape[:2]
    layout = build_layout_classifier(config.table_extraction).classify(
        build_table_extractor(config.table_extraction).extract(table_image), (width, height)
    )

    cropper_params = config.ocr.cropper.params if config.ocr.cropper else {}
    return PreparedPage(layout=layout, ocr_image=ocr_image, cropper=PaddedRegionCropper(**cropper_params),
                        scaled=display.metadata.get("working_resolution"))


def _row(name: str, text: str | None, region: LayoutRegion, role: str, source: str) -> dict:
    return {
        "image": name,
        "text": text,
        "kind": region.content_kind.value,
        "role": role,
        "source": source,
        "cell": region.id.split(":")[-1],
    }


def item_rows(page: PreparedPage) -> list[dict[CellRole, LayoutRegion]]:
    """The layout's item rows, top to bottom, each as role -> cell."""
    rows: dict[int, dict[CellRole, LayoutRegion]] = {}
    for region in page.layout.regions:
        if region.role in ITEM_ROLES and region.row is not None:
            rows.setdefault(region.row, {})[region.role] = region
    # By position: a completed grid's inferred rows are not numbered in page order.
    return [row for _, row in sorted(rows.items(), key=lambda kv: min(r.bbox.y for r in kv[1].values()))]


def _has_writing(page: PreparedPage, row: dict[CellRole, LayoutRegion]) -> bool:
    return any(not page.is_blank(region) for region in row.values())


def export_empty(page: PreparedPage, truth: InvoiceTruth, source: str, stem: str, out_dir: Path) -> list[dict]:
    """Crops of cells below the invoice's last item that hold no writing and
    are still not blank enough for the pipeline to skip.

    The service sends exactly these to the recognizer -- a dotted rule, the
    tail of the row above -- and a recognizer that answers them with digits
    invents a line item. Labelled `"text": ""` and `"empty": true`, they are
    what teaches it to say the cell is empty. Nothing is returned when the
    truth's items could not be paired with the first rows: which cells are
    empty is then not known.
    """
    ordered = item_rows(page)
    leading = ordered[: len(truth.items)]
    if len(leading) != len(truth.items) or not all(_has_writing(page, row) for row in leading):
        return []
    rows_out: list[dict] = []
    for row in ordered[len(leading):]:
        for region in row.values():
            if page.is_blank(region):
                continue
            name = page.write_crop(region, stem, out_dir)
            if name is not None:
                rows_out.append({**_row(name, "", region, region.role.value, source), "empty": True})
    return rows_out


def export_labelled(page: PreparedPage, truth: InvoiceTruth, source: str, stem: str, out_dir: Path) -> list[dict]:
    """Crops of the cells the truth file has a value for, labelled with it."""
    # Item rows, top to bottom. An invoice is filled from the top, so the
    # truth's N items belong to the first N rows -- provided each of them has
    # writing in it. Ink lower down does not make a row an item: on a form
    # whose writing sits on the row rules, letters and rule fragments spill
    # into the blank rows below, and no ink measure tells them apart from
    # faint writing (measured on both real forms).
    ordered = item_rows(page)
    leading = ordered[: len(truth.items)]
    if len(leading) == len(truth.items) and all(_has_writing(page, row) for row in leading):
        filled_rows = leading
        extra = sum(any(not page.is_blank(r) for r in row.values()) for row in ordered[len(leading):])
        if extra:
            print(f"  {extra} row(s) below the {len(leading)} item(s) show ink (stray marks or rule fragments); "
                  "not labelled.")
    else:
        filled_rows = [row for row in ordered if any(not page.is_blank(region) for region in row.values())]

    if len(filled_rows) != len(truth.items):
        # The item rows cannot be paired with the truth's items without risking
        # labels pasted onto the wrong cells. The header fields do not depend
        # on that pairing, so they keep their labels; item cells go out
        # unlabelled rather than not at all.
        print(
            f"  {len(filled_rows)} filled item row(s) detected but the truth has {len(truth.items)}: "
            "item cells exported unlabelled, header fields labelled."
        )
        headers = header_labels(truth)
        rows_out = export_unlabelled(page, source, stem, out_dir)
        for row in rows_out:
            row["text"] = headers.get(CellRole(row["role"]))
        return rows_out

    labelled: list[tuple[object, str, str]] = []
    for region in page.layout.regions:
        text = header_labels(truth).get(region.role)
        if text:
            labelled.append((region, text, region.role.value))
    for row, item in zip(filled_rows, truth.items):
        for role, field_name in ITEM_ROLES.items():
            region = row.get(role)
            value = item.product_name if field_name == "product_name" else item.text_of(field_name)
            if region is not None and value:
                labelled.append((region, value, role.value))

    rows_out: list[dict] = []
    for region, text, role in labelled:
        if page.is_blank(region):
            continue  # a label for a cell with no ink would teach the model to hallucinate
        name = page.write_crop(region, stem, out_dir)
        if name is not None:
            rows_out.append(_row(name, text, region, role, source))
    return rows_out


def export_unlabelled(page: PreparedPage, source: str, stem: str, out_dir: Path) -> list[dict]:
    """Every cell the pipeline would send to the recognizer, with no label."""
    rows_out: list[dict] = []
    ordered = sorted(page.layout.regions, key=lambda r: (r.bbox.y, -r.bbox.x))
    for region in ordered:
        if region.role in SKIPPED_ROLES or page.is_blank(region):
            continue
        name = page.write_crop(region, stem, out_dir)
        if name is not None:
            rows_out.append(_row(name, None, region, region.role.value, source))
    return rows_out


def export_one(truth: InvoiceTruth, config, images_dir: Path, out_dir: Path) -> list[dict]:
    image = cv2.imread(str(images_dir / truth.image))
    if image is None:
        print(f"  skipped: cannot read {images_dir / truth.image}")
        return []
    stem = Path(truth.image).parent.name or Path(truth.image).stem
    return export_labelled(prepare_page(image, config), truth, truth.image, stem, out_dir)


def export_photo(photo: Path, config, out_dir: Path, truth: InvoiceTruth | None = None) -> list[dict]:
    # imdecode rather than imread: imread cannot open a non-ASCII path on Windows.
    data = photo.read_bytes() if photo.is_file() else b""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        print(f"  cannot read {photo}")
        return []
    page = prepare_page(image, config)
    if page.scaled and page.scaled.get("scale") != 1.0:
        (w0, h0), (w1, h1) = page.scaled["from"], page.scaled["to"]
        print(f"  photo {w0}x{h0}, read at {w1}x{h1} (the pipeline's working resolution)")
    print(f"  layout: {page.layout.source}, {len(page.layout.regions)} box(es), "
          f"{sum(r.role is not CellRole.UNKNOWN for r in page.layout.regions)} classified")
    if truth is not None:
        return export_labelled(page, truth, photo.name, photo.stem, out_dir)
    return export_unlabelled(page, photo.name, photo.stem, out_dir)


def existing_source(out_dir: Path) -> str | None:
    """The photo whose crops `out_dir` already holds, if any."""
    labels = out_dir / "labels.jsonl"
    if not labels.exists():
        return None
    for line in labels.read_text(encoding="utf-8").splitlines():
        try:
            return json.loads(line).get("source")
        except ValueError:
            continue
    return None


def find_truth_for(photo: Path, truth_dir: Path) -> Path | None:
    """The ground-truth file describing `photo`, if there is one.

    A truth file names its image relative to the bills folder
    ("onlyone/img.png"); the photo matches when its path ends with that.
    """
    if not truth_dir.is_dir():
        return None
    photo_parts = [part.lower() for part in photo.resolve().parts]
    for path in sorted(truth_dir.glob("*.json")):
        try:
            image = json.loads(path.read_text(encoding="utf-8")).get("image") or ""
        except (OSError, ValueError):
            continue
        wanted = [part.lower() for part in Path(image.replace("\\", "/")).parts]
        if wanted and photo_parts[-len(wanted):] == wanted:
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/qwen_config.yaml")
    ap.add_argument("--images", default="bills")
    ap.add_argument("--truth", default="evaluation/ground_truth")
    ap.add_argument("--photo", type=Path, help="export one photo of your own instead of the truth files")
    ap.add_argument("--truth-file", type=Path,
                    help="with --photo: label its crops from this truth JSON (found automatically in --truth "
                         "when one of its files names this photo, e.g. bills/onlyone/img.png)")
    ap.add_argument("--out", default="datasets/crops")
    ap.add_argument("--overwrite", action="store_true",
                    help="with --photo: replace crops of a different photo already in --out")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.photo is not None:
        clash = existing_source(out_dir)
        if clash and clash != args.photo.name and not args.overwrite:
            print(
                f"{out_dir} already holds crops of '{clash}', including any answers typed into its "
                f"labels.jsonl. Use another --out (e.g. datasets\\{args.photo.parent.name or args.photo.stem}), "
                "or --overwrite to replace them.",
                file=sys.stderr,
            )
            return 2
        if args.overwrite and (out_dir / "images").is_dir():
            # Replacing, not merging: stale crops of another photo would sit
            # in images/ unreferenced and confuse anyone labelling by hand.
            for stale in (out_dir / "images").glob("*.png"):
                stale.unlink()
        truth_path = args.truth_file or find_truth_for(args.photo, Path(args.truth))
        truth = (
            InvoiceTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
            if truth_path else None
        )
        print(f"- {args.photo}")
        if truth_path and not args.truth_file:
            print(f"  labelled from {truth_path} (its 'image' is this photo)")
        rows = export_photo(args.photo, config, out_dir, truth)
        if truth is not None and not rows:
            # The truth's rows did not line up with the detected table. The
            # crops are still worth comparing models on, just without scores.
            print("  falling back to an unlabelled export of every readable cell")
            truth = None
            rows = export_photo(args.photo, config, out_dir, None)
        with (out_dir / "labels.jsonl").open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        labelled = sum(row["text"] is not None for row in rows)
        print(f"\n{len(rows)} crop(s) written to {out_dir}, {labelled} of them labelled.")
        if labelled < len(rows):
            print("Unlabelled crops have text: null in labels.jsonl -- type the right answer in to have them scored.")
        return 0 if rows else 1

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
