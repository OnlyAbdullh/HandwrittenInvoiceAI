"""Stage 1 (service venv): geometry and layout once per labelled invoice.

Writes, under experiments/cache/:

    pages/<invoice>_gray.png    the OCR-photometric page the recognizer crops from
    pages/<invoice>_color.png   the geometrically corrected page the UI shows
    regions.json                every labelled cell (bbox, role, kind, label, row),
                                the item rows, and optionally Surya's text boxes

The labelling is `tools/export_crops.export_labelled` itself, so the cells and
their labels are exactly the benchmark's.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\prepare.py [--surya]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from common import CACHE, CROPS, PAGES, REGIONS, ROOT, utf8_console

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from evaluation.schema import InvoiceTruth  # noqa: E402
from ocr.cropping.padded_cropper import PaddedRegionCropper  # noqa: E402
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder  # noqa: E402
from table_extraction.classifiers.factory import build_layout_classifier  # noqa: E402
from table_extraction.factory import build_table_extractor  # noqa: E402
from tools.export_crops import PreparedPage, export_labelled  # noqa: E402


@dataclass
class RecordingPage(PreparedPage):
    """A PreparedPage that remembers which region each exported crop came from."""

    written: list = field(default_factory=list)

    def write_crop(self, region, stem, out_dir):
        name = super().write_crop(region, stem, out_dir)
        if name is not None:
            self.written.append(region)
        return name


def prepare(image, config):
    display = PreprocessingPipelineBuilder.build(config.preprocessing.pipeline_geometric_steps()).run(
        ImagePayload(image=image)
    )
    ocr_image = PreprocessingPipelineBuilder.build(config.preprocessing.ocr_photometric_steps).run(display)
    table_image = PreprocessingPipelineBuilder.build(config.preprocessing.table_photometric_steps).run(display)
    height, width = display.image.shape[:2]
    layout = build_layout_classifier(config.table_extraction).classify(
        build_table_extractor(config.table_extraction).extract(table_image), (width, height)
    )
    params = config.ocr.cropper.params if config.ocr.cropper else {}
    page = RecordingPage(layout=layout, ocr_image=ocr_image, cropper=PaddedRegionCropper(**params))
    return display, page


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--surya", action="store_true", help="also run Surya's text detector on each page (CPU)")
    args = ap.parse_args()

    config = load_config(str(ROOT / args.config))
    PAGES.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REGIONS.read_text(encoding="utf-8")) if REGIONS.exists() else {}
    out: dict = {}

    detector = None
    if args.surya:
        from ocr.detectors.surya_detector import SuryaTextDetector

        detector = SuryaTextDetector()

    for path in sorted((ROOT / "evaluation" / "ground_truth").glob("*.json")):
        truth = InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
        name = Path(truth.image).parent.name
        image = cv2.imread(str(ROOT / "bills" / truth.image))
        display, page = prepare(image, config)

        gray = page.ocr_image.image
        cv2.imwrite(str(PAGES / f"{name}_gray.png"), gray)
        cv2.imwrite(str(PAGES / f"{name}_color.png"), display.image)

        rows = export_labelled(page, truth, truth.image, name, CROPS / "base_export")
        cells = []
        for region, row in zip(page.written, rows):
            box = region.bbox
            cells.append({
                "cell": row["cell"], "role": row["role"], "kind": row["kind"], "text": row["text"],
                "bbox": [box.x, box.y, box.w, box.h], "row": region.row if row["role"] in
                ("product_name", "quantity", "unit_price", "line_total") else None,
                "export": row["image"],
            })

        entry = {
            "image": truth.image, "form": truth.form,
            "page": [display.image.shape[1], display.image.shape[0]],
            "cells": cells,
            "surya_boxes": existing.get(name, {}).get("surya_boxes"),
        }
        if detector is not None:
            started = time.perf_counter()
            found = detector.detect(ImagePayload(image=display.image))
            entry["surya_boxes"] = [
                [r.bbox.x, r.bbox.y, r.bbox.x + r.bbox.w, r.bbox.y + r.bbox.h, r.confidence] for r in found.regions
            ]
            entry["surya_seconds"] = round(time.perf_counter() - started, 2)
        out[name] = entry
        print(f"{name}: page {entry['page']}, {len(cells)} labelled cells"
              + (f", {len(entry['surya_boxes'])} Surya boxes" if entry["surya_boxes"] else ""))

    CACHE.mkdir(parents=True, exist_ok=True)
    REGIONS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
