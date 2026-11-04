"""Record what the table extractor finds on a real photo, for the layout tests.

    python tools/record_layout_fixture.py config/default_config.yaml bills/onlyone/img.png tests/fixtures/form_b_cells.json

Runs the configured geometric and table-photometric preprocessing and the
configured extractor -- exactly what the pipeline does before classification --
and writes the cells it found as JSON. tests/test_real_form_layouts.py then
classifies that geometry without the photo or any model, so the classifier's
rules are pinned against boxes a camera really produced.

Re-record after changing the extractor or a geometric step: the fixture is
the geometry *those* produce, and an old one silently tests a page the
pipeline no longer sees. The fixture holds box coordinates only -- no pixels,
no text read from the invoice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder  # noqa: E402
from table_extraction.factory import build_table_extractor  # noqa: E402


def record(config_path: str, image_path: str, source_label: str | None = None) -> dict:
    config = load_config(config_path)
    image = cv2.imdecode(__import__("numpy").fromfile(image_path, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {image_path}")

    page = PreprocessingPipelineBuilder.build(config.preprocessing.geometric_steps).run(ImagePayload(image=image))
    table_input = PreprocessingPipelineBuilder.build(config.preprocessing.table_photometric_steps).run(page)
    result = build_table_extractor(config.table_extraction).extract(table_input)

    height, width = page.image.shape[:2]
    return {
        "source": source_label or Path(image_path).as_posix(),
        "image_size": [width, height],
        "extractor": result.extractor_name,
        "tables": [
            {
                "table_id": t.table_id,
                "bbox": {"x": t.bbox.x, "y": t.bbox.y, "w": t.bbox.w, "h": t.bbox.h},
                "cells": [
                    {
                        "bbox": {"x": c.bbox.x, "y": c.bbox.y, "w": c.bbox.w, "h": c.bbox.h},
                        "row": c.row, "col": c.col, "row_span": c.row_span, "col_span": c.col_span,
                    }
                    for c in t.cells
                ],
            }
            for t in result.tables
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("image")
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--source", help="what to record as the photo's name (default: the image path)")
    args = ap.parse_args(argv)

    data = record(args.config, args.image, args.source)
    args.fixture.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    cells = sum(len(t["cells"]) for t in data["tables"])
    print(f"{cells} cell(s) in {len(data['tables'])} table(s), page {data['image_size']}, -> {args.fixture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
