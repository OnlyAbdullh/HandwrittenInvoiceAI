"""Stage 2 (service venv): cut every labelled cell under each crop variant.

A variant is a crop recipe -- which page image, what padding, what enlargement,
what photometric treatment. `base` reproduces the shipped cropper exactly
(gray page, pad 10, LANCZOS x2); each other variant changes one thing.

Writes experiments/cache/crops/<variant>/{<invoice>_<cell>.png, manifest.jsonl}.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\make_crops.py [variant ...]
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from PIL import Image

from common import CROPS, PAGES, REGIONS, sha1_file, utf8_console, write_jsonl

from config.schema import StepConfig  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder  # noqa: E402

# name -> recipe. Keys:
#   src    gray | color
#   page   a pipeline step applied to the whole page before cropping (as the service would)
#   pad    px around the box (negative: inset); pad_x / pad_y override per axis;
#          pad_fx / pad_fy are fractions of the box's own width / height
#   tight  "ink" (shrink to the handwriting inside the box) | "surya" (Surya's boxes in the cell)
#   up     LANCZOS enlargement factor; "h<N>" enlarges to a height of N px
#   post   a treatment of the crop itself
#   unit   cell | row | page
BASE = {"src": "gray", "pad": 10, "up": 2, "unit": "cell"}
PIPELINE_STEPS = {
    "clahe": ("clahe", {"clip_limit": 2.0, "tile_grid_size": [8, 8]}),
    "bilateral": ("bilateral_filter", {"d": 5, "sigma_color": 50, "sigma_space": 50}),
    "adaptive": ("adaptive_threshold", {"block_size": 51, "c": 35}),
    "blur_divide": ("illumination_normalization_blur_divide", {"blur_kernel": 95}),
    "median": ("median_blur", {"k": 3}),
}
VARIANTS: dict[str, dict] = {
    "base": {},
    # -- bounding box ------------------------------------------------------
    "pad0": {"pad": 0},
    "pad4": {"pad": 4},
    "pad20": {"pad": 20},
    "pad30": {"pad": 30},
    "inset6": {"pad": -6},
    "expand_x25": {"pad_fx": 0.25},
    "expand_y25": {"pad_fy": 0.25},
    "expand_y40": {"pad_fy": 0.40},
    "ink_tight8": {"tight": "ink", "pad": 8},
    "ink_tight16": {"tight": "ink", "pad": 16},
    "ink_tight24": {"tight": "ink", "pad": 24},
    "surya_tight10": {"tight": "surya", "pad": 10},
    # Item cells widened to their column's own extent (median edges over the
    # rows): repairs a cell the grid completion left narrower than its column.
    "colfix": {"colfix": True},
    "colfix_pad20": {"colfix": True, "pad": 20},
    # -- resolution --------------------------------------------------------
    "up1": {"up": 1},
    "up3": {"up": 3},
    "up4": {"up": 4},
    "h128": {"up": "h128"},
    "h256": {"up": "h256"},
    "up2_cubic": {"resample": "bicubic"},
    # -- photometric -------------------------------------------------------
    "color": {"src": "color"},
    "clahe": {"page": "clahe"},
    "clahe_crop": {"post": "clahe"},
    "bilateral": {"page": "bilateral"},
    "median": {"page": "median"},
    "adaptive": {"page": "adaptive"},
    "blur_divide": {"page": "blur_divide"},
    "otsu": {"post": "otsu"},
    "nlm": {"post": "nlm"},
    "sharpen": {"post": "sharpen"},
    "stretch": {"post": "stretch"},
    "norules": {"post": "norules"},
    # -- larger units ------------------------------------------------------
    "row": {"unit": "row"},
    "row_up1": {"unit": "row", "up": 1},
    "row_color": {"unit": "row", "src": "color"},
    "page": {"unit": "page", "src": "color", "pad": 0, "up": 1},
}


def ink_mask(gray: np.ndarray, delta: int = 30) -> np.ndarray:
    """Handwriting-sized ink, the shipped blank-cell gate's definition: darker
    than the paper, with straight rules removed."""
    paper = np.percentile(gray, 85)
    mask = (gray < paper - delta).astype(np.uint8)
    if not mask.any():
        return mask
    h, w = mask.shape
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.4 * w)), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(0.5 * h))))
    rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal) | cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical)
    return mask & (1 - rules)


def ink_box(gray: np.ndarray, box: list[int], search_pad: int = 12) -> list[int] | None:
    """The handwriting's own bounding box in and just around a cell.

    Searches `search_pad` px beyond the box so a stroke that crosses the rule is
    followed rather than cut; specks (dotted rules) are ignored.
    """
    x, y, w, h = box
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = max(0, x - search_pad), max(0, y - search_pad), min(W, x + w + search_pad), min(H, y + h + search_pad)
    strokes = ink_mask(gray[y1:y2, x1:x2])
    count, _, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    keep = [s for s in stats[1:count] if s[cv2.CC_STAT_AREA] >= 12 and max(s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT]) >= 6]
    if not keep:
        return None
    left = min(s[cv2.CC_STAT_LEFT] for s in keep)
    top = min(s[cv2.CC_STAT_TOP] for s in keep)
    right = max(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in keep)
    bottom = max(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in keep)
    return [x1 + left, y1 + top, right - left, bottom - top]


def surya_box(box: list[int], surya_boxes: list[list]) -> list[int] | None:
    """Surya's text boxes clipped to the cell, merged -- the detector flow's
    `assign_to_layout`. None when Surya found no text in the cell."""
    x, y, w, h = box
    pieces = []
    for sx1, sy1, sx2, sy2, _ in surya_boxes or []:
        ix1, iy1, ix2, iy2 = max(x, sx1), max(y, sy1), min(x + w, sx2), min(y + h, sy2)
        if ix2 - ix1 < 6 or iy2 - iy1 < 6:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < 0.2 * (sx2 - sx1) * (sy2 - sy1) and (ix2 - ix1) * (iy2 - iy1) < 0.2 * w * h:
            continue  # a neighbour's box grazing this cell
        pieces.append((ix1, iy1, ix2, iy2))
    if not pieces:
        return None
    x1, y1 = min(p[0] for p in pieces), min(p[1] for p in pieces)
    x2, y2 = max(p[2] for p in pieces), max(p[3] for p in pieces)
    return [x1, y1, x2 - x1, y2 - y1]


def post_process(patch: np.ndarray, name: str | None) -> np.ndarray:
    if not name:
        return patch
    gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if name == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2)).apply(gray)
    if name == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if name == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if name == "sharpen":
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        return cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    if name == "stretch":
        lo, hi = np.percentile(gray, (1, 99))
        return np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    if name == "norules":
        # Paint long straight rules with the paper's brightness.
        paper = np.percentile(gray, 85)
        mask = (gray < paper - 30).astype(np.uint8)
        h, w = mask.shape
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(0.5 * w)), 1))
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(0.7 * h))))
        rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal) | cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical)
        rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
        return cv2.inpaint(gray, rules, 3, cv2.INPAINT_TELEA)
    raise ValueError(name)


def cut(page: np.ndarray, gray: np.ndarray, box: list[int], recipe: dict, surya_boxes) -> np.ndarray | None:
    x, y, w, h = box
    tight = recipe.get("tight")
    if tight == "ink":
        box = ink_box(gray, box) or box
    elif tight == "surya":
        box = surya_box(box, surya_boxes)
        if box is None:
            return None
    x, y, w, h = box
    pad = recipe["pad"]
    pad_x = recipe.get("pad_x", int(recipe["pad_fx"] * w) if "pad_fx" in recipe else pad)
    pad_y = recipe.get("pad_y", int(recipe["pad_fy"] * h) if "pad_fy" in recipe else pad)
    H, W = page.shape[:2]
    x1, y1, x2, y2 = max(0, x - pad_x), max(0, y - pad_y), min(W, x + w + pad_x), min(H, y + h + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    patch = post_process(page[y1:y2, x1:x2], recipe.get("post"))

    up = recipe["up"]
    factor = int(up[1:]) / patch.shape[0] if isinstance(up, str) else float(up)
    if factor != 1.0:
        pil = Image.fromarray(patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        resample = Image.Resampling.BICUBIC if recipe.get("resample") == "bicubic" else Image.Resampling.LANCZOS
        pil = pil.resize((max(8, round(pil.width * factor)), max(8, round(pil.height * factor))), resample)
        return np.asarray(pil)
    return patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def union(boxes: list[list[int]]) -> list[int]:
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def column_consensus(cells: list[dict]) -> list[dict]:
    """Each item cell with its x-extent replaced by its column's median edges."""
    edges: dict[str, list[tuple[int, int]]] = {}
    for cell in cells:
        if cell["row"] is not None:
            edges.setdefault(cell["role"], []).append((cell["bbox"][0], cell["bbox"][0] + cell["bbox"][2]))
    out = []
    for cell in cells:
        column = edges.get(cell["role"]) if cell["row"] is not None else None
        if column and len(column) >= 3:
            x1 = int(np.median([e[0] for e in column]))
            x2 = int(np.median([e[1] for e in column]))
            cell = dict(cell, bbox=[x1, cell["bbox"][1], x2 - x1, cell["bbox"][3]])
        out.append(cell)
    return out


def units(entry: dict, unit: str, colfix: bool = False) -> list[dict]:
    """What a variant cuts: the labelled cells, the item rows, or the page."""
    if unit == "cell":
        cells = column_consensus(entry["cells"]) if colfix else entry["cells"]
        return [dict(c, members=[c]) for c in cells]
    if unit == "row":
        rows: dict[int, list[dict]] = {}
        for cell in entry["cells"]:
            if cell["row"] is not None:
                rows.setdefault(cell["row"], []).append(cell)
        return [
            {"cell": f"row{row}", "role": "row", "kind": "row", "row": row, "text": None,
             "bbox": union([c["bbox"] for c in cells]), "members": cells}
            for row, cells in sorted(rows.items())
        ]
    width, height = entry["page"]
    return [{"cell": "page", "role": "page", "kind": "page", "row": None, "text": None,
             "bbox": [0, 0, width, height], "members": entry["cells"]}]


def build(variant: str, regions: dict) -> None:
    recipe = {**BASE, **VARIANTS[variant]}
    out_dir = CROPS / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, entry in regions.items():
        gray = cv2.imread(str(PAGES / f"{name}_gray.png"), cv2.IMREAD_GRAYSCALE)
        page = cv2.imread(str(PAGES / f"{name}_color.png")) if recipe["src"] == "color" else gray
        if recipe.get("page"):
            step, params = PIPELINE_STEPS[recipe["page"]]
            page = PreprocessingPipelineBuilder.build([StepConfig(name=step, enabled=True, params=params)]).run(
                ImagePayload(image=gray)
            ).image
        for unit in units(entry, recipe["unit"], recipe.get("colfix", False)):
            patch = cut(page, gray, unit["bbox"], recipe, entry.get("surya_boxes"))
            row = {
                "invoice": name, "cell": unit["cell"], "role": unit["role"], "kind": unit["kind"],
                "row": unit["row"], "text": unit["text"], "path": None, "sha1": None,
                "members": [{k: m[k] for k in ("cell", "role", "kind", "text")} for m in unit["members"]],
            }
            if patch is not None:
                path = out_dir / f"{name}_{unit['cell']}.png"
                Image.fromarray(patch).save(path)
                row.update(path=f"{variant}/{path.name}", sha1=sha1_file(path), size=[patch.shape[1], patch.shape[0]])
            rows.append(row)
    write_jsonl(out_dir / "manifest.jsonl", rows)
    missing = sum(r["path"] is None for r in rows)
    print(f"{variant}: {len(rows) - missing} crop(s)" + (f", {missing} missing" if missing else ""))


def main() -> int:
    utf8_console()
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    force = "--force" in sys.argv[1:]
    names = [a for a in sys.argv[1:] if a != "--force"] or list(VARIANTS)
    for variant in names:
        if (CROPS / variant / "manifest.jsonl").exists() and not force:
            print(f"{variant}: exists, kept (--force rebuilds)")
            continue
        if VARIANTS.get(variant, {}).get("tight") == "surya" and not all(e.get("surya_boxes") for e in regions.values()):
            print(f"{variant}: skipped, no Surya boxes in regions.json (run prepare.py --surya)")
            continue
        build(variant, regions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
