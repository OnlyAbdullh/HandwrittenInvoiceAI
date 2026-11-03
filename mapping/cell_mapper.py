"""Maps raw OCR fragments onto the page's labelled regions.

Because geometric correction runs once, upstream of both the OCR and table
branches (see orchestration.PipelineOrchestrator), every bbox coming out of
OCRResult and InvoiceLayout is already in the same coordinate space -- the
geometrically corrected display image. No coordinate remapping happens here,
only assignment.

* Assignment is by **identity** first, when the fragment carries a
  `source_id`. In the layout-driven flow the crop was cut from a specific
  region, so which region it belongs to is known exactly -- re-deriving it
  from geometry would be throwing away a fact in order to guess at it. If the
  `source_id` doesn't resolve to a known region, that's a bug upstream (OCR
  and layout came from different runs) and is NOT papered over with a
  geometric guess.

* Fragments with no `source_id` are scored (IOA / IOU / centre-distance
  closeness) against every region on the page. This still picks the
  containing region in the common case (IOA -> 1.0 when a region fully covers
  a fragment) but degrades gracefully when a fragment's box clips a divider or
  slightly overshoots its cell.

The trace of every decision goes to the module logger at DEBUG. It used to be
``print``, which raised ``UnicodeEncodeError`` on its first emoji whenever
stdout was redirected to a non-UTF-8 stream -- turning every extraction into a
500 for a service launched as a hidden child process.
"""

from __future__ import annotations

import logging

from core.domain.document import (
    DocumentElement,
    FreeFieldElement,
    StructuredDocument,
    TableCellElement,
)
from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.roles import CellRole, Zone

logger = logging.getLogger(__name__)


def _hull(fragments: list[OCRFragment]) -> BoundingBox:
    x1 = min(f.bbox.x for f in fragments)
    y1 = min(f.bbox.y for f in fragments)
    x2 = max(f.bbox.x + f.bbox.w for f in fragments)
    y2 = max(f.bbox.y + f.bbox.h for f in fragments)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _is_arabic_char(ch: str) -> bool:
    return ("؀" <= ch <= "ۿ") or ("ݐ" <= ch <= "ݿ") or ("ࢠ" <= ch <= "ࣿ")


def _is_latin_char(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _detect_text_direction(text: str) -> str:
    if not text:
        return "ltr"
    arabic_count = sum(1 for ch in text if _is_arabic_char(ch))
    latin_count = sum(1 for ch in text if _is_latin_char(ch))
    if arabic_count == 0 and latin_count == 0:
        return "ltr"
    return "rtl" if arabic_count >= latin_count else "ltr"


def merge_fragments_in_cell(fragments: list[OCRFragment]) -> str:
    """Join a region's fragments in reading order.

    Lines top to bottom; within a line, right to left when the line is mostly
    Arabic and left to right otherwise. A two-line description merged in the
    wrong order would be scored against the catalog as a different product.
    """
    fragments = [f for f in fragments if f.text and f.text.strip()]
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0].text.strip()

    heights = [f.bbox.h for f in fragments if f.bbox.h > 0]
    avg_h = (sum(heights) / len(heights)) if heights else 1.0
    line_threshold = max(avg_h * 0.6, 1.0)

    lines: list[list[OCRFragment]] = []
    for f in sorted(fragments, key=lambda f: f.bbox.y + f.bbox.h / 2.0):
        fy = f.bbox.y + f.bbox.h / 2.0
        for line in lines:
            line_y = sum(x.bbox.y + x.bbox.h / 2.0 for x in line) / len(line)
            if abs(fy - line_y) <= line_threshold:
                line.append(f)
                break
        else:
            lines.append([f])

    lines.sort(key=lambda line: sum(x.bbox.y + x.bbox.h / 2.0 for x in line) / len(line))

    out_lines = []
    for line in lines:
        rtl = _detect_text_direction(" ".join(x.text for x in line)) == "rtl"
        line_sorted = sorted(line, key=lambda x: x.bbox.x + x.bbox.w / 2.0, reverse=rtl)
        out_lines.append(" ".join(x.text.strip() for x in line_sorted))

    return "\n".join(out_lines)


class CellMapper:
    """Turns (OCR fragments, page layout) into one flat list of UI elements."""

    def __init__(self, w_ioa: float = 0.3, w_iou: float = 0.7, w_dist: float = 0.0, min_score: float = 0.25):
        self.w_ioa = w_ioa
        self.w_iou = w_iou
        self.w_dist = w_dist
        self.min_score = min_score

    def _score(self, frag_bbox: BoundingBox, region_bbox: BoundingBox) -> float:
        ioa = frag_bbox.ioa(region_bbox)
        iou = frag_bbox.iou(region_bbox)
        dist = frag_bbox.center_distance(region_bbox)
        norm_dist = dist / region_bbox.diagonal if region_bbox.diagonal > 0 else 1.0
        closeness = max(0.0, 1.0 - min(norm_dist, 1.0))
        return self.w_ioa * ioa + self.w_iou * iou + self.w_dist * closeness

    def map(self, ocr_result: OCRResult, layout: InvoiceLayout) -> StructuredDocument:
        regions = list(layout.regions)

        # Smallest region first: a fragment inside both a cell and a
        # whole-table box belongs to the cell, so the cell must win ties.
        order = sorted(range(len(regions)), key=lambda i: regions[i].area)
        by_id = {region.id: i for i, region in enumerate(regions)}

        per_region: dict[int, list[OCRFragment]] = {}
        unassigned: list[OCRFragment] = []

        for fragment in ocr_result.fragments:
            if not (fragment.text or "").strip():
                continue

            index = self._assign(fragment, regions, order, by_id)
            if index is None:
                unassigned.append(fragment)
            else:
                per_region.setdefault(index, []).append(fragment)

        # Every region becomes an element, including the ones nothing was read
        # in: the verification grid needs a box to click on for a quantity the
        # engine failed to read, and a missing element would shift the row.
        elements: list[DocumentElement] = [
            self._element(region, per_region.get(index, []))
            for index, region in enumerate(regions)
        ]

        for index, fragment in enumerate(unassigned):
            elements.append(
                FreeFieldElement(
                    id=f"free:{index}",
                    bbox=fragment.bbox,
                    fragments=[fragment],
                    merged_text=fragment.text.strip(),
                    role=CellRole.UNKNOWN,
                    zone=Zone.UNKNOWN,
                )
            )

        logger.debug(
            "Mapped %d fragment(s) onto %d region(s); %d left as free fields.",
            len(ocr_result.fragments),
            len(regions),
            len(unassigned),
        )
        return StructuredDocument(elements=elements)

    def _assign(
        self,
        fragment: OCRFragment,
        regions: list[LayoutRegion],
        order: list[int],
        by_id: dict[str, int],
    ) -> int | None:
        source_id = getattr(fragment, "source_id", None)
        if source_id is not None:
            index = by_id.get(source_id)
            if index is None:
                logger.warning(
                    "Fragment %r carries source_id %r, unknown to the layout; "
                    "left unassigned rather than guessed geometrically.",
                    fragment.text,
                    source_id,
                )
            return index

        best_index, best_score = None, -1.0
        for i in order:
            score = self._score(fragment.bbox, regions[i].bbox)
            if score > best_score:
                best_score, best_index = score, i

        if best_index is not None and best_score >= self.min_score:
            logger.debug(
                "Fragment %r -> region %r (score %.3f).",
                fragment.text,
                regions[best_index].id,
                best_score,
            )
            return best_index

        logger.debug(
            "Fragment %r: best score %.3f < %.2f; becomes a free field.",
            fragment.text,
            best_score,
            self.min_score,
        )
        return None

    def _element(self, region: LayoutRegion, fragments: list[OCRFragment]) -> DocumentElement:
        merged = merge_fragments_in_cell(fragments)

        if region.row is None or region.col is None or region.table_id is None:
            return FreeFieldElement(
                id=region.id,
                bbox=region.bbox,
                fragments=fragments,
                merged_text=merged,
                role=region.role,
                zone=region.zone,
            )

        return TableCellElement(
            id=region.id,
            table_id=region.table_id,
            row=region.row,
            col=region.col,
            bbox=region.bbox,
            fragments=fragments,
            merged_text=merged,
            role=region.role,
            zone=region.zone,
            row_span=region.row_span,
            col_span=region.col_span,
        )

    @staticmethod
    def bounds_of(fragments: list[OCRFragment]) -> BoundingBox | None:
        """Public helper for callers assembling their own elements from fragments."""
        return _hull(fragments) if fragments else None
