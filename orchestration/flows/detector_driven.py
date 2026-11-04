"""Detector finds the boxes, recognizer reads them.

This is what `HybridSuryaQwenEngine` did, unchanged in behaviour and split into
four collaborators the flow coordinates:

    detector.detect  ->  refiner.refine  ->  cropper.crop  ->  recognizer.read

Its strength is that it reads text wherever it appears, including outside any
printed box. Its two weaknesses are worth stating, because they are what the
layout-driven flow exists to avoid:

* The detector misses short isolated digits. A quantity of one or two characters
  is often never boxed, so it is never read -- and there is nothing in the
  output to say a value is missing rather than blank.
* Column boundaries have to be *declared*, as fractions of the page width, by
  someone who measured one printed form by hand.

`assign_to_layout: true` removes the second weakness when the printed form was
recognised: each detected box is clipped to every classified cell it overlaps,
and each piece takes that cell's content kind (so numbers get the digit prompt
and grammar) and its id (so it is mapped by identity). A box spanning two
columns is split at the printed rule. What Surya adds over the layout-driven
flow is then only the crop: tight around the ink it found, instead of the
whole cell. A cell in which it found nothing is not read -- the first weakness
stays, and is exactly what a comparison should show.
"""

from __future__ import annotations

from dataclasses import replace

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.ocr import OCRResult, TextRegion
from core.domain.roles import CellRole
from orchestration.flows.base import ExtractionFlow
from orchestration.flows.layout_driven import SKIPPED_ROLES
from orchestration.flows.registry import flow_registry


@flow_registry.register("detector_driven")
class DetectorDrivenFlow(ExtractionFlow):
    name = "detector_driven"
    REQUIRES = ("detector", "refiner", "cropper", "recognizer")

    @classmethod
    def from_config(cls, config, components, table_extractor, layout_classifier, cell_mapper):
        return cls(
            table_extractor,
            layout_classifier,
            cell_mapper,
            detector=components.detector,
            refiner=components.refiner,
            cropper=components.cropper,
            recognizer=components.recognizer,
            **config.flow.params,
        )

    def __init__(
        self,
        table_extractor,
        layout_classifier,
        cell_mapper,
        detector,
        refiner,
        cropper,
        recognizer,
        assign_to_layout: bool = False,
        min_overlap: float = 0.3,
    ):
        super().__init__(table_extractor, layout_classifier, cell_mapper)
        self.detector = detector
        self.refiner = refiner
        self.cropper = cropper
        self.recognizer = recognizer
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError(f"detector_driven min_overlap must be in (0, 1], got {min_overlap}.")
        self.assign_to_layout = assign_to_layout
        # The share of a detected box that must lie inside a cell for that part
        # of it to be read as the cell.
        self.min_overlap = min_overlap

    def _read_text(
        self, payload: ImagePayload, layout: InvoiceLayout, debug=None
    ) -> OCRResult:
        height, width = payload.image.shape[:2]

        detection = self.detector.detect(payload)
        regions = self.refiner.refine(detection.regions, width, height)
        if self.assign_to_layout:
            regions = assign_to_layout(regions, layout, self.min_overlap)

        return self._crop_and_read(
            payload,
            regions,
            engine_name=f"{self.detector.name}+{self.recognizer.name}",
            raw_output={
                "detector": detection.raw_detector_output,
                "detected": len(detection.regions),
                "after_refine": len(regions),
            },
            debug=debug
        )

    def _warnings(self, layout: InvoiceLayout, ocr_result: OCRResult) -> list[str]:
        if not ocr_result.fragments:
            return ["The text detector found nothing on this page."]
        return []

    def components(self) -> list[object]:
        return super().components() + [
            self.detector,
            self.refiner,
            self.cropper,
            self.recognizer,
        ]

    @property
    def engine_name(self) -> str:
        return f"{self.detector.name}+{self.recognizer.name}"

def _intersection(a: BoundingBox, b: BoundingBox) -> BoundingBox | None:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(x=left, y=top, w=right - left, h=bottom - top)


def assign_to_layout(regions: list[TextRegion], layout: InvoiceLayout, min_overlap: float) -> list[TextRegion]:
    """Each detected box, cut into the classified cells it overlaps.

    A piece is kept when at least `min_overlap` of the detected box lies in
    that cell; it takes the cell's content kind and id. A box over printed
    captions only is dropped, as the layout-driven flow skips them. A box
    outside every classified cell is kept as it was -- text outside the grid
    is what a detector is for.
    """
    cells = [r for r in layout.regions if r.role is not CellRole.UNKNOWN]
    out: list[TextRegion] = []
    for region in regions:
        area = max(1, region.bbox.w * region.bbox.h)
        pieces = []
        for cell in cells:
            piece = _intersection(region.bbox, cell.bbox)
            if piece is not None and piece.w * piece.h >= min_overlap * area:
                pieces.append((cell, piece))
        if not pieces:
            out.append(region)
            continue
        out += [
            replace(region, bbox=piece, content_kind=cell.content_kind, source_id=cell.id)
            for cell, piece in pieces
            if cell.role not in SKIPPED_ROLES
        ]
    return out
