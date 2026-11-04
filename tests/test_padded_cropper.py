"""The cropper's blank-cell gate: what counts as writing.

A cell nobody wrote in must never reach the model -- each one costs a model
call and comes back as an invented reading, i.e. a phantom row. On a form whose
rows are printed as dotted lines, a cell moved onto its writing holds the dots
and a rule; those are not writing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.domain.geometry import BoundingBox
from ocr.cropping.padded_cropper import PaddedRegionCropper


def cell(width: int = 160, height: int = 84) -> np.ndarray:
    return np.full((height, width), 230, np.uint8)


def dotted_rule(page: np.ndarray, y: int) -> None:
    for x in range(4, page.shape[1] - 4, 9):
        cv2.circle(page, (x, y), 2, 60, -1)


def gate() -> PaddedRegionCropper:
    return PaddedRegionCropper(min_ink_ratio=0.01)


def is_blank(page: np.ndarray) -> bool:
    h, w = page.shape[:2]
    return gate()._is_blank(page, BoundingBox(x=0, y=0, w=w, h=h))


class TestBlankGate:
    def test_a_dotted_rule_alone_is_blank(self):
        page = cell()
        dotted_rule(page, 50)
        assert is_blank(page)

    def test_a_solid_rule_joined_to_the_borders_is_blank(self):
        page = cell()
        cv2.line(page, (0, 60), (159, 60), 40, 3)
        cv2.line(page, (12, 0), (12, 83), 40, 3)
        cv2.line(page, (148, 0), (148, 83), 40, 3)
        assert is_blank(page)

    def test_a_handwritten_figure_on_the_dotted_rule_is_writing(self):
        page = cell()
        dotted_rule(page, 50)
        cv2.line(page, (60, 18), (64, 52), 40, 3)       # "١"
        cv2.polylines(page, [np.array([[80, 20], [100, 22], [96, 52]])], False, 40, 3)  # "٦"-ish
        assert not is_blank(page)

    def test_small_writing_in_a_tall_box_is_still_writing(self):
        page = cell(width=400, height=160)
        cv2.putText(page, "12 12 12", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 40, 2)
        assert not is_blank(page)

    def test_the_old_pixel_count_is_available(self):
        page = cell()
        dotted_rule(page, 50)
        legacy = PaddedRegionCropper(min_ink_ratio=0.01, min_stroke_height=0.0)
        assert not legacy._is_blank(page, BoundingBox(x=0, y=0, w=160, h=84))

    @pytest.mark.parametrize("params", [{"min_stroke_height": 1.0}, {"min_stroke_height": -0.1}, {"max_stroke_px": 0}])
    def test_bad_parameters_are_refused(self, params):
        with pytest.raises(ValueError):
            PaddedRegionCropper(**params)
