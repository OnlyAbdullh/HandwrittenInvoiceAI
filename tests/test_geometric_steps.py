"""The two coordinate-changing steps.

These run before everything else and their output *is* the image the UI shows
and every bbox is measured against, so a bad decision here is not recoverable
downstream — and, worse, is invisible: a wrongly cropped page is still a
perfectly valid image.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from config.loader import load_config
from core.domain.image_payload import ImagePayload, PipelineContext
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from preprocessing.steps.geometric.deskew import HoughDeskewStep
from preprocessing.steps.geometric.perspective_correction import (
    PerspectiveCorrectionStep,
)
from preprocessing.steps.geometric.working_resolution import WorkingResolutionStep


def context(image: np.ndarray) -> PipelineContext:
    return PipelineContext(payload=ImagePayload(image=image))


def blank(height: int = 900, width: int = 700, value: int = 255) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestPerspectiveCorrection:
    def test_crops_a_page_photographed_on_a_dark_desk(self):
        photo = blank(1400, 1000, value=90)
        cv2.rectangle(photo, (70, 90), (930, 1310), (255, 255, 255), -1)

        result = PerspectiveCorrectionStep(margin=0.0).apply(context(photo))
        height, width = result.payload.image.shape[:2]

        # The desk is gone; roughly the page remains.
        assert 1150 < height < 1300
        assert 800 < width < 930

    def test_keeps_what_lies_just_outside_the_detected_outline(self):
        """The outline found is not always the page.

        On form B -- white paper on a white cloth -- the paper's edge barely
        shows, and the strongest outline is the table's printed border. Warping
        exactly to it cut away the invoice number, the customer and the date
        printed above the table, and most of the totals strip below it: no
        reader could ever recover them. The margin keeps a band of the photo
        around whatever outline was found.
        """
        photo = blank(1400, 1000, value=250)  # paper the same colour as the desk
        cv2.rectangle(photo, (150, 300), (850, 1200), (0, 0, 0), 4)  # the table's border
        cv2.putText(photo, "No 0001901", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        cropped = PerspectiveCorrectionStep(margin=0.0).apply(context(photo.copy())).payload.image
        kept = PerspectiveCorrectionStep(margin=0.15).apply(context(photo.copy())).payload.image

        def ink_above_table(image, top):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return int((gray[: max(top - 10, 0)] < 128).sum())

        assert ink_above_table(cropped, 0) == 0  # the number is gone
        table_top = round(0.15 * 900)  # margin x the outline's height
        assert ink_above_table(kept, table_top) > 500  # the number survives

    def test_the_margin_is_measured_from_the_outline_and_recorded(self):
        photo = blank(1400, 1000, value=90)
        cv2.rectangle(photo, (70, 90), (930, 1310), (255, 255, 255), -1)

        result = PerspectiveCorrectionStep(margin=0.1).apply(context(photo))
        height, width = result.payload.image.shape[:2]
        plain = PerspectiveCorrectionStep(margin=0.0).apply(context(photo.copy())).payload.image

        assert (height, width) == (plain.shape[0] + 2 * round(0.1 * plain.shape[0]),
                                   plain.shape[1] + 2 * round(0.1 * plain.shape[1]))
        assert result.payload.metadata["perspective_correction"]["margin"] == 0.1

    def test_an_out_of_range_margin_is_rejected(self):
        with pytest.raises(ValueError):
            PerspectiveCorrectionStep(margin=0.8)

    def test_refuses_to_crop_to_a_blob_of_text(self):
        """The guard that matters most.

        On a borderless receipt the largest closed contour is a word, not the
        page edge. Warping to it replaced a 1400x1000 invoice with a 37x57
        thumbnail — and because the result is still a valid image, nothing
        downstream could tell that the invoice had been thrown away.
        """
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        result = PerspectiveCorrectionStep().apply(context(page))

        assert result.payload.image.shape[:2] == (1400, 1000)
        assert "skipped" in result.payload.metadata["perspective_correction"]

    def test_a_blank_page_passes_through(self):
        page = blank()
        result = PerspectiveCorrectionStep().apply(context(page))

        assert result.payload.image.shape[:2] == (900, 700)

    def test_the_floor_is_tunable(self):
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # A floor of 0 restores the old, unguarded behaviour, which is what
        # makes this a policy rather than a hardcoded rule.
        result = PerspectiveCorrectionStep(min_area_ratio=0.0).apply(context(page))
        assert result.payload.image.shape[:2] != (1400, 1000)

    def test_an_empty_image_is_an_error(self):
        with pytest.raises(ValueError):
            PerspectiveCorrectionStep().apply(context(np.zeros((0, 0, 3), np.uint8)))


def photographed_at_an_angle() -> tuple[np.ndarray, np.ndarray]:
    """A white page with one ruled line near its top, on a dark desk, seen
    from one side: a trapezoid, its top edge sloping about four
    degrees. Returns the photo and the page's corners (top-left, top-right,
    bottom-right, bottom-left)."""
    flat = blank(1100, 800)
    cv2.line(flat, (0, 110), (799, 110), (0, 0, 0), 5)
    corners = np.array([[120, 220], [900, 160], [930, 1300], [90, 1240]], dtype=np.float32)
    flat_corners = np.array([[0, 0], [799, 0], [799, 1099], [0, 1099]], dtype=np.float32)
    photo = cv2.warpPerspective(
        flat, cv2.getPerspectiveTransform(flat_corners, corners), (1000, 1400),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 60, 60),
    )
    return photo, corners


class TestPerspectiveIsRemoved:
    """A page photographed from an angle is a trapezoid, not a rotated
    rectangle. Fitted with a rectangle it stayed a trapezoid: on a real photo
    (bills/abd) the table's top rule still sloped four degrees after this step,
    the grid finder lost the rows under it, and every header field was then
    read from the wrong place."""

    @staticmethod
    def _desk_in_corners(page: np.ndarray, inset: int = 12) -> int:
        """How many of the page's four corners still show the dark desk."""
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        points = [(inset, inset), (inset, width - inset), (height - inset, inset), (height - inset, width - inset)]
        return sum(int(gray[y, x] < 128) for y, x in points)

    def test_the_four_corners_of_the_page_become_the_corners_of_the_image(self):
        photo, _ = photographed_at_an_angle()
        result = PerspectiveCorrectionStep(margin=0.0).apply(context(photo))

        assert result.payload.metadata["perspective_correction"]["fit"] == "quadrilateral"
        assert self._desk_in_corners(result.payload.image) == 0

    def test_a_rectangle_fit_leaves_the_trapezoid(self):
        """The old behaviour, kept selectable so the difference stays measurable."""
        photo, _ = photographed_at_an_angle()
        result = PerspectiveCorrectionStep(margin=0.0, fit="rectangle").apply(context(photo))

        assert result.payload.metadata["perspective_correction"]["fit"] == "rectangle"
        assert self._desk_in_corners(result.payload.image) >= 2

    def test_a_ruled_line_across_the_page_comes_out_level(self):
        photo, _ = photographed_at_an_angle()

        def slope_of_the_rule(fit: str) -> float:
            page = PerspectiveCorrectionStep(margin=0.0, fit=fit).apply(context(photo.copy())).payload.image
            gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            band = gray[: int(0.3 * height)]

            def rule_y(x: int) -> float:
                # Black, not merely dark: the desk shows in a rectangle fit's corners.
                return float(np.mean(np.nonzero(band[:, x] < 35)[0]))

            return abs(rule_y(int(0.15 * width)) - rule_y(int(0.85 * width)))

        assert slope_of_the_rule("quadrilateral") <= 3
        assert slope_of_the_rule("rectangle") >= 15

    def test_the_transform_maps_the_photo_onto_the_page(self):
        """Every bbox the pipeline reports is in the corrected page's space, so
        the recorded transform has to be the warp that was applied."""
        photo, corners = photographed_at_an_angle()
        result = PerspectiveCorrectionStep(margin=0.1).apply(context(photo))
        height, width = result.payload.image.shape[:2]

        x, y = result.payload.transform.apply_to_point(float(corners[0][0]), float(corners[0][1]))
        # The page's top-left corner lands one margin in from the image's.
        assert abs(x - width * 0.1 / 1.2) <= 6
        assert abs(y - height * 0.1 / 1.2) <= 6

    def test_an_outline_that_is_not_four_sided_falls_back_to_the_rectangle(self):
        photo = blank(1400, 1000, value=60)
        cv2.circle(photo, (500, 700), 420, (255, 255, 255), -1)

        result = PerspectiveCorrectionStep(margin=0.0).apply(context(photo))

        assert result.payload.metadata["perspective_correction"]["fit"] == "rectangle"

    def test_an_unknown_fit_is_rejected(self):
        with pytest.raises(ValueError):
            PerspectiveCorrectionStep(fit="triangle")

    def test_nothing_is_drawn_beyond_the_edge_of_the_photo(self):
        """The margin reaches past the photo when the page fills it. The
        photo's border used to be stretched across that margin, so print cut
        by the edge became straight dark streaks; the grid finder closed boxes
        between them and the classifier read them as the totals strip."""
        photo = blank(1400, 1000, value=250)
        cv2.rectangle(photo, (40, 60), (960, 1340), (0, 0, 0), 4)  # the outline that is found
        for x in range(200, 800, 60):  # print the photo's bottom edge cuts through
            cv2.line(photo, (x, 1360), (x, 1399), (0, 0, 0), 8)

        page = PerspectiveCorrectionStep(margin=0.15).apply(context(photo)).payload.image
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(int)

        beyond = gray[-40]  # a row well past the photo's bottom edge
        assert np.abs(np.diff(beyond)).max() < 40


class TestDeskew:
    def test_straightens_a_rotated_page(self):
        page = blank(800, 800)
        for y in range(200, 600, 60):
            cv2.line(page, (100, y), (700, y), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((400, 400), -5.0, 1.0),
            (800, 800),
            borderValue=(255, 255, 255),
        )

        result = HoughDeskewStep().apply(context(rotated))
        angle = result.payload.metadata["deskew_hough"]["median_angle_deg"]

        # Rotated by -5°, so the correction is about +5° back.
        assert 3.5 < angle < 6.5

    def test_a_page_with_no_straight_lines_passes_through(self):
        """Nothing to measure is not a failure.

        A borderless receipt, or a page of loose handwriting, simply has no long
        straight edges. Raising here would fail the whole extraction over a step
        that had no work to do.
        """
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        result = HoughDeskewStep().apply(context(page))

        assert result.payload.image.shape[:2] == (1400, 1000)
        assert result.payload.metadata["deskew_hough"]["num_lines_used"] == 0

    def test_a_blank_page_passes_through(self):
        result = HoughDeskewStep().apply(context(blank()))
        assert result.payload.image.shape[:2] == (900, 700)

    def test_refuses_to_rotate_beyond_the_maximum_angle(self):
        """The clamp applies to the median, not to each line.

        It used to be per-line, which silently threw away every *vertical* rule
        on a ruled invoice — half the available evidence. Now every segment
        votes and the clamp guards the answer instead: with max_angle_deg at 1°
        the measurement is rejected and the page is left exactly as it came in.
        """
        page = blank(800, 800)
        for x in range(200, 600, 60):
            cv2.line(page, (x, 100), (x + 300, 700), (0, 0, 0), 3)

        result = HoughDeskewStep(max_angle_deg=1.0).apply(context(page))
        meta = result.payload.metadata["deskew_hough"]

        assert result.payload.image.shape[:2] == (800, 800)
        assert "skipped" in meta
        assert meta["num_lines_used"] > 0

    def test_vertical_rules_vote_for_the_same_tilt(self):
        """The reason folding exists.

        A ruled invoice carries as many vertical rules as horizontal ones. The
        old per-line filter discarded every one of them, so a form whose
        verticals are the crisper feature was measured off whichever horizontals
        happened to survive. Folding modulo 90 makes a vertical rule vote for
        the tilt it actually describes.
        """
        page = blank(800, 800)
        for x in range(200, 640, 60):
            cv2.line(page, (x, 120), (x, 680), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((400, 400), -4.0, 1.0),
            (800, 800),
            borderValue=(255, 255, 255),
        )

        folded = HoughDeskewStep().apply(context(rotated))
        assert 2.5 < folded.payload.metadata["deskew_hough"]["median_angle_deg"] < 5.5

        # Without folding the same page yields nothing to measure at all.
        unfolded = HoughDeskewStep(fold_to_90=False).apply(context(rotated))
        assert unfolded.payload.metadata["deskew_hough"]["num_lines_used"] == 0

    def test_expanding_the_canvas_keeps_the_corners(self):
        """Rotating into the original frame clips the corners, and on a page
        that fills the photo the clipped corner is where the table's outer rule
        lives."""
        page = blank(800, 600)
        for y in range(150, 500, 50):
            cv2.line(page, (60, y), (540, y), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((300, 400), -6.0, 1.0),
            (600, 800),
            borderValue=(255, 255, 255),
        )

        expanded = HoughDeskewStep().apply(context(rotated)).payload.image
        clipped = HoughDeskewStep(expand_canvas=False).apply(context(rotated)).payload.image

        assert expanded.shape[0] > clipped.shape[0]
        assert expanded.shape[1] > clipped.shape[1]
        assert clipped.shape[:2] == (800, 600)

    def test_a_page_already_level_is_not_resampled(self):
        """A sub-degree correction costs sharpness and buys nothing."""
        page = blank(800, 800)
        for y in range(200, 600, 60):
            cv2.line(page, (100, y), (700, y), (0, 0, 0), 3)

        result = HoughDeskewStep().apply(context(page))

        assert np.array_equal(result.payload.image, page)
        assert "skipped" in result.payload.metadata["deskew_hough"]


class TestWorkingResolution:
    """A large photo is shrunk to the size every pixel threshold was tuned at."""

    def test_a_large_photo_is_shrunk_to_the_working_long_side(self):
        result = WorkingResolutionStep(max_long_side=2000).apply(context(blank(4000, 3000)))
        assert result.payload.image.shape[:2] == (2000, 1500)
        assert result.payload.metadata["working_resolution"]["scale"] == 0.5

    def test_the_transform_maps_the_photo_onto_the_shrunk_page(self):
        result = WorkingResolutionStep(max_long_side=2000).apply(context(blank(4000, 3001)))
        height, width = result.payload.image.shape[:2]
        x, y = result.payload.transform.apply_to_point(3001, 4000)
        assert abs(x - width) < 1e-6 and abs(y - height) < 1e-6

    def test_a_photo_at_or_below_the_working_size_is_not_enlarged(self):
        photo = blank(1863, 1466)
        result = WorkingResolutionStep(max_long_side=2000).apply(context(photo))
        assert result.payload.image.shape == photo.shape
        assert np.allclose(result.payload.transform.matrix, np.eye(3))

    def test_an_absurd_working_size_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            WorkingResolutionStep(max_long_side=100)

    def test_it_runs_first_in_the_geometric_stage(self):

        config = load_config("config/default_config.yaml")
        steps = config.preprocessing.pipeline_geometric_steps()
        assert steps[0].name == "working_resolution" and steps[0].params == {"max_long_side": 2000}
        assert [s.name for s in steps[1:]] == [s.name for s in config.preprocessing.geometric_steps]

        config.preprocessing.working_long_side = None
        assert config.preprocessing.pipeline_geometric_steps() == config.preprocessing.geometric_steps

    def test_the_same_page_photographed_larger_gives_the_same_page(self):
        """The whole geometric stage, at the photo's size and at twice it."""
        photo = blank(2000, 1500, value=90)
        cv2.rectangle(photo, (120, 150), (1380, 1850), (255, 255, 255), -1)
        cv2.rectangle(photo, (300, 500), (1200, 1500), (0, 0, 0), 3)
        big = cv2.resize(photo, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        stage = PreprocessingPipelineBuilder.build(
            load_config("config/default_config.yaml").preprocessing.pipeline_geometric_steps()
        )
        small_page = stage.run(ImagePayload(image=photo)).image
        big_page = stage.run(ImagePayload(image=big)).image
        assert abs(small_page.shape[0] - big_page.shape[0]) <= 4
        assert abs(small_page.shape[1] - big_page.shape[1]) <= 4
