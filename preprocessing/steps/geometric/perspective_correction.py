"""Find the page's outline and warp it flat.

Canny + dilate + findContours gives the strongest closed outline on the photo
(the page's edge, or -- white paper on a white cloth -- the table's printed
border). The outline is then fitted with **four corners of its own** and those
are mapped onto a rectangle, which is what removes perspective.

It used to be fitted with `minAreaRect`, a rotated *rectangle*. A rectangle can
only undo rotation: a page photographed from an angle is a trapezoid, and
warping its bounding rectangle left the trapezoid exactly as it was. Measured on
the three real photos, opposite sides of the outline differed by 6 to 10%.
On `bills/abd` that left the table's top rule sloping 4 degrees after "perspective
correction"; the grid finder's horizontal kernel cannot follow a rule that
steep, so the printed caption row was never detected, the first handwritten row
was labelled the caption row, and every header field was read from the wrong
place. See `outline_corners`.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.geometry import Transform
from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry

QUADRILATERAL = "quadrilateral"
RECTANGLE = "rectangle"

# A four-corner fit is believed only when it covers at least this share of the
# outline's convex hull. A page with a folded corner, or an outline that is
# really five-sided, loses a large piece to the best four corners; warping to
# them would shear the page, and a rotated rectangle is the safer reading.
MIN_QUAD_HULL_COVERAGE = 0.85


def _order_points(pts: np.ndarray) -> np.ndarray:
    """getPerspectiveTransform needs corners in a CONSISTENT order:
    top-left, top-right, bottom-right, bottom-left. minAreaRect/boxPoints
    doesn't guarantee this order, so sort manually.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return rect


def _intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    """Where two lines, each (vx, vy, x0, y0), cross; None when parallel."""
    (ax, ay, px, py), (bx, by, qx, qy) = line_a, line_b
    denominator = ax * by - ay * bx
    if abs(denominator) < 1e-9:
        return None
    t = ((qx - px) * by - (qy - py) * bx) / denominator
    return np.array([px + t * ax, py + t * ay], dtype="float32")


def _boundary_points(contour: np.ndarray) -> np.ndarray:
    """Every pixel of the outline, as (x, y) rows.

    `findContours` stores a straight run as its two ends, so a long straight
    side -- exactly what a line is to be fitted to -- is a handful of points.
    Drawing the outline and reading it back gives the side pixel by pixel.
    """
    x, y, width, height = cv2.boundingRect(contour)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, 1, offset=(-x, -y))
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs + x, ys + y]).astype("float32")


def _refine_corners(outline: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Corners as the crossings of lines fitted to the outline's four sides.

    The simplified polygon's corners are points *of the outline*, and a printed
    border has rounded corners: each one sits on the arc, inside the true
    corner. A line fitted to the middle of each side ignores the arcs -- and,
    fitted robustly, the handwriting or row numbers that touch the border --
    and two such lines cross where the corner would be if it were sharp. Any
    side or corner that cannot be fitted keeps the polygon's own.
    """
    lines: list[np.ndarray | None] = []
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        side = end - start
        length = float(np.linalg.norm(side))
        if length < 1.0:
            return corners
        direction = side / length
        offsets = outline - start
        along = offsets @ direction
        across = np.abs(offsets @ np.array([-direction[1], direction[0]], dtype="float32"))
        # The middle of the side, near it: the rounded corners and anything
        # that belongs to a neighbouring side are left out.
        on_side = outline[(along > 0.15 * length) & (along < 0.85 * length) & (across < 0.02 * length)]
        if len(on_side) < 0.25 * length:
            lines.append(None)  # mostly missing: not a side a line can be trusted on
            continue
        lines.append(cv2.fitLine(on_side, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4))

    refined = corners.copy()
    for i in range(4):
        before, after = lines[i - 1], lines[i]  # the two sides meeting at corner i
        if before is None or after is None:
            continue
        crossing = _intersection(before, after)
        # A crossing far from the polygon's corner is a bad fit, not a corner.
        limit = 0.05 * min(np.linalg.norm(corners[i] - corners[i - 1]), np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
        if crossing is not None and np.linalg.norm(crossing - corners[i]) <= limit:
            refined[i] = crossing
    return refined


def outline_corners(contour: np.ndarray) -> np.ndarray | None:
    """The outline's own four corners (unordered), or None when it is not a quadrilateral.

    The convex hull is simplified with a growing tolerance until four vertices
    remain: handwriting that touches the border, and the border's rounded
    corners, are small against the page and go first. The fit is rejected when
    it loses too much of the hull (`MIN_QUAD_HULL_COVERAGE`).
    """
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    if hull_area <= 0 or perimeter <= 0:
        return None

    for tolerance in np.linspace(0.005, 0.08, 31):
        polygon = cv2.approxPolyDP(hull, float(tolerance) * perimeter, True)
        if len(polygon) < 4:
            return None
        if len(polygon) == 4:
            corners = polygon.reshape(4, 2).astype("float32")
            if cv2.contourArea(corners) < MIN_QUAD_HULL_COVERAGE * hull_area:
                return None
            refined = _refine_corners(_boundary_points(contour), corners)
            return refined if cv2.isContourConvex(refined.reshape(4, 1, 2)) else corners
    return None


@step_registry.register("perspective_correction")
class PerspectiveCorrectionStep:
    name = "perspective_correction"

    def __init__(
        self,
        canny_low: int = 30,
        canny_high: int = 150,
        min_area_ratio: float = 0.2,
        margin: float = 0.15,
        fit: str = QUADRILATERAL,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high

        # The smallest fraction of the frame a detected quad may cover and still
        # be believed to be the page. A real page edge fills most of the photo;
        # anything much smaller is a paragraph, a stamp or a logo whose outline
        # happened to be the largest closed contour. Warping to one of those
        # throws the invoice away and leaves a thumbnail of a word behind --
        # measured on a borderless receipt, a 1400x1000 page came out 37x57 --
        # and nothing downstream can tell that it happened, because a cropped
        # image is still a perfectly valid image.
        self.min_area_ratio = min_area_ratio

        if not 0.0 <= margin <= 0.5:
            raise ValueError(f"perspective_correction margin must be in [0, 0.5], got {margin}.")
        self.margin = margin

        # `quadrilateral` warps the outline's own four corners, which removes
        # perspective; an outline that is not four-sided falls back to its
        # rotated bounding rectangle. `rectangle` always uses the rectangle --
        # what this step did before, kept so the difference stays measurable.
        if fit not in (QUADRILATERAL, RECTANGLE):
            raise ValueError(
                f"perspective_correction fit must be '{QUADRILATERAL}' or '{RECTANGLE}', got {fit!r}."
            )
        self.fit = fit

    @staticmethod
    def _soften_beyond_photo(
        warped: np.ndarray, source_shape: tuple[int, int], matrix: np.ndarray, size: tuple[int, int]
    ) -> np.ndarray:
        """Blur the part of the page that lies outside the photo.

        Replicating the photo's border avoids one false rule along its edge
        but draws many across it: wherever the edge cuts through print or a
        desk's grain, each dark pixel is stretched into a straight streak the
        full depth of the margin. On `bills/abd` the grid finder closed boxes
        between those streaks and the classifier read them as the totals
        strip. Nothing out there is the invoice, so it is averaged flat.
        """
        inside = cv2.warpPerspective(
            np.full(source_shape, 255, dtype=np.uint8), matrix, size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        beyond = inside == 0
        if not beyond.any():
            return warped
        kernel = max(31, int(0.05 * max(size))) | 1
        warped[beyond] = cv2.blur(warped, (kernel, kernel))[beyond]
        return warped

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("perspective_correction received an empty image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # An evenly-lit scan with no visible page edge -- the paper fills the
            # frame, so there is nothing to crop to. That is a perfectly good
            # invoice, and failing here would fail the whole extraction over a
            # step that simply had no work to do.
            ctx.payload.metadata[self.name] = {
                "skipped": "no page contour detected; image passed through unchanged"
            }
            return ctx

        paper_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(paper_contour)
        box = cv2.boxPoints(rect).astype("float32")

        corners = outline_corners(paper_contour) if self.fit == QUADRILATERAL else None
        fitted = QUADRILATERAL if corners is not None else RECTANGLE
        ordered = _order_points(corners if corners is not None else box)
        (tl, tr, br, bl) = ordered

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        if max_width <= 0 or max_height <= 0:
            # A degenerate quad -- a single line or point of detected edge. Warping
            # to it would destroy the page, so keep the original, same as when no
            # contour was found at all.
            ctx.payload.metadata[self.name] = {
                "skipped": "detected page contour is degenerate; image passed through unchanged"
            }
            return ctx

        source_height, source_width = image.shape[:2]
        area_ratio = (max_width * max_height) / float(source_width * source_height)

        if area_ratio < self.min_area_ratio:
            # Not the page. See min_area_ratio in __init__ for why this guard has
            # to exist: without it the step silently replaces the invoice with a
            # crop of whatever had the strongest outline.
            ctx.payload.metadata[self.name] = {
                "skipped": (
                    f"largest contour covers only {area_ratio:.1%} of the frame, "
                    f"below the {self.min_area_ratio:.0%} floor; it is not the page edge"
                ),
                "rejected_area_ratio": area_ratio,
            }
            return ctx

        pad_x = int(round(self.margin * max_width))
        pad_y = int(round(self.margin * max_height))
        dst = np.array(
            [
                [pad_x, pad_y],
                [pad_x + max_width - 1, pad_y],
                [pad_x + max_width - 1, pad_y + max_height - 1],
                [pad_x, pad_y + max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        size = (max_width + 2 * pad_x, max_height + 2 * pad_y)
        warped = cv2.warpPerspective(
            image,
            matrix,
            size,
            # Beyond the photo's own edge there is nothing; replicating its
            # border keeps that from drawing a ruled line the grid finder
            # would take for part of the form.
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped = self._soften_beyond_photo(warped, image.shape[:2], matrix, size)

        debug_img = image.copy()
        cv2.drawContours(debug_img, [paper_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [ordered.astype(int)], -1, (0, 0, 255), 3)
        for (x, y) in ordered:
            cv2.circle(debug_img, (int(x), int(y)), 8, (255, 0, 0), -1)

        ctx.payload.image = warped
        ctx.payload.transform = ctx.payload.transform.then(
            Transform(matrix=matrix.astype(np.float64))
        )
        ctx.payload.metadata["perspective_correction"] = {
            "corners": ordered.tolist(),
            "rect_angle": rect[2],
            "margin": self.margin,
            "fit": fitted,
        }
        ctx.payload.debug_images.append(("edges", edges))
        ctx.payload.debug_images.append(("edges_dilated", edges_dilated))
        ctx.payload.debug_images.append(("contour_overlay", debug_img))

        return ctx
