"""Shrink a large photo to the working resolution before anything else looks at it.

Every pixel-valued threshold downstream -- the page outline's Canny and dilation,
deskew's minimum line length, the table extractor's cell sizes and stack
filter, the layout classifier's row pitch -- was tuned on photos whose long
side is about 2000 px. The same invoice photographed at 4000 px is, to those
thresholds, a different page: measured on `bills/sahror` scaled up 2x, the
extractor found 78 boxes, the classifier labelled 42 of them, and 36 cells were
cut instead of 21. On the laptop that is exactly what happened to the full-size
phone photos (53 crops, 26 of them alignable) while the 2000 px copies gave 41
of 41.

Scaling here, once, at the very start of the geometric stage, keeps every one of
those thresholds meaningful without touching any of them. It is a geometric
step -- it changes coordinates -- so its transform is recorded like the others,
and every box the pipeline reports is in the space of the page it produces.

It only ever shrinks, with area averaging (no aliasing in thin strokes). A
photo already at or below the working size is read as it is: enlarging adds no
detail, and it measured worse -- `bills/onlyone` at 0.7x enlarged back to
2000 px lost its item rows, and a 1863 px synthetic form A enlarged by 7% cut
63 cells instead of 57 -- while shrinking `bills/*` from 1.5x and 2x gave the
same 41 labelled cells as the 2000 px originals.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.geometry import Transform
from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry

MIN_LONG_SIDE = 500


@step_registry.register("working_resolution")
class WorkingResolutionStep:
    name = "working_resolution"

    def __init__(self, max_long_side: int = 2000):
        if int(max_long_side) < MIN_LONG_SIDE:
            raise ValueError(
                f"working_resolution max_long_side must be at least {MIN_LONG_SIDE}, got {max_long_side}."
            )
        self.max_long_side = int(max_long_side)

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("working_resolution received an empty image.")

        height, width = image.shape[:2]
        current = max(height, width)
        if current <= self.max_long_side:
            ctx.payload.metadata[self.name] = {"scale": 1.0, "from": [width, height], "to": [width, height]}
            return ctx

        scale = self.max_long_side / float(current)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        ctx.payload.image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # The exact per-axis factors, so a point maps onto the resized pixel
        # grid even though each side was rounded separately.
        sx, sy = new_size[0] / float(width), new_size[1] / float(height)
        ctx.payload.transform = ctx.payload.transform.then(
            Transform(matrix=np.diag([sx, sy, 1.0]).astype(np.float64))
        )
        ctx.payload.metadata[self.name] = {"scale": round(scale, 4), "from": [width, height], "to": list(new_size)}
        return ctx
