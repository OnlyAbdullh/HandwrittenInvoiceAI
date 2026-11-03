"""Cut each region out of the page and prepare it for a recognizer.

Extracted from `HybridSuryaQwenEngine`, where the three tuning decisions below
were buried among the detector and recognizer calls. They are shared unchanged
by both region-based flows, which is exactly why they belong in one place:

* **pad** -- Surya crops tight and a printed cell rule sits right against the
  ink. Either way the glyphs need room; a crop that clips an ascender reads as a
  different letter.
* **upscale** -- small crops read measurably better enlarged. LANCZOS, because
  a handwritten stroke is exactly the high-frequency content a cheaper filter
  smears.
* **min_side** -- below this there is nothing to read, and each crop costs a
  full model call.

The crop is returned at its padded, upscaled size while the region keeps its
original page coordinates. Nothing downstream measures anything off the crop, so
the two never need to agree.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from core.domain.image_payload import ImagePayload
from core.domain.ocr import RegionCrop, TextRegion
from ocr.cropping.registry import cropper_registry


@cropper_registry.register("padded_crop")
class PaddedRegionCropper:
    name = "padded_crop"

    def __init__(
        self,
        pad: int = 10,
        upscale: int = 2,
        min_side: int = 8,
        min_ink_ratio: float = 0.0,
        ink_delta: int = 30,
        ink_inset: int = 8,
    ):
        if upscale < 1:
            raise ValueError(f"padded_crop upscale must be >= 1, got {upscale}.")
        if ink_inset < 0:
            raise ValueError(f"padded_crop ink_inset must be >= 0, got {ink_inset}.")
        self.pad = pad
        self.upscale = upscale
        self.min_side = min_side
        self.min_ink_ratio = float(min_ink_ratio)
        self.ink_delta = int(ink_delta)

        # Blankness is judged on the region's *interior*, this many pixels in
        # from its box. A printed cell's own border lines put 9-20% "ink" into
        # the padded crop of a cell nobody wrote in, which made the gate useless:
        # on bills/sahror every blank cell scored >= 0.09 padded, and exactly
        # 0.000 once the border is excluded, against >= 0.022 for the faintest
        # filled cell.
        self.ink_inset = ink_inset

    def ink_ratio(self, patch: np.ndarray) -> float:
        gray = patch if patch.ndim == 2 else patch[..., :3].mean(axis=2)
        paper = np.percentile(gray, 85)
        return float((gray < paper - self.ink_delta).mean())

    def _is_blank(self, page: np.ndarray, box) -> bool:
        inset = min(self.ink_inset, box.w // 4, box.h // 4)
        interior = page[box.y + inset : box.y + box.h - inset, box.x + inset : box.x + box.w - inset]
        return interior.size == 0 or self.ink_ratio(interior) < self.min_ink_ratio

    def crop(
        self, image: ImagePayload, regions: Sequence[TextRegion]
    ) -> list[RegionCrop]:
        page = image.image
        h, w = page.shape[:2]

        crops: list[RegionCrop] = []
        for region in regions:
            box = region.bbox
            x1 = max(0, box.x - self.pad)
            y1 = max(0, box.y - self.pad)
            x2 = min(w, box.x + box.w + self.pad)
            y2 = min(h, box.y + box.h + self.pad)

            if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
                continue

            patch = page[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            if self.min_ink_ratio > 0 and self._is_blank(page, box):
                continue

            if self.upscale > 1:
                patch = self._upscale(patch)

            crops.append(RegionCrop(region=region, image=patch))

        return crops

    def _upscale(self, patch: np.ndarray) -> np.ndarray:
        # Via PIL rather than cv2.resize because the recognizers take PIL images
        # anyway, and LANCZOS here matches what the hybrid engine was measured
        # with -- changing the resampler silently changes recognition quality.
        pil = Image.fromarray(patch)
        pil = pil.resize(
            (pil.width * self.upscale, pil.height * self.upscale),
            Image.Resampling.LANCZOS,
        )
        return np.asarray(pil)