"""Surya's text detector, on its own.

Extracted from `HybridSuryaQwenEngine`, where it was fused with cropping and
Qwen recognition in one class. Surya's *recognizer* produced Chinese glyphs and
LaTeX on this project's handwritten Arabic; its *detector* placed boxes
correctly. Keeping only the detector is the whole point, and it can only be
"kept" separately if it is separate.

Known limit, stated plainly because it changes how results should be read: the
detector misses short isolated digits. On the sample invoice, quantities of one
or two characters were never boxed, so they never reached the recognizer at all.
That is the specific weakness the layout-driven flow does not have -- a printed
cell exists whether or not anything was detected inside it.
"""

from __future__ import annotations

import time

from surya.detection import DetectionPredictor

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import DetectionResult, TextRegion
from core.domain.roles import ContentKind
from ocr.detectors.registry import detector_registry
from ocr.engines.qwen_vlm_engine import to_pil


@detector_registry.register("surya_detector")
class SuryaTextDetector:
    name = "surya_detector"

    def __init__(
        self,
        min_side: int = 12,
        ignore_beyond_x: float | None = None,
        ignore_before_y: float | None = None,
        ignore_after_y: float | None = None,
        min_confidence: float | None = None,
    ):
        # Ignore specks: a box a few pixels on a side is dust or a printing
        # artefact, and reading it costs a full model call.
        self.min_side = min_side

        # Page fractions outside which detections are dropped (box centre):
        # a printed margin, a letterhead band, a footer the recognizer should
        # never be pointed at.
        for name, value in (
            ("ignore_beyond_x", ignore_beyond_x),
            ("ignore_before_y", ignore_before_y),
            ("ignore_after_y", ignore_after_y),
            ("min_confidence", min_confidence),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"surya_detector: {name} must be in [0, 1], got {value}.")
        if ignore_before_y is not None and ignore_after_y is not None and ignore_before_y >= ignore_after_y:
            raise ValueError("surya_detector: ignore_before_y must be below ignore_after_y.")

        self.ignore_beyond_x = ignore_beyond_x
        self.ignore_before_y = ignore_before_y
        self.ignore_after_y = ignore_after_y

        # Boxes the detector itself was unsure of. Each kept box costs a model
        # call and a weak one is most often a printed rule or dot pattern.
        self.min_confidence = min_confidence

        self._predictor: DetectionPredictor | None = None

    @property
    def predictor(self) -> DetectionPredictor:
        """Built on first use, not in __init__.

        Constructing it loads weights, and the pipeline pool builds an
        orchestrator to answer questions like "what config would you run?"
        without ever calling it.
        """
        if self._predictor is None:
            self._predictor = DetectionPredictor()
        return self._predictor

    def is_ready(self) -> bool:
        return self._predictor is not None

    def warmup(self) -> None:
        _ = self.predictor

    def detect(self, image: ImagePayload) -> DetectionResult:
        pil = to_pil(image.image)
        w, h = pil.size

        started = time.perf_counter()
        det = self.predictor([pil])[0]
        elapsed = time.perf_counter() - started

        regions: list[TextRegion] = []
        for box in det.bboxes:
            x1, y1, x2, y2 = (int(v) for v in box.bbox)
            confidence = getattr(box, "confidence", None)
            if not self._keep(x1, y1, x2, y2, confidence, w, h):
                continue

            regions.append(
                TextRegion(
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    # Detection confidence, not recognition confidence. The flow
                    # stores it as the fragment's `detection_confidence`; it is
                    # never reported as how well the text was read.
                    confidence=confidence,
                    content_kind=ContentKind.ANY,
                )
            )

        return DetectionResult(
            regions=regions,
            detector_name=self.name,
            raw_detector_output={"det": det, "detect_s": round(elapsed, 3)},
        )

    def _keep(
        self, x1: int, y1: int, x2: int, y2: int, confidence: float | None, w: int, h: int
    ) -> bool:
        if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
            return False

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if self.ignore_beyond_x is not None and cx > self.ignore_beyond_x * w:
            return False
        if self.ignore_before_y is not None and cy < self.ignore_before_y * h:
            return False
        if self.ignore_after_y is not None and cy > self.ignore_after_y * h:
            return False

        # A detector that reports no score is not filtered on one.
        if self.min_confidence is not None and confidence is not None:
            return confidence >= self.min_confidence
        return True
