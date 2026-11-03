"""Qwen as a TextRecognizer.

A thin adapter over :class:`~ocr.engines.qwen_vlm_engine.QwenVLMEngine`, not a
reimplementation. That class owns the transports and their awkward parts
(worker startup, timeouts, thread safety, UTF-8 on Windows).

What this adds is the seam: it turns each crop's
:class:`~core.domain.roles.ContentKind` into the right prompt and decoding mode,
batches crops, and rejects readings that cannot be what the cell holds (see
:mod:`ocr.recognizers.output_validation`). The recognizer knows about *kinds of
characters* and nothing about invoices.
"""

from __future__ import annotations

import logging
from typing import Sequence

from PIL import Image

from core.domain.ocr import Recognition, RegionCrop
from core.domain.roles import ContentKind
from ocr.engines.qwen_vlm_engine import PROMPTS, QwenVLMEngine, to_pil
from ocr.recognizers import output_validation
from ocr.recognizers.registry import recognizer_registry

logger = logging.getLogger(__name__)


@recognizer_registry.register("qwen")
class QwenRecognizer:
    name = "qwen"

    def __init__(
        self,
        default_kind: str = ContentKind.ANY.value,
        batch_size: int = 4,
        model_path: str = "OCR_Models/Arabic-Qwen3.5-OCR-v4",
        backend: str = "subprocess",
        host: str = "http://localhost:11434",
        model_name: str = "arabic-ocr",
        max_new_tokens: int = 64,
        worker_python: str = "qwen_venv/Scripts/python.exe",
        worker_script: str = "tools/qwen_worker.py",
        arch: str | None = None,
        timeout_s: float = 120.0,
        startup_timeout_s: float = 600.0,
    ):
        missing = [k.value for k in ContentKind if k.value not in PROMPTS]
        if missing:
            raise ValueError(
                f"qwen recognizer: no prompt defined for content kind(s) {missing}. "
                "Add them to PROMPTS in ocr/engines/qwen_vlm_engine.py."
            )
        if batch_size < 1:
            raise ValueError(f"qwen recognizer: batch_size must be >= 1, got {batch_size}.")

        self.default_kind = ContentKind(default_kind)
        self.batch_size = batch_size
        self._engine = QwenVLMEngine(
            model_path=model_path,
            backend=backend,
            host=host,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            worker_python=worker_python,
            worker_script=worker_script,
            arch=arch,
            timeout_s=timeout_s,
            startup_timeout_s=startup_timeout_s,
        )

    @property
    def engine(self) -> QwenVLMEngine:
        return self._engine

    def is_ready(self) -> bool:
        return self._engine.is_ready()

    def warmup(self) -> None:
        self._engine.warmup()

    def close(self) -> None:
        self._engine.close()

    # ------------------------------------------------------------------

    def _kind(self, crop: RegionCrop) -> ContentKind:
        return crop.content_kind or self.default_kind

    def read(self, crop: RegionCrop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops: Sequence[RegionCrop]) -> list[Recognition]:
        results: list[Recognition] = []
        for start in range(0, len(crops), self.batch_size):
            results.extend(self._read_chunk(crops[start : start + self.batch_size]))
        return results

    def _read_chunk(self, crops: Sequence[RegionCrop]) -> list[Recognition]:
        items = [
            (crop.image if isinstance(crop.image, Image.Image) else to_pil(crop.image),
             self._kind(crop).value)
            for crop in crops
        ]

        try:
            raw = self._engine.read_batch(items)
        except Exception as exc:  # noqa: BLE001
            if len(items) == 1:
                # One unreadable field must not abort a whole invoice: the
                # other cells are still worth having, and this one comes back
                # empty and flagged for review rather than guessed.
                logger.warning(
                    "qwen failed to read region at %s: %s",
                    crops[0].region.bbox.as_tuple(),
                    exc,
                )
                return [Recognition("")]
            logger.warning("qwen batch of %d failed (%s); retrying one at a time.", len(items), exc)
            return [self._read_chunk([crop])[0] for crop in crops]

        return [self._validated(crop, reading) for crop, reading in zip(crops, raw)]

    def _validated(self, crop: RegionCrop, reading: Recognition) -> Recognition:
        kind = self._kind(crop)
        text = output_validation.clean(reading.text, kind)

        if text is None:
            logger.info(
                "Rejected a %s reading at %s as not fitting the cell: %r",
                kind.value,
                crop.region.bbox.as_tuple(),
                reading.text,
            )
            return Recognition("", confidence=0.0)
        if not text:
            return Recognition("")
        return Recognition(text, confidence=reading.confidence)
