"""Deterministic fake recognizer, for exercising the flows without a model.

Same rationale as the `stub` OCR engine: the detect/crop/recognize machinery,
the layout classifier, the mapper and the parser all need to be testable on a
machine with no weights and no second virtualenv. This reads nothing -- it
derives a stable string from the crop's pixels and its declared content kind, so
the same page always yields the same answer.

Like `stub`, it is **never** a fallback for a real recognizer that failed.
Selecting it is always explicit. Fabricated data presented as a reading is worse
than an error.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from core.domain.ocr import Recognition, RegionCrop
from core.domain.roles import ContentKind
from ocr.recognizers.registry import recognizer_registry

_WORDS = ("سكر", "أرز", "زيت", "شاي", "حليب", "ورق", "قلم")


@recognizer_registry.register("echo")
class EchoRecognizer:
    name = "echo"

    def __init__(self, blank_ratio: float = 0.0):
        # How often to answer with "" instead, so the empty-cell path gets
        # exercised too. 0 by default: a test that wants blanks asks for them.
        self.blank_ratio = blank_ratio

    def _seed(self, crop: RegionCrop) -> int:
        payload = np.ascontiguousarray(crop.image).tobytes()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:4], "big")

    def _text(self, crop: RegionCrop, seed: int) -> str:
        kind = crop.content_kind
        if kind in (ContentKind.NUMBER, ContentKind.CODE):
            return str(seed % 1000)
        if kind is ContentKind.DATE:
            return f"2026/{seed % 12 + 1}/{seed % 28 + 1}"
        if kind is ContentKind.ARABIC_TEXT:
            return f"{_WORDS[seed % len(_WORDS)]} {seed % 100}"
        return f"txt{seed % 10000}"

    def read(self, crop: RegionCrop) -> Recognition:
        seed = self._seed(crop)

        if self.blank_ratio and (seed % 100) < self.blank_ratio * 100:
            return Recognition("")

        # A fabricated reading carries a fabricated -- but deterministic and
        # in-range -- score, so the confidence plumbing is exercised end to end.
        return Recognition(self._text(crop, seed), confidence=0.5 + (seed % 50) / 100)

    def read_all(self, crops: Sequence[RegionCrop]) -> list[Recognition]:
        return [self.read(crop) for crop in crops]
