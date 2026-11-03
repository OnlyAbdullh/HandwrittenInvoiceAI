"""The half of a VLM recognizer that does not depend on how the model is run.

A vision-language model reads a crop when it is shown the crop and told what
kind of characters to expect. Everything around that one call is the same
whichever runtime answers it -- transformers in a worker process, or llama.cpp
behind a local server:

* turning each crop's :class:`~core.domain.roles.ContentKind` into the kind the
  backend picks its prompt and decoding mode from,
* chunking crops into batches without ever reordering or dropping one,
* isolating a failure to the one crop that caused it, and
* rejecting readings that cannot be what the cell holds
  (:mod:`ocr.recognizers.output_validation`).

That is this class. A concrete recognizer supplies only a *backend*: any object
with ``read_batch([(PIL image, kind), ...]) -> list[Recognition]`` plus the
optional ``is_ready`` / ``warmup`` / ``close`` lifecycle hooks the flows probe.
"""

from __future__ import annotations

import logging
from typing import Mapping, Protocol, Sequence

from PIL import Image

from core.domain.ocr import Recognition, RegionCrop
from core.domain.roles import ContentKind
from ocr.recognizers import output_validation

logger = logging.getLogger(__name__)


class VisionBackend(Protocol):
    """What a prompted recognizer needs from the runtime behind it."""

    def read_batch(self, items: Sequence[tuple[Image.Image, str]]) -> list[Recognition]: ...


def require_prompt_for_every_kind(prompts: Mapping[str, str], recognizer: str) -> None:
    """Fail at construction, not mid-page, when a content kind has no prompt.

    A missing prompt would silently fall back to the generic one for every cell
    of that kind -- the digits-only instruction would simply stop being sent.
    """
    missing = [k.value for k in ContentKind if k.value not in prompts]
    if missing:
        raise ValueError(
            f"{recognizer} recognizer: no prompt defined for content kind(s) {missing}."
        )


def check_batch_settings(recognizer: str, default_kind: str, batch_size: int) -> ContentKind:
    """Validate the two settings every prompted recognizer shares."""
    if batch_size < 1:
        raise ValueError(f"{recognizer} recognizer: batch_size must be >= 1, got {batch_size}.")
    return ContentKind(default_kind)


def to_pil(image) -> Image.Image:
    """A crop as the RGB PIL image every VLM backend takes."""
    if isinstance(image, Image.Image):
        return image
    # Imported here: the numpy conversion lives with the Qwen engine, which
    # owns the one copy of it.
    from ocr.engines.qwen_vlm_engine import to_pil as array_to_pil

    return array_to_pil(image)


class PromptedRecognizer:
    """Batching, failure isolation and output validation for a VLM backend."""

    name = "prompted"

    def __init__(self, backend: VisionBackend, default_kind: ContentKind, batch_size: int):
        self._backend = backend
        self.default_kind = default_kind
        self.batch_size = batch_size

    # -- lifecycle, delegated -------------------------------------------

    def is_ready(self) -> bool:
        probe = getattr(self._backend, "is_ready", None)
        return bool(probe()) if callable(probe) else True

    def warmup(self) -> None:
        warm = getattr(self._backend, "warmup", None)
        if callable(warm):
            warm()

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    # -- reading ----------------------------------------------------------

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
        items = [(to_pil(crop.image), self._kind(crop).value) for crop in crops]

        try:
            raw = self._backend.read_batch(items)
        except Exception as exc:  # noqa: BLE001
            if len(items) == 1:
                # One unreadable field must not abort a whole invoice: the
                # other cells are still worth having, and this one comes back
                # empty and flagged for review rather than guessed.
                logger.warning(
                    "%s failed to read region at %s: %s",
                    self.name,
                    crops[0].region.bbox.as_tuple(),
                    exc,
                )
                return [Recognition("")]
            logger.warning(
                "%s batch of %d failed (%s); retrying one at a time.", self.name, len(items), exc
            )
            return [self._read_chunk([crop])[0] for crop in crops]

        if len(raw) != len(crops):
            raise RuntimeError(
                f"{self.name} backend returned {len(raw)} reading(s) for {len(crops)} crop(s)."
            )
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
