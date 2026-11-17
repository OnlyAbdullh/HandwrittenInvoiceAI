"""A recognizer that remembers every reading it has made, on disk.

A measurement tool, not a service feature. Reading one invoice takes the VLM
tens of seconds, and almost every experiment after the first re-reads crops it
has already read: a change to the parser or to the arithmetic repair changes no
crop at all, and a change to the layout changes a few. Wrapped in this, a
re-run reads only the crops that are new, a run that was stopped half way
resumes where it stopped, and the model is not even loaded when nothing is new.

    recognizer:
      name: cached
      params:
        cache_dir: evaluation/reading_cache
        recognizer: {name: qwen, params: {...}}

`tools/evaluate.py --reading-cache DIR` wraps the configured recognizer this
way without editing a config.

A reading is keyed by the crop's exact pixels, its content kind, and the wrapped
recognizer's full specification (its name and every parameter, the model path
included), so a different model, or the same model configured differently,
never answers from another's readings. What the key cannot see is code: after
changing a prompt or the decoding policy, pass a new `salt` or delete the
directory.

A Decorator: it implements the :class:`TextRecognizer` protocol it wraps, so no
flow knows the difference.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.domain.ocr import Recognition, RegionCrop
from core.exceptions import ConfigurationError
from core.interfaces.text_recognizer import TextRecognizer
from ocr.recognizers.registry import recognizer_registry

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _fingerprint(value: object) -> list | None:
    """Sizes and modification times of the file, or of the files directly in
    the folder, that a parameter names; None when it names neither.

    Paths in a configuration are written relative to the project, so both that
    reading and the plain one are tried.
    """
    if not isinstance(value, str) or not value or len(value) > 1024:
        return None
    for path in (Path(value), _PROJECT_ROOT / value):
        try:
            if path.is_file():
                stat = path.stat()
                return [[path.name, stat.st_size, stat.st_mtime_ns]]
            if path.is_dir():
                return sorted(
                    [entry.name, entry.stat().st_size, entry.stat().st_mtime_ns]
                    for entry in path.iterdir() if entry.is_file()
                )
        except OSError:
            continue
    return None


@recognizer_registry.register("cached")
class CachedRecognizer:
    def __init__(self, recognizer: Mapping[str, Any], cache_dir: str, salt: str = ""):
        if not isinstance(recognizer, Mapping) or not isinstance(recognizer.get("name"), str):
            raise ConfigurationError("cached recognizer: `recognizer` must be {name: <recognizer>, params: {...}}.")
        if recognizer["name"] == "cached":
            raise ConfigurationError("cached recognizer: it cannot wrap another cached recognizer.")
        if not cache_dir:
            raise ConfigurationError("cached recognizer: `cache_dir` is required.")

        self._inner_name: str = recognizer["name"]
        self._inner_params: dict = dict(recognizer.get("params") or {})
        self._inner: TextRecognizer | None = None
        self._dir = Path(cache_dir)
        # Everything that decides what the wrapped recognizer would answer,
        # as far as configuration can say -- and, for a parameter that names a
        # folder or file, what is in it: a model fine-tuned again into the same
        # folder is a different model under the same name.
        self._identity = json.dumps(
            {
                "name": self._inner_name,
                "params": self._inner_params,
                "files": {key: mark for key, value in self._inner_params.items() if (mark := _fingerprint(value))},
                "salt": salt,
            },
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")

        # Reported as the wrapped recognizer: a report says which model read
        # the page, and that is not "cached".
        self.name = self._inner_name
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------

    def _key(self, crop: RegionCrop) -> str:
        image = np.ascontiguousarray(crop.image)
        digest = hashlib.sha256(self._identity)
        digest.update(crop.content_kind.value.encode("utf-8"))
        digest.update(str(image.shape).encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def _load(self, key: str) -> Recognition | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return Recognition(text=entry["text"], confidence=entry.get("confidence"))
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Unreadable cached reading %s; reading the crop again.", path)
            return None

    def _store(self, key: str, crop: RegionCrop, reading: Recognition) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "text": reading.text,
            "confidence": reading.confidence,
            "kind": crop.content_kind.value,
            "recognizer": self._inner_name,
        }
        # Written whole, then renamed: a run stopped mid-write must not leave
        # half a file that a later run trusts.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _recognizer(self) -> TextRecognizer:
        # Built on the first crop nobody has read: a run answered entirely
        # from the cache never loads a model.
        if self._inner is None:
            self._inner = recognizer_registry.create(self._inner_name, **self._inner_params)
        return self._inner

    # ------------------------------------------------------------------

    def read(self, crop: RegionCrop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops: Sequence[RegionCrop]) -> list[Recognition]:
        keys = [self._key(crop) for crop in crops]
        readings: list[Recognition | None] = [self._load(key) for key in keys]

        unread = [i for i, reading in enumerate(readings) if reading is None]
        self.hits += len(crops) - len(unread)
        self.misses += len(unread)
        if unread:
            fresh = self._recognizer().read_all([crops[i] for i in unread])
            for i, reading in zip(unread, fresh):
                readings[i] = reading
                self._store(keys[i], crops[i], reading)
        return [reading for reading in readings if reading is not None]

    # Lifecycle hooks the flows probe. Warming up would load the model whether
    # or not anything is left to read, which is the cost this exists to avoid.

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
