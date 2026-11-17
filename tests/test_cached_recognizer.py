"""The reading cache: a crop is read by the model once, ever.

It exists for measurement (tools/evaluate.py --reading-cache). What matters is
that it can never answer with a reading that was made of something else: other
pixels, another kind of cell, or another model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.domain.geometry import BoundingBox
from core.domain.ocr import Recognition, RegionCrop, TextRegion
from core.domain.roles import ContentKind
from core.exceptions import ConfigurationError
from ocr.recognizers.cached_recognizer import CachedRecognizer
from ocr.recognizers.registry import recognizer_registry

BUILT: list[str] = []
READ: list[int] = []


@recognizer_registry.register("counting_for_cache_tests")
class CountingRecognizer:
    """Answers with the crop's first pixel and records every call."""

    name = "counting_for_cache_tests"

    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        BUILT.append(suffix)

    def read(self, crop: RegionCrop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops) -> list[Recognition]:
        READ.append(len(crops))
        return [Recognition(f"{int(c.image.flat[0])}{self.suffix}", confidence=0.9) for c in crops]


@pytest.fixture(autouse=True)
def reset_counters():
    BUILT.clear()
    READ.clear()


def crop(value: int, kind: ContentKind = ContentKind.NUMBER) -> RegionCrop:
    region = TextRegion(bbox=BoundingBox(0, 0, 4, 4), content_kind=kind)
    return RegionCrop(region=region, image=np.full((4, 4), value, dtype=np.uint8))


def cached(tmp_path, **params) -> CachedRecognizer:
    return CachedRecognizer(
        recognizer={"name": "counting_for_cache_tests", "params": params}, cache_dir=str(tmp_path / "cache")
    )


def test_a_crop_is_read_by_the_model_once(tmp_path):
    first = cached(tmp_path).read_all([crop(1), crop(2)])
    assert [r.text for r in first] == ["1", "2"] and READ == [2]

    again = cached(tmp_path)  # a later run: a new process, the same folder
    assert [(r.text, r.confidence) for r in again.read_all([crop(1), crop(2)])] == [("1", 0.9), ("2", 0.9)]
    assert READ == [2]  # nothing was read again
    assert (again.hits, again.misses) == (2, 0)


def test_the_model_is_not_loaded_when_every_crop_is_cached(tmp_path):
    cached(tmp_path).read(crop(7))
    assert BUILT == [""]

    cached(tmp_path).read(crop(7))
    assert BUILT == [""]  # the second recognizer was never built


def test_only_the_new_crops_are_read_and_the_order_is_kept(tmp_path):
    recognizer = cached(tmp_path)
    recognizer.read_all([crop(2)])

    readings = recognizer.read_all([crop(1), crop(2), crop(3)])

    assert [r.text for r in readings] == ["1", "2", "3"]
    assert READ == [1, 2]  # the second call read crops 1 and 3 only


def test_the_same_pixels_as_another_kind_of_cell_are_read_again(tmp_path):
    """The kind selects the prompt and the decoding constraint."""
    recognizer = cached(tmp_path)
    recognizer.read(crop(5, ContentKind.NUMBER))
    recognizer.read(crop(5, ContentKind.ARABIC_TEXT))
    assert READ == [1, 1]


def test_a_differently_configured_model_never_answers_from_anothers_readings(tmp_path):
    assert cached(tmp_path).read(crop(9)).text == "9"
    assert cached(tmp_path, suffix="!").read(crop(9)).text == "9!"


def test_a_model_changed_in_place_is_read_again(tmp_path):
    """A fine-tune written again into the same folder is another model under
    the same name: its parameters are identical, its weights are not."""
    import os

    weights = tmp_path / "model" / "adapter.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"first training run")
    folder = str(weights.parent)

    cached(tmp_path, suffix=folder).read(crop(6))
    cached(tmp_path, suffix=folder).read(crop(6))
    assert READ == [1]  # the same files: answered from the cache

    weights.write_bytes(b"second training run, longer")
    os.utime(weights, ns=(1, 1))
    cached(tmp_path, suffix=folder).read(crop(6))
    assert READ == [1, 1]


def test_a_blank_reading_is_remembered_as_blank(tmp_path):
    class Blank(CountingRecognizer):
        def read_all(self, crops):
            READ.append(len(crops))
            return [Recognition("") for _ in crops]

    recognizer_registry.register("blank_for_cache_tests")(Blank)
    spec = {"name": "blank_for_cache_tests", "params": {}}
    first = CachedRecognizer(recognizer=spec, cache_dir=str(tmp_path))
    assert first.read(crop(3)).text == ""
    assert CachedRecognizer(recognizer=spec, cache_dir=str(tmp_path)).read(crop(3)) == Recognition("")
    assert READ == [1]


def test_a_damaged_entry_is_read_again_not_trusted(tmp_path):
    recognizer = cached(tmp_path)
    recognizer.read(crop(4))
    entry = next((tmp_path / "cache").rglob("*.json"))
    entry.write_text("{not json", encoding="utf-8")

    assert cached(tmp_path).read(crop(4)).text == "4"
    assert READ == [1, 1]
    assert json.loads(entry.read_text(encoding="utf-8"))["text"] == "4"  # and repaired


def test_it_reports_as_the_model_it_wraps(tmp_path):
    assert cached(tmp_path).name == "counting_for_cache_tests"


@pytest.mark.parametrize("spec", [
    {"params": {}},
    "qwen",
    {"name": "cached", "params": {}},
])
def test_a_malformed_specification_is_rejected(tmp_path, spec):
    with pytest.raises(ConfigurationError):
        CachedRecognizer(recognizer=spec, cache_dir=str(tmp_path))


def test_a_cache_directory_is_required():
    with pytest.raises(ConfigurationError):
        CachedRecognizer(recognizer={"name": "echo"}, cache_dir="")
