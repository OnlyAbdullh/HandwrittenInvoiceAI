"""The routed recognizer: a model per cell kind, a second model when unsure."""

from __future__ import annotations

import numpy as np
import pytest

from core.domain.geometry import BoundingBox
from core.domain.ocr import Recognition, RegionCrop, TextRegion
from core.domain.roles import ContentKind
from core.exceptions import ConfigurationError
from ocr.recognizers.registry import recognizer_registry

CALLS: list[tuple[str, int]] = []
CLOSED: list[str] = []


class Scripted:
    """Answers `<label>:<kind>` with a fixed confidence, and records each batch."""

    name = "scripted"

    def __init__(self, label: str, confidence: float | None = 0.9, unsure_kinds: tuple = ()):
        self.label, self.confidence, self.unsure_kinds = label, confidence, tuple(unsure_kinds)

    def read_all(self, crops):
        CALLS.append((self.label, len(crops)))
        return [Recognition(f"{self.label}:{c.content_kind.value}",
                            0.1 if c.content_kind.value in self.unsure_kinds else self.confidence)
                for c in crops]

    def read(self, crop):
        return self.read_all([crop])[0]

    def close(self):
        CLOSED.append(self.label)


try:
    recognizer_registry.register("test_scripted")(Scripted)
except ValueError:
    pass  # already registered by an earlier import of this module


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    CLOSED.clear()


def crop(kind: ContentKind) -> RegionCrop:
    return RegionCrop(region=TextRegion(bbox=BoundingBox(0, 0, 10, 10), content_kind=kind),
                      image=np.zeros((10, 10, 3), np.uint8))


def spec(label: str, **params) -> dict:
    return {"name": "test_scripted", "params": {"label": label, **params}}


def build(**params):
    return recognizer_registry.create("routed", **params)


def test_each_kind_goes_to_its_model_and_order_is_kept():
    r = build(recognizers={"digits": spec("digits"), "words": spec("words")},
              default="digits", routes={"arabic_text": "words"})
    kinds = [ContentKind.NUMBER, ContentKind.ARABIC_TEXT, ContentKind.DATE, ContentKind.ARABIC_TEXT]

    texts = [x.text for x in r.read_all([crop(k) for k in kinds])]

    assert texts == ["digits:number", "words:arabic_text", "digits:date", "words:arabic_text"]
    assert sorted(CALLS) == [("digits", 2), ("words", 2)]  # one batch per model, not one call per cell


def test_unsure_readings_are_read_again_by_the_fallback_model():
    r = build(recognizers={"main": spec("main", unsure_kinds=("arabic_text",)), "second": spec("second")},
              default="main", fallback={"to": "second", "below_confidence": 0.5})

    texts = [x.text for x in r.read_all([crop(ContentKind.NUMBER), crop(ContentKind.ARABIC_TEXT)])]

    assert texts == ["main:number", "second:arabic_text"]
    assert CALLS == [("main", 2), ("second", 1)]


def test_unknown_confidence_counts_as_unsure():
    r = build(recognizers={"main": spec("main", confidence=None), "second": spec("second")},
              default="main", fallback={"to": "second", "below_confidence": 0.5})
    assert r.read(crop(ContentKind.NUMBER)).text == "second:number"


def test_a_cell_the_fallback_model_already_read_is_not_read_twice():
    r = build(recognizers={"main": spec("main"), "second": spec("second", confidence=0.1)},
              default="main", routes={"arabic_text": "second"}, fallback={"to": "second", "below_confidence": 0.5})
    r.read_all([crop(ContentKind.ARABIC_TEXT)])
    assert CALLS == [("second", 1)]


@pytest.mark.parametrize(("params", "message"), [
    ({"recognizers": {}, "default": "a"}, "names no recognizer"),
    ({"recognizers": {"a": spec("a")}, "default": "b"}, "not defined"),
    ({"recognizers": {"a": spec("a")}, "default": "a", "routes": {"price": "a"}}, "unknown cell kind"),
    ({"recognizers": {"a": spec("a"), "b": spec("b")}, "default": "a"}, "never used"),
    ({"recognizers": {"a": spec("a")}, "default": "a", "fallback": {"to": "a"}}, "below_confidence"),
    ({"recognizers": {"a": spec("a")}, "default": "a", "fallback": {"to": "a", "below_confidence": 2}}, r"\(0, 1\]"),
    ({"recognizers": {"a": {"name": "routed"}}, "default": "a"}, "cannot contain"),
])
def test_configuration_mistakes_fail_before_any_model_loads(params, message):
    with pytest.raises(ConfigurationError, match=message):
        build(**params)


def test_a_member_that_fails_to_build_closes_the_ones_already_built():
    with pytest.raises(ConfigurationError):
        build(recognizers={"a": spec("a"), "b": {"name": "test_scripted", "params": {"label": "b", "bogus": 1}}},
              default="a", routes={"number": "b"})
    assert CLOSED == ["a"]


def test_closing_closes_every_member():
    build(recognizers={"a": spec("a"), "b": spec("b")}, default="a", routes={"date": "b"}).close()
    assert sorted(CLOSED) == ["a", "b"]


def test_echo_members_run_through_the_real_layout_flow():
    """The composite is an ordinary recognizer to the pipeline."""
    r = build(recognizers={"x": {"name": "echo"}, "y": {"name": "echo", "params": {"blank_ratio": 1.0}}},
              default="x", routes={"arabic_text": "y"})
    assert r.read(crop(ContentKind.ARABIC_TEXT)).text == ""
    assert r.read(crop(ContentKind.NUMBER)).text != ""
    assert r.is_ready()



def test_the_layout_flow_runs_end_to_end_through_the_routed_recognizer():
    from config.schema import ComponentConfig
    from core.domain.image_payload import ImagePayload
    from orchestration.pipeline_orchestrator import PipelineOrchestrator
    from tests.test_coordinate_space import config, tilted_invoice

    cfg = config()
    cfg.ocr.recognizer = ComponentConfig(name="routed", params={
        "recognizers": {"main": {"name": "echo"}, "names": {"name": "echo", "params": {"blank_ratio": 1.0}}},
        "default": "main", "routes": {"arabic_text": "names"},
    })
    orchestrator = PipelineOrchestrator(cfg)
    result = orchestrator.run(ImagePayload(image=tilted_invoice()), write_debug_images=False, persist=False).result

    assert orchestrator.engine_name == "layout+routed"
    assert result.ocr_engine == "layout+routed" and result.elements, "the grid's cells were read"
