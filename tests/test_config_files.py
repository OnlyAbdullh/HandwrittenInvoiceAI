"""Every shipped YAML config loads, and every component it names would build.

The cheap components are built for real. The expensive ones (OCR engine, text
detector, recognizer) are not -- that would load model weights -- but their
parameters are checked against their constructors, which is exactly the class
of mistake that used to ship: `column_kinds: [ ... ]`, or keys a component
silently ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ocr.cropping  # noqa: F401
import ocr.recognizers  # noqa: F401
import ocr.refiners  # noqa: F401
from config.loader import expand_env, load_config
from core.exceptions import ConfigurationError
from core.registry import check_params
from ocr.engines import stub_engine  # noqa: F401
from ocr.factory import build_region_cropper, build_region_refiner
from ocr.recognizers.registry import recognizer_registry
from ocr.registry import engine_registry
from orchestration.flows.registry import flow_registry
from output.factory import build_output_formatter
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from string_matching.factory import build_string_matcher
from table_extraction.classifiers.factory import build_layout_classifier
from table_extraction.factory import build_table_extractor

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", ["nuqta_config.yaml", "qwen_config.yaml", "qwen_detector_config.yaml", "surya_config.yaml"])
def test_every_approach_reads_the_same_page_as_the_default(name: str):
    """tools/compare_pipelines.py compares *reading* approaches. If one config
    preprocesses the page or labels the layout differently, the comparison
    measures that instead -- the Surya configs had drifted that way (binarized
    pages, columns measured on form A only)."""
    default = load_config(CONFIG_DIR / "default_config.yaml")
    other = load_config(CONFIG_DIR / name)
    assert other.preprocessing == default.preprocessing
    assert other.table_extraction == default.table_extraction


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads_and_builds(path: Path):
    config = load_config(path)

    for steps in (
        config.preprocessing.geometric_steps,
        config.preprocessing.ocr_photometric_steps,
        config.preprocessing.table_photometric_steps,
    ):
        PreprocessingPipelineBuilder.build(steps)

    build_table_extractor(config.table_extraction)
    build_layout_classifier(config.table_extraction)
    build_string_matcher(config.string_matching)
    build_output_formatter(config.output)
    flow_cls = flow_registry.get(config.flow.name)
    check_params("flow", config.flow.name, flow_cls.__init__, config.flow.params)

    if config.ocr.cropper:
        build_region_cropper(config.ocr, config.flow.name)
    if config.ocr.refiner:
        build_region_refiner(config.ocr, config.flow.name)

    if config.ocr.recognizer:
        name = config.ocr.recognizer.name
        check_params("recognizer", name, recognizer_registry.get(name), config.ocr.recognizer.params)
    if config.ocr.engine and config.ocr.engine in engine_registry.available():
        check_params("engine", config.ocr.engine, engine_registry.get(config.ocr.engine), config.ocr.engine_params)
    if config.ocr.detector:
        detector_registry = pytest.importorskip("ocr.detectors.registry").detector_registry
        pytest.importorskip("ocr.detectors.surya_detector")
        name = config.ocr.detector.name
        check_params("detector", name, detector_registry.get(name), config.ocr.detector.params)


def test_no_config_points_at_a_developer_machine():
    """Machine paths belong in the environment, not in committed files."""
    for path in CONFIGS:
        text = path.read_text(encoding="utf-8")
        assert ":\\\\" not in text and ":/" not in text.replace("http://", ""), path.name


class TestEnvExpansion:
    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("INVOICE_AI_TEST_X", raising=False)
        assert expand_env("${INVOICE_AI_TEST_X:-fallback/path}") == "fallback/path"

    def test_environment_wins(self, monkeypatch):
        monkeypatch.setenv("INVOICE_AI_TEST_X", "D:/models/q")
        assert expand_env({"a": ["${INVOICE_AI_TEST_X:-x}"]}) == {"a": ["D:/models/q"]}

    def test_unset_without_default_is_an_error(self, monkeypatch):
        monkeypatch.delenv("INVOICE_AI_TEST_X", raising=False)
        with pytest.raises(ValueError, match="INVOICE_AI_TEST_X"):
            expand_env("${INVOICE_AI_TEST_X}")

    def test_non_strings_pass_through(self):
        assert expand_env({"n": 3, "b": True, "none": None}) == {"n": 3, "b": True, "none": None}


class TestStrictParams:
    def test_an_unknown_component_param_is_rejected_naming_the_valid_ones(self):
        from ocr.cropping.registry import cropper_registry

        with pytest.raises(ConfigurationError, match="Unknown parameter.*snap_to_columns.*Accepted:.*min_ink_ratio"):
            cropper_registry.create("padded_crop", snap_to_columns=True)

    def test_an_unknown_flow_param_is_rejected(self):
        from mapping.cell_mapper import CellMapper
        from orchestration.flows.factory import build_flow

        config = load_config(CONFIG_DIR / "surya_config.yaml")
        config.ocr.engine = "stub"
        config.flow.params = {"bogus": 1}
        with pytest.raises(ConfigurationError, match="bogus"):
            build_flow(config, CellMapper())
