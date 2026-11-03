"""The llamacpp recognizer inside the real pipeline, from the shipped config.

`config/nuqta_config.yaml` is loaded as shipped -- only the two model paths come
from the environment, the way a user sets them -- and run end to end over a
page drawn from the geometry of a real photographed form A
(tests/fixtures/form_a_cells.json). Preprocessing, grid detection,
classification, cropping, the blank-cell gate, output validation, parsing,
reconciliation and the wire formatter are all the real ones. Only the model's
reading comes from the stand-in server.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from api.pipeline_pool import PipelinePool
from config.loader import load_config
from core.domain.catalog import Catalogs, NamedEntry
from core.domain.image_payload import ImagePayload
from core.domain.ocr import RegionCrop, TextRegion
from core.domain.geometry import BoundingBox
from core.domain.roles import ContentKind
from ocr.decoding_policy import DATE_CHARS, NUMBER_CHARS
from ocr.llamacpp.client import charset_grammar
from ocr.recognizers import llamacpp_recognizer
from ocr.recognizers.registry import recognizer_registry
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from tests.llamacpp_support import fake_launcher, read_record, write_model, write_projector

ROOT = Path(__file__).resolve().parent.parent
NUQTA_CONFIG = ROOT / "config" / "nuqta_config.yaml"
FORM_A = Path(__file__).parent / "fixtures" / "form_a_cells.json"
EXPECTED_CONFIDENCE = round(math.sqrt(0.9 * 0.8), 4)


@pytest.fixture
def nuqta_env(tmp_path, monkeypatch):
    """The two model paths set the way the setup guide says, and the stand-in server."""
    monkeypatch.setenv("INVOICE_AI_NUQTA_MODEL", str(write_model(tmp_path / "nuqta-ocr-4b-Q4_K_M.gguf")))
    monkeypatch.setenv("INVOICE_AI_NUQTA_MMPROJ", str(write_projector(tmp_path / "nuqta-ocr-4b-mmproj-f16.gguf")))
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(record))
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    monkeypatch.setattr(llamacpp_recognizer, "_launcher", lambda server_path: fake_launcher())
    return record


def nuqta_config():
    config = load_config(NUQTA_CONFIG)
    config.debug_dir = None
    return config


def form_a_photo() -> np.ndarray:
    """Form A's ruled boxes, every box written in, on a darker background so
    perspective correction finds the paper edge as it would on a photo."""
    data = json.loads(FORM_A.read_text(encoding="utf-8"))
    width, height = data["image_size"]
    page = np.full((height, width, 3), 255, np.uint8)
    for table in data["tables"]:
        for cell in table["cells"]:
            b = cell["bbox"]
            cv2.rectangle(page, (b["x"] - 2, b["y"] - 2), (b["x"] + b["w"] + 1, b["y"] + b["h"] + 1), (0, 0, 0), 3)
            # Handwriting stand-in: a sparse row of small glyphs across the
            # box -- enough ink for the cropper's blank-cell threshold even in
            # the large header boxes, and nothing a line or box detector could
            # mistake for a rule.
            for x in range(b["x"] + 12, b["x"] + b["w"] - 30, 55):
                cv2.putText(page, "12", (x, b["y"] + b["h"] // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 160), 2)
    margin = 60
    photo = np.full((height + 2 * margin, width + 2 * margin, 3), 70, np.uint8)
    photo[margin : margin + height, margin : margin + width] = page
    return photo


def test_the_shipped_config_selects_llamacpp_and_names_no_machine_path():
    config = load_config(NUQTA_CONFIG)
    assert config.flow.name == "layout_driven"
    assert config.ocr.recognizer.name == "llamacpp"
    assert config.ocr.recognizer.params["gpu_layers"] == "auto"
    # Same pipeline as the default config apart from the recognizer.
    default = load_config(ROOT / "config" / "default_config.yaml")
    assert config.preprocessing == default.preprocessing
    assert config.table_extraction == default.table_extraction
    assert config.ocr.cropper == default.ocr.cropper


def test_an_invoice_runs_end_to_end_through_the_shipped_config(nuqta_env):
    orchestrator = PipelineOrchestrator(nuqta_config())
    try:
        run = orchestrator.run(
            ImagePayload(image=form_a_photo()),
            catalogs=Catalogs(products=[NamedEntry(name="بوري شبابي زارا", entry_id=4)]),
            write_debug_images=False,
            persist=False,
        )
    finally:
        orchestrator.flow.recognizer.close()

    assert run.layout.source == "contour_based+bill_layout"
    assert orchestrator.engine_name == "layout+llamacpp"
    body = run.output

    # Header fields, each read with its own kind's prompt and validated.
    assert body["invoice_id"]["value"] == "00010"  # caption stripped from the code cell
    assert body["date"]["value"] == "2026-08-07"
    assert body["invoice_id"]["ocr_confidence"] == EXPECTED_CONFIDENCE

    # Line items: ten written rows on the form, every one read.
    products = body["products"]
    assert len(products) == 10
    first = products[0]
    assert first["quantity"]["value"] == 12
    assert first["quantity"]["ocr_confidence"] == EXPECTED_CONFIDENCE
    assert first["product_name"]["original_value"] == "بوري شبابي زارا"
    assert first["product_name"]["results"][0] == {
        "id": "4", "value": "بوري شبابي زارا", "string_matching_score": 1.0,
    }
    assert body["total_invoice_price"]["value"] == 12.0

    # One request per readable cell; printed captions and line numbers skipped.
    # Form A has 76 boxes: 6 column captions, 3 footer captions and 10 printed
    # line numbers are never read, leaving 57.
    sent = [e for e in read_record(nuqta_env) if e["event"] == "completion"]
    grammars = [e["body"].get("grammar") for e in sent]
    assert len(sent) == run.timings["crop_count"] == 57
    assert grammars.count(charset_grammar(NUMBER_CHARS)) == 31  # 10 rows x 3 + total in figures
    assert grammars.count(charset_grammar(DATE_CHARS)) == 1


def test_the_model_is_loaded_once_across_preprocessing_changes(nuqta_env, tmp_path, monkeypatch):
    config_file = tmp_path / "service.yaml"
    config_file.write_text(NUQTA_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    pool = PipelinePool(str(config_file))
    try:
        first = pool.get(pool.default_config)

        tweaked = pool.default_config.model_copy(deep=True)
        tweaked.preprocessing.ocr_photometric_steps[2].enabled = True  # CLAHE on
        second = pool.get(tweaked)

        assert first is not second
        assert first.flow.recognizer is second.flow.recognizer
        assert pool.is_ready()
        assert pool.engine_name == "layout+llamacpp"
        assert len([e for e in read_record(nuqta_env) if e["event"] == "launch"]) == 1

        server = first.flow.recognizer.server
    finally:
        pool.close()

    assert not server.is_alive()


def test_a_missing_model_is_reported_as_not_ready_not_as_a_bad_request(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOICE_AI_NUQTA_MODEL", str(tmp_path / "not-downloaded.gguf"))
    monkeypatch.setenv("INVOICE_AI_NUQTA_MMPROJ", str(tmp_path / "not-downloaded-mmproj.gguf"))
    pool = PipelinePool(str(NUQTA_CONFIG))

    # FileNotFoundError, which the HTTP layer maps to 503 ENGINE_NOT_READY.
    with pytest.raises(FileNotFoundError):
        pool.get(pool.default_config)
    assert pool.is_ready() is False


class TestTransformersBackendStillWorks:
    """The `qwen` recognizer now shares PromptedRecognizer with `llamacpp`; its
    subprocess transport must behave exactly as before."""

    def test_default_config_still_selects_the_transformers_recognizer(self):
        config = load_config(ROOT / "config" / "default_config.yaml")
        assert config.ocr.recognizer.name == "qwen"
        assert config.ocr.recognizer.params["backend"] == "subprocess"

    def test_qwen_recognizer_reads_through_its_worker(self, tmp_path):
        recognizer = recognizer_registry.create(
            "qwen",
            model_path=str(tmp_path),
            backend="subprocess",
            worker_python=sys.executable,
            worker_script=str(Path(__file__).parent / "fixtures" / "fake_qwen_worker.py"),
            timeout_s=10.0,
            startup_timeout_s=30.0,
        )
        try:
            crops = [
                RegionCrop(
                    region=TextRegion(bbox=BoundingBox(0, 0, 8, 8), content_kind=ContentKind.ARABIC_TEXT),
                    image=np.full((8, 8), shade, np.uint8),
                )
                for shade in (10, 200)
            ]
            readings = recognizer.read_all(crops)

            assert [len(r.text) for r in readings] == [16, 16]  # the fake worker's digests
            assert readings[0].text != readings[1].text
            assert all(r.confidence == 0.9 for r in readings)
            assert recognizer.is_ready()
        finally:
            recognizer.close()
        assert not recognizer.is_ready()
