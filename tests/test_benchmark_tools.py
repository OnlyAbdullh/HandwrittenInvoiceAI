"""The model-comparison tooling: crops from one photo, and the bake-off's bookkeeping.

The model-running half of tools/benchmark_recognizers.py needs torch and the
weights; what is tested here is everything around it -- what gets discovered
and how it is named, what is skipped and why, how results are ranked with and
without labels, and that the single-photo export hands it the same cells the
pipeline reads.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import pytest
from PIL import Image

from tests.llamacpp_support import write_model, write_projector

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    import sys

    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load("benchmark_recognizers")


@pytest.fixture(scope="module")
def export():
    return _load("export_crops")


@pytest.fixture
def models_dir(tmp_path):
    root = tmp_path / "OCR_Models"

    gguf = root / "Nuqta-Arabic-OCR-4B-GGUF"
    gguf.mkdir(parents=True)
    write_model(gguf / "nuqta-ocr-4b-Q4_K_M.gguf")
    write_model(gguf / "nuqta-ocr-4b-Q8_0.gguf")
    write_projector(gguf / "nuqta-ocr-4b-mmproj-f16.gguf")

    text_only = root / "Some-Text-GGUF"
    text_only.mkdir()
    write_model(text_only / "model.gguf")

    def hf(name: str, config: dict):
        folder = root / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")

    hf("Arabic-Qwen3.5-OCR-v4", {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})
    hf("Qari-OCR-v0.3", {"architectures": ["Qwen2VLForConditionalGeneration"], "model_type": "qwen2_vl"})
    hf("RayR1-trocr-base-arabic-handwritten", {"architectures": ["VisionEncoderDecoderModel"]})
    hf("gagan3012-Florence-2-FT-ArabicOCR", {"architectures": ["Florence2ForConditionalGeneration"], "model_type": "florence2"})
    hf("Mystery-Model", {"architectures": ["SomethingElse"]})
    (root / "Ali0044-Qalam_Net_V2").mkdir()
    return root


class TestDiscovery:
    def test_every_folder_is_classified_and_each_quantization_is_its_own_entry(self, bench, models_dir):
        found = {c.name: c for c in bench.discover(models_dir, None)}

        assert found["Nuqta-Arabic-OCR-4B-GGUF [nuqta-ocr-4b-Q4_K_M]"].family == "gguf"
        assert found["Nuqta-Arabic-OCR-4B-GGUF [nuqta-ocr-4b-Q8_0]"].mmproj.name == "nuqta-ocr-4b-mmproj-f16.gguf"
        assert "mmproj" not in " ".join(found)  # the projector is not a model of its own
        assert found["Arabic-Qwen3.5-OCR-v4"].family == "chat_vlm"
        assert found["Arabic-Qwen3.5-OCR-v4"].model_type == "qwen3_5"
        assert found["Qari-OCR-v0.3"].family == "chat_vlm"
        assert found["RayR1-trocr-base-arabic-handwritten"].family == "encoder_decoder"
        assert found["gagan3012-Florence-2-FT-ArabicOCR"].family == "florence2"

    def test_what_cannot_run_is_reported_with_a_reason(self, bench, models_dir):
        found = {c.name: c for c in bench.discover(models_dir, None)}

        assert "without an *mmproj*" in found["Some-Text-GGUF"].skipped_reason
        assert "no config.json" in found["Ali0044-Qalam_Net_V2"].skipped_reason
        assert "unrecognized architecture" in found["Mystery-Model"].skipped_reason

    def test_only_filters_by_name(self, bench, models_dir):
        names = [c.name for c in bench.discover(models_dir, ["nuqta", "qari"])]
        assert names == [
            "Nuqta-Arabic-OCR-4B-GGUF [nuqta-ocr-4b-Q4_K_M]",
            "Nuqta-Arabic-OCR-4B-GGUF [nuqta-ocr-4b-Q8_0]",
            "Qari-OCR-v0.3",
        ]

    def test_a_skipped_candidate_is_never_loaded(self, bench, models_dir):
        mystery = next(c for c in bench.discover(models_dir, None) if c.name == "Mystery-Model")
        result = bench.run_model(mystery, [], None, "generic", "llama-server", None)
        assert result.skipped_reason and not result.per_crop


def result(bench, name, readings):
    r = bench.ModelResult(name=name, family="chat_vlm", device="cpu")
    for expected, predicted, kind in readings:
        r.per_crop.append({
            "image": f"images/{len(r.per_crop)}.png", "cell": "r1c1", "kind": kind, "role": "quantity",
            "expected": expected, "predicted": predicted, "cer": bench.cer(predicted, expected),
            "valid": bench.passes_validation(predicted, kind), "confidence": None, "seconds": 0.5,
        })
    return r


class TestScoring:
    def test_cer_uses_the_pipelines_normalization_and_ignores_unlabelled_crops(self, bench):
        assert bench.cer("١٢", "12") == 0.0  # Arabic-Indic digits fold
        assert bench.cer("رقم الفاتورة : 00010", "00010") == 0.0  # printed caption dropped
        assert bench.cer("anything", None) is None

    def test_validation_is_the_pipelines_own(self, bench):
        assert bench.passes_validation("١٢", "number") is True
        assert bench.passes_validation("وهو نائم بكلمات", "number") is False

    def test_labelled_runs_are_ranked_by_error_rate(self, bench):
        good = result(bench, "good", [("12", "12", "number"), ("350", "350", "number")])
        bad = result(bench, "bad", [("12", "17", "number"), ("350", "3", "number")])
        ranked, by_cer = bench.rank([bad, good])
        assert by_cer and [r.name for r in ranked] == ["good", "bad"]
        assert good.exact_match_rate == 1.0

    def test_unlabelled_runs_are_ranked_by_validation_then_speed(self, bench):
        sane = result(bench, "sane", [(None, "12", "number"), (None, "350", "number")])
        prose = result(bench, "prose", [(None, "وهو نائم", "number"), (None, "12", "number")])
        ranked, by_cer = bench.rank([prose, sane])
        assert not by_cer and [r.name for r in ranked] == ["sane", "prose"]
        assert sane.valid_rate == 1.0 and prose.valid_rate == 0.5

    def test_the_html_report_shows_every_crop_beside_every_reading(self, bench):
        crops = [{"image": "images/0.png", "text": None, "kind": "number", "cell": "r1c1", "role": "quantity",
                  "pil": Image.new("RGB", (80, 30), "white")}]
        r = result(bench, "Nuqta [Q4_K_M]", [(None, "<b>١٢</b>", "number")])
        skipped = bench.ModelResult(name="Qalam", skipped_reason="no config.json")

        page = bench.to_html([r, skipped], crops, "pipeline", "datasets/my_invoice")

        assert "data:image/png;base64," in page
        assert "&lt;b&gt;١٢&lt;/b&gt;" in page  # readings are escaped, never rendered as markup
        assert "no label" in page and "no config.json" in page
        assert "Nuqta [Q4_K_M]" in page


class TestPhotoExport:
    def test_a_photo_yields_the_cells_the_pipeline_reads(self, export, tmp_path):
        from tests.test_llamacpp_pipeline import form_a_photo

        photo = tmp_path / "my invoice.jpg"
        cv2.imwrite(str(photo), form_a_photo())
        out = tmp_path / "crops"

        assert export.main(["--photo", str(photo), "--out", str(out), "--config", "config/nuqta_config.yaml"]) == 0

        rows = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 57  # form A: 76 boxes less captions and printed line numbers
        assert all(row["text"] is None for row in rows)
        assert {"invoice_number", "invoice_date", "quantity", "product_name"} <= {row["role"] for row in rows}
        assert all((out / row["image"]).exists() for row in rows)

    def test_an_unreadable_photo_fails_cleanly(self, export, tmp_path):
        assert export.main(["--photo", str(tmp_path / "missing.jpg"), "--out", str(tmp_path / "c")]) == 1
