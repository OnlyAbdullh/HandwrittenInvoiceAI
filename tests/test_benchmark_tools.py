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

    def test_a_model_kept_in_a_subfolder_is_found(self, bench, models_dir):
        nested = models_dir / "gagan3012-Florence-2-FT-ArabicOCR"
        inner = nested / "checkpoint"
        inner.mkdir()
        (nested / "config.json").rename(inner / "config.json")

        found = {c.name: c for c in bench.discover(models_dir, None)}

        assert found["gagan3012-Florence-2-FT-ArabicOCR"].family == "florence2"
        assert found["gagan3012-Florence-2-FT-ArabicOCR"].path == inner

    def test_a_folder_without_a_model_says_what_it_contains(self, bench, models_dir):
        (models_dir / "Ali0044-Qalam_Net_V2" / "model.h5").write_bytes(b"")
        found = {c.name: c for c in bench.discover(models_dir, None)}
        assert "contains: model.h5" in found["Ali0044-Qalam_Net_V2"].skipped_reason

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


class TestTruthLookup:
    def test_the_users_invoice_is_matched_to_its_ground_truth(self, export, tmp_path):
        found = export.find_truth_for(tmp_path / "bills" / "onlyone" / "img.png", ROOT / "evaluation" / "ground_truth")
        assert found is not None and found.name == "onlyone.json"

    def test_an_unknown_photo_has_no_truth(self, export, tmp_path):
        assert export.find_truth_for(tmp_path / "my_invoice.jpg", ROOT / "evaluation" / "ground_truth") is None

    def _truth(self, tmp_path, items: int) -> Path:
        truth_dir = tmp_path / "truth"
        truth_dir.mkdir()
        (truth_dir / "synthetic.json").write_text(json.dumps({
            "image": "synthetic/img.png", "invoice_number": "00010", "customer_name": "حسام المارديني",
            "city": "ريف دمشق", "total": 120,
            "items": [{"product_name": "بوري شبابي زارا", "quantity": 1, "unit_price": 12, "total_price": 12}] * items,
        }, ensure_ascii=False), encoding="utf-8")
        photo = tmp_path / "bills" / "synthetic" / "img.png"
        photo.parent.mkdir(parents=True)
        from tests.test_llamacpp_pipeline import form_a_photo

        cv2.imwrite(str(photo), form_a_photo())
        return photo

    def test_a_photo_with_truth_is_exported_labelled_without_asking(self, export, tmp_path):
        photo = self._truth(tmp_path, items=10)  # the synthetic form has ten written rows
        out = tmp_path / "crops"

        assert export.main(["--photo", str(photo), "--truth", str(tmp_path / "truth"), "--out", str(out),
                            "--config", "config/nuqta_config.yaml"]) == 0

        rows = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows and all(row["text"] for row in rows)
        assert {"00010", "12", "بوري شبابي زارا"} <= {row["text"] for row in rows}

    def test_the_truths_items_go_to_the_first_written_rows(self, export, tmp_path):
        # Ten written rows on the page, four items in the truth: an invoice is
        # filled from the top, so the four belong to the first four rows; ink
        # below them is not an item.
        photo = self._truth(tmp_path, items=4)
        out = tmp_path / "crops"

        assert export.main(["--photo", str(photo), "--truth", str(tmp_path / "truth"), "--out", str(out),
                            "--config", "config/nuqta_config.yaml"]) == 0

        rows = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        quantities = [row for row in rows if row["role"] == "quantity"]
        assert len(quantities) == 4 and all(row["text"] == "1" for row in quantities)
        assert {"00010", "120"} <= {row["text"] for row in rows}

    def test_truth_that_cannot_fit_the_table_keeps_only_header_labels(self, export, tmp_path):
        photo = self._truth(tmp_path, items=11)  # more items than the form has rows
        out = tmp_path / "crops"

        assert export.main(["--photo", str(photo), "--truth", str(tmp_path / "truth"), "--out", str(out),
                            "--config", "config/nuqta_config.yaml"]) == 0

        rows = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        by_role = {row["role"]: row["text"] for row in rows}
        assert len(rows) == 57  # every readable cell is still there to compare models on
        assert by_role["invoice_number"] == "00010"
        assert by_role["total_in_figures"] == "120"
        assert all(row["text"] is None for row in rows if row["role"] in ("quantity", "product_name"))


class TestIsolation:
    """Each model runs in its own process: a native crash costs that model only."""

    def test_a_crashing_model_becomes_a_row_not_the_end_of_the_run(self, bench, tmp_path):
        import sys

        candidate = bench.Candidate("Qari-OCR-v0.3", tmp_path, "chat_vlm")
        crash = [sys.executable, "-c", "import sys; sys.stderr.write('CUDA error: illegal memory access\\n'); sys.exit(3)"]

        result = bench.run_isolated(candidate, crash, tmp_path / "never-written.json")

        assert result.name == "Qari-OCR-v0.3"
        assert "crashed: exit code 3" in result.skipped_reason
        assert "illegal memory access" in result.skipped_reason

    def test_a_finished_child_hands_back_its_result(self, bench, tmp_path):
        import sys

        result_file = tmp_path / "result.json"
        payload = {"name": "m", "family": "gguf", "device": "llama.cpp", "load_s": 1.5, "skipped_reason": None,
                   "per_crop": [{"image": "a.png", "cer": 0.0, "valid": True, "seconds": 0.2, "confidence": 0.9}]}
        write = [sys.executable, "-c",
                 f"import json, pathlib; pathlib.Path({str(result_file)!r}).write_text(json.dumps({payload!r}))"]

        result = bench.run_isolated(bench.Candidate("m", tmp_path, "gguf"), write, result_file)

        assert result.skipped_reason is None
        assert result.exact_match_rate == 1.0 and result.load_s == 1.5

    @pytest.mark.parametrize("code, words", [
        (3221225477, "access violation"), (-1073741819, "access violation"),
        (-11, "segmentation fault"), (-9, "out-of-memory killer"), (7, "exit code 7"),
    ])
    def test_native_crash_codes_are_named(self, bench, code, words):
        assert words in bench.describe_exit(code)


class TestEmptyReadings:
    def test_an_empty_reading_of_a_written_cell_is_a_miss_not_a_pass(self, bench, monkeypatch, tmp_path):
        answers = iter(["", "EMPTY", "١٢", "وهو نائم"])

        class Stub(bench.Reader):
            def read(self, image, kind):
                return next(answers), None

        monkeypatch.setattr(bench, "build_reader", lambda *a, **k: Stub())
        crops = [{"image": f"{i}.png", "kind": "number", "text": None, "pil": Image.new("RGB", (10, 10))} for i in range(4)]

        result = bench.run_model(bench.Candidate("m", tmp_path, "chat_vlm"), crops, None, "generic", "", None)

        assert [c["empty"] for c in result.per_crop] == [True, True, False, False]
        assert [c["valid"] for c in result.per_crop] == [False, False, True, False]
        assert result.empty_rate == 0.5 and result.valid_rate == 0.25


class TestOverwriteGuard:
    def test_crops_of_another_photo_are_not_overwritten_without_asking(self, export, tmp_path):
        from tests.test_llamacpp_pipeline import form_a_photo

        out = tmp_path / "crops"
        out.mkdir()
        (out / "labels.jsonl").write_text(
            json.dumps({"image": "images/img_r1c1.png", "text": "typed by hand", "source": "other.png"}) + "\n",
            encoding="utf-8",
        )
        photo = tmp_path / "img.png"
        cv2.imwrite(str(photo), form_a_photo())

        assert export.main(["--photo", str(photo), "--out", str(out), "--config", "config/nuqta_config.yaml"]) == 2
        assert "typed by hand" in (out / "labels.jsonl").read_text(encoding="utf-8")

        assert export.main(["--photo", str(photo), "--out", str(out), "--overwrite",
                            "--config", "config/nuqta_config.yaml"]) == 0
        assert "typed by hand" not in (out / "labels.jsonl").read_text(encoding="utf-8")

    def test_re_exporting_the_same_photo_needs_no_flag(self, export, tmp_path):
        from tests.test_llamacpp_pipeline import form_a_photo

        photo = tmp_path / "img.png"
        cv2.imwrite(str(photo), form_a_photo())
        out = tmp_path / "crops"
        args = ["--photo", str(photo), "--out", str(out), "--config", "config/nuqta_config.yaml"]
        assert export.main(args) == 0
        assert export.main(args) == 0


def test_pipeline_mode_reads_nuqtas_settings_from_the_service_config(bench):
    params = bench.service_llamacpp_params(ROOT / "config" / "nuqta_config.yaml")

    assert params["repetition_penalty"] == 1.0
    assert params["image_min_tokens"] == "auto"
    assert params["prompts"]["number"].startswith("Transcribe the Arabic text")
    # Machine-specific entries are the benchmark's to supply, per candidate.
    assert not {"model_path", "mmproj_path", "server_path", "gpu_layers"} & set(params)


class TestRecognitionMetrics:
    def test_word_error_rate_counts_whole_words(self, bench):
        assert bench.wer("جاكيت صوف أزرق", "جاكيت صوف ازرق") == 0.0  # same after normalization
        assert bench.wer("جاكيت قطن أزرق", "جاكيت صوف أزرق") == pytest.approx(0.333, abs=0.001)
        assert bench.wer("x", None) is None

    def test_value_accuracy_is_strict_about_digits_and_the_decimal_point(self, bench):
        assert bench.value_correct("٥.٧٠", "5.7", "number", "unit_price") is True
        assert bench.value_correct("57", "5.7", "number", "unit_price") is False
        assert bench.value_correct("12", "12", "number", "quantity") is True
        assert bench.value_correct("13", "12", "number", "quantity") is False
        assert bench.value_correct("نص", "12", "arabic_text", "product_name") is None
        assert bench.value_correct("12", None, "number", "quantity") is None

    def test_results_break_down_per_kind_and_per_invoice(self, bench):
        r = bench.ModelResult(name="m")
        for source, kind, cer_ in (("a", "number", 0.0), ("a", "arabic_text", 0.5), ("b", "number", 0.2)):
            r.per_crop.append({"source": source, "kind": kind, "cer": cer_, "value_ok": None})
        assert r.breakdown("kind")["number"] == {"n": 2, "cer": 0.1, "exact": 0.5, "value_accuracy": None}
        assert set(r.breakdown("source")) == {"a", "b"}
        summary = r.summary()
        assert "by_kind" in summary and "by_invoice" in summary

    def test_peak_memory_of_this_process_is_measurable_here(self, bench):
        assert (bench.peak_rss_mb() or 0) > 0

    def test_several_invoices_get_distinct_keys(self, bench, tmp_path):
        for name in ("first", "second"):
            folder = tmp_path / name / "images"
            folder.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(folder / "0.png")
            (tmp_path / name / "labels.jsonl").write_text(
                json.dumps({"image": "images/0.png", "text": "1", "kind": "number"}) + "\n", encoding="utf-8")
        rows = bench.load_crops(tmp_path / "first") + bench.load_crops(tmp_path / "second")
        assert [r["key"] for r in rows] == ["first/images/0.png", "second/images/0.png"]
        assert [r["source"] for r in rows] == ["first", "second"]


class TestLabelsAndDates:
    def test_amount_labels_keep_their_cents(self, export):
        # ":g" used to write 214812.05 as "214812".
        assert export._number(214812.05) == "214812.05"
        assert export._number(5.7) == "5.7"
        assert export._number(212000.0) == "212000"

    def test_date_labels_are_written_day_first(self, export):
        assert export._written_date("2026-08-07") == "7/8/2026"

    def test_a_date_cell_is_right_when_it_names_the_same_day(self, bench):
        assert bench.value_correct("٢٠٢٦/٨/٧", "7/8/2026", "date", "invoice_date") is True
        assert bench.value_correct("7/8/2.26", "7/8/2026", "date", "invoice_date") is True
        assert bench.value_correct("8/7/2026", "7/8/2026", "date", "invoice_date") is False


class TestPerfectReadings:
    """evaluate.py --perfect-readings: every cell read as its export label."""

    def test_each_crop_is_read_as_the_label_of_its_region(self, tmp_path):
        import numpy as np

        from core.domain.geometry import BoundingBox
        from core.domain.ocr import RegionCrop, TextRegion

        evaluate = _load("evaluate")
        (tmp_path / "labels.jsonl").write_text(
            '{"image": "images/img_r3c1.png", "text": "212000"}\n'
            '{"image": "images/img_u0.png", "text": "0001901"}\n'
            '{"image": "images/img_r9c0.png", "text": null}\n',
            encoding="utf-8",
        )
        evaluate.PerfectReadings.labels = evaluate.load_export_labels(tmp_path)
        evaluate.PerfectReadings.unlabelled = []

        def crop(source_id):
            region = TextRegion(bbox=BoundingBox(0, 0, 4, 4), source_id=source_id)
            return RegionCrop(region=region, image=np.zeros((4, 4), np.uint8))

        readings = evaluate.PerfectReadings().read_all(
            [crop("t:r3c1"), crop("t:u0"), crop("t:r9c0"), crop("t:r1c4")]
        )
        assert [r.text for r in readings] == ["212000", "0001901", "", ""]
        assert evaluate.PerfectReadings.unlabelled == ["r9c0", "r1c4"]


def test_evaluate_prints_arabic_and_symbols_through_a_cp1252_pipe(monkeypatch):
    """On Windows a pipe defaults to cp1252: printing the report's "≥" there
    crashed every run of compare_pipelines.py after the evaluation was done."""
    import io
    import sys

    evaluate = _load("evaluate")
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    evaluate.utf8_console()
    print("≥ 0.75 أبو جمال")
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == "≥ 0.75 أبو جمال\n"
