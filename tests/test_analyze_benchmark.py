"""tools/analyze_benchmark.py: combinations are judged on held-out invoices."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def analysis():
    spec = importlib.util.spec_from_file_location("tools_analyze_benchmark", ROOT / "tools" / "analyze_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cell(source: str, i: int, kind: str, cer: float, confidence: float | None = None, value_ok=None) -> dict:
    return {"image": f"{source}/images/{i}.png", "source": source, "kind": kind, "cer": cer,
            "confidence": confidence, "value_ok": value_ok}


def entry(name: str, cells: list[dict]) -> dict:
    return {"name": name, "per_crop": cells}


def two_invoices(numbers_cer: float, text_cer: float, confidence=None) -> list[dict]:
    """Four cells per invoice: two numbers, two names."""
    out = []
    for source in ("a", "b"):
        out += [cell(source, 0, "number", numbers_cer, confidence, numbers_cer == 0.0),
                cell(source, 1, "number", numbers_cer, confidence, numbers_cer == 0.0),
                cell(source, 2, "arabic_text", text_cer, confidence),
                cell(source, 3, "arabic_text", text_cer, confidence)]
    return out


def write(tmp_path: Path, name: str, entries: list[dict]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


class TestLoading:
    def test_skipped_models_and_unlabelled_cells_are_left_out(self, analysis, tmp_path):
        path = write(tmp_path, "run", [
            entry("good", [cell("a", 0, "number", 0.0), {"image": "a/images/9.png", "cer": None}]),
            {"name": "broken", "skipped_reason": "no config.json", "per_crop": []},
        ])
        models = analysis.load_results([path])
        assert list(models) == ["good"]
        assert list(models["good"]) == ["a/images/0.png"]

    def test_the_same_model_in_two_runs_is_kept_apart(self, analysis, tmp_path):
        first = write(tmp_path, "first", [entry("m", [cell("a", 0, "number", 0.0)])])
        second = write(tmp_path, "second", [entry("m", [cell("a", 0, "number", 0.5)])])
        assert set(analysis.load_results([first, second])) == {"m", "m (second)"}

    def test_only_cells_every_model_read_are_compared(self, analysis):
        models = {"x": {"k1": {}, "k2": {}}, "y": {"k2": {}, "k3": {}}}
        assert analysis.common_cells(models) == ["k2"]


class TestFolds:
    def test_one_invoice_is_in_sample_and_says_so(self, analysis):
        models = {"m": {f"a/{i}": cell("a", i, "number", 0.0) for i in range(3)}}
        split, held_out = analysis.folds(models, sorted(models["m"]))
        assert not held_out and split[0][0] == split[0][1]

    def test_each_invoice_is_held_out_once_and_never_trained_on(self, analysis):
        models = {"m": {c["image"]: c for c in two_invoices(0.0, 0.0)}}
        split, held_out = analysis.folds(models, analysis.common_cells(models))
        assert held_out and len(split) == 2
        for train, test in split:
            assert not set(train) & set(test)
            assert len({k.split("/")[0] for k in test}) == 1


class TestStrategies:
    def models(self):
        # "digits" reads numbers perfectly, names badly; "words" the reverse.
        return {
            "digits": {c["image"]: c for c in two_invoices(0.0, 0.6)},
            "words": {c["image"]: c for c in two_invoices(0.5, 0.0)},
        }

    def test_routing_picks_each_kinds_best_model_on_the_other_invoice(self, analysis):
        models = self.models()
        records, choices = analysis.route_by_kind(models, analysis.common_cells(models))
        assert choices == {"number": ["digits", "digits"], "arabic_text": ["words", "words"]}
        assert analysis.score(records)["cer"] == 0.0

    def test_the_oracle_takes_the_best_reading_per_cell(self, analysis):
        models = self.models()
        assert analysis.score(analysis.oracle(models, analysis.common_cells(models)))["cer"] == 0.0

    def test_fallback_needs_the_primary_to_report_confidence(self, analysis):
        models = self.models()
        assert analysis.fallback(models, analysis.common_cells(models), "digits", "words") is None

    def test_fallback_threshold_is_learned_on_the_training_invoice(self, analysis):
        # The primary is unsure exactly where it is wrong (names), sure where right.
        primary = {}
        for c in two_invoices(0.0, 0.6):
            primary[c["image"]] = {**c, "confidence": 0.95 if c["kind"] == "number" else 0.3}
        models = {"primary": primary, "words": {c["image"]: c for c in two_invoices(0.5, 0.0)}}
        records, thresholds = analysis.fallback(models, analysis.common_cells(models), "primary", "words")
        assert analysis.score(records)["cer"] == 0.0
        assert all(0.3 < t <= 0.95 for t in thresholds)

    def test_scores(self, analysis):
        s = analysis.score([cell("a", 0, "number", 0.0, value_ok=True), cell("a", 1, "number", 0.5, value_ok=False),
                            cell("a", 2, "arabic_text", 0.0)])
        assert s == {"n": 3, "cer": 0.167, "exact": 0.667, "value_accuracy": 0.5}


class TestCalibration:
    def test_confident_wrong_readings_are_counted(self, analysis):
        cells = [cell("a", 0, "number", 0.0, 0.95), cell("a", 1, "number", 0.4, 0.9),
                 cell("a", 2, "number", 0.4, 0.2), cell("a", 3, "number", 0.0, 0.6)]
        models = {"m": {c["image"]: c for c in cells}}
        cal = analysis.calibration(models, sorted(models["m"]), "m")
        assert cal["wrong"] == 2 and cal["confidently_wrong"] == 1
        assert cal["accepted_wrong_rate"] == 0.25
        top = next(b for b in cal["bands"] if b["band"] == "0.90-1.00")
        assert top == {"band": "0.90-1.00", "n": 2, "exact": 0.5}

    def test_a_model_without_confidence_has_no_calibration(self, analysis):
        models = {"m": {"k": cell("a", 0, "number", 0.0)}}
        assert analysis.calibration(models, ["k"], "m") is None


class TestReport:
    def test_end_to_end_report_names_the_best_single_model_and_the_held_out_protocol(self, analysis, tmp_path):
        path = write(tmp_path, "run", [entry("digits", two_invoices(0.0, 0.6, confidence=0.9)),
                                       entry("words", two_invoices(0.5, 0.0))])
        out = tmp_path / "reports" / "analysis.md"
        assert analysis.main([str(path), "--out", str(out)]) == 0
        report = out.read_text(encoding="utf-8")
        assert "leave-one-invoice-out" in report
        assert "← best single" in report
        assert "Route by cell kind" in report and "Oracle" in report
        assert "words: reports no confidence" in report

    def test_one_invoice_is_labelled_in_sample(self, analysis, tmp_path):
        cells = [c for c in two_invoices(0.0, 0.0) if c["source"] == "a"]
        report = analysis.analyze({"m": {c["image"]: c for c in cells}})
        assert "Only one invoice" in report

    def test_nothing_to_compare_is_an_error_exit(self, analysis, tmp_path):
        path = write(tmp_path, "empty", [{"name": "x", "skipped_reason": "no weights"}])
        assert analysis.main([str(path)]) == 2
