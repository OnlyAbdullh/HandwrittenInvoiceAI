"""tools/compare_pipelines.py: one table across reading approaches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("tools_compare_pipelines", ROOT / "tools" / "compare_pipelines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summary(complete, header, numbers, seconds):
    return {"invoices": 2, "failed": 0, "complete_invoices": complete, "header_correct": header, "header_total": 9,
            "numeric_cells_correct": numbers, "numeric_cells_total": 24, "product_top1_correct": 8,
            "products_total": 8, "phantom_rows": 0, "missed_rows": 0, "mean_seconds": seconds}


def test_ranked_by_complete_invoices_then_fields_then_speed(compare):
    report = compare.to_markdown([
        {"config": "surya_config.yaml", "summary": summary(0, 5, 10, 30.0), "run": {}},
        {"config": "default_config.yaml", "summary": summary(1, 8, 22, 14.4), "run": {"peak_ram_mb": 2896.7}},
        {"config": "nuqta_config.yaml", "summary": summary(1, 8, 20, 100.0), "run": {}},
    ])
    rows = [line for line in report.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert [r.split("|")[3].strip() for r in rows] == [
        "`default_config.yaml`", "`nuqta_config.yaml`", "`surya_config.yaml`"]
    assert "Grid cells -> Qwen3.5-OCR-v4" in rows[0]


def test_an_approach_that_cannot_run_is_reported_with_its_reason(compare, tmp_path):
    result = compare.run_one(tmp_path / "missing_config.yaml", tmp_path / "out", [])
    assert "error" in result
    report = compare.to_markdown([result])
    assert "Could not run:" in report and "missing_config.yaml" in report
