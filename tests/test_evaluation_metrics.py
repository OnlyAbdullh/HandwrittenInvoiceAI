"""The evaluation scores what was hand-counted on a real extraction.

`sahror_output_before_fixes.json` is the wire response the pipeline produced for
bills/sahror before the audit fixes. It was graded by hand against the paper
(5 of 12 numeric cells right, 6 invented rows, date and total missing), and the
metrics must agree with that count.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import score_invoice, summarize
from evaluation.schema import InvoiceTruth, LineItemTruth

ROOT = Path(__file__).resolve().parent.parent


def load():
    output = json.loads((ROOT / "tests/fixtures/sahror_output_before_fixes.json").read_text(encoding="utf-8"))
    truth = InvoiceTruth.model_validate_json(
        (ROOT / "evaluation/ground_truth/sahror.json").read_text(encoding="utf-8")
    )
    return output, truth


def test_matches_the_hand_count_of_a_real_extraction():
    output, truth = load()
    score = score_invoice(output, truth)

    assert (score.numeric_cells_correct, score.numeric_cells_total) == (5, 12)
    assert score.rows_matched == 4
    assert score.phantom_rows == 6
    assert score.missed_rows == 0
    assert score.product_top1_correct == 4
    assert score.product_auto_accepted == 2

    header = {f.name: f for f in score.header}
    assert header["invoice_number"].correct
    assert not header["date"].correct
    assert not header["total"].correct
    assert header["customer_name"].correct and header["customer_name"].auto_accepted
    assert header["city"].correct and not header["city"].auto_accepted


def test_an_invented_row_does_not_match_through_its_zero_score_candidate():
    """The matcher always ranks something first; a 0.0 candidate is not a match."""
    invented = {
        "product_name": {"original_value": "أفكارك في جريدتي", "results": [{"value": "بوري شبابي زارا", "string_matching_score": 0.0}]},
        "quantity": {"value": 12}, "unit_price": {"value": None}, "total_price": {"value": None},
    }
    truth = InvoiceTruth(image="x", items=[LineItemTruth(product_name="بوري شبابي زارا", quantity=87, unit_price=5.95, total_price=517.65)])

    score = score_invoice({"products": [invented]}, truth)

    assert score.rows_matched == 0
    assert score.phantom_rows == 1 and score.missed_rows == 1


def test_null_truth_fields_are_not_scored():
    truth = InvoiceTruth(image="x", invoice_number="7", city=None, date=None, total=None)
    score = score_invoice({"invoice_id": {"value": "7"}}, truth)

    assert [f.name for f in score.header] == ["invoice_number"]
    assert summarize([score])["header_correct"] == 1


def _perfect_output(truth: InvoiceTruth) -> dict:
    def matched(name):
        return {"original_value": name, "results": [{"value": name, "string_matching_score": 1.0}]}

    return {
        "invoice_id": {"value": truth.invoice_number},
        "date": {"value": truth.date},
        "total_invoice_price": {"value": truth.total},
        "customer_name": matched(truth.customer_name),
        "city": matched(truth.city),
        "products": [
            {"product_name": matched(i.product_name), "quantity": {"value": i.quantity},
             "unit_price": {"value": i.unit_price}, "total_price": {"value": i.total_price}}
            for i in truth.items
        ],
    }


def test_a_perfect_extraction_is_a_complete_invoice_with_nothing_to_review():
    _, truth = load()
    score = score_invoice(_perfect_output(truth), truth)
    assert score.complete and score.complete_without_review
    summary = summarize([score])
    assert summary["complete_invoices"] == 1 and summary["complete_without_review"] == 1


def test_one_wrong_digit_makes_the_invoice_incomplete_and_shows_in_its_column():
    _, truth = load()
    output = _perfect_output(truth)
    output["products"][1]["unit_price"]["value"] = 999
    score = score_invoice(output, truth)

    assert not score.complete
    fields = summarize([score])["by_field"]
    assert fields["unit_price"] == {"correct": len(truth.items) - 1, "total": len(truth.items)}
    assert fields["quantity"] == {"correct": len(truth.items), "total": len(truth.items)}
    assert fields["total"] == {"correct": 1, "total": 1}


def test_a_right_but_unsure_match_is_complete_yet_still_needs_review():
    _, truth = load()
    output = _perfect_output(truth)
    output["city"]["results"][0]["string_matching_score"] = 0.6
    score = score_invoice(output, truth)
    assert score.complete and not score.complete_without_review


def test_the_report_carries_per_field_rows_and_run_measurements():
    from evaluation.metrics import to_markdown

    output, truth = load()
    report = to_markdown([score_invoice(output, truth, seconds=3.0)], "x.yaml",
                         {"load_seconds": 12.5, "peak_ram_mb": 2048.0, "vram_mb": None})
    assert "### Per field" in report and "| unit_price |" in report
    assert "| Pipeline load seconds | 12.5 |" in report
    assert "Peak RAM" in report and "VRAM" not in report  # unknown is omitted, never invented


def test_an_unparsed_reading_is_shown_beside_the_null_it_became():
    from evaluation.metrics import to_markdown

    truth = InvoiceTruth(image="x", date="2026-08-07", total=10.0)
    score = score_invoice({"date": {"value": None}, "total_invoice_price": {"value": None}}, truth,
                          readings={"date": "2.26/8/7?", "total": None})
    report = to_markdown([score], "x.yaml")
    assert "None (read: '2.26/8/7?')" in report
