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
