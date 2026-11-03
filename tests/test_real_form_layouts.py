"""Classification of real photographed forms, from recorded extractor output.

The synthetic forms in test_layout_classifier.py pin the rules; these pin the
rules *against geometry a camera actually produced* -- skewed rows, ragged
widths, a stray box below the totals strip. Each fixture is the cell list the
`contour_based` extractor returned for one bill under bills/, recorded so the
test needs neither the image nor a model.

Re-record a fixture after changing the extractor or the geometric steps: run
`tools/record_layout_fixture.py <config> <image> <fixture>`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.domain.geometry import BoundingBox
from core.domain.roles import CellRole
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from table_extraction.classifiers.bill_layout_classifier import BillLayoutClassifier

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> tuple[TableExtractionResult, tuple[int, int]]:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    tables = [
        TableStructure(
            table_id=t["table_id"],
            bbox=BoundingBox(**t["bbox"]),
            cells=[
                TableCell(
                    bbox=BoundingBox(**c["bbox"]),
                    row=c["row"],
                    col=c["col"],
                    table_id=t["table_id"],
                    row_span=c.get("row_span", 1),
                    col_span=c.get("col_span", 1),
                )
                for c in t["cells"]
            ],
        )
        for t in data["tables"]
    ]
    width, height = data["image_size"]
    return TableExtractionResult(tables=tables, extractor_name=data["extractor"]), (width, height)


def role_at(layout, x: int, y: int) -> CellRole:
    """The role of the region whose box contains (x, y)."""
    hits = [
        r for r in layout.regions
        if r.bbox.x <= x <= r.bbox.x + r.bbox.w and r.bbox.y <= y <= r.bbox.y + r.bbox.h
    ]
    assert len(hits) == 1, f"expected one region at ({x}, {y}), found {len(hits)}"
    return hits[0].role


@pytest.fixture(scope="module")
def form_a():
    table, size = load_fixture("form_a_cells.json")
    return BillLayoutClassifier().classify(table, size)


class TestFormA:
    """bills/sahror: the 6-column, 10-row sales invoice the defaults describe."""

    def test_the_handwritten_total_box_is_read_as_the_total(self, form_a):
        # Footer strip, left to right: words | "كتابة :" | figures | "رقماً :" | "السعر الإجمالي".
        # The figures box (x 554-770) holds "٢١٤٨١٢,٠٥" on the paper. It used to
        # be labelled a caption and skipped, so the invoice total was always null.
        assert role_at(form_a, 660, 1420) is CellRole.TOTAL_IN_FIGURES
        assert role_at(form_a, 250, 1420) is CellRole.TOTAL_IN_WORDS
        assert role_at(form_a, 500, 1420) is CellRole.LABEL
        assert role_at(form_a, 840, 1420) is CellRole.LABEL
        assert role_at(form_a, 1100, 1420) is CellRole.LABEL

    def test_header_fields(self, form_a):
        assert role_at(form_a, 270, 160) is CellRole.INVOICE_NUMBER
        assert role_at(form_a, 300, 350) is CellRole.CITY
        assert role_at(form_a, 650, 350) is CellRole.INVOICE_DATE
        assert role_at(form_a, 1000, 340) is CellRole.MERCHANT_NAME

    def test_ten_item_rows_with_columns_in_rtl_order(self, form_a):
        for role, x in (
            (CellRole.NOTES, 230),
            (CellRole.LINE_TOTAL, 450),
            (CellRole.UNIT_PRICE, 660),
            (CellRole.QUANTITY, 830),
            (CellRole.PRODUCT_NAME, 1050),
            (CellRole.LINE_NUMBER, 1245),
        ):
            assert len(form_a.with_role(role)) == 10, role
            assert role_at(form_a, x, 570) is role

    def test_the_printed_column_header_row_is_not_an_item(self, form_a):
        assert len(form_a.with_role(CellRole.COLUMN_HEADER)) == 6


@pytest.fixture(scope="module")
def two_form_classifier():
    from config.loader import load_config

    params = load_config("config/qwen_config.yaml").table_extraction.classifier_params
    return BillLayoutClassifier(**params)


class TestFormProfiles:
    """Both printed forms through the one configured classifier."""

    def test_form_a_still_selects_its_own_profile(self, two_form_classifier):
        table, size = load_fixture("form_a_cells.json")
        layout = two_form_classifier.classify(table, size)

        assert layout.source.endswith("+bill_layout")
        assert role_at(layout, 660, 1420) is CellRole.TOTAL_IN_FIGURES

    def test_form_b_is_recognised_by_its_four_column_grid(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        assert layout.source.endswith("[form_b_bilingual]")

    def test_form_b_columns_and_total(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        # Row 3 of the recorded grid: an item row (qty 192, price 5.70).
        assert role_at(layout, 140, 400) is CellRole.UNIT_PRICE
        assert role_at(layout, 350, 400) is CellRole.QUANTITY
        assert role_at(layout, 800, 410) is CellRole.PRODUCT_NAME
        assert role_at(layout, 1370, 410) is CellRole.LINE_TOTAL
        # The handwritten sum, in the box below the table.
        assert role_at(layout, 680, 1824) is CellRole.TOTAL_AMOUNT

    def test_the_first_detected_row_of_form_b_is_an_item_not_a_header(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        assert not layout.with_role(CellRole.COLUMN_HEADER)
        assert len(layout.with_role(CellRole.QUANTITY)) >= 10
