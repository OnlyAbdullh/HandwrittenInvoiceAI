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
from core.domain.roles import CellRole, ContentKind, Zone
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

        # The second item row (6 x 200 = 1200) on the recorded page.
        assert role_at(layout, 313, 463) is CellRole.UNIT_PRICE
        assert role_at(layout, 478, 465) is CellRole.QUANTITY
        assert role_at(layout, 859, 471) is CellRole.PRODUCT_NAME
        assert role_at(layout, 1267, 479) is CellRole.LINE_TOTAL
        # The handwritten sum: the wide box of the totals strip ("المجموع"),
        # not the small empty box beside it.
        assert role_at(layout, 680, 1650) is CellRole.TOTAL_AMOUNT

    def test_the_printed_caption_row_of_form_b_is_a_header_not_an_item(self, two_form_classifier):
        """With the page kept around the table, the caption row (Price |
        Quantity | Description | Total) is inside the detected outline."""
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        assert len(layout.with_role(CellRole.COLUMN_HEADER)) >= 4
        assert role_at(layout, 316, 290) is CellRole.COLUMN_HEADER

    def test_the_unruled_header_fields_are_read_from_where_the_form_prints_them(self, two_form_classifier):
        """Invoice number, customer and date sit on dotted lines above the
        table, with no box around them. They were never read at all."""
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        number, customer, date = (layout.with_role(r) for r in (
            CellRole.INVOICE_NUMBER, CellRole.MERCHANT_NAME, CellRole.INVOICE_DATE))
        assert len(number) == len(customer) == len(date) == 1
        table_top = layout.table_bbox.y
        for region in (number[0], customer[0], date[0]):
            assert region.zone is Zone.HEADER
            assert region.bbox.y + region.bbox.h <= table_top  # entirely above the table
        # Right to left as printed: customer on the right, date on the left.
        assert customer[0].bbox.x > date[0].bbox.x + date[0].bbox.w - 1
        assert date[0].content_kind is ContentKind.DATE
        assert number[0].content_kind is ContentKind.CODE


class TestGridCompletion:
    """Form B from the recorded real photo. Its first item row's contours do
    not all close (the price box of item 1 is found merged with the caption
    above it). Without completion the item table started at the second row and
    every row was labelled one place off; the first item -- 16 x 13250 = 212000
    -- was never read.
    """

    def test_the_first_item_row_is_read(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        # Row 1 of the paper, at y ~385 on the recorded page.
        assert role_at(layout, 313, 385) is CellRole.UNIT_PRICE
        assert role_at(layout, 478, 382) is CellRole.QUANTITY
        assert role_at(layout, 859, 389) is CellRole.PRODUCT_NAME
        assert role_at(layout, 1200, 400) is CellRole.LINE_TOTAL

    def test_all_fifteen_rows_and_nothing_else(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        for role in (CellRole.UNIT_PRICE, CellRole.QUANTITY, CellRole.PRODUCT_NAME, CellRole.LINE_TOTAL):
            assert len(layout.with_role(role)) == 15, role
        # Inferred boxes are marked, so a render or a log can tell them apart.
        assert 0 < sum(r.id.endswith("s") for r in layout.regions) <= 4
        # The totals strip is still the totals strip, not a sixteenth item.
        assert role_at(layout, 680, 1650) is CellRole.TOTAL_AMOUNT

    def test_every_item_row_has_one_row_number(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        rows = {}
        for region in layout.regions:
            if region.role in (CellRole.UNIT_PRICE, CellRole.QUANTITY, CellRole.PRODUCT_NAME, CellRole.LINE_TOTAL):
                rows.setdefault(region.row, []).append(region.role)
        assert len(rows) == 15 and all(len(roles) == 4 for roles in rows.values())

    def test_a_box_spanning_rows_is_not_an_item_cell(self, two_form_classifier):
        """A 141x142 box closed between the "Total" caption and the first item's
        rule was labelled a second line total for item 1."""
        table, size = load_fixture("form_b_cells.json")
        layout = two_form_classifier.classify(table, size)

        heights = sorted(r.bbox.h for r in layout.with_role(CellRole.LINE_TOTAL))
        assert heights[-1] < 1.5 * heights[len(heights) // 2]

    def test_form_a_needs_nothing_inferred(self, two_form_classifier):
        table, size = load_fixture("form_a_cells.json")
        layout = two_form_classifier.classify(table, size)
        assert not any(r.id.endswith("s") for r in layout.regions)

    def test_the_vertical_shift_moves_item_boxes_only(self):
        from config.loader import load_config

        params = dict(load_config("config/qwen_config.yaml").table_extraction.classifier_params)
        table, size = load_fixture("form_a_cells.json")
        plain = BillLayoutClassifier(**params).classify(table, size)
        shifted = BillLayoutClassifier(**{**params, "item_vertical_shift": 0.25}).classify(table, size)

        before = {r.id: r.bbox for r in plain.regions}
        for region in shifted.regions:
            moved = region.bbox.y - before[region.id].y
            if region.role in (CellRole.QUANTITY, CellRole.PRODUCT_NAME):
                assert moved == round(0.25 * region.bbox.h)
            elif region.role in (CellRole.INVOICE_NUMBER, CellRole.TOTAL_IN_FIGURES, CellRole.COLUMN_HEADER):
                assert moved == 0


class TestUnruledFieldConfig:
    def test_a_declared_field_outside_the_photo_is_dropped_not_invented(self):
        table, size = load_fixture("form_b_cells.json")
        layout = BillLayoutClassifier(
            table_row_count=15, table_col_count=4,
            column_roles=["unit_price", "quantity", "product_name", "line_total"],
            unruled_fields=[{"role": "invoice_date", "x": [0.1, 0.2], "y": [-5.0, -4.0]}],
        ).classify(table, size)
        assert not layout.with_role(CellRole.INVOICE_DATE)

    @pytest.mark.parametrize("field", [
        {"role": "invoice_date", "x": [0.3, 0.2], "y": [-0.1, 0.0]},
        {"role": "no_such_role", "x": [0.1, 0.2], "y": [-0.1, 0.0]},
        {"role": "invoice_date", "x": [0.1, 0.2]},
        {"role": "invoice_date", "x": [0.1, 0.2], "y": [-0.1, 0.0], "colour": "red"},
    ])
    def test_a_malformed_field_fails_when_the_config_loads(self, field):
        with pytest.raises(ValueError):
            BillLayoutClassifier(unruled_fields=[field])


class TestFormBCropPlacement:
    """Where form B's writing actually sits, as the configured profile crops it."""

    def _layouts(self, two_form_classifier):
        table, size = load_fixture("form_b_cells.json")
        # The same profile without the crop placement settings.
        plain = BillLayoutClassifier(
            table_row_count=15, table_col_count=4,
            column_roles=["unit_price", "quantity", "product_name", "line_total"],
            header_roles=[], footer_roles=["total_amount", "label"],
            printed_header_row=True, footer_from_below=False, ragged_row_slack=1, complete_grid=True,
        ).classify(table, size)
        configured = two_form_classifier.classify(table, size)
        assert configured.source.endswith("[form_b_bilingual]")
        return plain, configured

    def test_item_crops_move_down_onto_the_writing(self, two_form_classifier):
        plain, configured = self._layouts(two_form_classifier)
        before = {r.id: r.bbox for r in plain.regions}
        for region in configured.with_role(CellRole.QUANTITY):
            assert region.bbox.y - before[region.id].y == round(0.25 * region.bbox.h)

    def test_the_total_is_cropped_to_the_figures_not_the_whole_strip(self, two_form_classifier):
        plain, configured = self._layouts(two_form_classifier)
        strip = plain.with_role(CellRole.TOTAL_AMOUNT)[0].bbox
        figures = configured.with_role(CellRole.TOTAL_AMOUNT)[0].bbox
        assert figures.x == strip.x + round(0.22 * strip.w)
        assert figures.x + figures.w == strip.x + round(0.84 * strip.w)
        assert figures.y + figures.h == strip.y + strip.h
        # "المجموع" (right) and "فقط لا غير" (left) are outside the crop.
        assert figures.x + figures.w < strip.x + strip.w - 100
        assert figures.x > strip.x + 100

    def test_the_date_crop_does_not_reach_the_invoice_number_line(self, two_form_classifier):
        _, configured = self._layouts(two_form_classifier)
        number = configured.with_role(CellRole.INVOICE_NUMBER)[0].bbox
        date = configured.with_role(CellRole.INVOICE_DATE)[0].bbox
        assert date.y >= number.y + number.h


class TestFieldInsetConfig:
    @pytest.mark.parametrize("inset", [
        {"x": [0.8, 0.2]},
        {"x": [-0.1, 0.5]},
        {"y": [0.0, 1.5]},
        {"x": [0.1, 0.9], "z": [0, 1]},
        {"x": "wide"},
    ])
    def test_a_malformed_inset_fails_when_the_config_loads(self, inset):
        with pytest.raises(ValueError):
            BillLayoutClassifier(field_insets={"total_amount": inset})

    def test_an_unknown_role_fails(self):
        with pytest.raises(ValueError):
            BillLayoutClassifier(field_insets={"no_such_role": {"x": [0.1, 0.9]}})
