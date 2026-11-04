"""Arithmetic reconciliation of line items, per invoice/reconciliation.py.

Each case here is a reading these recognizers actually produce. The stage's
whole premise is that the *digits* are usually right and the separator usually
is not, so the tests are written as "this is what came back, this is what the
row must end up saying".
"""

from __future__ import annotations

import pytest

from core.domain.invoice import (
    ExtractedValue,
    InvoiceDraft,
    InvoiceLineItem,
    WarningCodes,
)
from invoice.reconciliation import (
    decimal_candidates,
    dot_run_candidates,
    observed_decimals,
    reconcile,
)


def row(quantity=None, unit_price=None, total=None, index=0) -> InvoiceLineItem:
    """One line item, built from raw cell readings.

    `raw` is what the recognizer returned and `value` what the parser made of
    it -- None where the strict number rules refused the reading, which is
    exactly the case reconciliation exists for.
    """

    def cell(reading):
        if reading is None:
            return ExtractedValue()

        raw, value = reading if isinstance(reading, tuple) else (str(reading), reading)
        return ExtractedValue(value=value, confidence=0.9, raw=raw)

    item = InvoiceLineItem(row_index=index)
    item.quantity = cell(quantity)
    item.unit_price = cell(unit_price)
    item.total_price = cell(total)
    return item


def reconciled(item: InvoiceLineItem) -> InvoiceLineItem:
    reconcile(InvoiceDraft(line_items=[item]))
    return item


class TestDecimalCandidates:
    def test_every_legal_separator_position(self):
        assert decimal_candidates("51765") == [51765.0, 5176.5, 517.65]

    def test_a_short_sequence_runs_out_of_positions(self):
        assert decimal_candidates("5") == [5.0]

    def test_an_empty_sequence_offers_nothing(self):
        assert decimal_candidates("") == []


class TestDotRunCandidates:
    """These models write trailing zeros as dots."""

    def test_expands_a_trailing_dot_run(self):
        assert dot_run_candidates("212...") == ["212", "2120", "21200", "212000", "2120000"]

    def test_a_reading_with_no_trailing_dots_is_left_alone(self):
        assert dot_run_candidates("212") == ["212"]

    def test_a_cell_with_no_digits_offers_nothing(self):
        assert dot_run_candidates("ثلاثون") == []


class TestObservedDecimals:
    def test_reads_the_fraction_length_off_a_clean_reading(self):
        assert observed_decimals("517.65") == 2
        assert observed_decimals("5176.5") == 1

    def test_a_multi_separator_reading_carries_no_signal(self):
        # Dot-pattern noise, not formatting. Trusting it would mislead.
        assert observed_decimals("٥-١-٧,٦,٥") is None

    def test_no_separator_at_all(self):
        assert observed_decimals("51765") is None


class TestRepairingSeparators:
    def test_finds_the_decimal_position_the_equation_agrees_with(self):
        """The classic case: three readings, one consistent placement."""
        item = reconciled(row(quantity=("3", 3), unit_price=("517.65", 517.65),
                              total=("155295", 155295.0)))

        assert item.unit_price.value == pytest.approx(517.65)
        assert item.quantity.value == 3
        assert item.total_price.value == pytest.approx(1552.95)
        assert item.arithmetic_ok

    def test_recovers_a_cell_the_strict_parser_refused(self):
        """"٥-١-٧,٦,٥" parses to nothing but its digits are 517.65 exactly."""
        item = reconciled(row(quantity=("3", 3), unit_price=("٥-١-٧,٦,٥", None),
                              total=("1552.95", 1552.95)))

        assert item.unit_price.value == pytest.approx(517.65)
        assert item.arithmetic_ok

    def test_a_row_that_already_multiplies_out_is_left_alone(self):
        item = reconciled(row(quantity=("2", 2), unit_price=("150.00", 150.0),
                              total=("300.00", 300.0)))

        assert item.unit_price.value == pytest.approx(150.0)
        assert item.quantity.value == 2
        assert item.total_price.value == pytest.approx(300.0)
        assert item.arithmetic_ok


class TestDerivingAMissingCell:
    def test_computes_the_total_from_price_and_quantity(self):
        item = reconciled(row(quantity=("4", 4), unit_price=("25.50", 25.5)))

        assert item.total_price.value == pytest.approx(102.0)

    def test_computes_the_price_from_total_and_quantity(self):
        item = reconciled(row(quantity=("4", 4), total=("102.00", 102.0)))

        assert item.unit_price.value == pytest.approx(25.5)

    def test_one_known_cell_derives_nothing(self):
        """Two knowns is the minimum. One number cannot imply the other two."""
        item = reconciled(row(quantity=("4", 4)))

        assert item.unit_price.value is None
        assert item.total_price.value is None


class TestRefusingToGuess:
    def test_an_irreconcilable_row_keeps_what_was_read(self):
        """No decimal placement works, and the derived value explains nothing.

        The row is left exactly as the recognizer produced it and flagged. A
        confident wrong number is worse than a visibly unresolved one -- the
        verification screen is where this gets settled.
        """
        item = reconciled(row(quantity=("2", 2), unit_price=("7", 7.0),
                              total=("9999999", 9999999.0)))

        assert item.quantity.value == 2
        assert item.unit_price.value == pytest.approx(7.0)
        assert not item.arithmetic_ok

    def test_an_empty_row_is_untouched(self):
        item = reconciled(row())

        assert item.quantity.value is None
        assert item.unit_price.value is None
        assert item.total_price.value is None
        # Nothing to disagree about.
        assert item.arithmetic_ok


class TestWarnings:
    def test_a_repaired_row_no_longer_reports_a_mismatch(self):
        """The parser raised the warning against the row as it was read."""
        item = row(quantity=("3", 3), unit_price=("517.65", 517.65),
                   total=("155295", 155295.0))
        item.arithmetic_ok = False

        draft = InvoiceDraft(line_items=[item])
        reconcile(draft)

        codes = [warning.code for warning in draft.warnings]
        assert WarningCodes.ARITHMETIC_MISMATCH not in codes

    def test_an_unresolved_row_reports_one_exactly_once(self):
        item = row(quantity=("2", 2), unit_price=("7", 7.0), total=("9999999", 9999999.0))

        draft = InvoiceDraft(line_items=[item])
        reconcile(draft)
        reconcile(draft)  # Re-running must not accumulate duplicates.

        codes = [warning.code for warning in draft.warnings]
        assert codes.count(WarningCodes.ARITHMETIC_MISMATCH) == 1


class TestHeaderTotal:
    def _draft(self, total_raw, line_totals):
        from core.domain.invoice import ExtractedValue, InvoiceDraft, InvoiceLineItem

        draft = InvoiceDraft()
        draft.header.total_amount = ExtractedValue(value=float(digit_only(total_raw)), raw=total_raw)
        for i, amount in enumerate(line_totals):
            draft.line_items.append(InvoiceLineItem(row_index=i, total_price=ExtractedValue(value=amount)))
        return draft

    def test_a_dropped_decimal_separator_is_restored_from_the_line_totals(self):
        from invoice.reconciliation import reconcile

        draft = self._draft("٢١٤٨١٢٠٥", [212000.0, 1200.0, 1094.4, 517.65])
        reconcile(draft)
        assert draft.header.total_amount.value == 214812.05

    def test_a_total_that_matches_no_placement_is_left_alone(self):
        from invoice.reconciliation import reconcile

        draft = self._draft("999", [10.0, 20.0])
        reconcile(draft)
        assert draft.header.total_amount.value == 999.0


class TestHeaderTotalFromVerifiedRows:
    """The total reading is one or two digits off a sum every row verifies."""

    def _draft(self, total_raw):
        from string_matching.normalization import normalize_price

        draft = InvoiceDraft(line_items=[
            row(16, 13250, 212000, index=0),
            row(6, 200, 1200, index=1),
            row(192, 5.7, 1094.4, index=2),
            row(87, 5.95, 517.65, index=3),
        ])
        draft.header.total_amount = ExtractedValue(value=normalize_price(total_raw), raw=total_raw)
        return draft

    @pytest.mark.parametrize("reading", ["۴۱۴۸۱۲۰۵", "٢١٤٨١٢٫٠"])  # both laptop readings of 214812.05
    def test_a_digit_misread_or_dropped_is_corrected_and_flagged(self, reading):
        draft = reconcile(self._draft(reading))
        assert draft.header.total_amount.value == 214812.05
        flagged = [w for w in draft.warnings if w.field == "total_amount"]
        assert [w.code for w in flagged] == [WarningCodes.MANUAL_REVIEW_REQUIRED]

    def test_a_reading_of_a_different_number_is_left_alone(self):
        draft = reconcile(self._draft("٥١"))
        assert draft.header.total_amount.value == 51.0
        assert WarningCodes.TOTAL_MISMATCH in [w.code for w in draft.warnings]

    def test_not_when_a_row_is_unverified(self):
        draft = self._draft("۴۱۴۸۱۲۰۵")
        draft.line_items[1].quantity = ExtractedValue()  # 1200 is no longer confirmed by its row
        reconcile(draft)
        assert draft.header.total_amount.value == 41481205.0

    def test_a_stale_total_mismatch_is_withdrawn(self):
        from core.domain.invoice import InvoiceWarning

        draft = self._draft("٢١٤٨١٢,٠٥")
        draft.warnings.append(InvoiceWarning(code=WarningCodes.TOTAL_MISMATCH, field="total_amount", message="stale"))
        reconcile(draft)
        assert draft.header.total_amount.value == 214812.05
        assert WarningCodes.TOTAL_MISMATCH not in [w.code for w in draft.warnings]


def digit_only(text):
    from string_matching.normalization import digit_sequence

    return digit_sequence(text)


class TestHandwrittenMisreads:
    """Rows as the laptop read them (Arabic-Qwen3.5-OCR-v4 on bills/sahror and
    bills/onlyone), and what each must become. The readings are real."""

    @staticmethod
    def _row(raw):
        from string_matching.normalization import parse_number_strict

        item = InvoiceLineItem(row_index=0)
        item.quantity, item.unit_price, item.total_price = (
            ExtractedValue(value=parse_number_strict(r), confidence=0.9, raw=r) for r in raw
        )
        return item

    @pytest.mark.parametrize(("raw", "truth"), [
        # a run of dot-zeros counted one short, and "13250" never read as 1325.0
        (("16", "١٣٢٥٠", "٢١٢٠٠"), (16, 13250, 212000)),
        # the decimal comma read as a zero, and a digit misread in the total
        (("192", "٥٠٧", "1٠٤٠٤"), (192, 5.7, 1094.4)),
        # the dot-shaped zero read as a separator
        (("192", "٥٠٧", "1.٩٤٤"), (192, 5.7, 1094.4)),
        (("87", "٥٠٩٥", "517.65"), (87, 5.95, 517.65)),
        # the total misread: the other two cells, taken as read, say what it is
        (("16", "13250", "١٢٠٠"), (16, 13250, 212000)),
    ])
    def test_the_row_comes_out_right(self, raw, truth):
        draft = reconcile(InvoiceDraft(line_items=[self._row(raw)]))
        item = draft.line_items[0]
        assert (item.quantity.value, item.unit_price.value, item.total_price.value) == pytest.approx(truth)
        assert item.arithmetic_ok

    def test_a_digit_correction_is_flagged_for_review(self):
        draft = reconcile(InvoiceDraft(line_items=[self._row(("87", "٥٠٩٥", "517.65"))]))
        flagged = [w.field for w in draft.warnings if w.code == WarningCodes.MANUAL_REVIEW_REQUIRED]
        assert flagged == ["line_items[0].unit_price"]

    def test_a_separator_or_dot_run_repair_is_not_flagged(self):
        separator = self._row(("87", "٥,٩٥", "٥١٧٦٥"))
        dots = self._row(("16", "13250", "212..."))
        draft = reconcile(InvoiceDraft(line_items=[separator, dots]))
        assert (separator.unit_price.value, separator.total_price.value) == (5.95, 517.65)
        assert dots.total_price.value == 212000.0
        assert not [w for w in draft.warnings if w.code == WarningCodes.MANUAL_REVIEW_REQUIRED]

    def test_running_twice_does_not_duplicate_the_flag(self):
        draft = InvoiceDraft(line_items=[self._row(("87", "٥٠٩٥", "517.65"))])
        reconcile(draft)
        reconcile(draft)
        assert len([w for w in draft.warnings if w.code == WarningCodes.MANUAL_REVIEW_REQUIRED]) <= 1

    def test_a_row_that_already_multiplies_out_is_left_alone(self):
        # Wrong on the paper (200 x 6 = 1200) but consistent as read: arithmetic
        # cannot see it, and must not invent a different consistent row.
        draft = reconcile(InvoiceDraft(line_items=[self._row(("6", "٢٠١", "١٢٠٦"))]))
        item = draft.line_items[0]
        assert (item.unit_price.value, item.total_price.value) == (201.0, 1206.0)

    def test_a_decimal_part_of_zeros_is_never_inserted(self):
        from invoice.reconciliation import handwriting_candidates

        assert 1325.0 not in handwriting_candidates("١٣٢٥٠")
        assert 5.7 in handwriting_candidates("٥٠٧")
        assert 1094.4 in handwriting_candidates("1.٩٤٤")
        assert 212000.0 in handwriting_candidates("٢١٢٠٠")
