"""Arithmetic reconciliation of a parsed invoice's line items.

The recognizers on this project misplace the decimal separator far more often
than they misread a digit: a cell photographed across a printed dot pattern
comes back as ``"٥-١-٧,٦,٥"`` whose digit sequence, ``51765``, is exactly right.
So instead of trusting the separator, this stage tries every legal decimal
position and keeps the one combination that satisfies

    unit price x quantity = line total

That single equation both repairs the separators and verifies the reading in one
step, and it fills in a missing cell from the two that were read.

Two safety rules, each of which was the difference between a useful correction
and a confidently wrong number:

* A computed value replaces a *read* one only when the two are close --
  :func:`_explains_reading`. A derived value that looks nothing like what the
  model saw means the model read a different cell, not a mis-scaled one.
* Nothing here ever invents a value out of one number. Two knowns are the
  minimum, and a value derived from two unconfirmed cells is marked for review
  rather than accepted.

This runs after :class:`~invoice.parser.InvoiceParser`, on the draft it
produced, and is called once from
:meth:`~orchestration.pipeline_orchestrator.PipelineOrchestrator.run`. It
replaced the standalone ``postprocess/reconcile.py`` CLI, which reimplemented
its own digit normalization and never ran inside the pipeline at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.domain.invoice import (
    InvoiceDraft,
    InvoiceLineItem,
    InvoiceWarning,
    WarningCodes,
)
from string_matching.normalization import digit_sequence, fold_digits, fold_latin_lookalikes

# Slack for the decimal division, in currency units.
TOLERANCE = 0.02

# Amounts on these invoices carry at most two decimals, so the separator can sit
# in one of three places.
MAX_DECIMALS = 2

# How different a derived value may be from what was read, as a fraction of the
# longer digit sequence, before it stops being an explanation of that reading.
MAX_ERROR_RATE = 0.50

# The invoice total may be corrected to the verified line sum when the reading
# differs from it by at most this many digits (and by at most a quarter of them).
MAX_TOTAL_EDITS = 2

# The parser's tolerance for "the rows do not add up to the stated total".
TOTAL_MISMATCH_TOLERANCE = 0.05

_SEPARATORS = re.compile(r"[,.٫،\-_/\\|]")


@dataclass
class _Cell:
    """One numeric cell of a row, as reconciliation sees it."""

    name: str
    raw: str
    digits: str
    value: float | None

    @property
    def known(self) -> bool:
        return bool(self.digits)


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def decimal_candidates(digits: str, max_decimals: int = MAX_DECIMALS) -> list[float]:
    """Every value the digit sequence could be, by decimal position."""
    if not digits:
        return []

    values: list[float] = []
    for places in range(0, max_decimals + 1):
        if places == 0:
            candidate = float(digits)
        elif len(digits) > places:
            candidate = float(f"{digits[:-places]}.{digits[-places:]}")
        else:
            continue

        if candidate not in values:
            values.append(candidate)

    return values


def dot_run_candidates(raw: str | None) -> list[str]:
    """Digit sequences for a cell whose reading ends in a run of dots.

    These models write trailing zeros as dots: ``"212..."`` is 212000 and
    ``"١٢.."`` is 1200. The count does not reliably match the number of zeros,
    so every plausible expansion is offered and the equation picks.
    """
    digits = digit_sequence(raw)
    if not digits:
        return []

    tail = re.search(r"[.…]+\s*$", (raw or "").strip())
    if not tail:
        return [digits]

    dots = len(tail.group(0).replace(" ", ""))

    # One past the dot count: a three-zero amount is sometimes abbreviated to
    # two dots, so the run length is a hint rather than a count.
    sequences = [digits]
    for zeros in range(1, dots + 2):
        expanded = digits + "0" * zeros
        if expanded not in sequences:
            sequences.append(expanded)

    return sequences


def handwriting_candidates(raw: str | None) -> list[float]:
    """Values an amount's reading may stand for, from how handwritten Arabic
    figures are misread -- each one observed on the real invoices:

    * **the decimal comma read as a zero.** "٥,٩٥" came back as "٥٠٩٥" and
      "٥,٧٠" as "٥٠٧": the separator is a small mark beside the digits, and so
      is the dot-shaped zero "٠". An interior zero may be the separator, when
      the reading shows no separator of its own.
    * **the dot-shaped zero read as a separator.** "١٠٩٤,٤" came back as
      "1.٩٤٤": a dot between digits may be the zero.
    * **a run of dot-zeros counted short.** "٢١٢..." (212000) came back as
      "٢١٢٠٠", "١٢.." as "١٢٠". A reading ending in zeros may have lost one.
      Never the reverse: no reading was seen to gain a zero.

    The equation, not this list, decides; these only make the right value
    available to it.
    """
    text = fold_latin_lookalikes(fold_digits(raw or "")).strip()
    digits = digit_sequence(text)
    if not digits:
        return []

    values: list[float] = []

    def add(value: float) -> None:
        if value not in values:
            values.append(value)

    has_separator = observed_decimals(raw) is not None
    if not has_separator:
        for i in range(1, len(digits) - 1):
            decimals = len(digits) - i - 1
            if digits[i] == "0" and decimals <= MAX_DECIMALS and set(digits[i + 1:]) != {"0"}:
                add(float(f"{digits[:i]}.{digits[i + 1:]}"))

    for match in re.finditer(r"(?<=\d)[.·](?=\d)", text):
        zeroed = digit_sequence(text[: match.start()] + "0" + text[match.end():])
        for value in _scaled(zeroed, has_separator=False):
            add(value)

    if digits.endswith("0") and not re.search(r"[.…]\s*$", text):
        for value in _scaled(digits + "0", has_separator=False):
            add(value)
    return values


def _scaled(sequence: str, has_separator: bool) -> list[float]:
    """decimal_candidates, less the placements a reading without a separator
    cannot mean: "13250" is never 1325.0 -- nobody writes a decimal part of
    zeros without writing the separator before it."""
    if has_separator:
        return decimal_candidates(sequence)
    return [
        value for places, value in _placements(sequence)
        if places == 0 or set(sequence[-places:]) != {"0"}
    ]


def _placements(sequence: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for places in range(MAX_DECIMALS + 1):
        if places == 0:
            out.append((0, float(sequence)))
        elif len(sequence) > places:
            out.append((places, float(f"{sequence[:-places]}.{sequence[-places:]}")))
    return out


def _pool(cell: _Cell, integral: bool) -> list[float]:
    """Every value this cell could hold, best-guess order.

    `integral` for the quantity column: a quantity is a whole number by the
    project's normalization rules, so there is no separator to search for.
    """
    sequences = dot_run_candidates(cell.raw) or ([cell.digits] if cell.digits else [])

    if integral:
        return [float(sequence) for sequence in sequences if sequence]

    has_separator = observed_decimals(cell.raw) is not None
    values: list[float] = []
    for sequence in sequences:
        for value in _scaled(sequence, has_separator):
            if value not in values:
                values.append(value)
    for value in handwriting_candidates(cell.raw):
        if value not in values:
            values.append(value)

    return values


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def observed_decimals(raw: str | None) -> int | None:
    """How many digits followed the separator in the raw reading, if legible.

    The *position* of a separator is a clearer visual mark than the shape of a
    digit, so the models get it right more often than they get the digits right
    -- which is what disambiguates 517.65 from 5176.5 when both satisfy the
    equation. A reading with several separators is dot-pattern noise, not
    formatting, and its signal is discarded rather than trusted.
    """
    text = fold_latin_lookalikes(fold_digits(raw or ""))
    parts = [part for part in _SEPARATORS.split(text) if part.strip()]

    if len(parts) != 2:
        return None

    tail = re.sub(r"\D", "", parts[-1])
    return len(tail) if 1 <= len(tail) <= 2 else None


def _separator_target(raw: str | None) -> float | None:
    """The value the reading claims when its separator is taken at face value.

    Built from the digit sequence plus the separator position rather than by
    parsing, so it survives the noise around the number -- and so that a printed
    "150.00" claims 150.0 rather than the 0-decimal 150 that stripping trailing
    zeros would suggest.
    """
    wanted = observed_decimals(raw)
    if wanted is None:
        return None

    digits = digit_sequence(raw)
    if not digits:
        return None

    return int(digits) / (10**wanted)


def _matches_separator(value: float, raw: str | None) -> bool | None:
    """Whether `value` sits where the reading's separator says. None = no signal."""
    target = _separator_target(raw)
    if target is None:
        return None
    return abs(value - target) <= 1e-9


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current

    return previous[-1]


def _explains_reading(derived: float, observed: str) -> bool:
    """Whether `derived` is a plausible correction of what was actually read."""
    if not observed:
        return False

    digits = re.sub(r"\D", "", f"{derived:.2f}".rstrip("0").rstrip("."))
    span = max(len(digits), len(observed))
    return _levenshtein(digits, observed) / span <= MAX_ERROR_RATE


def _pick_by_separator(options: list[float], raw: str | None) -> float | None:
    """Break a tie between same-sequence values using the separator's position."""
    unique = sorted(set(options))
    if len(unique) == 1:
        return unique[0]

    target = _separator_target(raw)
    if target is None:
        return None

    matched = [value for value in unique if abs(value - target) <= 1e-9]
    return matched[0] if len(matched) == 1 else None


def _settled_value(cell: _Cell, pool: list[float]) -> float | None:
    """The one value this cell can be read as, or None if it is ambiguous.

    The parser's own reading wins when it had one -- it already applied the
    number rules to a cell those rules accepted. Only a cell they refused falls
    back to the separator evidence.
    """
    if cell.value is not None:
        return cell.value
    return _pick_by_separator(pool, cell.raw)


def _solve_missing(
    missing: str,
    prices: list[float],
    quantities: list[float],
    totals: list[float],
) -> set[float]:
    if missing == "total_price":
        return {round(p * q, 2) for p in prices for q in quantities}
    if missing == "unit_price":
        return {round(t / q, 2) for q in quantities for t in totals if q}
    if missing == "quantity":
        return {round(t / p, 2) for p in prices for t in totals if p}
    return set()


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def reconcile(draft: InvoiceDraft) -> InvoiceDraft:
    """Repair each line's numbers against ``price x quantity = total``.

    Mutates and returns `draft`. Rows the equation cannot settle are left
    exactly as the parser read them: an unresolved row belongs on the
    verification screen, and overwriting it with a guess is what this stage
    exists to avoid.
    """
    # Rows whose three numbers were all actually read. A cell this stage
    # derives makes its row multiply out by construction, which verifies
    # nothing, so only these rows can vouch for the invoice total.
    fully_read = [all(_has_reading(f) for f in (i.quantity, i.unit_price, i.total_price)) for i in draft.line_items]
    draft.warnings = [w for w in draft.warnings if not _is_row_correction(w)]
    corrected = False
    for item in draft.line_items:
        for name, raw, value in _reconcile_row(item):
            corrected = True
            # The digits changed, not only the separator: the row now adds up,
            # so nothing else would ask anyone to look at it.
            draft.warnings.append(InvoiceWarning(
                code=WarningCodes.MANUAL_REVIEW_REQUIRED,
                field=f"line_items[{item.row_index}].{name}",
                message=(f"Row {item.row_index + 1}: {name} was read as '{raw}' and corrected to {value} "
                         "so that quantity x unit price = line total. Confirm it against the paper."),
            ))

    # A row whose digits were just corrected adds up by construction too.
    _reconcile_total(draft, rows_verified=all(fully_read) and not corrected)
    _refresh_arithmetic_warnings(draft)
    _refresh_total_warning(draft)
    return draft


def _is_row_correction(warning: InvoiceWarning) -> bool:
    return (warning.code == WarningCodes.MANUAL_REVIEW_REQUIRED and warning.field is not None
            and warning.field.startswith("line_items[") and "corrected to" in warning.message)


def _has_reading(field) -> bool:
    raw = field.raw if field.raw is not None else field.value
    return bool(digit_sequence("" if raw is None else str(raw)))


def _reconcile_total(draft: InvoiceDraft, rows_verified: bool = False) -> None:
    """Repair the invoice total's decimal position against the line totals.

    A handwritten "214812.05" comes back as the bare digits "٢١٤٨١٢٠٥" -- the
    separator is the mark the model most often drops. The line totals, once
    reconciled, give an independent figure to check each decimal placement
    against. Only a placement that equals their sum is accepted; if none does,
    the reading is left as it was and the TOTAL_MISMATCH warning stands.
    """
    total = draft.header.total_amount
    amounts = [item.total_price.value for item in draft.line_items]
    if not amounts or any(a is None for a in amounts):
        return

    items_sum = round(sum(float(a) for a in amounts), 2)
    if isinstance(total.value, (int, float)) and abs(float(total.value) - items_sum) <= TOLERANCE:
        return

    raw = total.raw if total.raw is not None else total.value
    digits = digit_sequence("" if raw is None else str(raw))
    if not digits:
        return

    matches = [v for v in decimal_candidates(digits) if abs(v - items_sum) <= TOLERANCE]
    if len(matches) == 1:
        total.value = round(matches[0], 2)
        return

    # A digit misread or dropped, not only a separator: the laptop runs read
    # 214812.05 as "۴۱۴۸۱۲۰۵" (one digit wrong) and as "٢١٤٨١٢٫٠" (the last
    # digit dropped). The sum replaces the reading only when every row had all
    # three numbers read and each one multiplies out -- so every amount in the sum
    # was verified by its own row -- and the reading differs from the sum by
    # at most MAX_TOTAL_EDITS digits. It is still a correction, so it is
    # flagged for the verification screen rather than passed off as read.
    if not rows_verified or not all(_row_multiplies_out(item) for item in draft.line_items):
        return
    expected = digit_sequence(f"{items_sum:.2f}".rstrip("0").rstrip("."))
    edits = _levenshtein(expected, digits)
    if 0 < edits <= MAX_TOTAL_EDITS and edits / max(len(expected), len(digits)) <= 0.25:
        read_as = total.value
        total.value = items_sum
        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.MANUAL_REVIEW_REQUIRED,
                field="total_amount",
                message=(
                    f"The total was read as '{raw}' ({read_as}); every row multiplies out and the "
                    f"line totals sum to {items_sum:.2f}, which differs from the reading by "
                    f"{edits} digit(s), so the sum is offered. Confirm it against the paper."
                ),
            )
        )


def _refresh_total_warning(draft: InvoiceDraft) -> None:
    """Re-issue TOTAL_MISMATCH against the reconciled rows and total.

    The parser compares the stated total with the rows as they were read;
    after the rows and the total are repaired that comparison can be wrong in
    either direction.
    """
    draft.warnings = [w for w in draft.warnings if w.code != WarningCodes.TOTAL_MISMATCH]
    stated = draft.header.total_amount.value
    if not draft.line_items or not isinstance(stated, (int, float)):
        return
    items_sum = sum(float(item.total_price.value or 0.0) for item in draft.line_items)
    if abs(items_sum - float(stated)) > TOTAL_MISMATCH_TOLERANCE:
        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.TOTAL_MISMATCH,
                field="total_amount",
                message=f"Line items sum to {items_sum:.2f} but the stated total is {float(stated):.2f}.",
            )
        )


def _refresh_arithmetic_warnings(draft: InvoiceDraft) -> None:
    """Re-issue ARITHMETIC_MISMATCH against the reconciled numbers.

    The parser raised those warnings against the rows as they were read. A row
    this stage repaired is no longer mismatched, and leaving the old warning
    standing would put a contradiction into the run's own record.
    """
    draft.warnings = [
        warning
        for warning in draft.warnings
        if warning.code != WarningCodes.ARITHMETIC_MISMATCH
    ]

    for item in draft.line_items:
        if item.arithmetic_ok:
            continue

        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.ARITHMETIC_MISMATCH,
                field=f"line_items[{item.row_index}]",
                message=(
                    f"Row {item.row_index + 1}: quantity x unit price does not "
                    "equal the line total, and no decimal placement reconciles them."
                ),
            )
        )


def _cells(item: InvoiceLineItem) -> dict[str, _Cell]:
    def cell(name: str, field) -> _Cell:
        raw = field.raw if field.raw is not None else field.value
        text = "" if raw is None else str(raw)
        return _Cell(
            name=name,
            raw=text,
            digits=digit_sequence(text),
            value=float(field.value) if isinstance(field.value, (int, float)) else None,
        )

    return {
        "unit_price": cell("unit_price", item.unit_price),
        "quantity": cell("quantity", item.quantity),
        "total_price": cell("total_price", item.total_price),
    }


def _write_back(item: InvoiceLineItem, name: str, value: float) -> None:
    field = getattr(item, name)

    # Quantity stays an integer, matching normalize_quantity: a fractional
    # quantity here is a division artefact, not something the paper said.
    field.value = int(round(value)) if name == "quantity" else round(value, 2)


def _reconcile_row(item: InvoiceLineItem) -> list[tuple[str, str, float]]:
    cells = _cells(item)
    known = {name: cell for name, cell in cells.items() if cell.known}

    pools = {
        name: _pool(cell, integral=(name == "quantity")) for name, cell in cells.items()
    }

    changed: list[tuple[str, str, float]] = []
    if len(known) == 3:
        changed = _reconcile_complete_row(item, cells, pools)
    elif len(known) == 2:
        _derive_missing_cell(item, cells, pools)

    item.arithmetic_ok = _row_is_consistent(item)
    return changed


def _reconcile_complete_row(item, cells: dict[str, _Cell], pools) -> list[tuple[str, str, float]]:
    # A row whose own readings already multiply out needs nothing. Searching one
    # anyway would let a differently-scaled combination that also satisfies the
    # equation replace three correct numbers.
    if _row_multiplies_out(item):
        return []

    combinations = [
        (price, quantity, total)
        for price in pools["unit_price"]
        for quantity in pools["quantity"]
        for total in pools["total_price"]
        if abs(price * quantity - total) <= TOLERANCE
    ]

    if not combinations:
        return _repair_least_trusted_cell(item, cells, pools)

    if len(combinations) > 1:
        chosen = _best_scaled(combinations, cells)
        if chosen is None:
            return []
        combinations = [chosen]

    price, quantity, _ = combinations[0]

    # The product, not the read total: the match was accepted within tolerance,
    # so the read total may carry a stray digit the multiplication does not.
    changed = []
    for name, value in (("unit_price", price), ("quantity", quantity), ("total_price", round(price * quantity, 2))):
        if _digits_of(value) != cells[name].digits and not _only_trailing_zeros_added(value, cells[name].raw):
            changed.append((name, cells[name].raw, value))
        _write_back(item, name, value)
    return changed


def _only_trailing_zeros_added(value: float, raw: str) -> bool:
    """A reading ending in dots ("212...") whose value only spells out those
    dots as zeros has not been corrected, only read."""
    return bool(re.search(r"[.…]\s*$", raw or "")) and _digits_of(value) in dot_run_candidates(raw)


def _best_scaled(combinations, cells: dict[str, _Cell]):
    """Pick among combinations the equation cannot separate.

    5.95 x 87, 59.5 x 87 and 595 x 87 are all internally consistent; the
    equation fixes the ratios and says nothing about the scale. Two independent
    signals break the tie, and each cell contributes exactly one of them:

    * A cell whose reading carries **one** legible separator has already said
      where the point goes, and that is the stronger evidence.
    * A cell with no separator signal can only offer its literal reading -- the
      bare digit sequence, unscaled. "13250 x 16 = 212000" beats
      "1325 x 16 = 21200" on that basis.

    A cell never votes twice. Counting a legible cell's bare-digit reading as
    well would have it argue against its own separator and cancel out.
    """
    scored = []

    for price, quantity, total in combinations:
        score = 0

        for name, value in (
            ("unit_price", price),
            ("quantity", quantity),
            ("total_price", total),
        ):
            cell = cells[name]
            agrees = _matches_separator(value, cell.raw)

            if agrees is not None:
                score += 1 if agrees else -1
            elif cell.digits and abs(value - float(cell.digits)) <= TOLERANCE:
                score += 1

        scored.append((score, (price, quantity, total)))

    best = max(score for score, _ in scored)
    winners = [combo for score, combo in scored if score == best]

    # Ambiguous and unrankable: leave the row exactly as it was read. A coin
    # flip between two scales is the one case where the user must decide.
    return winners[0] if best > 0 and len(winners) == 1 else None


def _repair_least_trusted_cell(item, cells: dict[str, _Cell], pools) -> list[tuple[str, str, float]]:
    """No combination works, so one cell was misread: find which.

    Every read cell is tried as the misread one. Its value is derived from the
    other two (over their candidate values), and kept only if it still
    *explains* what was read (`_explains_reading`). The explanation closest to
    the reading wins -- fewest digits changed, then the most other cells taken
    at their literal reading -- and only if it is the single best; a tie is
    left for the user. The other two cells are written back at the values the
    derivation used, which is how "٥٠٧" becomes 5.7 when the line total says so.

    Returns the cells whose digits changed, for the review warning.
    """
    known = {name: cell for name, cell in cells.items() if cell.known}
    if len(known) < 3:
        return []

    scored = []
    for suspect, cell in known.items():
        others = [name for name in ("unit_price", "quantity", "total_price") if name != suspect]
        for a in pools[others[0]]:
            for b in pools[others[1]]:
                given = {others[0]: a, others[1]: b}
                options = _solve_missing(
                    suspect,
                    [given["unit_price"]] if "unit_price" in given else [],
                    [given["quantity"]] if "quantity" in given else [],
                    [given["total_price"]] if "total_price" in given else [],
                )
                for value in options:
                    if suspect == "quantity" and abs(value - round(value)) > 1e-9:
                        continue
                    if value <= 0 or not _explains_reading(value, cell.digits):
                        continue
                    distance = _levenshtein(_digits_of(value), cell.digits)
                    literal = sum(
                        cells[name].digits and abs(v - float(cells[name].digits)) <= TOLERANCE
                        for name, v in given.items()
                    )
                    scored.append(((distance, -literal), suspect, value, given))

    if not scored:
        return []
    scored.sort(key=lambda entry: entry[0])
    best_key = scored[0][0]
    best = {(suspect, value, tuple(sorted(given.items()))) for key, suspect, value, given in scored if key == best_key}
    if len(best) != 1:
        return []

    suspect, value, given = next(iter(best))
    changed = []
    for name, new in (*given, (suspect, value)):
        if _digits_of(new) != cells[name].digits:
            changed.append((name, cells[name].raw, new))
        _write_back(item, name, new)
    return changed


def _digits_of(value: float) -> str:
    return re.sub(r"\D", "", f"{value:.2f}".rstrip("0").rstrip("."))


def _derive_missing_cell(item, cells: dict[str, _Cell], pools) -> None:
    """Fill the one unread cell from the two that were read.

    Both sources have to settle on a single value first: multiplying two
    ambiguous readings together would produce a whole spread of "answers" and
    then pick one, which is guessing dressed as arithmetic.

    The result is mathematically sound and factually unconfirmed -- neither
    source cell was verified by anything. It is offered so the row is not left
    blank; the verification screen is where it gets confirmed.
    """
    missing = next(
        name
        for name in ("unit_price", "quantity", "total_price")
        if not cells[name].known
    )

    known = {
        name: _settled_value(cell, pools[name])
        for name, cell in cells.items()
        if name != missing
    }

    if any(value is None for value in known.values()):
        return

    options = _solve_missing(
        missing,
        [known["unit_price"]] if "unit_price" in known else [],
        [known["quantity"]] if "quantity" in known else [],
        [known["total_price"]] if "total_price" in known else [],
    )

    if len(options) == 1:
        _write_back(item, missing, options.pop())


def _purity(cell: _Cell) -> float:
    """Digits as a fraction of the raw reading -- a clean crop approaches 1.0."""
    raw = cell.raw.strip()
    return len(cell.digits) / len(raw) if raw else 0.0


def _row_multiplies_out(item: InvoiceLineItem) -> bool:
    """All three numbers present, and the equation already holds."""
    values = [item.quantity.value, item.unit_price.value, item.total_price.value]
    if any(value is None for value in values):
        return False

    quantity, price, total = (float(value) for value in values)
    return abs(quantity * price - total) <= TOLERANCE


def _row_is_consistent(item: InvoiceLineItem) -> bool:
    """What `arithmetic_ok` reports.

    An incomplete row counts as consistent: there is no disagreement to flag
    when one of the three numbers was never read, and reporting one would put an
    amber highlight on every blank cell of an unfilled form.
    """
    values = [item.quantity.value, item.unit_price.value, item.total_price.value]
    if any(value is None for value in values):
        return True

    return _row_multiplies_out(item)
