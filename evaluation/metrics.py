"""Scores one extraction against its ground truth.

Scoring reads the **wire response** (the `invoice_json` shape the desktop app
receives), not internal objects, so what is measured is what a user would get.

What is counted, and why each matters to an accountant:

* header fields -- exact after normalization (ISO date, numbers to the cent,
  invoice number with digits folded). A catalog-matched field (customer, city)
  is right when the top-ranked candidate is the true entry; whether that
  candidate also cleared the 0.75 review threshold is reported separately,
  because a correct candidate below it still costs the user a click.
* line items -- predicted rows are aligned to true rows by product name and
  numbers, so a missed row and an invented one are both visible:
  `phantom` rows were invented (blank lines read as items), `missed` rows were
  on the paper and are not in the output.
* numeric cells -- quantity, unit price and line total of every true row;
  a missed row counts all of its cells as wrong.
* name CER -- character error rate of the raw reading (`original_value`)
  against the true name, after the same normalization matching uses. This is
  the recognizer's quality before the catalog rescues it.
* per field -- every header field and every item column on its own, so "the
  totals are right but the dates are not" is visible rather than averaged away.
* complete invoices -- invoices where *everything* scored is right: every
  header field, every numeric cell, every product's top candidate, no row
  invented or missed. This is what "no correction needed" means to a user.
  `complete_without_review` additionally needs every catalog-matched field to
  have cleared the review threshold -- the invoice a user could accept as is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.schema import InvoiceTruth, LineItemTruth
from string_matching.algorithms import levenshtein_distance
from string_matching.normalization import fold_digits, normalize_text

AUTO_ACCEPT_THRESHOLD = 0.75
_MONEY_TOLERANCE = 0.005
_ROW_MATCH_THRESHOLD = 0.5
_NUMERIC_COLUMNS = ("quantity", "unit_price", "total_price")


@dataclass
class FieldScore:
    name: str
    expected: object
    predicted: object
    correct: bool
    auto_accepted: bool | None = None


@dataclass
class InvoiceScore:
    image: str
    form: str | None
    header: list[FieldScore] = field(default_factory=list)
    numeric_cells_total: int = 0
    numeric_cells_correct: int = 0
    # column -> [correct, total], for quantity / unit_price / total_price
    numeric_by_column: dict[str, list[int]] = field(default_factory=dict)
    products_total: int = 0
    product_top1_correct: int = 0
    product_auto_accepted: int = 0
    rows_expected: int = 0
    rows_predicted: int = 0
    rows_matched: int = 0
    name_cer: list[float] = field(default_factory=list)
    seconds: float = 0.0
    model_calls: int | None = None
    error: str | None = None

    @property
    def phantom_rows(self) -> int:
        return self.rows_predicted - self.rows_matched

    @property
    def missed_rows(self) -> int:
        return self.rows_expected - self.rows_matched

    @property
    def complete(self) -> bool:
        """Everything scored on this invoice is right (see the module docstring)."""
        return (
            self.error is None
            and all(f.correct for f in self.header)
            and self.numeric_cells_correct == self.numeric_cells_total
            and self.product_top1_correct == self.products_total
            and self.phantom_rows == 0
            and self.missed_rows == 0
        )

    @property
    def complete_without_review(self) -> bool:
        """Complete, and every catalog match cleared the review threshold."""
        return (
            self.complete
            and all(f.auto_accepted is not False for f in self.header)
            and self.product_auto_accepted == self.products_total
        )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _same_number(predicted, expected) -> bool:
    if expected is None:
        return True
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(expected)) <= _MONEY_TOLERANCE
    except (TypeError, ValueError):
        return False


def _similarity(a: str | None, b: str | None) -> float:
    left, right = normalize_text(a), normalize_text(b)
    if not left or not right:
        return 0.0
    return 1.0 - levenshtein_distance(left, right) / max(len(left), len(right))


def _cer(predicted: str | None, expected: str) -> float:
    truth = normalize_text(expected)
    if not truth:
        return 0.0
    return levenshtein_distance(normalize_text(predicted), truth) / len(truth)


def _top_candidate(matched_field: dict) -> tuple[str | None, float]:
    results = matched_field.get("results") or []
    if not results:
        return None, 0.0
    return results[0].get("value"), float(results[0].get("string_matching_score") or 0.0)


def _matched_field_score(name: str, matched_field: dict, expected: str) -> FieldScore:
    top, score = _top_candidate(matched_field)
    correct = top is not None and normalize_text(top) == normalize_text(expected)
    return FieldScore(
        name=name,
        expected=expected,
        predicted=f"{matched_field.get('original_value')!r} -> {top!r} ({score:.2f})",
        correct=correct,
        auto_accepted=correct and score >= AUTO_ACCEPT_THRESHOLD,
    )


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def _row_similarity(predicted: dict, truth: LineItemTruth) -> float:
    product = predicted.get("product_name") or {}
    top, top_score = _top_candidate(product)
    # The matcher always ranks *something* first, even for an invented row
    # (score 0.0), so the candidate counts only as strongly as it scored.
    name = _similarity(product.get("original_value"), truth.product_name)
    if top and normalize_text(top) == normalize_text(truth.product_name):
        name = max(name, top_score)

    pairs = (
        (predicted.get("quantity", {}).get("value"), truth.quantity),
        (predicted.get("unit_price", {}).get("value"), truth.unit_price),
        (predicted.get("total_price", {}).get("value"), truth.total_price),
    )
    known = [(p, t) for p, t in pairs if t is not None]
    numbers = sum(_same_number(p, t) for p, t in known) / len(known) if known else 0.0

    return max(name, numbers)


def _align(predicted: list[dict], truth: list[LineItemTruth]) -> list[tuple[int, int]]:
    """Greedy one-to-one alignment, best pairs first."""
    scored = sorted(
        (
            (_row_similarity(p, t), pi, ti)
            for pi, p in enumerate(predicted)
            for ti, t in enumerate(truth)
        ),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, pi, ti in scored:
        if score < _ROW_MATCH_THRESHOLD or pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((pi, ti))
    return pairs


# --------------------------------------------------------------------------
# One invoice
# --------------------------------------------------------------------------


def score_invoice(output: dict, truth: InvoiceTruth, seconds: float = 0.0, model_calls: int | None = None) -> InvoiceScore:
    score = InvoiceScore(image=truth.image, form=truth.form, seconds=seconds, model_calls=model_calls)

    if truth.invoice_number is not None:
        predicted = (output.get("invoice_id") or {}).get("value")
        score.header.append(FieldScore(
            "invoice_number", truth.invoice_number, predicted,
            fold_digits(str(predicted or "")).strip() == fold_digits(truth.invoice_number).strip(),
        ))
    if truth.date is not None:
        predicted = (output.get("date") or {}).get("value")
        score.header.append(FieldScore("date", truth.date, predicted, predicted == truth.date))
    if truth.total is not None:
        predicted = (output.get("total_invoice_price") or {}).get("value")
        score.header.append(FieldScore("total", truth.total, predicted, _same_number(predicted, truth.total)))
    if truth.customer_name is not None:
        customer = output.get("customer_name") or {}
        score.header.append(_matched_field_score("customer_name", customer, truth.customer_name))
        score.name_cer.append(_cer(customer.get("original_value"), truth.customer_name))
    if truth.city is not None:
        city = output.get("city") or {}
        score.header.append(_matched_field_score("city", city, truth.city))
        score.name_cer.append(_cer(city.get("original_value"), truth.city))

    predicted_rows = output.get("products") or []
    pairs = {ti: pi for pi, ti in _align(predicted_rows, truth.items)}

    score.rows_expected = len(truth.items)
    score.rows_predicted = len(predicted_rows)
    score.rows_matched = len(pairs)
    score.products_total = len(truth.items)

    for ti, item in enumerate(truth.items):
        cells = [c for c in (item.quantity, item.unit_price, item.total_price) if c is not None]
        score.numeric_cells_total += len(cells)

        pi = pairs.get(ti)
        for column in _NUMERIC_COLUMNS:
            expected = getattr(item, column)
            if expected is None:
                continue
            tally = score.numeric_by_column.setdefault(column, [0, 0])
            tally[1] += 1
            if pi is not None:
                tally[0] += _same_number((predicted_rows[pi].get(column) or {}).get("value"), expected)
        if pi is None:
            score.name_cer.append(1.0)
            continue

        row = predicted_rows[pi]
        score.numeric_cells_correct += sum((
            item.quantity is not None and _same_number(row["quantity"]["value"], item.quantity),
            item.unit_price is not None and _same_number(row["unit_price"]["value"], item.unit_price),
            item.total_price is not None and _same_number(row["total_price"]["value"], item.total_price),
        ))

        product = row.get("product_name") or {}
        top, top_score = _top_candidate(product)
        if top and normalize_text(top) == normalize_text(item.product_name):
            score.product_top1_correct += 1
            if top_score >= AUTO_ACCEPT_THRESHOLD:
                score.product_auto_accepted += 1
        score.name_cer.append(_cer(product.get("original_value"), item.product_name))

    return score


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.0f}%)" if denominator else "n/a"


def by_field(scores: list[InvoiceScore]) -> dict[str, dict[str, int]]:
    """{field: {"correct", "total"}} over every invoice that ran: each header
    field, each numeric item column, and the product's top-ranked candidate."""
    out: dict[str, dict[str, int]] = {}

    def add(name: str, correct: int, total: int) -> None:
        tally = out.setdefault(name, {"correct": 0, "total": 0})
        tally["correct"] += correct
        tally["total"] += total

    for s in scores:
        if s.error is not None:
            continue
        for f in s.header:
            add(f.name, int(f.correct), 1)
        for column, (correct, total) in s.numeric_by_column.items():
            add(column, correct, total)
        if s.products_total:
            add("product_name", s.product_top1_correct, s.products_total)
    return out


def summarize(scores: list[InvoiceScore]) -> dict:
    ok = [s for s in scores if s.error is None]
    header = [f for s in ok for f in s.header]
    cers = [c for s in ok for c in s.name_cer]
    calls = [s.model_calls for s in ok if s.model_calls is not None]
    seconds = [s.seconds for s in ok]
    return {
        "invoices": len(scores),
        "failed": len(scores) - len(ok),
        "header_correct": sum(f.correct for f in header),
        "header_total": len(header),
        "numeric_cells_correct": sum(s.numeric_cells_correct for s in ok),
        "numeric_cells_total": sum(s.numeric_cells_total for s in ok),
        "product_top1_correct": sum(s.product_top1_correct for s in ok),
        "product_auto_accepted": sum(s.product_auto_accepted for s in ok),
        "products_total": sum(s.products_total for s in ok),
        "phantom_rows": sum(s.phantom_rows for s in ok),
        "missed_rows": sum(s.missed_rows for s in ok),
        "mean_name_cer": round(sum(cers) / len(cers), 3) if cers else None,
        "mean_seconds": round(sum(seconds) / len(seconds), 1) if seconds else None,
        "max_seconds": round(max(seconds), 1) if seconds else None,
        "mean_model_calls": round(sum(calls) / len(calls), 1) if calls else None,
        "complete_invoices": sum(s.complete for s in ok),
        "complete_without_review": sum(s.complete_without_review for s in ok),
        "by_field": by_field(scores),
    }


def to_markdown(scores: list[InvoiceScore], config_name: str, run_info: dict | None = None) -> str:
    """The report. `run_info` (optional) carries what was measured about the
    run rather than any one invoice: model load time, peak RAM, VRAM, device."""
    summary = summarize(scores)
    lines = [
        f"# Evaluation — `{config_name}`",
        "",
        f"{summary['invoices']} invoice(s), {summary['failed']} failed to run.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Header fields correct | {_pct(summary['header_correct'], summary['header_total'])} |",
        f"| Numeric cells correct | {_pct(summary['numeric_cells_correct'], summary['numeric_cells_total'])} |",
        f"| Product top-1 correct | {_pct(summary['product_top1_correct'], summary['products_total'])} |",
        f"| Product auto-accepted (≥ {AUTO_ACCEPT_THRESHOLD}) | {_pct(summary['product_auto_accepted'], summary['products_total'])} |",
        f"| Phantom rows / missed rows | {summary['phantom_rows']} / {summary['missed_rows']} |",
        f"| Mean name CER (raw reading) | {summary['mean_name_cer']} |",
        f"| Complete invoices (everything right) | {_pct(summary['complete_invoices'], summary['invoices'])} |",
        f"| Complete and nothing to review | {_pct(summary['complete_without_review'], summary['invoices'])} |",
        f"| Mean / slowest seconds per invoice | {summary['mean_seconds']} / {summary['max_seconds']} |",
        f"| Mean model calls per invoice | {summary['mean_model_calls']} |",
    ]
    for key, label in (("load_seconds", "Pipeline load seconds"), ("peak_ram_mb", "Peak RAM, MB (incl. model server)"),
                       ("vram_mb", "VRAM held, MB"), ("device", "Device")):
        if run_info and run_info.get(key) is not None:
            lines.append(f"| {label} | {run_info[key]} |")
    lines += ["", "### Per field", "", "| Field | Correct |", "|---|---|"]
    for name, tally in summary["by_field"].items():
        lines.append(f"| {name} | {_pct(tally['correct'], tally['total'])} |")
    lines.append("")

    for s in scores:
        lines.append(f"## {s.image}" + (f" ({s.form})" if s.form else ""))
        lines.append("")
        if s.error:
            lines += [f"**Failed:** {s.error}", ""]
            continue
        lines += ["| Field | Expected | Predicted | OK |", "|---|---|---|---|"]
        for f in s.header:
            mark = "✅" if f.correct else "❌"
            if f.correct and f.auto_accepted is False:
                mark = "⚠️ below threshold"
            lines.append(f"| {f.name} | {f.expected} | {f.predicted} | {mark} |")
        lines += [
            "",
            (f"Rows: {s.rows_matched}/{s.rows_expected} matched, {s.phantom_rows} phantom, "
             f"{s.missed_rows} missed. Numeric cells: {_pct(s.numeric_cells_correct, s.numeric_cells_total)}. "
             f"{s.seconds:.1f}s, {s.model_calls if s.model_calls is not None else '?'} model calls. "
             f"Complete: {'yes' if s.complete else 'no'}."),
            "",
        ]
    return "\n".join(lines)
