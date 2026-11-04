"""Pre-processing applied to every string before it reaches a matcher.

Matching Arabic OCR output against a catalog only works if both sides are first
reduced to the same canonical form. These rules are the only implementation of
them: the desktop app used to mirror them in
``InvoiceDigitizationApp/Helpers/TextNormalizer.cs`` so it could re-match locally,
and that copy is deleted. Nothing here has to be kept in step with a second
version any more, which is the point -- the two drifted, invisibly, and a
merchant the service had matched came back unmatched in the app.

Three separate normalizations live here because they have genuinely different
rules:

* :func:`normalize_text`   -- for names (merchants, products, cities)
* :func:`normalize_quantity` / :func:`normalize_price` -- for numeric cells
* :func:`normalize_date`   -- for date cells

Nothing here is ever stored as the user-visible value. Normalization is for
comparison and parsing only; the raw OCR text always travels alongside so the
verification screen can show what the paper actually said.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

# --------------------------------------------------------------------------
# Digits
# --------------------------------------------------------------------------

# Arabic-Indic (U+0660-U+0669) and extended/Persian Arabic-Indic (U+06F0-U+06F9).
_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"

_DIGIT_MAP = {
    ord(ch): str(index)
    for group in (_ARABIC_INDIC, _EXTENDED_ARABIC_INDIC)
    for index, ch in enumerate(group)
}


def fold_digits(text: str) -> str:
    """Arabic-Indic digits to ASCII, leaving every other character alone."""
    return text.translate(_DIGIT_MAP)


# The VLM recognizers translate a handwritten digit into the Latin letter it
# looks like. Only the three that were actually measured on this project's
# invoices are folded -- adding more on a hunch would corrupt product names,
# which pass through the same normalizer.
_LATIN_LOOKALIKES = str.maketrans({"O": "5", "o": "5", "V": "7", "v": "7", "A": "8"})


def fold_latin_lookalikes(text: str | None) -> str:
    """Undo the three documented digit-to-Latin-letter mis-readings."""
    return (text or "").translate(_LATIN_LOOKALIKES)


def digit_sequence(text: str | None) -> str:
    """Every digit in `text`, in order, with nothing else.

    The reading a numeric cell is most likely to have got *right*: these models
    misplace separators far more often than they misread a digit, so the bare
    sequence is what the arithmetic reconciliation searches decimal positions
    over. See :mod:`invoice.reconciliation`.
    """
    return re.sub(r"\D", "", fold_latin_lookalikes(fold_digits(text or "")))


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

# Harakat (U+064B-U+0652) plus the tatweel/kashida (U+0640). Written as an
# explicit range because combining marks are invisible in an editor and are
# easily mangled by a copy-paste. The range deliberately stops above U+064A
# (yeh), which is a real letter and must survive.
_DIACRITICS = re.compile("[ً-ْـ]")

# Letter forms that vary freely between writers for the same word.
_LETTER_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        # Persian/Urdu letter forms the VLM emits for the same handwriting.
        "ی": "ي",
        "ې": "ي",
        "ک": "ك",
        "ۀ": "ه",
        "ھ": "ه",
        "ہ": "ه",
    }
)

# Punctuation, Arabic and Latin alike. Digits are deliberately NOT stripped:
# they are part of real product names ("سكر 1كغ", "A4 Paper").
_PUNCTUATION = re.compile(r"[.,;:!؟?،؛(){}\[\]<>\"'`«»/\\|_\-+*=&^%#@~$]")

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Fold a name to the canonical form used for every similarity comparison.

    Applies, in order: NFKC (collapses presentation forms such as ﻻ back to
    their canonical letters), Arabic-Indic digits to ASCII, diacritic and
    tatweel removal, alef/yeh/teh-marbuta folding, punctuation removal,
    whitespace collapse, and case folding.
    """
    if not text:
        return ""

    result = unicodedata.normalize("NFKC", text)
    result = fold_digits(result)
    result = _DIACRITICS.sub("", result)
    result = result.translate(_LETTER_FOLD)
    result = _PUNCTUATION.sub(" ", result)
    result = _WHITESPACE.sub(" ", result).strip()
    return result.casefold()


def normalize_words(text: str | None) -> list[str]:
    """:func:`normalize_text` split into words, for order-independent matching."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

# Currency markers that sit next to amounts on regional invoices and must not
# be read as part of the number.
_CURRENCY = re.compile(
    r"(?:ر\.س|ر\.ع|د\.أ|د\.ك|ج\.م|ل\.س|SAR|AED|JOD|KWD|EGP|SYP|USD|"
    r"ريال|درهم|دينار|جنيه|ليره|ليرة|\$|€|£)",
    re.IGNORECASE,
)

_NON_DIGIT = re.compile(r"[^0-9]")


def _clean_numeric_text(text: str) -> str:
    """Shared pre-cleaning: NFKC, fold digits, drop currency, unify separators."""
    cleaned = fold_digits(unicodedata.normalize("NFKC", text))
    cleaned = _CURRENCY.sub(" ", cleaned)
    # U+066B is unambiguously the decimal point and U+066C unambiguously a
    # thousands separator, so it is dropped: turning it into "," let the
    # "rightmost separator is the decimal" rule read ١٬٢٥٠ as 1.25.
    return cleaned.replace("٫", ".").replace("٬", "")


# A separator straight after a separator, before a digit: "214812,.5". Two
# separators in a row cannot both be separators, and the second one, a dot, is
# the Arabic-Indic zero "٠" -- which is written as a dot. The paper said
# ٢١٤٨١٢,٠٥; as read, this became 214812.5 (",." -> one separator) or
# 2148125 ("..": both dropped as thousands separators).
_ZERO_AFTER_SEPARATOR = re.compile(r"(?<=\d)([.,])[.·](?=\d)")


def _written_zeros(cleaned: str) -> str:
    return _ZERO_AFTER_SEPARATOR.sub(r"\g<1>0", cleaned)


def normalize_quantity(text: str | None) -> int | None:
    """Read a quantity cell as a whole number.

    Quantities on these invoices are always integers, so everything that is not
    a digit is discarded rather than interpreted. This is deliberate and is the
    rule the AI team specified: a comma or a dot in a handwritten quantity is
    almost always a mis-read stroke, not a decimal point, and the digit it was
    printed next to is nearly always a zero -- so ``٢.٠`` and ``٢،٠`` both read
    as ``20``, not ``2``.

    Returns None when the cell carries no digit at all.
    """
    if not text:
        return None

    digits = _NON_DIGIT.sub("", _clean_numeric_text(text))
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:  # pragma: no cover - unreachable while the regex holds
        return None


def normalize_price(text: str | None) -> float | None:
    """Read a price cell as a decimal number.

    Unlike a quantity, a price genuinely has a fractional part, so separators
    are resolved rather than stripped: the rightmost separator is the decimal
    point (a thousands separator is never last) and every other one is dropped.
    """
    if not text:
        return None

    cleaned = _written_zeros(_clean_numeric_text(text))

    match = re.search(r"-?\d[\d,.\s]*", cleaned)
    if not match:
        return None

    candidate = _resolve_separators(match.group(0).strip().replace(" ", ""))

    try:
        return float(candidate)
    except ValueError:
        return None


def parse_number_strict(text: str | None) -> float | None:
    """Parse text that is *entirely* a number, else None.

    Distinct from :func:`normalize_price`, which searches for a number anywhere
    in the string. Use this to answer "is this token a numeric cell?" -- a
    product name like ``سكر 1كغ`` or ``A4 Paper`` must not be classified as the
    number 1 or 4, which would both shift the column mapping and erase the
    product name from the row.
    """
    if not text:
        return None

    cleaned = _written_zeros(_clean_numeric_text(text).strip())
    if not cleaned:
        return None

    if not re.fullmatch(r"-?\d[\d,.\s]*", cleaned):
        return None

    candidate = _resolve_separators(cleaned.replace(" ", ""))

    try:
        return float(candidate)
    except ValueError:
        return None


def _resolve_separators(value: str) -> str:
    """Decide which of ``.`` and ``,`` is the decimal point.

    OCR reads "1,234.50" and "1.234,50" for the same amount depending on how the
    invoice was printed. The rightmost separator wins.
    """
    last_dot = value.rfind(".")
    last_comma = value.rfind(",")

    if last_dot == -1 and last_comma == -1:
        return value

    if last_comma > last_dot:
        # The comma is the decimal point: drop dots, then swap it for one.
        return value.replace(".", "").replace(",", ".")

    result = value.replace(",", "")

    # "1.234.500" -- more than one dot means they are all thousands separators.
    if result.count(".") > 1:
        result = result.replace(".", "")

    return result


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

# Any separator a writer might use between date components.
# "٫" (Arabic decimal separator) and "،" (Arabic comma) are in the date cells'
# decoding alphabet (ocr/decoding_policy.py), so a reading can contain them.
_DATE_SEPARATORS = re.compile(r"[-.\\_\s٫،]+")


def normalize_date_text(text: str | None) -> str:
    """Canonicalize a date cell's *text* to ``DD/MM/YYYY``-shaped input.

    Folds Arabic-Indic digits, rewrites every separator to ``/`` and removes the
    whitespace writers leave around the slashes. This does not validate the date
    -- :func:`parse_date` does that -- it only makes the string parseable.
    """
    if not text:
        return ""

    cleaned = fold_digits(unicodedata.normalize("NFKC", text)).strip()
    cleaned = _DATE_SEPARATORS.sub("/", cleaned)
    # Collapse the runs a stray separator can leave behind ("14//03///2026").
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    return cleaned.strip("/")


# Ordered most-specific first, so an unambiguous ISO date is never read as D/M/Y.
#
# Each pattern stands alone: it may not start or end inside a longer number or a
# longer run of separated numbers. Without that, "7/8/226" (a year missing a
# digit) matched "7/8/22" and came out as 2022-08-07, and "7/8/2/26" matched
# "8/2/26" -- both confident, wrong dates. Normalization turns spaces into "/",
# which is why a lone "/" before a pattern is allowed but "<digit>/" is not.
_ALONE_BEFORE = r"(?<!\d)(?<!\d/)"
_ALONE_AFTER = r"(?!\d)(?!/\d)"
_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(_ALONE_BEFORE + r"(\d{4})/(\d{1,2})/(\d{1,2})" + _ALONE_AFTER), "ymd"),
    (re.compile(_ALONE_BEFORE + r"(\d{1,2})/(\d{1,2})/(\d{4})" + _ALONE_AFTER), "dmy"),
    (re.compile(_ALONE_BEFORE + r"(\d{1,2})/(\d{1,2})/(\d{2})" + _ALONE_AFTER), "dmy2"),
]

# A dot between two digits. The Arabic-Indic zero "٠" is written as a dot, and a
# recognizer allowed both digits and "." in a date cell reads "٢٠٢٦" as "2.26"
# as often as not.
_DOT_BETWEEN_DIGITS = re.compile(r"(?<=\d)[.٫·](?=\d)")


# Invoices are not dated before this or in the far future. A handwritten
# "٢٠٢٦" read as "٢٠٦٦" is a valid calendar date and a wrong invoice.
MIN_PLAUSIBLE_YEAR = 2000


def is_plausible_date(value: date) -> bool:
    return MIN_PLAUSIBLE_YEAR <= value.year <= date.today().year + 1


def parse_date(text: str | None, plausible_only: bool = False) -> date | None:
    """Find a date in OCR text and return it, or None.

    With `plausible_only`, a date whose year falls outside
    [2000, next year] is treated as not found: such a reading is far more
    likely a mis-read digit than a real date, and reporting it would put a
    confident wrong value on the invoice.

    Day/month order is ambiguous in ``03/04/2026``. This assumes **day first**,
    the regional convention. When one component exceeds 12 the order is
    unambiguous and is corrected automatically.
    """
    if not text:
        return None

    found = _parse_date_text(normalize_date_text(text), plausible_only)
    if found is not None:
        return found

    # Nothing parsed as written. A dot between two digits may be a zero (see
    # _DOT_BETWEEN_DIGITS): try each such dot, and every one of them, as "0".
    # Accepted only when every reading that parses names the same date, so a
    # repair never chooses between two different answers.
    raw = fold_digits(unicodedata.normalize("NFKC", text))
    dots = [m.start() for m in _DOT_BETWEEN_DIGITS.finditer(raw)]
    if not dots or len(dots) > 4:
        return None
    variants = {raw[:i] + "0" + raw[i + 1:] for i in dots}
    variants.add(_DOT_BETWEEN_DIGITS.sub("0", raw))
    dates = {d for v in variants if (d := _parse_date_text(normalize_date_text(v), plausible_only)) is not None}
    return dates.pop() if len(dates) == 1 else None


def repair_year(text: str | None) -> date | None:
    """A date whose year was misread by one digit, when only one fix fits.

    The models read the handwritten year "٢٠٢٦" as "٣٠٢٦", "١٠٢٦" and "٢٠٤٦" on
    the real invoices: one digit wrong, the rest right. When the reading parses
    to an implausible year, every single-digit change of that year is tried,
    and the date is returned only if exactly one lands in the plausible range.
    "٣٠٢٦" -> 2026 (no other digit change is plausible); "٢٠٤٦" -> None,
    because 2006, 2016 and 2026 all are. The caller must flag the result for
    review: it is a correction, not a reading.
    """
    found = parse_date(text)
    if found is None or is_plausible_date(found):
        return None
    year = f"{found.year:04d}"
    fixes = set()
    for i, digit in enumerate(year):
        for replacement in "0123456789":
            if replacement == digit:
                continue
            try:
                candidate = found.replace(year=int(year[:i] + replacement + year[i + 1:]))
            except ValueError:
                continue  # 29 February in a non-leap year
            if is_plausible_date(candidate):
                fixes.add(candidate)
    return fixes.pop() if len(fixes) == 1 else None


def _parse_date_text(cleaned: str, plausible_only: bool) -> date | None:
    for pattern, order in _DATE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue

        a, b, c = (int(group) for group in match.groups())

        if order == "ymd":
            year, month, day = a, b, c
        else:
            day, month = a, b
            year = c if order == "dmy" else 2000 + c

            # A "day" over 12 must actually be the month.
            if month > 12 and day <= 12:
                day, month = month, day

        try:
            found = date(year, month, day)
        except ValueError:
            continue  # e.g. 31/02 -- keep looking.
        if plausible_only and not is_plausible_date(found):
            continue
        return found

    return None


def format_date(value: date | None) -> str | None:
    """ISO-8601, the only date format the API contract emits."""
    return value.isoformat() if value else None
