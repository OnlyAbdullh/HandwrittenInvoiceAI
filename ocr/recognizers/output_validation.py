"""Rejects recognizer output that cannot be what the cell holds.

A small VLM shown a blank or illegible crop does not reliably answer "EMPTY":
measured on a real invoice, blank cells came back as invented Arabic prose
("وهو نائم بكلمات لا أفهمها ..."), stray markup ("</quote>"), and a plausible
"12" in every empty quantity box -- six phantom line items on a four-line
invoice. A reading that does not fit its cell's :class:`ContentKind` is
therefore treated as *no reading*: an empty cell the verification screen
flags, never a value someone has to notice is fabricated.

Shapes are checked on the value with any printed caption removed: several
forms print the caption in the same box ("رقم الفاتورة : 00010"), and the
caption is not part of the value.
"""

from __future__ import annotations

import re

from core.domain.roles import ContentKind
from string_matching.normalization import fold_digits, fold_latin_lookalikes

_EMPTY_SENTINEL = "EMPTY"

# Chat/markup debris a VLM emits around (or instead of) an answer.
_ARTEFACTS = re.compile(r"</?[A-Za-z_][\w\-]*>|```")

# Quotes are stripped only when they wrap the whole reading. A lone trailing
# quote is part of what the model saw -- `٥"` was its reading of `٢٠٠`, the two
# zero-dots merged into one mark -- and dropping it would turn an unreadable
# cell into a confident, wrong `5`.
_WRAPPING_QUOTES = re.compile(r"^[\"'«“](.*)[\"'»”]$", re.DOTALL)

_CAPTION_SEPARATORS = (":", "：")

_NUMBER = re.compile(r"[\d\s.,٫٬،\-]+")
_DATE = re.compile(r"[\d\s/\-.٫،,]+")
_CODE = re.compile(r"[\w\s\-/.]+", re.ASCII)

_MAX_NUMBER_DIGITS = 15
_MAX_CODE_LENGTH = 24
_MAX_DATE_LENGTH = 24
_MAX_TEXT_WORDS = {ContentKind.ARABIC_TEXT: 8, ContentKind.ANY: 20}
_MAX_TEXT_CHARS = {ContentKind.ARABIC_TEXT: 60, ContentKind.ANY: 160}


def _value_part(text: str) -> str:
    """The value with a leading printed caption removed, if there is one."""
    for separator in _CAPTION_SEPARATORS:
        head, found, tail = text.partition(separator)
        if found and tail.strip():
            return tail.strip()
    return text


def clean(text: str | None, kind: ContentKind) -> str | None:
    """The reading as it should be kept, "" for a blank cell, None if rejected.

    None means the output was something, but not something this cell can
    hold -- the caller records it as empty with zero confidence.
    """
    cleaned = _ARTEFACTS.sub("", (text or "")).strip()
    wrapped = _WRAPPING_QUOTES.fullmatch(cleaned)
    if wrapped:
        cleaned = wrapped.group(1).strip()
    if not cleaned or cleaned.upper() == _EMPTY_SENTINEL:
        return ""

    if kind in (ContentKind.NUMBER, ContentKind.DATE, ContentKind.CODE):
        value = _value_part(cleaned)
        folded = fold_latin_lookalikes(fold_digits(value))

        if kind is ContentKind.NUMBER:
            digits = sum(ch.isdigit() for ch in folded)
            ok = _NUMBER.fullmatch(folded) is not None and 1 <= digits <= _MAX_NUMBER_DIGITS
        elif kind is ContentKind.DATE:
            digits = sum(ch.isdigit() for ch in folded)
            ok = _DATE.fullmatch(folded) is not None and digits >= 3 and len(folded) <= _MAX_DATE_LENGTH
        else:
            ok = (
                _CODE.fullmatch(fold_digits(value)) is not None
                and any(ch.isalnum() for ch in value)
                and len(value) <= _MAX_CODE_LENGTH
            )
        return value if ok else None

    # Free text: a cell holds a name or a short description, not a paragraph.
    if not any(ch.isalnum() for ch in cleaned):
        return None
    words = len(cleaned.split())
    if words > _MAX_TEXT_WORDS.get(kind, 20) or len(cleaned) > _MAX_TEXT_CHARS.get(kind, 160):
        return None
    return cleaned
