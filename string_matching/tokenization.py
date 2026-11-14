"""Splitting a product name into the words that identify it.

:func:`string_matching.normalization.normalize_words` splits on whitespace
only, which is right for a merchant's name and wrong for a product's: a
catalog writes ``4سم``, ``38/48ماركة`` and ``باكستاني46``, and each of those is
a size glued to a word. Read as one token, the size can never be compared with
the size on the invoice, and sizes are what tell most sibling products apart.

Digits and letters are therefore separated here. A run of digits is always its
own token, which is also what lets the matcher compare numbers exactly while
comparing words approximately.
"""

from __future__ import annotations

import re

from string_matching.normalization import normalize_text

# A boundary between a digit and anything that is neither a digit nor a space.
_DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\d\s])|(?<=[^\d\s])(?=\d)")


def split_words(text: str | None) -> list[str]:
    """`text` normalized and split into words, with digit runs set apart.

    ``"قشاط 4سم"`` -> ``["قشاط", "4", "سم"]``; ``"38/48ماركة G-5"`` ->
    ``["38", "48", "ماركه", "g", "5"]``.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    return _DIGIT_BOUNDARY.sub(" ", normalized).split()


def is_number(word: str) -> bool:
    """Whether a token is a run of digits -- a size, a width, a model number."""
    return word.isdigit()
