"""Recognizer output that cannot be what the cell holds is rejected.

Every rejected example below is a reading the VLM actually produced for a
*blank* cell of bills/sahror; every accepted one is a real reading of a filled
cell.
"""

from __future__ import annotations

import pytest

from core.domain.roles import ContentKind
from ocr.recognizers.output_validation import clean


@pytest.mark.parametrize(
    "text, kind, expected",
    [
        # Real readings, kept.
        ("١۶", ContentKind.NUMBER, "١۶"),
        ("٥,٩٥", ContentKind.NUMBER, "٥,٩٥"),
        ("٥۱٧, ٦۵", ContentKind.NUMBER, "٥۱٧, ٦۵"),
        ("٢١٢...", ContentKind.NUMBER, "٢١٢..."),
        ("رقم الفاتورة : 00010", ContentKind.CODE, "00010"),
        ("A-1903", ContentKind.CODE, "A-1903"),
        ("التاريخ : 7/8/2026", ContentKind.DATE, "7/8/2026"),
        ("بنطلون كمان كبير", ContentKind.ARABIC_TEXT, "بنطلون كمان كبير"),
        ("الاسم : حسام المارديني", ContentKind.ARABIC_TEXT, "الاسم : حسام المارديني"),
        # Blank, said plainly.
        ("EMPTY", ContentKind.NUMBER, ""),
        ("  ", ContentKind.ARABIC_TEXT, ""),
        ("</quote>", ContentKind.NUMBER, ""),
    ],
)
def test_kept_or_blank(text, kind, expected):
    assert clean(text, kind) == expected


@pytest.mark.parametrize(
    "text, kind",
    [
        # Hallucinated on blank cells of bills/sahror.
        ("وهو نائم  بكلمات لا أفهمها مثل انقض بغـلس له الضابط لزمتك. سألت راجح هل بلغ أصحابنا", ContentKind.ARABIC_TEXT),
        ("أبو عدي بن قريش بن لؤي بن قبيات بن جرجس بن هلال بن نوح بن عمطى", ContentKind.NUMBER),
        ("هذه الحافنة هي حاملة العرق", ContentKind.NUMBER),
        ("...</quote>", ContentKind.ARABIC_TEXT),
        ('٥"', ContentKind.NUMBER),
        ("٢»١» ...", ContentKind.NUMBER),
        ("التاريخ : /٨/", ContentKind.DATE),
        ("1234567890123456", ContentKind.NUMBER),
    ],
)
def test_rejected(text, kind):
    assert clean(text, kind) is None
