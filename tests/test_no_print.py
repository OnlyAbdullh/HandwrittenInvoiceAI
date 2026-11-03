"""Production code logs; it never prints.

`CellMapper.map` used to print an emoji-decorated trace of every decision. On
Windows, with stdout redirected (how a desktop app launches a hidden service),
stdout is the ANSI code page, and the first emoji raised UnicodeEncodeError --
every extraction became a 500. pytest captures stdout as UTF-8, so the suite
never saw it. Both halves are pinned here: no print calls exist, and the mapper
survives a stdout that cannot encode Arabic.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest

from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.roles import CellRole, Zone
from mapping.cell_mapper import CellMapper

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGES = (
    "api", "config", "core", "invoice", "mapping", "ocr", "orchestration",
    "output", "persistence", "preprocessing", "string_matching", "table_extraction",
)


def _print_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_no_print_calls_in_production_packages():
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for package in PRODUCTION_PACKAGES
        for path in sorted((ROOT / package).rglob("*.py"))
        for line in _print_calls(path)
    ]
    assert not offenders, "use logging instead of print: " + ", ".join(offenders)


def test_mapping_survives_a_stdout_that_cannot_encode_arabic(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    region = LayoutRegion(
        id="t:r0c0",
        bbox=BoundingBox(x=0, y=0, w=100, h=40),
        role=CellRole.PRODUCT_NAME,
        zone=Zone.TABLE,
        row=0,
        col=0,
        table_id="t",
    )
    fragments = [
        OCRFragment(text="بنطلون كتان", bbox=BoundingBox(x=5, y=5, w=60, h=20), source_id="t:r0c0"),
        OCRFragment(text="مکبیر", bbox=BoundingBox(x=10, y=10, w=30, h=20)),
    ]

    document = CellMapper().map(OCRResult(fragments=fragments, engine_name="t"), InvoiceLayout(regions=[region]))

    assert document.elements[0].merged_text.startswith("بنطلون")
