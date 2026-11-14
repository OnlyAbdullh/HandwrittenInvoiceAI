"""Reading a product catalog from the files it is kept in.

The service never reads a catalog from disk -- the desktop app sends its live
Products table with every request. The tools do: to check a catalog before it
is loaded into the app, and to measure matching against it.

Three formats: a JSON list of names, the evaluation catalog
(`evaluation/catalog.json`, its `products` list), and an Excel workbook whose
first sheet has the product name in its first column under one header row.
The workbook is read with the standard library, so the tools need no
spreadsheet package the service itself would never use.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from core.domain.catalog import NamedEntry

_SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def load_products(path: str | Path) -> list[NamedEntry]:
    """The products in `path`, with ids: the file's own, else 1-based positions."""
    file = Path(path)
    if file.suffix.lower() == ".xlsx":
        names = [row[0] for row in _first_sheet_rows(file)[1:] if row and row[0]]
        return [NamedEntry(name=name.strip(), entry_id=i) for i, name in enumerate(names, 1)]

    data = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("products", [])

    products = []
    for position, item in enumerate(data, 1):
        if isinstance(item, str):
            products.append(NamedEntry(name=item, entry_id=position))
        else:
            products.append(
                NamedEntry(
                    name=item["name"],
                    entry_id=item.get("id", position),
                    aliases=list(item.get("aliases", [])),
                )
            )
    return products


def _first_sheet_rows(file: Path) -> list[list[str | None]]:
    with zipfile.ZipFile(file) as archive:
        strings = _shared_strings(archive)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relations = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relations}

        first = next(iter(workbook.find(f"{_SHEET}sheets")))
        target = targets[first.attrib[f"{_REL}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet = ElementTree.fromstring(archive.read(target))

    rows: list[list[str | None]] = []
    for row in sheet.iter(f"{_SHEET}row"):
        cells: dict[int, str] = {}
        for cell in row.findall(f"{_SHEET}c"):
            value = _cell_text(cell, strings)
            if value is not None:
                cells[_column(cell.attrib["r"])] = value
        if cells:
            rows.append([cells.get(i) for i in range(max(cells) + 1)])
    return rows


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.iter(f"{_SHEET}t"))
        for item in root.findall(f"{_SHEET}si")
    ]


def _cell_text(cell: ElementTree.Element, strings: list[str]) -> str | None:
    kind = cell.attrib.get("t")
    value = cell.find(f"{_SHEET}v")
    if kind == "s" and value is not None:
        return strings[int(value.text)]
    if kind == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{_SHEET}t"))
    return value.text if value is not None else None


def _column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - 64
    return number - 1
