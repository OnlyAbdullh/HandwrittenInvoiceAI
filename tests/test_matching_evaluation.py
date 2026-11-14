"""The tools that measure product matching: catalog files, scoring, simulation."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path

import pytest

from core.domain.catalog import NamedEntry
from evaluation.catalog_io import load_products
from evaluation.matching_benchmark import Case, render_table, score_matcher
from evaluation.reading_simulation import ErrorChannel, shorten
from string_matching.product_matchers.coverage_matcher import CoverageProductMatcher
from string_matching.product_matchers.word_overlap_matcher import WordOverlapProductMatcher
from string_matching.product_vocabulary import ProductVocabulary

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workbook(path: Path, rows: list[list[str]]) -> None:
    """A minimal .xlsx with one sheet, written the way Excel writes strings."""
    strings = list(dict.fromkeys(cell for row in rows for cell in row))
    sheet_rows = "".join(
        f'<row r="{r}">'
        + "".join(
            f'<c r="{chr(65 + c)}{r}" t="s"><v>{strings.index(cell)}</v></c>'
            for c, cell in enumerate(row)
        )
        + "</row>"
        for r, row in enumerate(rows, 1)
    )
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relations = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{namespace}" xmlns:r="{relations}">'
            '<sheets><sheet name="s" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{namespace}">'
            + "".join(f"<si><t>{text}</t></si>" for text in strings)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{namespace}"><sheetData>{sheet_rows}</sheetData></worksheet>',
        )


class TestCatalogFiles:
    def test_a_workbook_is_read_without_a_spreadsheet_package(self, tmp_path):
        path = tmp_path / "catalog.xlsx"
        _workbook(path, [["اسم المادة", "الوحدة"], ["قميص كتان", "قطعة"], ["قشاط محير", "دزينة"]])

        products = load_products(path)

        assert [p.name for p in products] == ["قميص كتان", "قشاط محير"]
        assert [p.entry_id for p in products] == [1, 2]

    def test_json_lists_and_the_evaluation_catalog_are_read(self, tmp_path):
        names = tmp_path / "names.json"
        names.write_text(json.dumps(["قميص كتان", "قشاط محير"]), encoding="utf-8")
        assert [p.name for p in load_products(names)] == ["قميص كتان", "قشاط محير"]

        products = load_products(ROOT / "evaluation" / "catalog.json")
        assert products[0].entry_id == 1 and products[0].name


class TestScoring:
    PRODUCTS = [
        NamedEntry(name="بنطال جينز بوري 38-48 دودج", entry_id=1),
        NamedEntry(name="بنطال جينز بوري 42-52 دودج", entry_id=2),
        NamedEntry(name="قميص ساتان وطني", entry_id=3),
    ]

    def test_ranking_and_acceptance_are_counted_apart(self):
        cases = [
            Case("بنطال جينز بوري 42-52 دودج", 1),   # read in full
            Case("بنطال جينز دودج", 0),              # shorthand, two products fit
            Case("جاكيت جلد", None),                 # not in the catalog
        ]
        score = score_matcher(CoverageProductMatcher(), self.PRODUCTS, cases, top_k=5)

        assert (score.in_catalog, score.not_in_catalog) == (2, 1)
        assert score.top1 >= 1 and score.in_list == 2
        assert score.accepted == 1 and score.wrong_accepts == 0
        assert score.not_in_catalog_accepted == 0
        assert score.outcomes["full_match"] == 1

    def test_removing_the_intended_product_shows_look_alike_acceptance(self):
        """The measurement the old matcher fails: with 42-52 gone from the
        catalog, it accepts 38-48 for a reading that says 42-52."""
        cases = [Case("بنطال جينز بوري 42-52 دودج", 1)]

        old = score_matcher(WordOverlapProductMatcher(), self.PRODUCTS, cases)
        new = score_matcher(CoverageProductMatcher(), self.PRODUCTS, cases)

        assert (old.removed_cases, old.removed_accepted) == (1, 1)
        assert (new.removed_cases, new.removed_accepted) == (1, 0)

    def test_the_table_names_every_matcher(self):
        cases = [Case("قميص ساتان وطني", 2)]
        scores = {
            "word_overlap": score_matcher(WordOverlapProductMatcher(), self.PRODUCTS, cases),
            "coverage": score_matcher(CoverageProductMatcher(), self.PRODUCTS, cases),
        }
        table = render_table("t", scores, top_k=5)

        assert "| word_overlap | 1/1 (100.0%)" in table
        assert "Outcomes of `coverage`: full_match 1." in table


class TestSimulation:
    PAIRS = [("باب", "بات")] * 30 + [("باب", "باب")] * 30 + [("دار", "دا ر")] * 10

    def test_the_channel_reproduces_what_was_measured(self):
        channel = ErrorChannel(self.PAIRS)

        # "ب" was read wrong 30 times in 120, always as "ت".
        assert channel.error_rate("ب") == pytest.approx(0.25)
        assert set(channel.misread["ب"]) == {"ت"}
        assert channel.split_rate > 0

        rng = random.Random(1)
        readings = {channel.corrupt(rng, "باب") for _ in range(200)}
        assert "باب" in readings and len(readings) > 1
        assert all(set(r) <= set("بات ر") for r in readings)

    def test_no_noise_returns_the_text(self):
        assert ErrorChannel(self.PAIRS).corrupt(random.Random(1), "باب", scale=0) == "باب"

    def test_a_reading_of_something_else_teaches_nothing(self):
        channel = ErrorChannel([("باب", "سيارة كبيرة جدا")])
        assert channel.pairs_used == 0

    def test_shortening_keeps_order_and_skips_names_too_short_to_shorten(self):
        rng = random.Random(3)
        words = ["بنطال", "جينز", "بوري", "38", "48", "دودج"]

        assert shorten(rng, words, "full") == words
        assert shorten(rng, words, "no_numbers") == ["بنطال", "جينز", "بوري", "دودج"]
        half = shorten(rng, words, "half")
        assert len(half) == 3 and half == [w for w in words if w in half]
        assert shorten(rng, ["قشاط", "محير"], "two_words") is None
        assert shorten(rng, ["قشاط", "محير"], "no_numbers") is None
        with pytest.raises(ValueError):
            shorten(rng, words, "backwards")


class TestTools:
    def test_labels_resolve_to_catalog_positions(self):
        tool = _load("evaluate_matching")
        products = [NamedEntry(name="قميص كتان", entry_id=1)]
        pairs = [("قميص كنان", "قميص كتان"), ("جاكيت", "جاكيت جلد"), ("؟", None)]

        cases, catalog, added = tool.cases_from_readings(pairs, products, add_missing=False)
        assert [c.target for c in cases] == [0, None, None]
        assert added == [] and len(catalog) == 1

        cases, catalog, added = tool.cases_from_readings(pairs, products, add_missing=True)
        assert [c.target for c in cases] == [0, 1, None]
        assert added == ["جاكيت جلد"] and len(catalog) == 2

    def test_both_reading_formats_are_understood(self, tmp_path):
        tool = _load("evaluate_matching")
        (tmp_path / "mine.jsonl").write_text(
            json.dumps({"reading": "قميص كنان", "product": "قميص كتان"}, ensure_ascii=False)
            + "\n"
            + json.dumps(
                {"role": "product_name", "text": "قشاط محير", "pred_text": "قشاط مجير"},
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps({"role": "quantity", "text": "16", "pred_text": "١٦"}) + "\n",
            encoding="utf-8",
        )
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "dup.jsonl").write_text(
            json.dumps({"reading": "x", "product": "y"}) + "\n", encoding="utf-8"
        )

        assert tool.read_pairs(tmp_path, {"product_name"}) == [
            ("قميص كنان", "قميص كتان"),
            ("قشاط مجير", "قشاط محير"),
        ]

    def test_simulated_readings_cover_active_products_only(self):
        tool = _load("evaluate_matching")
        vocabulary = ProductVocabulary.from_groups(inactive_markers=["الغاء"])
        products = [
            NamedEntry(name="بنطال جينز بوري 38-48 دودج", entry_id=1),
            NamedEntry(name="الغاء 2", entry_id=2),
        ]
        cases = tool.simulated_cases(
            products, vocabulary, ErrorChannel([]), "no_numbers", 0.0, per_product=2, seed=1
        )

        assert [(c.reading, c.target) for c in cases] == [("بنطال جينز بوري دودج", 0)] * 2

    def test_the_catalog_check_finds_what_a_person_should_look_at(self):
        tool = _load("check_catalog")
        vocabulary = ProductVocabulary.from_groups(
            word_variants=[["لوجو", "لوغو"]], inactive_markers=["الغاء"]
        )
        names = [
            "الغاء 2",
            "قشاط 4 سم صيني",
            "قشاط صيني 4سم",
            "بنطال جينز 38-48 لوجو",
            "بنطال جينز 42-52 لوغو",
            "بنطال جامو",
            "بنطال جامبو",
            "قشاط محير",
            "قشاط مطاط محير",
        ]
        found = tool.catalog_findings(names, vocabulary)

        assert found["inactive"] == ["الغاء 2"] and found["active"] == 8
        assert found["duplicates"] == [["قشاط 4 سم صيني", "قشاط صيني 4سم"]]
        assert "قشاط صيني 4سم" in found["glued"]
        # Declared variants are one word; the undeclared look-alike is reported.
        assert [(a, b) for a, _, b, _ in found["near_words"]] == [("جامو", "جامبو")]
        assert ["بنطال جينز 38-48 لوجو", "بنطال جينز 42-52 لوغو"] in found["number_siblings"]
        assert ("قشاط محير", "قشاط مطاط محير") in found["contained"]

        report = tool.render(found, "test")
        assert "## Inactive rows (1)" in report and "جامو (1) / جامبو (1)" in report
