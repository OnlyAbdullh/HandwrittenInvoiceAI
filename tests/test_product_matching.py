"""Product matching: tokenizing, the vocabulary, ranking and the four outcomes."""

from __future__ import annotations

import pytest

from config.loader import load_config
from config.settings_contract import enforce_server_owned
from core.domain.catalog import (
    REVIEW_SCORE_THRESHOLD,
    Catalogs,
    FieldMatch,
    MatchOutcome,
    NamedEntry,
)
from core.domain.document import StructuredDocument, TableCellElement
from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment, OCRResult
from core.exceptions import ConfigurationError
from invoice.parser import InvoiceParser
from string_matching.catalog import match_product
from string_matching.factory import build_product_matcher
from string_matching.product_index import ProductIndexCache
from string_matching.product_matchers.coverage_matcher import (
    CoverageProductMatcher,
    word_similarity,
)
from string_matching.product_matchers.word_overlap_matcher import WordOverlapProductMatcher
from string_matching.product_vocabulary import ProductVocabulary
from string_matching.tokenization import split_words

THRESHOLD = REVIEW_SCORE_THRESHOLD * 100


def catalog(*names: str) -> list[NamedEntry]:
    return [NamedEntry(name=name, entry_id=i) for i, name in enumerate(names, 1)]


def accepted(match: FieldMatch) -> bool:
    return match.best is not None and match.best.similarity_score >= THRESHOLD


VOCABULARY = ProductVocabulary.from_groups(
    word_variants=[["بنطال", "بنطلون"], ["براشوت", "باراشوت"]],
    inactive_markers=["الغاء"],
)


class TestTokenization:
    def test_a_size_glued_to_a_word_is_set_apart(self):
        assert split_words("قشاط 4سم") == ["قشاط", "4", "سم"]
        assert split_words("بنطال جينز 38/48ماركة G-5") == [
            "بنطال", "جينز", "38", "48", "ماركه", "g", "5",
        ]

    def test_arabic_indic_digits_are_the_same_numbers(self):
        assert split_words("بنطال ٣٨-٤٨") == ["بنطال", "38", "48"]

    def test_nothing_to_split(self):
        assert split_words(None) == []
        assert split_words("  ") == []


class TestVocabulary:
    def test_a_catalog_word_is_stored_under_its_canonical_spelling(self):
        assert VOCABULARY.tokens("بنطلون كتان") == ["بنطال", "كتان"]

    def test_every_spelling_of_a_word_is_kept(self):
        assert VOCABULARY.spellings("بنطال") == ("بنطال", "بنطلون")
        assert VOCABULARY.spellings("كتان") == ("كتان",)

    def test_inactive_marker_is_found_however_it_is_written(self):
        assert VOCABULARY.is_inactive("الغاء 23")
        assert VOCABULARY.is_inactive("الغاء11")
        assert VOCABULARY.is_inactive("طاقية ماركات إلغاء")
        assert not VOCABULARY.is_inactive("طاقية كتان")

    def test_a_missing_file_is_an_empty_vocabulary(self, tmp_path):
        vocabulary = ProductVocabulary.load(tmp_path / "absent.json")
        assert vocabulary.tokens("بنطلون") == ["بنطلون"]
        assert not vocabulary.is_inactive("الغاء 2")

    def test_the_shipped_file_loads(self):
        vocabulary = ProductVocabulary.load("keywords/product_vocabulary.json")
        assert vocabulary.canonical("بنطلون") == "بنطال"
        assert vocabulary.is_inactive("الغاء 2")


class TestWordSimilarity:
    def test_a_number_resembles_numbers_and_never_words(self):
        # Enough to rank the nearest size first; acceptance needs it exact.
        assert word_similarity("38", "48") == pytest.approx(0.5)
        assert word_similarity("38", "38") == 1.0
        assert word_similarity("38", "سم") == 0.0

    def test_a_misread_letter_costs_a_fraction_of_the_word(self):
        assert word_similarity("كنان", "كتان") == pytest.approx(0.75)


class TestFullMatch:
    PRODUCTS = catalog(
        "بنطال جينز بوري 38-48 دودج",
        "بنطال جينز بوري 42-52 دودج",
        "قميص ساتان وطني",
    )

    def test_the_whole_name_in_any_order_is_accepted(self):
        match = CoverageProductMatcher().match("بوري جينز بنطال دودج 42-52", self.PRODUCTS)

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 2
        assert accepted(match)
        assert not match.requires_manual_review

    def test_misread_letters_do_not_stop_a_full_match(self):
        match = CoverageProductMatcher().match("بنطال جبنز بوري 42-52 دودح", self.PRODUCTS)

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 2
        # Accepted on the evidence, so reported at the threshold even though
        # two misread words pulled the ranking score under it.
        assert accepted(match)

    def test_a_size_the_catalog_lacks_is_not_accepted_as_its_sibling(self):
        """The reading names 46-56, which is not sold. The old matcher scored a
        sibling 0.8 and the app took it."""
        reading = "بنطال جينز بوري 46-56 دودج"

        old = WordOverlapProductMatcher().match(reading, self.PRODUCTS)
        assert accepted(old)

        new = CoverageProductMatcher().match(reading, self.PRODUCTS)
        assert new.outcome is MatchOutcome.UNEXPLAINED
        assert not accepted(new)
        assert new.requires_manual_review
        # The siblings are still offered, in order, for the user to choose from.
        assert {c.matched_id for c in new.candidates[:2]} == {1, 2}

    def test_a_misread_size_ranks_the_nearest_size_first_and_asks(self):
        """36-48 is not sold; 38-48 is one digit away and 42-52 is not close."""
        match = CoverageProductMatcher().match("بنطال جينز بوري 36-48 دودج", self.PRODUCTS)

        assert match.best.matched_id == 1
        assert match.outcome is MatchOutcome.UNEXPLAINED
        assert not accepted(match)

    def test_two_names_one_letter_apart_are_ambiguous(self):
        products = catalog("بنطال كتان جامو", "بنطال كتان جامبو")
        match = CoverageProductMatcher().match("بنطال كتان جاميو", products)

        assert match.outcome is MatchOutcome.AMBIGUOUS
        assert not accepted(match)


class TestShorthand:
    PRODUCTS = catalog(
        "بنطال جينز بوري 38-48 دودج",
        "بنطال جينز بوري 42-52 دودج",
        "بنطال كتان لاكوست رسمي 44-54 رومير",
        "قشاط جلد طبيعي كراشي",
    )

    def test_a_shortened_name_only_one_product_fits_is_ranked_first_for_review(self):
        match = CoverageProductMatcher().match("بنطال رومير", self.PRODUCTS)

        assert match.outcome is MatchOutcome.SHORTHAND_UNIQUE
        assert match.best.matched_id == 3
        assert not accepted(match)

    def test_a_shortened_name_several_products_fit_is_ambiguous(self):
        match = CoverageProductMatcher().match("بنطال جينز دودج", self.PRODUCTS)

        assert match.outcome is MatchOutcome.AMBIGUOUS
        assert {c.matched_id for c in match.candidates[:2]} == {1, 2}
        assert not accepted(match)

    def test_shorthand_ranks_the_long_product_above_short_unrelated_ones(self):
        """Dividing by the larger word count put any short name above the
        seven-word product a two-word shorthand was written for."""
        match = CoverageProductMatcher().match("لاكوست رومير", self.PRODUCTS)
        assert match.best.matched_id == 3

    def test_a_word_no_product_has_is_unexplained(self):
        match = CoverageProductMatcher().match("بنطال حرير", self.PRODUCTS)

        assert match.outcome is MatchOutcome.UNEXPLAINED
        assert not accepted(match)
        assert match.candidates


class TestSplitWords:
    def test_a_word_read_in_two_pieces_still_matches(self):
        products = catalog("باراشوت ملون", "طاقية كتان")
        match = CoverageProductMatcher().match("بارا شوت ملون", products)

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 1

    def test_joining_does_not_explain_a_leftover_word(self):
        """"براشوت" + "طون" still resembles "براشوت", which used to make the
        hat explain both written words."""
        products = catalog("طاقية براشوت", "باراشوت ملون")
        match = CoverageProductMatcher(VOCABULARY).match("باراشوت طون", products)

        assert match.best.matched_id == 2

    def test_numbers_are_never_joined(self):
        products = catalog("بنطال 3848", "بنطال 38-48")
        match = CoverageProductMatcher().match("بنطال 38 48", products)
        assert match.best.matched_id == 2


class TestVariants:
    def test_a_declared_variant_matches_in_full(self):
        products = catalog("بنطال كتان كبير", "قميص كتان")
        match = CoverageProductMatcher(VOCABULARY).match("بنطلون كتان كبير", products)

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 1

    def test_a_misread_variant_is_compared_with_the_spelling_it_resembles(self):
        """"نطلون" is one letter from "بنطلون" and three from "بنطال". Rewriting
        the catalog's word to its canonical spelling lost this match."""
        products = catalog("بنطلون كتان كبير", "جاكيت شامواه كبير")
        match = CoverageProductMatcher(VOCABULARY).match("نطلون كقاره كبير", products)

        assert match.best.matched_id == 1


class TestCatalogHandling:
    def test_inactive_products_are_never_offered(self):
        products = catalog("الغاء 23", "طاقية ماركات إلغاء", "طاقية كتان")
        match = CoverageProductMatcher(VOCABULARY).match("طاقية ماركات", products)

        assert [c.matched_id for c in match.candidates] == [3]

    def test_a_remembered_reading_sent_as_an_alias_matches_in_full(self):
        """How confirmed invoices will be used: a reading the user confirmed
        for a product is one more name it answers to."""
        products = [
            NamedEntry(name="بنطال كتان جيب كبير", entry_id=7, aliases=["بنطلون كتان كبير"]),
            NamedEntry(name="بنطال كتان جيب شبابي", entry_id=8),
        ]
        match = CoverageProductMatcher().match("بنطلون كتان كبير", products)

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 7
        assert match.best.matched_value == "بنطال كتان جيب كبير"
        assert match.best.matched_name == "بنطلون كتان كبير"
        assert len(match.candidates) == 2

    def test_nothing_read_or_nothing_to_match_gives_no_candidates(self):
        matcher = CoverageProductMatcher()

        assert matcher.match("", catalog("قميص كتان")).candidates == []
        assert matcher.match(None, catalog("قميص كتان")).candidates == []
        assert matcher.match("قميص", []).candidates == []
        assert matcher.match("قميص", []).requires_manual_review

    def test_candidates_are_ranked_and_limited(self):
        products = catalog("قميص كتان", "قميص ساتان", "قميص جوخ", "قميص كارو")
        match = CoverageProductMatcher().match("قميص", products, top_k=3)

        scores = [c.similarity_score for c in match.candidates]
        assert len(scores) == 3
        assert scores == sorted(scores, reverse=True)

    def test_only_a_full_match_reaches_the_review_threshold(self):
        products = catalog("قميص كتان", "قميص كتان شبابي", "قشاط محير")
        matcher = CoverageProductMatcher()

        for reading in ("قميص كتان", "قميص", "قشاط", "قميص حرير", "كتان شبابي"):
            match = matcher.match(reading, products)
            assert accepted(match) == (match.outcome is MatchOutcome.FULL_MATCH), reading

    def test_one_catalog_is_compiled_once(self):
        cache = ProductIndexCache(ProductVocabulary.empty())
        products = catalog("قميص كتان", "قشاط محير")

        assert cache.get(products) is cache.get(list(products))
        assert cache.get(products) is not cache.get(products[:1])


def _item_row(product_text: str) -> tuple[StructuredDocument, OCRResult]:
    """A two-row grid: printed column captions and one line item."""
    cells = [
        ("الصنف", 0, 0, 300, 200), ("الكمية", 0, 1, 200, 200),
        ("السعر", 0, 2, 100, 200), ("المبلغ", 0, 3, 20, 200),
        (product_text, 1, 0, 300, 240), ("2", 1, 1, 200, 240),
        ("1.25", 1, 2, 100, 240), ("2.50", 1, 3, 20, 240),
    ]
    elements = []
    for text, row, col, x, y in cells:
        bbox = BoundingBox(x=x, y=y, w=80, h=30)
        elements.append(
            TableCellElement(
                id=f"t1:r{row}c{col}", table_id="t1", row=row, col=col, bbox=bbox,
                fragments=[OCRFragment(text=text, bbox=bbox, confidence=0.9)],
                merged_text=text,
            )
        )
    document = StructuredDocument(elements=elements)
    fragments = [f for el in elements for f in el.fragments]
    return document, OCRResult(fragments=fragments, engine_name="test")


class TestParser:
    PRODUCTS = catalog("بنطال جينز بوري 38-48 دودج", "بنطال جينز بوري 42-52 دودج")

    def test_the_parser_uses_the_matcher_it_is_given(self):
        calls = []

        class Recording:
            def match(self, text, products, top_k):
                calls.append((text, len(products), top_k))
                return FieldMatch(ocr_raw_text=text, candidates=[])

        document, result = _item_row("بنطال جينز دودج")
        catalogs = Catalogs(products=self.PRODUCTS, top_k=7)
        InvoiceParser(product_matcher=Recording()).parse(document, result, catalogs)

        assert calls == [("بنطال جينز دودج", 2, 7)]

    def test_an_ambiguous_shorthand_is_offered_for_review_with_its_outcome_recorded(self):
        document, result = _item_row("بنطال جينز دودج")
        draft = InvoiceParser().parse(document, result, Catalogs(products=self.PRODUCTS))

        name = draft.line_items[0].product_name
        assert name.value == "بنطال جينز دودج"
        assert name.requires_manual_review and name.matched_id is None
        assert name.match_outcome == "ambiguous"
        assert name.to_wire()["match_outcome"] == "ambiguous"
        assert len(name.candidates) == 2

    def test_a_full_match_is_bound_to_its_product(self):
        document, result = _item_row("بنطال جينز بوري 42-52 دودج")
        draft = InvoiceParser().parse(document, result, Catalogs(products=self.PRODUCTS))

        name = draft.line_items[0].product_name
        assert name.matched_id == 2 and not name.requires_manual_review
        assert name.match_outcome == "full_match"


class TestWiring:
    def test_match_product_classifies_its_answer(self):
        match = match_product("جاكيت ازرق صوف", catalog("جاكيت صوف أزرق", "حذاء جلد أسود"))

        assert match.outcome is MatchOutcome.FULL_MATCH
        assert match.best.matched_id == 1

    def test_the_configured_matcher_is_built_with_the_vocabulary(self):
        config = load_config("config/default_config.yaml")
        matcher = build_product_matcher(config.string_matching)

        assert isinstance(matcher, CoverageProductMatcher)
        products = catalog("بنطال كتان كبير", "الغاء 2")
        assert [c.matched_id for c in matcher.match("بنطلون كتان كبير", products).candidates] == [1]

    def test_the_original_matcher_is_one_setting_away(self):
        config = load_config("config/default_config.yaml")
        config.string_matching.product_matcher = "word_overlap"

        matcher = build_product_matcher(config.string_matching)
        assert isinstance(matcher, WordOverlapProductMatcher)
        assert matcher.match("قميص كتان", catalog("قميص كتان")).outcome is None

    def test_an_unknown_parameter_is_refused(self):
        config = load_config("config/default_config.yaml")
        config.string_matching.product_matcher_params = {"coverage_wieght": 0.5}

        with pytest.raises(ConfigurationError):
            build_product_matcher(config.string_matching)

    def test_a_request_cannot_choose_the_matcher_or_the_vocabulary(self):
        server = load_config("config/default_config.yaml")
        request = server.model_copy(deep=True)
        request.string_matching.product_matcher = "word_overlap"
        request.string_matching.product_matcher_params = {"anything": 1}
        request.string_matching.product_vocabulary_path = "C:/elsewhere.json"

        enforce_server_owned(request, server)

        assert request.string_matching.product_matcher == "coverage"
        assert request.string_matching.product_matcher_params == {}
        assert (
            request.string_matching.product_vocabulary_path
            == server.string_matching.product_vocabulary_path
        )
