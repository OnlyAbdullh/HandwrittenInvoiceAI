"""Product matching that separates "which entry is closest" from "is it the one".

Two findings shaped this, both measured on the real 341-product catalog
(`docs/experiment-log.md`):

* **Writers shorten names.** Dividing the matched words by the larger word
  count -- what the original matcher does -- punishes every word the writer
  left out, so a three-word shorthand for a seven-word product ranks below
  short unrelated products. Asking instead *how much of what was written does
  this entry explain* keeps the right product in the list.
* **A score cannot say whether the closest entry is the product.** 226 of the
  341 names have another name scoring 0.75 or more against them: the same
  trousers in the next size range, the same shirt from another supplier. A
  clean reading of a product the catalog lacks therefore scores *higher*
  against its sibling than a noisy reading of a product it has.

So ranking and acceptance are two steps here. Ranking is a blend of word
coverage and the original score. Acceptance is an evidence check on the top
entry -- every written word explained, every word of the entry written,
numbers exactly -- and the answer is one of four
:class:`~core.domain.catalog.MatchOutcome` values rather than a number.

The wire format has no field for an outcome, and the desktop app accepts a
product when its top score reaches 0.75. The scores reported are therefore
adjusted so that reaching the threshold means exactly "full match": see
:meth:`CoverageProductMatcher._report`.

**Where history plugs in.** Nothing here needs past invoices. A reading the
user confirmed for a product is, structurally, one more name that product
answers to: every name of an entry competes and the entry keeps its best, so
once the app sends confirmed readings as a product's aliases they match in
full with no change to this class. Signals that reorder an ambiguous group --
what this customer bought before, the price a product usually sells at --
belong between :meth:`_rank` and :meth:`_decide`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from core.domain.catalog import (
    REVIEW_SCORE_THRESHOLD,
    CatalogEntry,
    FieldMatch,
    MatchOutcome,
)
from string_matching.algorithms import DEFAULT_TOP_K, ScoredCandidate, levenshtein_distance
from string_matching.product_index import IndexedName, ProductIndex, ProductIndexCache
from string_matching.product_vocabulary import ProductVocabulary
from string_matching.registry import product_matcher_registry
from string_matching.tokenization import is_number, split_words

# The highest score reported for an answer that needs review.
_BELOW_THRESHOLD = REVIEW_SCORE_THRESHOLD - 0.01


@lru_cache(maxsize=200_000)
def word_similarity(written: str, catalog_word: str) -> float:
    """How alike a written word and a catalog word are, in [0, 1].

    This is the *ranking* similarity. Two numbers are compared like two words,
    so a size read with one digit wrong still points at the nearest sibling;
    whether a number counts as *read* is a different question, answered
    exactly (:func:`_is_evidence`). A number never resembles a word.

    Cached because the same pairs recur: a batch of invoices is one writer,
    one catalog, and the same few dozen words.
    """
    if written == catalog_word:
        return 1.0
    if is_number(written) != is_number(catalog_word):
        return 0.0

    longest = max(len(written), len(catalog_word))
    return (longest - levenshtein_distance(written, catalog_word)) / longest


@dataclass(frozen=True)
class _Scored:
    """One catalog entry's answer to a query, on its best-scoring name."""

    name: IndexedName
    score: float

    # Every written word is explained by a word of this name.
    fits: bool

    # Every word of this name was written.
    complete: bool


@product_matcher_registry.register("coverage")
class CoverageProductMatcher:
    """Ranks by how much of the written text an entry explains, then classifies.

    `coverage_weight` blends word coverage with the original score (coverage
    times the ratio of word counts when the entry is the longer). At 0 this
    ranks as the original matcher did; at 1 an entry's unwritten words cost
    nothing. The default was chosen on simulated shorthand -- re-tune it on
    labelled invoices with `tools/evaluate_matching.py`.

    `word_threshold` is the similarity a word must reach to count as explained
    or written.

    `tie_margin` is how close a second full match must score for the two to be
    reported as ambiguous instead of the first being accepted.
    """

    def __init__(
        self,
        vocabulary: ProductVocabulary | None = None,
        coverage_weight: float = 0.7,
        word_threshold: float = 0.5,
        tie_margin: float = 0.05,
    ):
        if not 0.0 <= coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be between 0 and 1.")

        self._vocabulary = vocabulary or ProductVocabulary.empty()
        self._indexes = ProductIndexCache(self._vocabulary)
        self._coverage_weight = coverage_weight
        self._word_threshold = word_threshold
        self._tie_margin = tie_margin

    # ------------------------------------------------------------------
    # The interface
    # ------------------------------------------------------------------

    def match(
        self,
        text: str | None,
        products: Sequence[CatalogEntry],
        top_k: int = DEFAULT_TOP_K,
    ) -> FieldMatch:
        # Written words are compared as written -- see ProductVocabulary on why
        # they are not rewritten to canonical spellings first.
        written = split_words(text)
        index = self._indexes.get(products)

        if not written or not len(index):
            return FieldMatch(ocr_raw_text=text or "", candidates=[])

        ranked = self._rank(written, index)
        outcome = self._decide(ranked)

        return FieldMatch(
            ocr_raw_text=text or "",
            candidates=self._report(ranked[:top_k], outcome),
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, written: list[str], index: ProductIndex) -> list[_Scored]:
        """Every active entry, best first, each on its best-scoring name."""
        joined = _joined_pairs(written)
        units = (*written, *joined.values())

        # One row per catalog word: its similarity to each written word, then
        # to each joined pair, taking the best of the word's spellings.
        similarity = {
            word: [
                max(word_similarity(unit, spelling) for spelling in spellings)
                for unit in units
            ]
            for word, spellings in index.spellings.items()
        }

        best: dict[int, _Scored] = {}
        order: list[int] = []

        for name in index.names:
            scored = self._score(name, written, list(joined), similarity)
            key = id(name.entry)
            current = best.get(key)

            if current is None:
                order.append(key)
                best[key] = scored
            elif scored.score > current.score:
                best[key] = scored

        # Stable, so ties keep the catalog's own order and repeated runs over
        # an unchanged catalog rank identically.
        return sorted((best[key] for key in order), key=lambda s: s.score, reverse=True)

    def _score(
        self,
        name: IndexedName,
        written: list[str],
        pair_starts: list[int],
        similarity: dict[str, list[float]],
    ) -> _Scored:
        word_count = len(written)
        rows = [similarity[word] for word in name.tokens]

        # How well this name explains each written word on its own, and each
        # joined pair as one word.
        single = [max(row[i] for row in rows) for i in range(word_count)]
        pair = {
            start: max(row[word_count + k] for row in rows)
            for k, start in enumerate(pair_starts)
        }

        reading = _best_reading(single, pair)

        explained = [0.0] * word_count
        used: list[int] = []
        for start, is_pair in reading:
            if is_pair:
                explained[start] = explained[start + 1] = pair[start]
                used.append(word_count + pair_starts.index(start))
            else:
                explained[start] = single[start]
                used.append(start)

        coverage = sum(explained) / word_count

        # The original matcher's score, on these words: matched words over the
        # larger word count. Equal to coverage unless the entry is the longer.
        # A joined pair is one written word, which is what the entry sees.
        overlap = coverage * len(reading) / max(len(reading), len(name.tokens))

        return _Scored(
            name=name,
            score=self._coverage_weight * coverage + (1.0 - self._coverage_weight) * overlap,
            fits=all(
                self._is_evidence(score, word) for score, word in zip(explained, written)
            ),
            complete=all(
                self._is_evidence(max(row[unit] for unit in used), word)
                for row, word in zip(rows, name.tokens)
            ),
        )

    def _is_evidence(self, similarity: float, word: str) -> bool:
        """Whether a similarity is enough to say `word` was explained or written.

        A word may be misread and still be that word. A number may not: edit
        distance calls "38" and "48" half alike, and on an invoice they are two
        different products.
        """
        if is_number(word):
            return similarity == 1.0
        return similarity >= self._word_threshold

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def _decide(self, ranked: list[_Scored]) -> MatchOutcome:
        top, rest = ranked[0], ranked[1:]

        if not top.fits:
            return MatchOutcome.UNEXPLAINED

        if top.complete:
            # Two catalog names can both be written in full by one reading when
            # they differ by a word one letter apart (جامو / جامبو).
            tied = any(
                other.fits and other.complete and top.score - other.score <= self._tie_margin
                for other in rest
            )
            return MatchOutcome.AMBIGUOUS if tied else MatchOutcome.FULL_MATCH

        # A shortened name. Whether it identifies a product depends on how
        # many products it could be short for.
        if any(other.fits for other in rest):
            return MatchOutcome.AMBIGUOUS
        return MatchOutcome.SHORTHAND_UNIQUE

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _report(self, ranked: list[_Scored], outcome: MatchOutcome) -> list[ScoredCandidate]:
        """The ranked candidates, scored so the review threshold means "full match".

        The desktop app accepts a product whose top score reaches the review
        threshold and asks the user otherwise; that rule lives there, in one
        place, and the wire carries no outcome. So:

        * a full match is reported at no less than the threshold -- the
          evidence check accepted it, and a ranking score lowered by a few
          misread letters must not overrule that;
        * anything else is reported below it. When the ranking score would
          have crossed it, the whole list is scaled down together, which
          keeps the order and the proportions between candidates.
        """
        top = ranked[0].score
        scale = 1.0
        if outcome is not MatchOutcome.FULL_MATCH and top >= REVIEW_SCORE_THRESHOLD:
            scale = _BELOW_THRESHOLD / top

        candidates = []
        for position, scored in enumerate(ranked):
            score = scored.score * scale
            if position == 0 and outcome is MatchOutcome.FULL_MATCH:
                score = max(score, REVIEW_SCORE_THRESHOLD)

            entry = scored.name.entry
            candidates.append(
                ScoredCandidate(
                    matched_value=(entry.name or "").strip() or scored.name.name,
                    similarity_score=round(score * 100, 2),
                    matched_id=entry.entry_id,
                    matched_name=scored.name.name,
                )
            )
        return candidates


def _joined_pairs(written: list[str]) -> dict[int, str]:
    """Each adjacent pair of written words joined into one, by its start position.

    The recognizers split a handwritten word in two about once in ten
    ("بارا شوت") and were never seen joining two. Numbers are never joined:
    "38 48" is two sizes.
    """
    return {
        position: written[position] + written[position + 1]
        for position in range(len(written) - 1)
        if not is_number(written[position]) and not is_number(written[position + 1])
    }


def _best_reading(single: list[float], pair: dict[int, float]) -> list[tuple[int, bool]]:
    """Read the written words as words and joined pairs, whichever explains most.

    Returns `(start position, is a joined pair)` for each unit, in order. A
    pair is used only when the name explains it better as one word than as
    two, so the two readings are alternatives rather than each word taking the
    better of its own scores. Taking the better of each let a name holding
    "براشوت" explain the word after it too: "براشوت" joined to a short word
    still resembles "براشوت".
    """
    count = len(single)

    # best[i]: the highest total over the first i written words.
    best = [0.0] * (count + 1)
    took_pair = [False] * (count + 1)

    for end in range(1, count + 1):
        best[end] = best[end - 1] + single[end - 1]
        start = end - 2
        if start in pair and best[start] + 2 * pair[start] > best[end]:
            best[end] = best[start] + 2 * pair[start]
            took_pair[end] = True

    reading: list[tuple[int, bool]] = []
    end = count
    while end > 0:
        if took_pair[end]:
            reading.append((end - 2, True))
            end -= 2
        else:
            reading.append((end - 1, False))
            end -= 1
    return reading[::-1]
