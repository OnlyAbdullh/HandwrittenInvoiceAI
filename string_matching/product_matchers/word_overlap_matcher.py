from __future__ import annotations

from typing import Sequence

from core.domain.catalog import CatalogEntry, FieldMatch
from string_matching.algorithms import DEFAULT_TOP_K, product_match_similarity
from string_matching.product_vocabulary import ProductVocabulary
from string_matching.registry import product_matcher_registry


@product_matcher_registry.register("word_overlap")
class WordOverlapProductMatcher:
    """The original product matcher: uniform word overlap over the larger count.

    Kept as the measured baseline and as a one-line rollback
    (`string_matching.product_matcher: word_overlap`). It ranks well when the
    whole catalog name is written and poorly when it is shortened, and its
    score alone decides acceptance, so a sibling product scoring above the
    review threshold is accepted. `docs/experiment-log.md` has the comparison.

    The vocabulary is accepted and ignored, so both matchers are built the
    same way and this one stays exactly what it was.
    """

    def __init__(self, vocabulary: ProductVocabulary | None = None):
        del vocabulary

    def match(
        self,
        text: str | None,
        products: Sequence[CatalogEntry],
        top_k: int = DEFAULT_TOP_K,
    ) -> FieldMatch:
        # Imported here: string_matching.catalog imports the matchers package
        # to supply match_product's default.
        from string_matching.catalog import top_entry_matches

        return FieldMatch(
            ocr_raw_text=text or "",
            candidates=top_entry_matches(text, products, product_match_similarity, top_k),
        )
