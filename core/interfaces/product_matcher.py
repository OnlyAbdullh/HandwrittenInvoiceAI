from __future__ import annotations

from typing import Protocol, Sequence

from core.domain.catalog import CatalogEntry, FieldMatch


class ProductMatcher(Protocol):
    """Ranks the request's product catalog for one line item's name.

    Separate from :class:`core.interfaces.string_matcher.StringMatcher`, which
    corrects any element against the shared keyword dictionary: this one
    matches *records* -- it returns ids -- and it answers a narrower question,
    "which product was written here", for which a product catalog's structure
    (sibling sizes, shortened names) matters.
    """

    def match(
        self, text: str | None, products: Sequence[CatalogEntry], top_k: int
    ) -> FieldMatch: ...
