"""A product catalog, compiled once for matching.

The catalog arrives with every request -- it is the desktop app's live Products
table -- but it rarely changes between two invoices of a batch. Tokenizing 300
names for each of an invoice's ten rows would be the same work thirty times, so
the compiled form is built once per distinct catalog and reused
(:class:`ProductIndexCache`).

Compiling also decides what takes part at all: a row marked cancelled is
dropped here, so no matcher has to remember to skip it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Sequence

from core.domain.catalog import CatalogEntry
from string_matching.product_vocabulary import ProductVocabulary


@dataclass(frozen=True)
class IndexedName:
    """One name a product answers to, tokenized."""

    entry: CatalogEntry
    name: str
    tokens: tuple[str, ...]


class ProductIndex:
    """The active entries of a catalog, their tokens, and the words they use."""

    def __init__(self, products: Sequence[CatalogEntry], vocabulary: ProductVocabulary):
        names: list[IndexedName] = []

        for entry in products:
            for name in entry.names:
                if vocabulary.is_inactive(name):
                    continue
                tokens = tuple(vocabulary.tokens(name))
                if tokens:
                    names.append(IndexedName(entry=entry, name=name, tokens=tokens))

        self.names: tuple[IndexedName, ...] = tuple(names)

        # Every distinct word of the catalog. A query word is compared with
        # each of these once, however many products share it -- which is what
        # keeps the cost tied to the size of the vocabulary (about 300 words
        # for 340 products) rather than to products x words.
        self.lexicon: tuple[str, ...] = tuple(
            dict.fromkeys(token for name in names for token in name.tokens)
        )

        # Each catalog word with every way it may be spelled.
        self.spellings: dict[str, tuple[str, ...]] = {
            word: vocabulary.spellings(word) for word in self.lexicon
        }

    def __len__(self) -> int:
        return len(self.names)


def catalog_key(products: Sequence[CatalogEntry]) -> Hashable:
    """A value equal for two catalogs exactly when they compile to the same index."""
    return tuple((entry.entry_id, tuple(entry.names)) for entry in products)


class ProductIndexCache:
    """Keeps the last few compiled catalogs.

    Lock-guarded: uvicorn runs sync handlers on a thread pool, and two requests
    of one batch arrive with the same catalog at the same time.
    """

    def __init__(self, vocabulary: ProductVocabulary, capacity: int = 4):
        self._vocabulary = vocabulary
        self._capacity = capacity
        self._lock = threading.Lock()
        self._indexes: OrderedDict[Hashable, ProductIndex] = OrderedDict()

    def get(self, products: Sequence[CatalogEntry]) -> ProductIndex:
        key = catalog_key(products)

        with self._lock:
            index = self._indexes.get(key)
            if index is not None:
                self._indexes.move_to_end(key)
                return index

        # Built outside the lock: compiling is pure, so two threads building
        # the same index at once waste a little work and corrupt nothing.
        index = ProductIndex(products, self._vocabulary)

        with self._lock:
            self._indexes[key] = index
            self._indexes.move_to_end(key)
            while len(self._indexes) > self._capacity:
                self._indexes.popitem(last=False)

        return index
