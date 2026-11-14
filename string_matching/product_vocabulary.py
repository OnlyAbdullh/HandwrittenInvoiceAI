"""What an installation knows about how its product names are worded.

Two facts, both about words rather than about any one product:

* **Word variants.** The catalog spells the same word several ways (``لوغو`` /
  ``لوجو``, ``داود`` / ``داوود``) and writers use synonyms the catalog does not
  (``بنطلون`` for ``بنطال``). Declaring them once per word is cheaper and safer
  than adding alternative product names: one line covers every product
  containing the word, where duplicate names would have to be written out
  product by product.

  A catalog word is stored under its canonical spelling, and a written word is
  compared with **every** spelling of it (:meth:`ProductVocabulary.spellings`).
  Rewriting the written word to the canonical one instead does not work on a
  misreading: ``نطلون`` is one letter from ``بنطلون`` and three from ``بنطال``,
  so a reading that was nearly right would be scored as nearly wrong.
* **Inactive markers.** A catalog row containing ``الغاء`` is a cancelled
  product. It must never be offered as a match.

The data lives in a JSON file the machine owns
(``string_matching.product_vocabulary_path``), not in a request: it describes
the language of the trade, which does not change between two invoices.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from string_matching.tokenization import split_words

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductVocabulary:
    """Word variants and inactive markers, already normalized."""

    # Normalized variant spelling -> normalized canonical spelling.
    variants: Mapping[str, str] = field(default_factory=dict)
    inactive_markers: frozenset[str] = frozenset()

    # Canonical spelling -> every spelling of the word, canonical first.
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "ProductVocabulary":
        return cls()

    @classmethod
    def from_groups(
        cls,
        word_variants: Iterable[Sequence[str]] = (),
        inactive_markers: Iterable[str] = (),
    ) -> "ProductVocabulary":
        """Build from groups of equivalent spellings, canonical spelling first."""
        variants: dict[str, str] = {}
        groups: dict[str, tuple[str, ...]] = {}

        for group in word_variants:
            words = [word for spelling in group for word in split_words(spelling)[:1]]
            words = list(dict.fromkeys(words))
            if len(words) < 2:
                continue
            canonical = words[0]
            groups[canonical] = tuple(words)
            for word in words[1:]:
                variants[word] = canonical

        markers = frozenset(
            word for marker in inactive_markers for word in split_words(marker)
        )
        return cls(variants=variants, inactive_markers=markers, groups=groups)

    @classmethod
    def load(cls, path: str | Path) -> "ProductVocabulary":
        """Read the vocabulary file; a missing one is an empty vocabulary.

        Matching works without it, only less well, so a missing file must not
        stop the service -- the same rule the keyword dictionary follows.
        """
        file = Path(path)
        if not file.exists():
            logger.warning("Product vocabulary '%s' not found; no variants folded.", path)
            return cls.empty()

        data = json.loads(file.read_text(encoding="utf-8"))
        return cls.from_groups(
            word_variants=data.get("word_variants", []),
            inactive_markers=data.get("inactive_markers", []),
        )

    def canonical(self, word: str) -> str:
        """The canonical spelling of an already-normalized word."""
        return self.variants.get(word, word)

    def spellings(self, canonical: str) -> tuple[str, ...]:
        """Every spelling of a canonical word, itself first."""
        return self.groups.get(canonical, (canonical,))

    def tokens(self, text: str | None) -> list[str]:
        """A *catalog* name as canonical words. Written text is not folded."""
        return [self.canonical(word) for word in split_words(text)]

    def is_inactive(self, name: str | None) -> bool:
        """Whether a catalog name is marked as cancelled."""
        if not self.inactive_markers:
            return False
        return any(word in self.inactive_markers for word in split_words(name))
