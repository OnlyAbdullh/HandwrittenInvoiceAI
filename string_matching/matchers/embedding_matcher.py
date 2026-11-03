from __future__ import annotations

from core.domain.matching import KeywordDictionary, MatchResult
from core.exceptions import NotImplementedStrategyError
from string_matching.registry import matcher_registry


@matcher_registry.register("embedding")
class EmbeddingMatcher:
    """Registered but deliberately unbuilt: semantic nearest-neighbour matching.

    The API refuses it at validation time (`UNIMPLEMENTED_MATCHERS` in
    config/settings_contract.py). Raising `NotImplementedStrategyError` -- not a
    plain `NotImplementedError` -- is what lets the orchestrator degrade the
    keyword stage instead of failing the whole run if a YAML file selects it.
    """

    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult:
        raise NotImplementedStrategyError(
            "string_matching", "embedding", "Use 'levenshtein' or 'order_independent'."
        )
