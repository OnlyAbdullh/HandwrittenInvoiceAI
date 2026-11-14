from __future__ import annotations

import string_matching.matchers  # noqa: F401  (triggers matcher self-registration)
import string_matching.product_matchers  # noqa: F401  (same, for product matchers)
from config.schema import StringMatchingConfig
from core.interfaces.product_matcher import ProductMatcher
from core.interfaces.string_matcher import StringMatcher
from string_matching.product_vocabulary import ProductVocabulary
from string_matching.registry import matcher_registry, product_matcher_registry


def build_string_matcher(config: StringMatchingConfig) -> StringMatcher:
    return matcher_registry.create(config.algorithm, **config.algorithm_params)


def build_product_matcher(config: StringMatchingConfig) -> ProductMatcher:
    """The product matcher the configuration names, with this machine's vocabulary."""
    return product_matcher_registry.create(
        config.product_matcher,
        vocabulary=ProductVocabulary.load(config.product_vocabulary_path),
        **config.product_matcher_params,
    )
