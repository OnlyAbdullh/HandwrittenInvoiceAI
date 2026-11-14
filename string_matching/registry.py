from __future__ import annotations

from core.interfaces.product_matcher import ProductMatcher
from core.interfaces.string_matcher import StringMatcher
from core.registry import Registry

matcher_registry: Registry[StringMatcher] = Registry(kind="string matcher")

# Product-catalog matchers are their own family: they match records and
# classify the answer, where a string matcher corrects text against keywords.
product_matcher_registry: Registry[ProductMatcher] = Registry(kind="product matcher")
