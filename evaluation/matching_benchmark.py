"""Scoring a product matcher on readings whose intended product is known.

A matcher is judged on two different things, and one number hides the
difference:

* **Ranking** -- is the intended product first, or at least in the list the
  user picks from.
* **Acceptance** -- when the top score reaches the review threshold the
  desktop app takes the product without asking. A wrong product accepted that
  way is the failure that matters: nobody looks at it again.

Acceptance is also measured with the intended product *removed* from the
catalog. A product the catalog does not have yet is an ordinary event, and a
matcher that then accepts its nearest sibling has corrupted the invoice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from core.domain.catalog import REVIEW_SCORE_THRESHOLD, CatalogEntry
from core.interfaces.product_matcher import ProductMatcher


@dataclass(frozen=True)
class Case:
    """One reading and the catalog entry it was meant to be.

    `target` is a position in the product list, or None for a reading whose
    product is not in the catalog: nothing should be accepted for it.
    """

    reading: str
    target: int | None


@dataclass
class MatchingScore:
    cases: int = 0
    in_catalog: int = 0
    top1: int = 0
    in_list: int = 0
    accepted: int = 0
    wrong_accepts: int = 0

    # Readings whose product is not in the catalog, and how many of them had
    # something accepted anyway.
    not_in_catalog: int = 0
    not_in_catalog_accepted: int = 0

    # With the intended entry removed: how often a look-alike is accepted.
    removed_cases: int = 0
    removed_accepted: int = 0

    outcomes: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "in_catalog": self.in_catalog,
            "top1": self.top1,
            "in_list": self.in_list,
            "accepted": self.accepted,
            "wrong_accepts": self.wrong_accepts,
            "not_in_catalog": self.not_in_catalog,
            "not_in_catalog_accepted": self.not_in_catalog_accepted,
            "removed_cases": self.removed_cases,
            "removed_accepted": self.removed_accepted,
            "outcomes": dict(self.outcomes),
        }


def _accepted(match) -> bool:
    best = match.best
    return best is not None and best.score_fraction >= REVIEW_SCORE_THRESHOLD


def score_matcher(
    matcher: ProductMatcher,
    products: Sequence[CatalogEntry],
    cases: Sequence[Case],
    top_k: int = 5,
    with_removal: bool = True,
) -> MatchingScore:
    """Run every case through `matcher` and count what happened."""
    score = MatchingScore(cases=len(cases))
    products = list(products)

    for case in cases:
        match = matcher.match(case.reading, products, top_k)
        accepted = _accepted(match)
        score.outcomes[match.outcome.value if match.outcome else "unclassified"] += 1

        if case.target is None:
            score.not_in_catalog += 1
            score.not_in_catalog_accepted += accepted
            continue

        target = products[case.target]
        ranked = [candidate.matched_id for candidate in match.candidates]

        score.in_catalog += 1
        score.top1 += bool(ranked) and ranked[0] == target.entry_id
        score.in_list += target.entry_id in ranked
        score.accepted += accepted
        score.wrong_accepts += accepted and ranked[0] != target.entry_id

    if with_removal:
        # Grouped by target, so a matcher that caches its compiled catalog
        # compiles each reduced catalog once.
        for target in sorted({case.target for case in cases if case.target is not None}):
            reduced = products[:target] + products[target + 1:]
            for case in cases:
                if case.target == target:
                    score.removed_cases += 1
                    score.removed_accepted += _accepted(matcher.match(case.reading, reduced, top_k))

    return score


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.1%})" if whole else "n/a"


def render_table(title: str, scores: dict[str, MatchingScore], top_k: int) -> str:
    """One markdown table comparing matchers on the same cases."""
    lines = [
        f"### {title}",
        "",
        f"| Matcher | Top-1 | In top {top_k} | Accepted (≥ {REVIEW_SCORE_THRESHOLD}) | Wrong accepts "
        "| Accepted with intended product removed | Accepted, product not in catalog |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in scores.items():
        lines.append(
            f"| {name} | {_share(s.top1, s.in_catalog)} | {_share(s.in_list, s.in_catalog)} "
            f"| {_share(s.accepted, s.in_catalog)} | {_share(s.wrong_accepts, s.in_catalog)} "
            f"| {_share(s.removed_accepted, s.removed_cases)} "
            f"| {_share(s.not_in_catalog_accepted, s.not_in_catalog)} |"
        )

    classified = {n: s for n, s in scores.items() if set(s.outcomes) - {"unclassified"}}
    for name, s in classified.items():
        mix = ", ".join(f"{outcome} {count}" for outcome, count in s.outcomes.most_common())
        lines += ["", f"Outcomes of `{name}`: {mix}."]
    return "\n".join(lines)
