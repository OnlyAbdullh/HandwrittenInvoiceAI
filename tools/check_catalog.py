"""Report what in a product catalog will make matching harder.

    python tools/check_catalog.py keywords.xlsx
    python tools/check_catalog.py keywords.xlsx --out catalog_report.md

Reads the catalog the way the product matcher does -- same tokenizer, same
vocabulary file -- and lists what a person should look at before the catalog
goes into the desktop app:

* rows marked inactive, which never take part in matching;
* names that are the same once normalized;
* words one letter apart that are not declared variants of each other. Each is
  either one word spelled two ways (add it to `word_variants` in
  `keywords/product_vocabulary.json`) or two different words (leave it);
* products that no reading of the name can tell apart from a sibling without
  one particular word: names differing only in their numbers, and names whose
  words are all contained in a longer name. These are not errors. They are
  where a misread size or a shortened name leads to the wrong product, and
  where the matcher will ask instead of deciding.

Nothing is changed. The catalog belongs to the desktop app's database.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.catalog_io import load_products  # noqa: E402
from string_matching.algorithms import levenshtein_distance  # noqa: E402
from string_matching.normalization import normalize_text, normalize_words  # noqa: E402
from string_matching.product_vocabulary import ProductVocabulary  # noqa: E402
from string_matching.tokenization import is_number  # noqa: E402

# Shorter words are one letter apart by accident too often to be worth listing.
MIN_VARIANT_LENGTH = 4


def _has_glued_number(name: str) -> bool:
    """A number written against an Arabic word ("4سم"), not a code like "X8"."""
    return any(
        any(ch.isdigit() for ch in word) and any("؀" <= ch <= "ۿ" for ch in word)
        for word in normalize_words(name)
    )


def catalog_findings(names: list[str], vocabulary: ProductVocabulary) -> dict:
    """Everything the report prints, as data."""
    inactive = [name for name in names if vocabulary.is_inactive(name)]
    active = [name for name in names if not vocabulary.is_inactive(name)]
    tokens = {name: vocabulary.tokens(name) for name in active}

    same_words: dict[tuple, list[str]] = defaultdict(list)
    same_but_numbers: dict[tuple, list[str]] = defaultdict(list)
    for name, words in tokens.items():
        same_words[tuple(sorted(words))].append(name)
        same_but_numbers[tuple(sorted(w for w in words if not is_number(w)))].append(name)

    word_sets = {name: set(words) for name, words in tokens.items()}
    contained = sorted(
        (short, long)
        for short, long in combinations(active, 2)
        for short, long in ((short, long), (long, short))
        if word_sets[short] < word_sets[long]
    )

    usage = Counter(word for words in tokens.values() for word in set(words))
    words = [
        word
        for word in usage
        if len(word) >= MIN_VARIANT_LENGTH and not any(ch.isdigit() for ch in word)
    ]
    near = sorted(
        (
            (a, usage[a], b, usage[b])
            for a, b in combinations(words, 2)
            if abs(len(a) - len(b)) <= 1 and levenshtein_distance(a, b) == 1
        ),
        key=lambda pair: -(pair[1] + pair[3]),
    )

    return {
        "total": len(names),
        "active": len(active),
        "inactive": inactive,
        "distinct_words": len(usage),
        "glued": [name for name in active if _has_glued_number(name)],
        "leading_number": [name for name in active if tokens[name] and is_number(tokens[name][0])],
        "duplicates": [group for group in same_words.values() if len(group) > 1],
        "number_siblings": [group for group in same_but_numbers.values() if len(group) > 1],
        "contained": contained,
        "near_words": near,
    }


def render(findings: dict, source: str) -> str:
    f = findings
    lines = [
        f"# Catalog check: {source}",
        "",
        f"{f['total']} products, {f['active']} active, {len(f['inactive'])} inactive. "
        f"{f['distinct_words']} distinct words in the active names.",
    ]

    def section(title: str, note: str, rows: list[str]) -> None:
        lines.extend(["", f"## {title} ({len(rows)})", "", note, ""])
        lines.extend(f"- {row}" for row in rows) if rows else lines.append("None.")

    section(
        "Inactive rows",
        "Excluded from matching. Better not to send them from the app at all.",
        f["inactive"],
    )
    section(
        "Same name twice",
        "The same words once normalized and variants are folded. One of each should go.",
        [" = ".join(group) for group in f["duplicates"]],
    )
    section(
        "Words one letter apart",
        "One word spelled two ways, or two words? Add real variants to "
        "`word_variants`; the number is how many products use each.",
        [f"{a} ({na}) / {b} ({nb})" for a, na, b, nb in f["near_words"]],
    )
    section(
        "Names starting with a number",
        "A code written in front of the name is matched as part of it.",
        f["leading_number"],
    )
    section(
        "Numbers glued to words",
        "Read correctly -- the matcher separates them -- but worth tidying at the source.",
        f["glued"],
    )
    section(
        "Products that differ only in their numbers",
        "A misread or unwritten size leaves these indistinguishable.",
        [" | ".join(group) for group in f["number_siblings"]],
    )
    section(
        "Names contained in a longer name",
        "Written in full, the short name is also a shortened form of the long one.",
        [f"{short}  ⊂  {long}" for short, long in f["contained"]],
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalog", help=".xlsx or .json product catalog")
    p.add_argument("--vocabulary", default=str(PROJECT_ROOT / "keywords" / "product_vocabulary.json"))
    p.add_argument("--out", help="write the report here instead of printing it")
    args = p.parse_args()

    names = [product.name for product in load_products(args.catalog)]
    report = render(
        catalog_findings(names, ProductVocabulary.load(args.vocabulary)),
        Path(args.catalog).name,
    )

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
