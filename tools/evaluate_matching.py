"""Compare product matchers on a catalog, without running any model.

    # Real readings: every labelled product-name reading the experiments cached
    python tools/evaluate_matching.py --catalog keywords.xlsx \
        --readings experiments/predictions --add-missing-products

    # Your own labelled readings, one JSON object per line:
    #   {"reading": "بنطلون كنان كبير", "product": "بنطال كتان جيب كبير"}
    #   {"reading": "قميص حرير", "product": null}        <- not in the catalog
    python tools/evaluate_matching.py --catalog keywords.xlsx --readings labelled.jsonl

    # Simulated readings of every catalog name, full and shortened
    python tools/evaluate_matching.py --catalog keywords.xlsx --simulate \
        --channel-from experiments/predictions

Matching is pure text, so a comparison takes seconds and repeats exactly. What
is counted, and why acceptance is measured apart from ranking, is in
`evaluation/matching_benchmark.py`; what the simulation assumes is in
`evaluation/reading_simulation.py` -- read both before quoting a number.

`--readings` takes a `.jsonl` file or a folder of them. Two line formats are
understood: the one above, and the experiment cache's own
(`role` / `text` / `pred_text`), of which only product-name cells are used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import string_matching.product_matchers  # noqa: E402,F401  (registers the matchers)
from core.domain.catalog import NamedEntry  # noqa: E402
from evaluation.catalog_io import load_products  # noqa: E402
from evaluation.matching_benchmark import Case, render_table, score_matcher  # noqa: E402
from evaluation.reading_simulation import SHORTHAND_MODES, ErrorChannel, shorten  # noqa: E402
from string_matching.normalization import normalize_text  # noqa: E402
from string_matching.product_vocabulary import ProductVocabulary  # noqa: E402
from string_matching.registry import product_matcher_registry  # noqa: E402
from string_matching.tokenization import split_words  # noqa: E402

_TEXT_ROLES = {"product_name", "merchant_name", "city"}


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    # The experiment folder keeps a reading cache and model combinations beside
    # the per-model readings; both repeat readings already counted.
    return sorted(
        file
        for file in path.rglob("*.jsonl")
        if not {"cache", "combo"} & set(file.relative_to(path).parts)
    )


def read_pairs(path: Path, roles: set[str]) -> list[tuple[str, str | None]]:
    """(reading, label) pairs from one file or a folder of them."""
    pairs: list[tuple[str, str | None]] = []

    for file in _jsonl_files(path):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)

            if "reading" in row:
                pairs.append((row["reading"], row.get("product")))
            elif row.get("role") in roles and (row.get("pred_text") or "").strip():
                pairs.append((row["pred_text"], row.get("text")))

    return pairs


def cases_from_readings(
    pairs: list[tuple[str, str | None]],
    products: list[NamedEntry],
    add_missing: bool,
) -> tuple[list[Case], list[NamedEntry], list[str]]:
    """Resolve each label to a catalog position.

    A label that is not a catalog name is, by default, a reading whose product
    is not in the catalog. With `add_missing` it is added as a product instead
    -- for labelled invoices that were written before the catalog existed.
    """
    products = list(products)
    position = {normalize_text(p.name): i for i, p in enumerate(products)}
    added: list[str] = []
    cases: list[Case] = []

    for reading, label in pairs:
        key = normalize_text(label)
        if key and key not in position and add_missing:
            position[key] = len(products)
            products.append(NamedEntry(name=label, entry_id=10_000 + len(added)))
            added.append(label)
        cases.append(Case(reading=reading, target=position.get(key) if key else None))

    return cases, products, added


def simulated_cases(
    products: list[NamedEntry],
    vocabulary: ProductVocabulary,
    channel: ErrorChannel,
    mode: str,
    noise: float,
    per_product: int,
    seed: int,
) -> list[Case]:
    """`per_product` readings of every active catalog name, shortened by `mode`."""
    rng = random.Random(seed)
    cases: list[Case] = []

    for position, product in enumerate(products):
        if vocabulary.is_inactive(product.name):
            continue
        words = split_words(product.name)
        for _ in range(per_product):
            kept = shorten(rng, words, mode)
            if kept:
                cases.append(Case(channel.corrupt(rng, " ".join(kept), noise), position))

    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True, help=".xlsx or .json product catalog")
    p.add_argument("--vocabulary", default="keywords/product_vocabulary.json")
    p.add_argument("--matchers", nargs="+", default=["word_overlap", "coverage"],
                   help=f"registered: {', '.join(product_matcher_registry.available())}")
    p.add_argument("--params", default="{}",
                   help='JSON of constructor parameters per matcher, e.g. \'{"coverage": {"coverage_weight": 0.5}}\'')
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--readings", help="labelled readings: a .jsonl file or a folder of them")
    p.add_argument("--add-missing-products", action="store_true",
                   help="add labels that are not catalog names to the catalog")
    p.add_argument("--simulate", action="store_true", help="simulate readings of every catalog name")
    p.add_argument("--channel-from", default="experiments/predictions",
                   help="labelled readings the simulated misreadings are estimated from")
    p.add_argument("--modes", nargs="+", default=list(SHORTHAND_MODES), choices=SHORTHAND_MODES)
    p.add_argument("--noise", nargs="+", type=float, default=[1.0],
                   help="multiples of the measured error rate; 0 is no misreading")
    p.add_argument("--per-product", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-removal", action="store_true",
                   help="skip the intended-product-removed measurement (the slow part)")
    p.add_argument("--report", help="also write the counts to this JSON file")
    return p.parse_args()


def main() -> int:
    utf8_console()
    args = parse_args()
    if not args.readings and not args.simulate:
        print("Nothing to do: give --readings, --simulate, or both.", file=sys.stderr)
        return 2

    def anchored(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path

    products = load_products(anchored(args.catalog))
    vocabulary = ProductVocabulary.load(anchored(args.vocabulary))
    params = json.loads(args.params)
    matchers = {
        name: product_matcher_registry.create(name, vocabulary=vocabulary, **params.get(name, {}))
        for name in args.matchers
    }

    active = sum(not vocabulary.is_inactive(p.name) for p in products)
    print(f"Catalog: {len(products)} products, {active} active. Vocabulary: "
          f"{len(vocabulary.variants)} word variants.\n")

    report: dict[str, dict] = {}

    def run(title: str, cases: list[Case], catalog: list[NamedEntry]) -> None:
        scores = {
            name: score_matcher(matcher, catalog, cases, args.top_k, not args.no_removal)
            for name, matcher in matchers.items()
        }
        print(render_table(title, scores, args.top_k), end="\n\n")
        report[title] = {name: score.as_dict() for name, score in scores.items()}

    if args.readings:
        pairs = read_pairs(anchored(args.readings), {"product_name"})
        cases, catalog, added = cases_from_readings(pairs, products, args.add_missing_products)
        if added:
            print(f"Added {len(added)} labelled product(s) the catalog lacks: {', '.join(added)}\n")
        run(f"Real readings ({len(cases)})", cases, catalog)

    if args.simulate:
        channel = ErrorChannel(
            (label, reading)
            for reading, label in read_pairs(anchored(args.channel_from), _TEXT_ROLES)
            if label
        )
        print(f"Misreadings estimated from {channel.pairs_used} real readings: "
              f"average letter error {channel.average_error:.1%}, "
              f"word split at {channel.split_rate:.1%} of characters.\n")
        for mode in args.modes:
            for noise in args.noise:
                cases = simulated_cases(
                    products, vocabulary, channel, mode, noise, args.per_product, args.seed
                )
                run(f"Simulated, written form: {mode}, noise x{noise:g} ({len(cases)} readings)",
                    cases, products)

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"Counts written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
