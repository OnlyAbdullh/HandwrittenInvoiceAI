"""Would combining models beat the best single one? Answered from benchmark results.

    python tools/analyze_benchmark.py evaluation/reports/model_benchmark.json
    python tools/analyze_benchmark.py run_a.json run_b.json --out evaluation/reports/analysis.md

Reads the per-cell results `tools/benchmark_recognizers.py` saved -- no model
runs here -- and evaluates, on the cells every compared model read:

* **Each model alone** -- CER, exact match, number-value accuracy.
* **Routing by cell kind** (one model for numbers, another for names...). The
  model for each kind is chosen on the *other* invoices and scored on the held-out
  one (leave-one-invoice-out), so the routing is not graded on the cells it was
  tuned on. With a single invoice there is nothing to hold out, and the result is
  labelled in-sample -- optimistic by construction.
* **Confidence fallback** -- read with model A; when A's confidence is below a
  threshold, use model B's reading instead. The threshold is chosen on the other
  invoices, the same way. Only models that report a confidence can be A.
* **Oracle** -- the best model per cell, chosen with the answer in hand. Not
  achievable; it bounds what any combination of these models could reach.
* **Calibration** -- for each model with confidences: exact-match rate per
  confidence band, and how many wrong readings came with confidence >= 0.75
  (the review threshold), i.e. would have been accepted without review.

A combination is worth building only if it beats the best single model on
held-out invoices by more than noise -- with a few dozen cells, a gap of one or
two cells is noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REVIEW_THRESHOLD = 0.75
CONFIDENCE_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_results(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """{model name: {cell key: per-cell record}} for every model that ran.

    Only labelled cells are kept: without a label there is nothing to compare.
    """
    models: dict[str, dict[str, dict]] = {}
    for path in paths:
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if entry.get("skipped_reason") or not entry.get("per_crop"):
                continue
            name = entry["name"]
            if name in models:
                name = f"{name} ({path.stem})"
            models[name] = {c["image"]: c for c in entry["per_crop"] if c.get("cer") is not None}
    return models


def common_cells(models: dict[str, dict[str, dict]]) -> list[str]:
    """Cells every model has a scored reading for -- the only fair comparison set."""
    keys = None
    for cells in models.values():
        keys = set(cells) if keys is None else keys & set(cells)
    return sorted(keys or [])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "cer": None, "exact": None, "value_accuracy": None}
    values = [r for r in records if r.get("value_ok") is not None]
    return {
        "n": len(records),
        "cer": round(sum(r["cer"] for r in records) / len(records), 3),
        "exact": round(sum(r["cer"] == 0.0 for r in records) / len(records), 3),
        "value_accuracy": round(sum(bool(r["value_ok"]) for r in values) / len(values), 3) if values else None,
    }


def folds(models: dict[str, dict[str, dict]], keys: list[str]) -> tuple[list[tuple[list[str], list[str]]], bool]:
    """(train, test) key lists, leave-one-invoice-out; and whether it is held out.

    With one invoice, train == test: the result is in-sample and says so.
    """
    any_model = next(iter(models.values()))
    by_source: dict[str, list[str]] = {}
    for key in keys:
        by_source.setdefault(str(any_model[key].get("source")), []).append(key)
    if len(by_source) < 2:
        return [(keys, keys)], False
    return [
        ([k for s, ks in by_source.items() if s != held_out for k in ks], test)
        for held_out, test in by_source.items()
    ], True


def mean_cer(models, name: str, keys: list[str]) -> float:
    return sum(models[name][k]["cer"] for k in keys) / len(keys) if keys else float("inf")


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def route_by_kind(models, keys) -> tuple[list[dict], dict]:
    """Per kind, the model with the lowest CER on the training invoices."""
    chosen_records: list[dict] = []
    choices: dict[str, list[str]] = {}
    split, _ = folds(models, keys)
    for train, test in split:
        best_overall = min(models, key=lambda m: mean_cer(models, m, train))
        kinds = {models[best_overall][k].get("kind") for k in test}
        for kind in kinds:
            train_kind = [k for k in train if models[best_overall][k].get("kind") == kind]
            pick = (min(models, key=lambda m: mean_cer(models, m, train_kind)) if train_kind else best_overall)
            choices.setdefault(str(kind), []).append(pick)
            chosen_records += [models[pick][k] for k in test if models[pick][k].get("kind") == kind]
    return chosen_records, choices


def fallback(models, keys, primary: str, secondary: str) -> tuple[list[dict], list[float]] | None:
    """Primary's reading unless its confidence is below a threshold chosen on training data."""
    if not all(models[primary][k].get("confidence") is not None for k in keys):
        return None
    records: list[dict] = []
    thresholds: list[float] = []
    split, _ = folds(models, keys)
    for train, test in split:
        candidates = sorted({0.0, *(models[primary][k]["confidence"] for k in train), 1.01})

        def cer_at(threshold: float, cells: list[str]) -> float:
            picks = [models[secondary if models[primary][k]["confidence"] < threshold else primary][k] for k in cells]
            return sum(r["cer"] for r in picks) / len(picks)

        threshold = min(candidates, key=lambda t: (cer_at(t, train), t))
        thresholds.append(round(threshold, 3))
        records += [models[secondary if models[primary][k]["confidence"] < threshold else primary][k] for k in test]
    return records, thresholds


def oracle(models, keys) -> list[dict]:
    return [min((models[m][k] for m in models), key=lambda r: r["cer"]) for k in keys]


def calibration(models, keys, name: str) -> dict | None:
    records = [models[name][k] for k in keys if models[name][k].get("confidence") is not None]
    if not records:
        return None
    bands = []
    for low, high in CONFIDENCE_BANDS:
        inside = [r for r in records if low <= r["confidence"] < high]
        bands.append({
            "band": f"{low:.2f}-{min(high, 1.0):.2f}",
            "n": len(inside),
            "exact": round(sum(r["cer"] == 0.0 for r in inside) / len(inside), 3) if inside else None,
        })
    wrong = [r for r in records if r["cer"] != 0.0]
    confident_wrong = [r for r in wrong if r["confidence"] >= REVIEW_THRESHOLD]
    return {
        "bands": bands,
        "wrong": len(wrong),
        "confidently_wrong": len(confident_wrong),
        "accepted_wrong_rate": round(len(confident_wrong) / len(records), 3),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def analyze(models: dict[str, dict[str, dict]]) -> str:
    keys = common_cells(models)
    if not keys:
        return "No labelled cell was read by every model; nothing to compare."
    _, held_out = folds(models, keys)
    sources = sorted({str(next(iter(models.values()))[k].get("source")) for k in keys})
    lines = [
        "# Combining recognizers: is any combination better than the best single model?",
        "",
        (f"{len(keys)} labelled cell(s) read by all {len(models)} model(s), from {len(sources)} invoice(s): "
         f"{', '.join(sources)}."),
        "",
        ("Combinations are chosen on the other invoices and scored on the held-out one (leave-one-invoice-out)."
         if held_out else
         "**Only one invoice**: combinations are chosen and scored on the same cells (in-sample) -- an "
         "optimistic upper estimate, not evidence. Label a second invoice to get a held-out result."),
        "",
        "| Strategy | Cells | CER | Exact | Number value accuracy |",
        "|---|---|---|---|---|",
    ]

    def row(label: str, s: dict) -> None:
        lines.append(f"| {label} | {s['n']} | {s['cer']} | {s['exact']} | {s['value_accuracy']} |")

    singles = {m: score([models[m][k] for k in keys]) for m in models}
    best_single = min(singles, key=lambda m: singles[m]["cer"])
    for m in sorted(singles, key=lambda m: singles[m]["cer"]):
        row(f"{m} alone" + ("  ← best single" if m == best_single else ""), singles[m])

    if len(models) >= 2:
        routed, choices = route_by_kind(models, keys)
        row("Route by cell kind", score(routed))
        fallbacks = []
        for primary in models:
            for secondary in models:
                if primary == secondary:
                    continue
                result = fallback(models, keys, primary, secondary)
                if result is not None:
                    fallbacks.append((primary, secondary, result))
        for primary, secondary, (records, thresholds) in sorted(fallbacks, key=lambda f: score(f[2][0])["cer"])[:3]:
            row(f"{primary}, falling back to {secondary} below confidence {thresholds}", score(records))
        row("Oracle (best per cell, with hindsight -- not achievable)", score(oracle(models, keys)))

        lines += ["", "Routing chose, per kind (one entry per fold):", ""]
        for kind, picks in sorted(choices.items()):
            lines.append(f"- {kind}: {', '.join(picks)}")

    lines += ["", "## Confidence calibration", "",
              (f"Exact-match rate per confidence band. 'Accepted wrong' = wrong readings with confidence >= "
               f"{REVIEW_THRESHOLD}, which the review threshold would let through unflagged."), ""]
    any_calibrated = False
    for m in models:
        cal = calibration(models, keys, m)
        if cal is None:
            lines.append(f"- {m}: reports no confidence.")
            continue
        any_calibrated = True
        bands = "; ".join(f"{b['band']}: {b['exact']} (n={b['n']})" for b in cal["bands"] if b["n"])
        lines.append(f"- {m}: {bands}. Wrong readings: {cal['wrong']}, of which accepted wrong "
                     f"(confidence >= {REVIEW_THRESHOLD}): {cal['confidently_wrong']} "
                     f"({cal['accepted_wrong_rate']:.1%} of all cells).")
    if any_calibrated:
        lines += ["", ("Confidence is the model's own token probability, not a calibrated probability of being "
                       "right; these bands are how it actually behaved on these cells.")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path, help="model_benchmark.json file(s)")
    ap.add_argument("--out", type=Path, help="also write the report here (Markdown)")
    args = ap.parse_args(argv)

    models = load_results(args.results)
    if not models:
        print("No model with labelled, scored cells in these results.", file=sys.stderr)
        return 2
    report = analyze(models)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
