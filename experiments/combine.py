"""Stage 5 (service venv): combinations of cached readings, with no new inference.

Each combination is written as a predictions file under predictions/combo/, so
score.py ranks it beside the single-model configurations. Seconds are summed
over the readings a combination actually needs.

Also prints, for the reports:
  * calibration -- does a model's confidence separate right from wrong readings;
  * agreement   -- does agreement between two readings do so;
  * leave-one-invoice-out choices, for anything tuned on these cells.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\combine.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from common import EXP, PRED, is_empty, read_jsonl, score_cell, utf8_console, write_jsonl

from tools import benchmark_recognizers as bench  # noqa: E402

_SEPARATORS = re.compile(r"[,٫،٬.]+")


def vote_key(text: str | None, kind: str) -> str:
    """What two readings must share to count as the same reading."""
    if is_empty(text):
        return ""
    if kind in ("number", "date"):
        folded = (text or "").translate(bench._DIGIT_MAP).replace(" ", "")
        return _SEPARATORS.sub(".", folded).strip(".")
    return bench.normalize_text(text)


def load(model: str, variant: str) -> dict[tuple, dict]:
    rows = read_jsonl(PRED / model / f"{variant}.jsonl")
    if not rows:
        raise FileNotFoundError(f"no predictions for {model}/{variant}")
    out = {}
    for row in rows:
        row["correct"] = score_cell(row.get("pred_text"), row["text"], row["kind"], row["role"])["correct"]
        out[(row["invoice"], row["cell"])] = row
    return out


def emit(name: str, base: dict[tuple, dict], chosen: dict[tuple, dict], seconds: dict[tuple, float], notes: str) -> dict:
    rows = []
    for key, row in base.items():
        pick = chosen[key]
        rows.append({**{k: row[k] for k in ("invoice", "cell", "role", "kind", "row", "text", "members")},
                     "pred_text": pick.get("pred_text"), "confidence": pick.get("confidence"),
                     "error": pick.get("error"), "seconds": round(seconds[key], 3), "notes": notes,
                     "picked_from": f"{pick.get('model')}/{pick.get('_variant', '')}"})
    write_jsonl(PRED / "combo" / f"{name}.jsonl", rows)
    correct = sum(score_cell(r["pred_text"], r["text"], r["kind"], r["role"])["correct"] for r in rows)
    return {"name": name, "correct": correct, "n": len(rows)}


def accuracy(readings: dict[tuple, dict], invoice: str | None = None) -> float:
    rows = [r for (inv, _), r in readings.items() if invoice is None or inv == invoice]
    return sum(r["correct"] for r in rows) / max(len(rows), 1)


# -- the combinations ------------------------------------------------------

def vote(name: str, views: list[tuple[str, str]], notes: str = "") -> dict:
    """Majority vote over several readings of each cell; ties go to the first view."""
    loaded = [load(m, v) for m, v in views]
    chosen, seconds = {}, {}
    for key in loaded[0]:
        rows = [view[key] for view in loaded]
        keys = [vote_key(r.get("pred_text"), r["kind"]) for r in rows]
        counts = Counter(k for k in keys if k)
        if counts:
            best = max(counts.values())
            winner = next(k for k in keys if k and counts[k] == best)
            chosen[key] = rows[keys.index(winner)]
        else:
            chosen[key] = rows[0]
        seconds[key] = sum(r["seconds"] for r in rows)
    return emit(name, loaded[0], chosen, seconds, notes or "vote: " + ", ".join(f"{m}/{v}" for m, v in views))


def vote_loo(name: str, model: str, variants: list[str], k: int = 3) -> dict:
    """Vote over the k crop variants that scored best on the *other* invoice,
    so the choice of views is never made on the cells it is scored on."""
    loaded = {v: load(model, v) for v in variants}
    base = loaded["base"]
    invoices = sorted({key[0] for key in base})
    chosen, seconds, picked = {}, {}, {}
    for held_out in invoices:
        train = [i for i in invoices if i != held_out]
        ranked = sorted(variants, key=lambda v: (-sum(accuracy(loaded[v], i) for i in train), v != "base", v))
        views = picked[held_out] = ranked[:k]
        for key in (key for key in base if key[0] == held_out):
            rows = [loaded[v][key] for v in views]
            keys = [vote_key(r.get("pred_text"), r["kind"]) for r in rows]
            counts = Counter(x for x in keys if x)
            best = max(counts.values()) if counts else 0
            winner = next((x for x in keys if x and counts[x] == best), None)
            chosen[key] = rows[keys.index(winner)] if winner else rows[0]
            seconds[key] = sum(r["seconds"] for r in rows)
    return {**emit(name, base, chosen, seconds, f"vote over top-{k} variants chosen on the other invoice: {picked}"),
            "views": picked}


def route(name: str, default: tuple[str, str], routes: dict[str, tuple[str, str]]) -> dict:
    """Each cell is read once, by the model configured for its content kind
    (the service's `routed` recognizer without a fallback)."""
    base = load(*default)
    by_kind = {kind: load(*view) for kind, view in routes.items()}
    chosen = {key: by_kind.get(row["kind"], base)[key] for key, row in base.items()}
    seconds = {key: chosen[key]["seconds"] for key in base}
    return emit(name, base, chosen, seconds, f"routed by kind: default {default}, {routes}")


def max_confidence(name: str, a: tuple[str, str], b: tuple[str, str]) -> dict:
    """P9: both models read every cell; keep the more confident reading."""
    first, second = load(*a), load(*b)
    chosen = {k: (first[k] if (first[k].get("confidence") or 0) >= (second[k].get("confidence") or 0) else second[k]) for k in first}
    seconds = {k: first[k]["seconds"] + second[k]["seconds"] for k in first}
    return emit(name, first, chosen, seconds, f"higher confidence of {a} and {b}")


def best_threshold(first: dict, second: dict, keys: list[tuple]) -> float:
    candidates = sorted({first[k].get("confidence") or 0.0 for k in keys}) + [1.01]
    scored = [(sum((second[k] if (first[k].get("confidence") or 0) < t else first[k])["correct"] for k in keys), -t) for t in [0.0] + candidates]
    return -max(scored)[1]


def fallback(name: str, a: tuple[str, str], b: tuple[str, str]) -> dict:
    """P9: the primary reads every cell; below a confidence threshold the cell is
    re-read by the secondary. The threshold for each invoice is learned on the other."""
    first, second = load(*a), load(*b)
    invoices = sorted({k[0] for k in first})
    chosen, seconds, thresholds = {}, {}, {}
    for held_out in invoices:
        train = [k for k in first if k[0] != held_out] or list(first)
        t = thresholds[held_out] = best_threshold(first, second, train)
        for key in (k for k in first if k[0] == held_out):
            low = (first[key].get("confidence") or 0) < t
            chosen[key] = second[key] if low else first[key]
            seconds[key] = first[key]["seconds"] + (second[key]["seconds"] if low else 0.0)
    return {**emit(name, first, chosen, seconds, f"fallback {a}->{b}, leave-one-invoice-out thresholds {thresholds}"),
            "thresholds": thresholds}


def disagreement(name: str, a: tuple[str, str], b: tuple[str, str], referee: tuple[str, str]) -> dict:
    """P10: two readers; where they disagree a third reads the cell and the
    majority wins; with three different answers the primary's stands."""
    first, second, third = load(*a), load(*b), load(*referee)
    chosen, seconds, reread = {}, {}, 0
    for key, row in first.items():
        kind = row["kind"]
        ka, kb = vote_key(row.get("pred_text"), kind), vote_key(second[key].get("pred_text"), kind)
        seconds[key] = row["seconds"] + second[key]["seconds"]
        if ka == kb:
            chosen[key] = row
            continue
        reread += 1
        seconds[key] += third[key]["seconds"]
        kc = vote_key(third[key].get("pred_text"), kind)
        chosen[key] = second[key] if (kc and kc == kb) else row
    return {**emit(name, first, chosen, seconds, f"disagreement {a} vs {b}, referee {referee}; {reread} cell(s) re-read"),
            "reread": reread}


def oracle(name: str, views: list[tuple[str, str]]) -> dict:
    loaded = [load(m, v) for m, v in views]
    chosen = {k: next((view[k] for view in loaded if view[k]["correct"]), loaded[0][k]) for k in loaded[0]}
    seconds = {k: sum(view[k]["seconds"] for view in loaded) for k in loaded[0]}
    return emit(name, loaded[0], chosen, seconds, "ORACLE (upper bound, not a pipeline): " + ", ".join(f"{m}/{v}" for m, v in views))


# -- diagnostics -----------------------------------------------------------

def calibration(model: str, variant: str) -> dict:
    rows = list(load(model, variant).values())
    scored = [(r.get("confidence"), r["correct"]) for r in rows if r.get("confidence") is not None]
    right = [c for c, ok in scored if ok]
    wrong = [c for c, ok in scored if not ok]
    # AUROC: the chance a right reading outranks a wrong one by confidence.
    pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in right for b in wrong]
    bands = {}
    for low, high in ((0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 0.99), (0.99, 1.01)):
        inside = [ok for c, ok in scored if low <= c < high]
        bands[f"{low:.2f}-{min(high, 1.0):.2f}"] = f"{sum(inside)}/{len(inside)}"
    return {"model": f"{model}/{variant}", "auroc": round(sum(pairs) / len(pairs), 3) if pairs else None,
            "wrong_at_>=0.75": sum(1 for c in wrong if c >= 0.75), "wrong_at_>=0.90": sum(1 for c in wrong if c >= 0.9),
            "n": len(scored), "correct_by_band": bands}


def agreement(a: tuple[str, str], b: tuple[str, str]) -> dict:
    first, second = load(*a), load(*b)
    same = [k for k in first if vote_key(first[k].get("pred_text"), first[k]["kind"]) == vote_key(second[k].get("pred_text"), first[k]["kind"])]
    differ = [k for k in first if k not in same]
    return {"pair": f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}",
            "agree": len(same), "agree_and_right": sum(first[k]["correct"] for k in same),
            "disagree": len(differ), "first_right": sum(first[k]["correct"] for k in differ),
            "second_right": sum(second[k]["correct"] for k in differ)}


def leave_one_invoice_out(model: str, variants: list[str]) -> dict:
    """Pick the best variant on one invoice, score it on the other."""
    loaded = {v: load(model, v) for v in variants}
    invoices = sorted({k[0] for k in next(iter(loaded.values()))})
    out = {}
    for held_out in invoices:
        train = [i for i in invoices if i != held_out]
        best = max(variants, key=lambda v: (sum(accuracy(loaded[v], i) for i in train), v == "base"))
        out[held_out] = {"chosen_on_others": best, "held_out_acc": round(accuracy(loaded[best], held_out), 3),
                         "base_held_out_acc": round(accuracy(loaded["base"], held_out), 3)}
    return out


def available(model: str) -> list[str]:
    return sorted(p.stem for p in (PRED / model).glob("*.jsonl"))


def load_cell_variants(model: str) -> list[str]:
    """Crop variants of the labelled cells themselves, read with the service's
    prompts and no missing crop: the ones a vote may draw on."""
    out = []
    for name in available(model):
        if name.startswith(("row", "page")) or "_p_" in name or name.endswith("unconstrained"):
            continue
        rows = read_jsonl(PRED / model / f"{name}.jsonl")
        if rows and all(r.get("path") for r in rows):
            out.append(name)
    return out


def main() -> int:
    utf8_console()
    plan = json.loads((EXP / "combinations.json").read_text(encoding="utf-8"))
    report: dict = {"combinations": [], "calibration": [], "agreement": [], "loo_variant_choice": {}}
    for spec in plan.get("vote", []):
        report["combinations"].append(vote(spec["name"], [tuple(v) for v in spec["views"]]))
    for spec in plan.get("vote_loo", []):
        variants = [v for v in available(spec["model"]) if v in load_cell_variants(spec["model"])]
        report["combinations"].append(vote_loo(spec["name"], spec["model"], variants, spec.get("k", 3)))
    for spec in plan.get("route", []):
        report["combinations"].append(route(spec["name"], tuple(spec["default"]), {k: tuple(v) for k, v in spec["routes"].items()}))
    for spec in plan.get("max_confidence", []):
        report["combinations"].append(max_confidence(spec["name"], tuple(spec["a"]), tuple(spec["b"])))
    for spec in plan.get("fallback", []):
        report["combinations"].append(fallback(spec["name"], tuple(spec["a"]), tuple(spec["b"])))
    for spec in plan.get("disagreement", []):
        report["combinations"].append(disagreement(spec["name"], tuple(spec["a"]), tuple(spec["b"]), tuple(spec["referee"])))
    for spec in plan.get("oracle", []):
        report["combinations"].append(oracle(spec["name"], [tuple(v) for v in spec["views"]]))
    for model, variant in plan.get("calibration", []):
        report["calibration"].append(calibration(model, variant))
    for a, b in plan.get("agreement", []):
        report["agreement"].append(agreement(tuple(a), tuple(b)))
    for model, variants in plan.get("loo", {}).items():
        cell_variants = variants or load_cell_variants(model)
        report["loo_variant_choice"][model] = leave_one_invoice_out(model, cell_variants)

    (EXP / "predictions" / "combination_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
