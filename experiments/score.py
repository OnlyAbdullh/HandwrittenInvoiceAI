"""Stage 4 (service venv): score every predictions file into results.csv.

One row per (model, variant). Cell-level variants are scored per cell; row- and
page-level variants are scored by looking for each labelled value inside the
transcription (lenient: position is not checked), since a free transcription
carries no cell boundaries.

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\score.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean

from common import EXP, ITEM_ROLES, NUMBER_ROLES, PRED, read_jsonl, score_cell, utf8_console

from string_matching.normalization import normalize_price  # noqa: E402
from tools import benchmark_recognizers as bench  # noqa: E402

FIELDS = [
    "pipeline", "model", "variant", "unit", "n_cells", "cell_acc", "exact_norm", "exact_raw", "cer_norm", "cer_raw",
    "wer_norm", "number_value_acc", "name_cer", "qty_acc", "unit_price_acc", "line_total_acc", "invoice_total_acc",
    "invoice_number_acc", "date_acc", "name_exact", "row_acc", "invoice_acc", "failure_rate", "s_per_cell",
    "s_per_invoice", "acc_sahror", "acc_onlyone", "notes",
]

E2E_FIELDS = [
    "e2e_header", "e2e_numbers", "e2e_product_top1", "e2e_product_auto_accepted", "e2e_complete_invoices",
    "e2e_name_cer", "e2e_wrong_fields", "real_run_s_per_invoice", "real_run_peak_ram_mb", "real_run_report",
]

_NUMBER_TOKEN =re.compile(r"[0-9٠-٩۰-۹][0-9٠-٩۰-۹.,٫٬،]*")


def contained(member: dict, transcription: str) -> dict:
    """Score one labelled value against a free transcription of a row or page."""
    text, kind, role = member["text"], member["kind"], member["role"]
    found = ""
    if kind in ("number", "date", "code"):
        for token in _NUMBER_TOKEN.findall(transcription.replace("/", "/")):
            if score_cell(token, text, kind, role)["correct"]:
                found = token
                break
        if kind == "date" and not found:
            for token in re.findall(r"[0-9٠-٩۰-۹]+\s*/\s*[0-9٠-٩۰-۹]+\s*/\s*[0-9٠-٩۰-۹]+", transcription):
                if score_cell(token, text, kind, role)["correct"]:
                    found = token
                    break
    else:
        truth = bench.normalize_text(text)
        if truth and truth in bench.normalize_text(transcription):
            found = text
    return score_cell(found, text, kind, role)


def cell_scores(rows: list[dict]) -> list[dict]:
    """One score dict per labelled cell, whatever unit the variant read."""
    out = []
    for row in rows:
        predicted = row.get("pred_text") or ""
        if row["role"] in ("row", "page"):
            for member in row["members"]:
                score = contained(member, predicted)
                # CER of a contained value is 0 or 1 by construction; not comparable.
                score.update(cer=None, wer=None, cer_raw=None)
                out.append({**score, "invoice": row["invoice"], "role": member["role"], "kind": member["kind"],
                            "row": row["row"] if row["role"] == "row" else member.get("row"), "cell": member["cell"],
                            "seconds": row["seconds"] / len(row["members"]), "failed": bool(row.get("error"))})
        else:
            score = score_cell(predicted, row["text"], row["kind"], row["role"])
            out.append({**score, "invoice": row["invoice"], "role": row["role"], "kind": row["kind"], "row": row["row"],
                        "cell": row["cell"], "seconds": row["seconds"], "failed": bool(row.get("error")),
                        "confidence": row.get("confidence"), "pred": predicted, "truth": row["text"]})
    return out


def _rate(scores: list[dict], key: str = "correct") -> float | None:
    values = [s[key] for s in scores if s.get(key) is not None]
    return round(mean(1.0 if v else 0.0 for v in values), 3) if values else None


def _mean(scores: list[dict], key: str) -> float | None:
    values = [s[key] for s in scores if s.get(key) is not None]
    return round(mean(values), 3) if values else None


def summarize(scores: list[dict]) -> dict:
    by_role = lambda *roles: [s for s in scores if s["role"] in roles]
    invoices = sorted({s["invoice"] for s in scores})
    rows: dict[tuple, list[bool]] = {}
    for s in scores:
        if s["role"] in ITEM_ROLES and s["row"] is not None:
            rows.setdefault((s["invoice"], s["row"]), []).append(s["correct"])
    per_invoice = {name: [s for s in scores if s["invoice"] == name] for name in invoices}
    return {
        "n_cells": len(scores),
        "cell_acc": _rate(scores),
        "exact_norm": _rate(scores, "exact"),
        "exact_raw": _rate(scores, "exact_raw"),
        "cer_norm": _mean(scores, "cer"),
        "cer_raw": _mean(scores, "cer_raw"),
        "wer_norm": _mean(scores, "wer"),
        "number_value_acc": _rate(by_role(*NUMBER_ROLES)),
        "name_cer": _mean(by_role("product_name", "merchant_name", "city"), "cer"),
        "qty_acc": _rate(by_role("quantity")),
        "unit_price_acc": _rate(by_role("unit_price")),
        "line_total_acc": _rate(by_role("line_total")),
        "invoice_total_acc": _rate(by_role("total_amount", "total_in_figures")),
        "invoice_number_acc": _rate(by_role("invoice_number")),
        "date_acc": _rate(by_role("invoice_date")),
        "name_exact": _rate(by_role("product_name", "merchant_name", "city")),
        "row_acc": round(mean(1.0 if all(v) else 0.0 for v in rows.values()), 3) if rows else None,
        "invoice_acc": round(mean(1.0 if all(s["correct"] for s in cells) else 0.0 for cells in per_invoice.values()), 3),
        "failure_rate": round(mean(1.0 if (s["failed"] or s["empty"]) else 0.0 for s in scores), 3),
        "s_per_cell": _mean(scores, "seconds"),
        "s_per_invoice": round(sum(s["seconds"] for s in scores) / max(len(invoices), 1), 2),
        "acc_sahror": _rate(per_invoice.get("sahror", [])),
        "acc_onlyone": _rate(per_invoice.get("onlyone", [])),
    }


PIPELINE_IDS = {
    ("qwen35", "page"): "P1", ("qwen35", "row"): "P2", ("qwen35", "base"): "P3/P6 (baseline)",
    ("qwen35", "surya_tight10"): "P4", ("nuqta_q4", "surya_tight10"): "P5", ("nuqta_q4", "base"): "P7",
}


def pipeline_id(model: str, variant: str) -> str:
    for (m, v), pid in PIPELINE_IDS.items():
        if model == m and (variant == v or variant.startswith(v + "_") and v in ("page", "row")):
            return pid
    return ""


def main() -> int:
    utf8_console()
    table = []
    for path in sorted(PRED.glob("*/*.jsonl")):
        if path.parent.name == "cache":
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        summary = summarize(cell_scores(rows))
        unit = rows[0]["role"] if rows[0]["role"] in ("row", "page") else "cell"
        table.append({"pipeline": pipeline_id(path.parent.name, path.stem), "model": path.parent.name,
                      "variant": path.stem, "unit": unit, **summary, "notes": rows[0].get("notes", "")})

    # The whole pipeline on the same readings (e2e_replay.py), and the real runs
    # that were timed (measured.json), beside the per-cell scores.
    replay_path, measured_path = EXP / "e2e_replay.csv", EXP / "measured.json"
    replay = {}
    if replay_path.exists():
        with replay_path.open(encoding="utf-8", newline="") as source:
            replay = {(r["model"], r["variant"]): r for r in csv.DictReader(source)}
    measured = json.loads(measured_path.read_text(encoding="utf-8")) if measured_path.exists() else {}
    for row in table:
        if row["model"] == "combo":
            row["pipeline"] = row["variant"].split("_", 1)[0]
        e2e = replay.get((row["model"], row["variant"]))
        if e2e:
            row.update({
                "e2e_header": f"{e2e['header_correct']}/{e2e['header_total']}",
                "e2e_numbers": f"{e2e['numeric_cells_correct']}/{e2e['numeric_cells_total']}",
                "e2e_product_top1": f"{e2e['product_top1_correct']}/{e2e['products_total']}",
                "e2e_product_auto_accepted": f"{e2e['product_auto_accepted']}/{e2e['products_total']}",
                "e2e_complete_invoices": f"{e2e['complete_invoices']}/2",
                "e2e_name_cer": e2e["mean_name_cer"],
                "e2e_wrong_fields": e2e["wrong_fields"],
            })
        row.update(measured.get(f"{row['model']}/{row['variant']}", {}))

    table.sort(key=lambda r: (-(r["cell_acc"] or 0), r["cer_norm"] if r["cer_norm"] is not None else 9))
    with (EXP / "results.csv").open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS + E2E_FIELDS)
        writer.writeheader()
        writer.writerows(table)

    print(f"{'model':<12}{'variant':<22}{'acc':>6}{'num':>6}{'cer':>7}{'rawcer':>7}{'nameCER':>8}{'row':>6}{'fail':>6}{'s/cell':>7}  sah/only")
    for r in table:
        fmt = lambda v, w: f"{v:>{w}.3f}" if isinstance(v, float) else f"{'-':>{w}}"
        print(f"{r['model']:<12}{r['variant']:<22}{fmt(r['cell_acc'], 6)}{fmt(r['number_value_acc'], 6)}{fmt(r['cer_norm'], 7)}"
              f"{fmt(r['cer_raw'], 7)}{fmt(r['name_cer'], 8)}{fmt(r['row_acc'], 6)}{fmt(r['failure_rate'], 6)}{fmt(r['s_per_cell'], 7)}"
              f"  {r['acc_sahror']}/{r['acc_onlyone']}")
    print(f"\n{len(table)} configuration(s) -> {EXP / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
