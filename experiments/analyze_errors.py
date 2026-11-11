"""Stage 7 (service venv): what kind of error each wrong reading is.

Rule-based, on the reading against its label, so the counts are reproducible:

  numbers  separator    the digits are right, the decimal separator is missing or misplaced
           comma_as_zero the decimal comma was read as a zero (5,7 -> 507)
           zero_count   right once zeros are ignored: dot-shaped zeros counted short or long
           cut_or_extra the label's digits are a strict part of the reading, or the reverse
           wrong_digit  anything else
  text     hamza_alef / ta_marbuta / dots  a single confusable-letter class explains it
           characters   up to a third of the characters differ
           split_merge  right once spaces are ignored
           hallucinated more than a third differs, or the reading is much longer than the label
  any      empty        nothing was read

    D:\\OCR\\.venv\\Scripts\\python.exe experiments\\analyze_errors.py qwen35/base nuqta_q4/base
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

from common import PRED, is_empty, read_jsonl, score_cell, utf8_console

from tools import benchmark_recognizers as bench  # noqa: E402

_DOT_GROUPS = ["بتثنيئ", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة"]
_DOT_FOLD = {ch: group[0] for group in _DOT_GROUPS for ch in group}


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text.translate(bench._DIGIT_MAP))


def classify(predicted: str | None, expected: str, kind: str, role: str) -> str:
    if is_empty(predicted):
        return "empty"
    predicted = bench.strip_caption(predicted)
    if kind in ("number", "date", "code"):
        got, want = _digits(predicted), _digits(expected)
        if got == want:
            return "separator"
        if kind == "number" and "." in expected:
            whole, _, frac = expected.partition(".")
            if got in (whole + "0" + frac, whole + "0" + frac + "0", whole + "0" + frac.rstrip("0")):
                return "comma_as_zero"
        if got.replace("0", "") == want.replace("0", ""):
            return "zero_count"
        if got and want and (got in want or want in got):
            return "cut_or_extra"
        return "wrong_digit"
    got, want = bench.normalize_text(predicted), bench.normalize_text(expected)
    if got.replace(" ", "") == want.replace(" ", ""):
        return "split_merge"
    distance = bench.levenshtein(got, want)
    if distance > max(len(want) / 3, 2) or len(got) > 1.5 * len(want):
        return "hallucinated"
    if "".join(_DOT_FOLD.get(c, c) for c in got) == "".join(_DOT_FOLD.get(c, c) for c in want):
        return "dots"
    return "characters"


def main() -> int:
    utf8_console()
    report = {}
    for target in sys.argv[1:]:
        model, variant = target.split("/", 1)
        counts: Counter = Counter()
        by_role: dict[str, Counter] = {}
        examples: dict[str, list[str]] = {}
        rows = read_jsonl(PRED / model / f"{variant}.jsonl")
        for row in rows:
            if score_cell(row.get("pred_text"), row["text"], row["kind"], row["role"])["correct"]:
                continue
            category = classify(row.get("pred_text"), row["text"], row["kind"], row["role"])
            counts[category] += 1
            by_role.setdefault(row["role"], Counter())[category] += 1
            examples.setdefault(category, []).append(f"{row['invoice']}/{row['cell']}: {row['text']} -> {row.get('pred_text')}")
        report[target] = {"cells": len(rows), "wrong": sum(counts.values()), "by_category": dict(counts.most_common()),
                          "by_role": {role: dict(c) for role, c in by_role.items()}, "examples": examples}
        print(f"{target}: {sum(counts.values())}/{len(rows)} wrong  {dict(counts.most_common())}")
    (PRED / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
