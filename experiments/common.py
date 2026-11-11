"""Shared paths, caching and scoring for the pipeline experiments.

Standard library only at import time: the readers run in the VLM virtualenv,
which has none of the service's dependencies. Scoring helpers that need the
project (`value_correct`) import it lazily.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
CACHE = EXP / "cache"            # pages, crops: reproducible, git-ignored
PAGES = CACHE / "pages"
CROPS = CACHE / "crops"
PRED = EXP / "predictions"       # every model reading, kept
READ_CACHE = PRED / "cache"
REGIONS = CACHE / "regions.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUMBER_ROLES = ("quantity", "unit_price", "line_total", "total_amount", "total_in_figures")
ITEM_ROLES = ("product_name", "quantity", "unit_price", "line_total")


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file. A line that does not parse is skipped: the only
    way one arises is a process killed mid-append, and that reading was never
    completed -- it is simply read again on the next run."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            print(f"warning: skipped an incomplete line in {path.name}", file=sys.stderr)
    return rows


STOP_FILE = EXP / "STOP"


def stop_requested() -> bool:
    """True once experiments/STOP exists: readers finish the crop in hand, keep
    everything read so far and exit. Delete the file to run again."""
    return STOP_FILE.exists()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def manifest(variant: str) -> list[dict]:
    return read_jsonl(CROPS / variant / "manifest.jsonl")


def pipeline_prompts() -> dict[str, str]:
    """The service's per-kind prompts, read without importing the engine module
    (which needs dependencies the VLM venv lacks)."""
    tree = ast.parse((ROOT / "ocr" / "engines" / "qwen_vlm_engine.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("PROMPTS not found in ocr/engines/qwen_vlm_engine.py")


class ReadCache:
    """Append-only reading cache: one JSONL per model tag, keyed by crop content
    and every setting that changes the answer."""

    def __init__(self, model_tag: str):
        self.path = READ_CACHE / f"{model_tag}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {row["key"]: row for row in read_jsonl(self.path)}

    @staticmethod
    def key(image_sha1: str, kind: str, prompt: str, settings: str) -> str:
        return hashlib.sha1(f"{image_sha1}|{kind}|{prompt}|{settings}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, row: dict) -> None:
        row = {"key": key, **row}
        self.data[key] = row
        # One complete line per reading, flushed to the disk before the next
        # crop is read: an interruption costs at most the reading in progress.
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())


# -- scoring ---------------------------------------------------------------

def _bench():
    from tools import benchmark_recognizers as bench  # project metrics, one definition

    return bench


def is_empty(text: str | None) -> bool:
    return not (text or "").strip() or (text or "").strip().upper() == "EMPTY"


_RAW_WS = re.compile(r"\s+")


def raw_cer(predicted: str | None, expected: str) -> float:
    """CER with no normalization beyond whitespace collapsing: digit script,
    separators, hamza/ta-marbuta forms and captions all count."""
    bench = _bench()
    guess = _RAW_WS.sub(" ", (predicted or "").strip())
    truth = _RAW_WS.sub(" ", expected.strip())
    return round(bench.levenshtein(guess, truth) / max(len(truth), 1), 3)


def score_cell(predicted: str | None, expected: str, kind: str, role: str) -> dict:
    """Every per-cell measure for one reading."""
    bench = _bench()
    predicted = "" if is_empty(predicted) else (predicted or "")
    norm_exact = bench.normalize_text(predicted) == bench.normalize_text(expected)
    value = bench.value_correct(predicted, expected, kind, role)
    return {
        "cer": bench.cer(predicted, expected),
        "wer": bench.wer(predicted, expected),
        "cer_raw": raw_cer(predicted, expected),
        "exact": norm_exact,
        "exact_raw": _RAW_WS.sub(" ", predicted.strip()) == expected.strip(),
        "value": value,
        # The headline: a number or date is right when its value is; text when
        # it matches after the pipeline's own normalization.
        "correct": bool(value) if value is not None else norm_exact,
        "empty": not predicted,
    }
