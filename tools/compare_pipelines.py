"""Run every reading approach on the labelled invoices and rank them.

    python tools/compare_pipelines.py
    python tools/compare_pipelines.py --configs config/default_config.yaml config/qwen_detector_config.yaml

Each config is evaluated by `tools/evaluate.py` in its own process -- one
model's memory, or a native crash, never reaches the next -- against the same
truth files and catalogs. The approaches shipped as configs:

| Config | How a page is read |
|---|---|
| default_config.yaml | the printed grid's cells, read by Arabic-Qwen3.5-OCR-v4 |
| nuqta_config.yaml | the printed grid's cells, read by Nuqta (llama.cpp) |
| qwen_detector_config.yaml | Surya finds the ink, each box is assigned to its cell, Qwen reads it |
| surya_config.yaml | Surya detects and reads the whole page by itself |

An approach that cannot run on this machine (Surya not installed, a model
missing) is reported with the reason, never skipped silently. Writes
`evaluation/reports/pipeline_comparison.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = (
    "config/default_config.yaml",
    "config/nuqta_config.yaml",
    "config/qwen_detector_config.yaml",
    "config/surya_config.yaml",
)

APPROACH = {
    "default_config": "Grid cells -> Qwen3.5-OCR-v4",
    "qwen_config": "Grid cells -> Qwen3.5-OCR-v4",
    "nuqta_config": "Grid cells -> Nuqta (llama.cpp)",
    "qwen_detector_config": "Surya boxes -> cells -> Qwen3.5-OCR-v4",
    "surya_config": "Surya alone (whole page)",
}


def run_one(config: Path, out_root: Path, passthrough: list[str]) -> dict:
    """Evaluate one config in a child process; its summary, or why it failed."""
    out_dir = out_root / config.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    command = [sys.executable, str(PROJECT_ROOT / "tools" / "evaluate.py"), "--config", str(config),
               "--out", str(out_dir), "--label", "compare", *passthrough]
    # The child prints Arabic and "≥"; on Windows a pipe defaults to cp1252,
    # which cannot encode them. UTF-8 mode for the child, UTF-8 when reading it.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=PROJECT_ROOT, env=env, check=False)
    wall = round(time.perf_counter() - started, 1)

    # A report that was written counts even if the process failed afterwards.
    reports = sorted(out_dir.glob("*.json"))
    if not reports:
        tail = [line for line in (proc.stderr or proc.stdout or "").strip().splitlines() if line.strip()]
        return {"config": config.name, "error": tail[-1] if tail else f"exit code {proc.returncode}", "wall": wall}

    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    summary = report["summary"]
    if summary.get("failed") and summary["failed"] == summary.get("invoices"):
        reasons = {inv.get("error") for inv in report.get("invoices", []) if inv.get("error")}
        return {"config": config.name, "error": "; ".join(sorted(reasons)) or "every invoice failed", "wall": wall}
    return {"config": config.name, "summary": summary, "run": report.get("run", {}), "wall": wall}


def _fraction(correct: int, total: int) -> str:
    return f"{correct}/{total}" if total else "—"


def to_markdown(results: list[dict]) -> str:
    ok = [r for r in results if "summary" in r]
    failed = [r for r in results if "summary" not in r]

    def rank_key(r: dict):
        s = r["summary"]
        right = s["header_correct"] + s["numeric_cells_correct"] + s["product_top1_correct"]
        return (-s["complete_invoices"], -right, s["phantom_rows"] + s["missed_rows"], s["mean_seconds"] or 1e9)

    ok.sort(key=rank_key)
    lines = [
        "# Reading approaches compared",
        "",
        "Same labelled invoices, same catalogs, each config in its own process (`tools/evaluate.py`).",
        "Ranked by complete invoices, then fields right, then phantom and missed rows, then speed.",
        "",
        ("| Rank | Approach | Config | Complete | Header | Numbers | Product top-1 | Phantom / missed rows "
         "| s / invoice | Load s | Peak RAM MB |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(ok, 1):
        s, run = r["summary"], r["run"]
        stem = Path(r["config"]).stem
        lines.append(
            f"| {rank} | {APPROACH.get(stem, stem)} | `{r['config']}` "
            f"| {_fraction(s['complete_invoices'], s['invoices'] - s['failed'])} "
            f"| {_fraction(s['header_correct'], s['header_total'])} "
            f"| {_fraction(s['numeric_cells_correct'], s['numeric_cells_total'])} "
            f"| {_fraction(s['product_top1_correct'], s['products_total'])} "
            f"| {s['phantom_rows']} / {s['missed_rows']} | {s['mean_seconds']} "
            f"| {run.get('load_seconds', '—')} | {run.get('peak_ram_mb') or '—'} |"
        )
    if failed:
        lines += ["", "Could not run:", ""]
        for r in failed:
            stem = Path(r["config"]).stem
            lines.append(f"- {APPROACH.get(stem, stem)} (`{r['config']}`): {r['error']}")
    lines += ["", ("Each approach's full report (per field, per number, raw readings) is under "
                   "`evaluation/reports/compare/<config>/`.")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", type=Path, default=[Path(c) for c in DEFAULT_CONFIGS])
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports"))
    ap.add_argument("--images", default=None, help="passed to evaluate.py")
    ap.add_argument("--truth", default=None, help="passed to evaluate.py")
    args = ap.parse_args(argv)

    passthrough: list[str] = []
    if args.images:
        passthrough += ["--images", args.images]
    if args.truth:
        passthrough += ["--truth", args.truth]

    results = []
    for config in args.configs:
        print(f"- {config} ...", end="", flush=True)
        result = run_one(config, args.out / "compare", passthrough)
        print(f" {result['wall']}s" + (f"  (could not run: {result['error']})" if "error" in result else ""))
        results.append(result)

    report = to_markdown(results)
    print()
    print(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pipeline_comparison.md").write_text(report, encoding="utf-8")
    print(f"\nReport: {args.out / 'pipeline_comparison.md'}")
    return 0 if any("summary" in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
