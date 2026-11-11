"""Run every automated experiment step in order, skipping what is already done.

A step is done when all of its output files exist, so the state lives in the
results themselves and nothing can claim a step finished that did not. Inside a
step, every model reading is cached as it is made (predictions/cache/), so an
interrupted step resumes at the first crop it has not read.

    python experiments\\run_all.py            # run / resume
    python experiments\\run_all.py --status   # what is done, what is left; runs nothing

To stop cleanly while it runs: create the file experiments\\STOP (the reader
finishes the crop in hand and exits). Ctrl+C or a shutdown is also safe -- at
most the one reading in progress is lost. Delete STOP before resuming.

Interpreters: INVOICE_AI_SERVICE_PYTHON (default D:\\OCR\\.venv) for the steps
that need OpenCV and the project, INVOICE_AI_VLM_PYTHON (default
D:\\OCR\\.venv_vlm) for the transformers model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
SERVICE = os.environ.get("INVOICE_AI_SERVICE_PYTHON", r"D:\OCR\.venv\Scripts\python.exe")
VLM = os.environ.get("INVOICE_AI_VLM_PYTHON", r"D:\OCR\.venv_vlm\Scripts\python.exe")
STATE = EXP / "state.json"

CELL_VARIANTS = [
    "base", "pad0", "pad4", "pad20", "pad30", "inset6", "expand_x25", "expand_y25", "expand_y40",
    "ink_tight8", "ink_tight16", "ink_tight24", "surya_tight10", "colfix", "colfix_pad20",
    "up1", "up3", "up4", "h128", "h256", "up2_cubic",
    "color", "clahe", "clahe_crop", "bilateral", "median", "adaptive", "blur_divide", "otsu", "nlm",
    "sharpen", "stretch", "norules",
]
UNIT_VARIANTS = ["row", "row_up1", "row_color", "page"]
QWEN25 = "OCR_Models/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3"


def preds(model: str, names: list[str]) -> list[str]:
    return [f"predictions/{model}/{name}.jsonl" for name in names]


# (id, interpreter, arguments, outputs relative to experiments/, what it measures)
STEPS = [
    ("prepare", SERVICE, ["prepare.py", "--surya"], ["cache/regions.json"],
     "geometry, layout, labelled cells, Surya text boxes"),
    ("crops", SERVICE, ["make_crops.py"],
     [f"cache/crops/{v}/manifest.jsonl" for v in CELL_VARIANTS + UNIT_VARIANTS],
     "every crop variant"),
    ("qwen35_variants", VLM, ["read_qwen.py", "--variants", *CELL_VARIANTS, *UNIT_VARIANTS],
     preds("qwen35", CELL_VARIANTS + UNIT_VARIANTS),
     "Qwen3.5-OCR-v4 on every cell/row/page variant (P1, P2, P3/P6, P4; bbox, resolution, preprocessing)"),
    ("qwen35_prompt_card", VLM, ["read_qwen.py", "--variants", "base", "--prompts-file", "prompts/card.json", "--suffix", "p_card"],
     preds("qwen35", ["base_p_card"]), "the model card's own prompt for every kind"),
    ("qwen35_prompt_hint", VLM, ["read_qwen.py", "--variants", "base", "--prompts-file", "prompts/hint.json", "--suffix", "p_hint"],
     preds("qwen35", ["base_p_hint"]), "number prompt explaining the dot-zero and the decimal comma"),
    ("qwen35_prompt_short", VLM, ["read_qwen.py", "--variants", "base", "--prompts-file", "prompts/short.json", "--suffix", "p_short"],
     preds("qwen35", ["base_p_short"]), "short prompts"),
    ("qwen35_unconstrained", VLM, ["read_qwen.py", "--variants", "base", "--unconstrained", "--suffix", "unconstrained"],
     preds("qwen35", ["base_unconstrained"]), "numbers and dates without the character-set constraint"),
    ("qwen35_hires_1280", VLM, ["read_qwen.py", "--variants", "page", "row", "--max-side", "1280", "--suffix", "s1280"],
     preds("qwen35", ["page_s1280", "row_s1280"]), "full page and rows sent at 1280 px instead of 768"),
    ("qwen35_hires_1536", VLM, ["read_qwen.py", "--variants", "page", "row", "--max-side", "1536", "--suffix", "s1536"],
     preds("qwen35", ["page_s1536", "row_s1536"]), "full page and rows sent at 1536 px"),
    ("qwen25_cells", VLM, ["read_qwen.py", "--model", QWEN25, "--tag", "qwen25", "--variants", "base", "colfix"],
     preds("qwen25", ["base", "colfix"]), "Qwen2.5-VL-3B 4-bit: a third reader for disagreement voting"),
    ("nuqta_q4_cells", SERVICE, ["read_nuqta.py", "--variants", "base", "colfix", "surya_tight10"],
     preds("nuqta_q4", ["base", "colfix", "surya_tight10"]), "Nuqta Q4_K_M on grid cells (P7) and Surya boxes (P5)"),
    ("nuqta_q8_cells", SERVICE, ["read_nuqta.py", "--quant", "Q8_0", "--tag", "nuqta_q8", "--variants", "base"],
     preds("nuqta_q8", ["base"]), "Nuqta Q8_0 on grid cells"),
]


def missing(outputs: list[str]) -> list[str]:
    return [o for o in outputs if not (EXP / o).exists()]


def status() -> list[dict]:
    rows = []
    for step_id, _, args, outputs, what in STEPS:
        left = missing(outputs)
        rows.append({"step": step_id, "done": not left, "outputs_done": len(outputs) - len(left),
                     "outputs_total": len(outputs), "command": " ".join(args), "measures": what})
    return rows


def save_state(note: str) -> None:
    cache = EXP / "predictions" / "cache"
    counts = {p.stem: sum(1 for line in p.open(encoding="utf-8") if line.strip())
              for p in sorted(cache.glob("*.jsonl")) if "checkpoint" not in p.name} if cache.exists() else {}
    STATE.write_text(json.dumps({
        "saved_at": datetime.now().isoformat(timespec="seconds"), "note": note,
        "cached_readings": counts, "steps": status(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    if "--status" in sys.argv:
        for row in status():
            mark = "done" if row["done"] else f"{row['outputs_done']}/{row['outputs_total']}"
            print(f"{row['step']:<24}{mark:<8}{row['measures']}")
        save_state("status check")
        return 0

    for step_id, python, args, outputs, what in STEPS:
        if (EXP / "STOP").exists():
            print("experiments/STOP exists: delete it to run.")
            save_state("stopped by STOP file")
            return 1
        if not missing(outputs):
            print(f"[skip] {step_id}")
            continue
        print(f"[run ] {step_id}: {what}", flush=True)
        started = time.perf_counter()
        code = subprocess.run([python, *args], cwd=EXP).returncode
        left = missing(outputs)
        print(f"[{'done' if not left else 'INCOMPLETE'}] {step_id} in {time.perf_counter() - started:.0f}s"
              + (f", exit code {code}, {len(left)} output(s) missing" if left else ""), flush=True)
        save_state(f"after {step_id}")
        # A failed step is recorded and the run goes on: the steps are independent.

    subprocess.run([SERVICE, "score.py"], cwd=EXP)
    save_state("run_all finished")
    left = [row["step"] for row in status() if not row["done"]]
    print("\nAll automated steps done." if not left else f"\nNot finished: {', '.join(left)}")
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
