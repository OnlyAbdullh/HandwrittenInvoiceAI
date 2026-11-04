# Experiment log

One entry per experiment, kept whether or not it helped. Each entry gives:

- **Data**: what it was run on.
- **Where**: the user's laptop (RTX 3050 4 GB, 24 GB RAM, Ryzen 5 5600H), or the
  cloud development environment, which has no real model weights.
- **Result**: what was measured.
- **Decision**: what was kept.

Cloud results that used stand-in models verify *plumbing only*, never accuracy.

---

### 2026-11 · All models in `OCR_Models/`, pipeline mode
- **Data:** `bills/sahror`, 20 labelled cells.
- **Where:** laptop, CUDA.
- **Result:**

  | Model | CER |
  |---|---|
  | Arabic-Qwen3.5-OCR-v4 | **0.143** (exact 0.45, 0.97 s/cell) |
  | Qwen2.5-VL-3B 4-bit v3 | 0.262 |
  | Qwen3-VL-2B Persian-Arabic | 0.371 |
  | Nuqta Q4_K_M (first settings) | 0.379 |
  | TrOCR variants | > 1.1 |
  | Qari-OCR v0.3 | 1.22 |
  | Qalam_Net_V2 | 1.76 |
  | arabic-large-nougat | 4.12 |

  Some models were skipped (no `config.json`) or crashed natively. Process
  isolation was added so a crash costs only that model.
- **Decision:** keep Arabic-Qwen3.5-OCR-v4 as the default
  (`default_config.yaml`, `qwen_config.yaml`).

### 2026-11 · Nuqta read with its model card's settings
- **Hypothesis:** Nuqta was handicapped by our settings:
  - `image_min_tokens: 64` overrode its projector's ~1 MP floor (136 vs. 1,099
    prompt tokens for a 240×120 cell, measured on a real server);
  - it was given the Qwen-tuned Arabic prompts;
  - a repeat penalty of 1.1 was on.
- **Change:**
  - `image_min_tokens: auto` (the projector's declared floor);
  - the card's prompt;
  - `repetition_penalty: 1.0`.
- **Data:** the same 20 cells, laptop.
- **Result:** CER 0.379 → **0.239**, exact 0.30. Time went from 1.1 to 3.05 s/cell,
  because each cell is now about 1 MP of image tokens.
- **Decision:** keep the model-card settings in `nuqta_config.yaml`. Nuqta is now
  2nd, but still behind Qwen3.5-OCR-v4 and 3× slower. It is not the default.

### 2026-11 · Form B lost its first item row
- **Problem:** on `bills/onlyone`, `contour_based` did not close the first item
  row's cells (an open contour), so every item row was shifted, and the export
  could not label any item cell.
- **Tried and rejected** (each on both real photos, rendered with
  `tools/render_layout.py`):
  - `grid_line` extractor: lost other rows;
  - morphological gap closing: merged handwriting into rules;
  - extending rule ends: false cells;
  - four ink measures for a blank-cell gate: none separated spillover from
    writing.
- **Kept:** grid completion in `bill_layout` (`complete_grid: true`):
  - missing item rows are rebuilt from the measured row pitch inside the table
    outline;
  - fragments smaller than half a column's typical cell are ignored (a
    handwriting loop had been taken for the product cell).
- **Result, cloud on the real photos:**
  - form B now has 15 rows, aligned, with its first item row read;
  - `onlyone` exports 16 labelled cells;
  - form A is unchanged.

  Pinned by `tests/test_real_form_layouts.py::TestGridCompletion`.
- **Open:** phantom rows from spillover, and crops including the tails of the row
  above (`item_vertical_shift` exists for an A/B).

### 2026-11 · Evaluation chain end to end
- **Where:** cloud. A real `llama-server` (llama.cpp, built from source) with a tiny
  random-weight Qwen3.5 stand-in. HF stand-ins for the transformers families.
- **Checked:**
  - export of both real photos;
  - benchmark over both invoices (36 labelled cells): per-kind and per-invoice
    breakdowns, WER, value accuracy, and peak RAM measured from the server process;
  - `analyze_benchmark.py`: leave-one-invoice-out over 2 folds;
  - `evaluate.py` on both invoices: 23.5 model calls per invoice, per-field table,
    peak RAM of the process plus `llama-server` = 272 MB for the stand-in, and no
    orphaned server afterwards.
- **Accuracy:** none. The stand-ins read garbage by design.

### Pending: needs the laptop
- Re-run the benchmark on both invoices, then the combination analysis:

  ```powershell
  & D:\OCR\.venv_vlm\Scripts\python.exe tools\benchmark_recognizers.py --crops datasets\sahror datasets\onlyone --mode pipeline
  python tools\analyze_benchmark.py evaluation\reports\model_benchmark.json
  ```

- `evaluate.py` with `default_config.yaml` and `nuqta_config.yaml`, on GPU and
  CPU-only.
- Download and benchmark `sherif1313/Arabic-English-handwritten-OCR-v3`.
