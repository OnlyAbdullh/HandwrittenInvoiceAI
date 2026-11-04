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

### 2026-11-04 · Benchmark across both invoices, combination analysis, first invoice evaluation
- **Data:** `sahror` + `onlyone`, 34 cells read by all 8 runnable models.
- **Where:** laptop.
- **Result, per cell:**

  | Model | CER | Exact | s/cell |
  |---|---|---|---|
  | Arabic-Qwen3.5-OCR-v4 | **0.500** | 0.265 | 1.96 |
  | Nuqta Q4_K_M | 0.508 | 0.176 | 3.87 |
  | Nuqta Q8_0 | 0.519 | 0.206 | 6.57 |
  | Qwen2.5-VL-3B 4-bit v3 | 0.562 | 0.147 | 1.75 |
  | Qwen3-VL-2B | 0.666 | 0.206 | 2.79 |
  | Qari / TrOCR models | > 1.3 | — | — |

  The ranking held. Absolute error is far higher than on `sahror` alone (0.143):
  form B's cells are harder. One label was wrong in this run (the `sahror` total,
  see below).
- **Result, combinations, leave-one-invoice-out:**
  - **Routing by kind:** CER 0.585. Worse than the best single model.
  - **Fallback, Qwen3.5 → Qwen2.5-VL-3B below a confidence threshold:** CER 0.468,
    exact 11/34 vs. 9/34. That is two cells, and the learned threshold was
    0.942 on one fold and 0.655 on the other: unstable. The two models together
    also exceed 4 GB of VRAM.
  - **Oracle:** 0.357. Even perfect per-cell choice among these models leaves
    most of the error.
- **Result, calibration:** for every model, **44–53% of all cells were wrong
  readings with confidence ≥ 0.75**. For Qwen3.5, only 8 of its 19 cells
  (42%) at ≥ 0.9 were exact. A model's own confidence is not a safe
  auto-accept signal on these invoices.
- **Result, invoice evaluation:** `nuqta_config.yaml`, before the fixes below:

  | Measure | Result |
  |---|---|
  | Header fields | 4/9 |
  | Numeric cells | 9/24 |
  | Product top-1 | 7/8 |
  | Seconds per invoice | 60 |
  | Peak RAM, process + `llama-server` | 10.2 GB, unsplit; may include the memory-mapped model |

  `default_config.yaml` did not start: `INVOICE_AI_VLM_PYTHON` was not set.
- **Decision:** no combination enabled; Qwen3.5-OCR-v4 stays the default. Do
  not treat a model's confidence ≥ 0.75 as "no review needed" for numbers.

### 2026-11-04 · Why form B's header and total failed, and the date and total parsing
Found by reading the evaluation against the photos (cloud, real photos, no model).

- **Perspective correction cropped form B to its table.** White paper on a
  white cloth leaves no paper edge, so the strongest outline was the table's
  border. Everything outside it was cut away before reading: the invoice
  number, customer, date and the written total. The customer came back as the
  printed column captions. **Fix:** `margin` (default 0.15) keeps a band of the
  photo around the outline. Form A's layout is unchanged apart from the offset
  (checked on the real photo).
- **The form-B profile was tuned on the cropped page.** It now sets
  `printed_header_row: true` and takes the totals from the row after the items.
  Its unruled header (number, customer, date on dotted lines) is declared as
  `unruled_fields` relative to the table. A box closed between the "Total"
  caption and item 1 was taken as a second line total; cells taller than 1.5
  rows are now not item cells. The fixture was re-recorded with
  `tools/record_layout_fixture.py`.
- **Date parsing.** `7/8/226` was parsed as 2022-08-07 and `7/8/2.26` as
  2026-02-08: confident, wrong dates. Patterns may no longer start or end inside
  a longer number. A dot between digits is tried as the dot-shaped zero "٠", and
  only accepted when every reading agrees.
- **Amounts.** `214812,.5` (the paper: ٢١٤٨١٢,٠٥) parsed as 214812.5, and
  `214812..5` as 2,148,125. A dot straight after a separator is now the zero.
- **Benchmark labels:**
  - Amounts were written with `:g`, which keeps six significant digits, so
    214812.05 was labelled "214812".
  - Date cells were never exported or scored.

  Both are fixed. The labelled set is now 41 cells: 21 from `sahror`, 20 from
  `onlyone`, including form B's number, customer, date and total.
- **What these fixes do to accuracy is not measured yet.** Re-run on the laptop
  (below).

### 2026-11-04 · Re-run after the form-B fixes
- **Data:** `sahror` + `onlyone`, re-exported on the laptop: 53 crops, **26
  labelled**. The same two photos give 41 crops, all 41 labelled, in the cloud.
  The laptop run therefore covers mostly `sahror` and only part of `onlyone` (see
  the next entry for why). The per-invoice split is in the report's breakdown.
- **Where:** laptop, CUDA.
- **Result, per cell (26 cells):**

  | Model | CER | Exact | Value acc. | s/cell |
  |---|---|---|---|---|
  | Arabic-Qwen3.5-OCR-v4 | **0.245** | **0.346** | **0.412** | 1.99 |
  | Nuqta Q8_0 | 0.321 | 0.269 | 0.353 | 6.22 |
  | Nuqta Q4_K_M | 0.368 | 0.231 | 0.294 | 3.83 |
  | Qwen3-VL-2B | 0.452 | 0.269 | 0.235 | 3.97 |
  | Qwen2.5-VL-3B 4-bit v3 | 0.562 | 0.192 | 0.176 | 2.12 |
  | TrOCR ×2, Qari | > 1.1 | ≤ 0.04 | 0 | — |

  Not comparable with the 34-cell run: the cell set changed (labels fixed,
  different `onlyone` coverage). The ranking held.
- **Result, combinations, leave-one-invoice-out:**
  - **Routing by kind:** CER 0.481, worse; value accuracy 0.471 vs. 0.412, i.e.
    one more number cell right out of 17. Noise.
  - **Confidence fallback:** every learned threshold was 0, so no fallback
    helped. The report listed three no-op fallback rows. Fixed: such rows are no
    longer listed.
  - **Oracle:** 0.125.
- **Result, calibration:** Qwen3.5 accepted 7 wrong cells at ≥ 0.75 (27% of all
  cells), down from 47% on the earlier cell set. At 0.90–1.00 it was exact on 8
  of 12. Still not a safe auto-accept.
- **Result, invoice evaluation** (laptop photos, at full size, before the next
  entry's fix):

  | Measure | `default_config` (Qwen3.5) | `nuqta_config` |
  |---|---|---|
  | Header fields | 4/9 | 5/9 |
  | Numeric cells | **10/24** | 5/24 |
  | Product top-1 | 5/8 | 6/8 |
  | Phantom / missed rows | 6 / 3 | 1 / 2 |
  | Seconds per invoice | **38.2** | 105.5 |
  | Peak RAM | 3.5 GB (worker 3.0 GB) | 9.9 GB (`llama-server` 9.4 GB) |

  - `onlyone` (form B) with Qwen3.5: 6 phantom rows, no invoice number, and the
    total read as `٥١`. This is a broken layout, not a recognition failure.
  - Dates: the years were misread as `٢٠٤٦`, `٣٠٢٦` and `١٠٢٦` (truth 2026).
    These are now refused as implausible rather than parsed into a wrong date.
    Nuqta read `٢٠٢٦/٨/٧` correctly on `onlyone`.
  - The `sahror` total was read as `۴۱۴۸۱۲۰۵` (Qwen3.5) and `٢١٤٨١٢٫٠` (Nuqta),
    each one digit off 214812.05.
- **Decision:** Qwen3.5-OCR-v4 stays the default (best per cell, 2.8× faster per
  invoice, a third of the RAM). No combination enabled.

### 2026-11-04 · Photo resolution changed the layout
- **Problem:** the laptop's photos are the full-size originals, while the copies
  used for tuning have a long side of 2000 px. Every pixel threshold in the
  geometric and layout stages was tuned at 2000 px.
- **Reproduced in the cloud** by scaling the tuning copies:
  - `sahror` at 2×: 78 boxes, 42 classified, 36 crops (instead of 76/76/21);
  - `onlyone` at 1.5×: 29 crops.
- **Fix:** `working_resolution`, the first geometric step. It shrinks any photo
  whose long side exceeds `preprocessing.working_long_side` (default 2000;
  server-owned; `null` turns it off) and records the transform. It never
  enlarges a photo: enlarging measured worse (`onlyone` at 0.7× enlarged lost its
  item rows; a 1863 px synthetic form A cut 63 cells instead of 57).
- **Result, cloud, real photos rescaled:** 21/21 and 20/20 labelled cells at
  0.7×, 1.5× and 2×. At 3× (an upscaled copy shrunk back, so blurred twice),
  `onlyone` gives 17/17: its last item row is still fragile, a known open item.
  `export_crops.py` prints the scale it applied.
- **Also:** the invoice total may now be corrected to the line sum when it is one
  or two digits off it (not only a misplaced separator). This applies only when
  every row's three numbers were read and multiply out. The correction is flagged
  `MANUAL_REVIEW_REQUIRED` on `total_amount`. A stale `TOTAL_MISMATCH` from before
  reconciliation is re-evaluated.
- **Not measured yet** with the real models.

### 2026-11-04 · Re-run at the working resolution (41 of 41 cells)
- **Data:** `sahror` + `onlyone`, 41 labelled cells, all read by 8 models.
  The resolution fix worked on the laptop: 41 crops, 41 labelled.
- **Where:** laptop, CUDA.
- **Result, per cell:**

  | Model | CER | Exact | Value acc. (numbers, dates) | s/cell |
  |---|---|---|---|---|
  | Nuqta Q8_0 | **0.245** | 0.293 | 0.321 | 4.86 |
  | Arabic-Qwen3.5-OCR-v4 | 0.276 | 0.293 | **0.393** | **0.92** |
  | Nuqta Q4_K_M | 0.279 | 0.293 | 0.321 | 2.70 |
  | Qwen2.5-VL-3B 4-bit v3 | 0.423 | 0.244 | 0.250 | 0.99 |
  | Qwen3-VL-2B | 0.561 | 0.293 | 0.250 | 1.66 |
  | TrOCR ×2, Qari | > 1.3 | ≤ 0.02 | 0 | — |

- **Result, combinations, leave-one-invoice-out:**
  - **Routing by kind:** 0.315. Worse.
  - **Fallback, Qwen3.5 → Nuqta Q8:** CER 0.260, but value accuracy fell
    0.393 → 0.357, and the thresholds were unstable (0.643 / 0.998).
  - **Oracle:** 0.081.
- **Result, calibration:** 39% of Qwen3.5's cells were wrong at confidence ≥ 0.75;
  32–34% for Nuqta.
- **Result, invoice evaluation,** `default_config.yaml`:
  - header 4/9, numeric 12/24, product top-1 **8/8**;
  - rows 8/8 matched, 3 phantom, 0 missed;
  - **17.2 s per invoice** (38 s before: the photos are now read at 2000 px);
  - 3.6 GB peak RAM.

  Failures by field:
  - **Dates:** both years misread (`٢٠٤٦`, `٣٠٢٦`), so null.
  - **Totals:** `٢١٤٨٢٠٥` on `sahror` (a digit dropped; the rows did not all
    verify, so it was not corrected); `3148150 … ٥٠١` on `onlyone`, read from the
    whole totals strip.
  - **Invoice number:** `0001901 001` on `onlyone`.
- **Decision:** Qwen3.5-OCR-v4 stays the default.
  - Nuqta Q8's lead is 0.03 CER on 41 cells, about one cell's worth.
  - Qwen3.5 reads more numbers exactly (0.393 vs. 0.321), and numbers are what
    the arithmetic checks need.
  - Qwen3.5 is 5× faster per cell. Nuqta Q8 also needs more VRAM than the 4 GB
    card has, so it is partly on the CPU.

  Revisit when more invoices are labelled.

### 2026-11-04 · Form B crops: what the model was actually shown
Inspected every number crop of both invoices (cloud, real photos, no model).
Form A's crops are clean: its errors are the handwriting itself (the dot-shaped
zero, the year). Form B's crops had three problems:

- **Item cells.** The writing sits on each row's dotted rule, so the printed
  cell held the tails of the row above and cut off the bottom of its own digits.
  Compared `item_vertical_shift` at 0, 0.2, 0.3 and 0.4: at 0.2–0.3 every crop
  holds its own digits whole and nothing else; at 0.4 the row below starts to
  appear. **Set to 0.25** in the form-B profile.
- **Total.** The crop was the whole totals strip: "المجموع", a dotted line,
  "فقط لا غير" and the edge of the row above. New `bill_layout` option
  `field_insets` narrows a classified box to where the writing is, as fractions
  of that box. Form B: `total_amount: {x: [0.22, 0.84], y: [0.18, 1.0]}`.
- **Date.** Its box overlapped the bottom of the invoice-number line and clipped
  the "التاريخ" caption. Tightened: date `y: [-0.055, -0.002]`, `x: [-0.06, 0.235]`;
  invoice number `y: [-0.11, -0.058]`.

`evaluate.py` now lists every wrong quantity, price and line total with the raw
reading, so the next run shows whether a number was misread, lost its separator,
or came from a row the layout missed.

**Not measured yet** with the models.

### 2026-11-04 · After the form-B crop fixes
- **Where:** laptop, CUDA. 41 labelled cells.
- **Result, per cell:**

  | Model | CER | Exact | s/cell |
  |---|---|---|---|
  | **Arabic-Qwen3.5-OCR-v4** | **0.189** (from 0.276) | **0.366** | 1.05 |
  | Nuqta Q8_0 | 0.259 | 0.317 | 4.74 |
  | Nuqta Q4_K_M | 0.280 | 0.317 | 2.68 |
  | Qwen2.5-VL-3B 4-bit v3 | 0.298 (from 0.423) | 0.293 | 0.95 |

  Qwen3-VL-2B and Qari crashed natively this run (access violation); the
  benchmark isolates that.

  The crop fix helped every model on form B. Qwen3.5 is again clearly first.
- **Result, invoice evaluation:**
  - header 5/9 (the `onlyone` date is now right);
  - numeric 12/24;
  - product top-1 8/8;
  - **11 phantom rows** on `onlyone`, and 66 model calls instead of about 23.
- **The per-number table showed why each number failed** (raw readings):
  - **Decimal comma read as a zero:** `٥٠٧` for 5.7 and `٥٠٩٥` for 5.95.
  - **Dot-shaped zero read as a separator:** `1.٩٤٤` for 1094.4.
  - **Dot-zeros counted short:** `٢١٢٠٠` for 212000, `١٢٠` for 1200.
  - **A wrong decimal point inserted:** `١٣٢٥٠` became 1325.0, because 16 ×
    1325.0 = 21200 matched the short total.
  - **Plain misreads:** `٥٠` for 200, `٢٠١ × 6 = ١٢٠٦` for 200 × 6 = 1200.

### 2026-11-04 · Phantom rows, number repair, invoice-number sign
Cloud, real photos, with the laptop's readings replayed. Not yet measured with a
model.

- **Phantom rows: the blank-cell gate counted the dotted rule.** Moving form B's
  item boxes down put each row's dotted rule, and the table's bottom rule, inside
  the measured interior. All 11 empty rows then looked written. The gate now
  counts only *strokes*: ink left after straight rules are removed by
  morphological opening, in pieces at least 20% of the cell's height (capped at
  14 px).
  - Measured, both photos: empty cells ≤ 0.0064, the faintest written cell
    0.022, against the 0.01 threshold.
  - `onlyone` now sends no empty row.
- **Perfect-readings check** (`evaluate.py --perfect-readings`): with every
  cell read as its label, both invoices come out 100% right (9/9 header, 24/24
  numbers, 0 phantom rows). Layout, parser and arithmetic are sound, and every
  remaining error is recognition. Three unclassified boxes on form B are still
  read (3 model calls); their readings do not affect the result.
- **Number repair**, replaying the 8 rows the laptop read:

  | | Before | After |
  |---|---|---|
  | Rows right | 2 | **6** |

  - New candidate values from the misreads above:
    - an interior ٠ may be the decimal comma;
    - a dot between digits may be a zero;
    - a reading ending in zeros may have lost one.
  - A reading without a separator never gets a decimal part of zeros, so
    `13250` is never 1325.0.
  - When no combination fits, every cell is tried as the misread one, not just
    the longest. The closest unique explanation wins, and the cells it used are
    written back too.
  - A correction that changes digits, not just the separator, raises
    `MANUAL_REVIEW_REQUIRED` on that cell: the row now adds up, so nothing else
    would flag it.
  - Still wrong: `٥٠` for 200 becomes 20 (one digit off its reading, flagged), and
    `٢٠١ × 6 = ١٢٠٦` is self-consistent, so arithmetic cannot see it. With
    perfect readings, nothing changes.
- **Invoice number:** a printed number sign (`№`, read as `N/`) is stripped:
  `N/ 0001901` → `0001901`.

### Pending: needs the laptop
- Re-export both invoices at the working resolution (`export_crops.py` should
  now print `read at ...x2000` and 41 labelled cells), re-run the benchmark on
  both invoices, then the combination analysis:

  ```powershell
  & D:\OCR\.venv_vlm\Scripts\python.exe tools\benchmark_recognizers.py --crops datasets\sahror datasets\onlyone --mode pipeline
  python tools\analyze_benchmark.py evaluation\reports\model_benchmark.json
  ```

- `evaluate.py` with `default_config.yaml` and `nuqta_config.yaml`, on GPU and
  CPU-only.
- Download and benchmark `sherif1313/Arabic-English-handwritten-OCR-v3`.
