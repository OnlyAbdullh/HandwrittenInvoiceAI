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

### 2026-11-04 · Number repair measured; the invoice total as a fourth equation
- **Laptop result, `default_config.yaml`:** exactly what the replay predicted.
  - **Numbers 20/24** (from 12/24);
  - **header 6/9** (both invoice numbers right now);
  - phantom rows **0** (from 11);
  - 22 model calls per invoice, **14.7 s per invoice**, 3.75 GB peak RAM.
- **Left:** both totals, the `sahror` year (`٣٠٢٦`), and row 2 on both invoices
  (6 × 200 = 1200 read as `٥٠`/`١٢٠` and as `٢٠١`/`١٢٠٦`).
- **Change (cloud, laptop readings replayed):**
  - **The invoice total as a fourth equation.** Each row that does not check out
    is tried as *the* wrong one, with the others at their repaired values. Its
    line total must agree with its own reading (as misread in the known ways)
    *and* make the invoice total agree with the total's reading (≤ 2 digits).
    Only a unique answer is written.
    - `sahror`: row 2 → 6 × 200 = 1200, and the total `٢١٤٨٢٠٥` → 214812.05.
    - `onlyone`: its total reading `٥١٤٨١٤٠٠` is too far from any sum, so
      nothing changes. Its row 2 (`٢٠١ × 6 = ١٢٠٦`) multiplies out as misread;
      no check here can see it.
  - **Year repair.** An implausible year is fixed only when exactly one
    single-digit change is plausible: `٣٠٢٦` → 2026, `١٠٢٦` → 2026. `٢٠٤٦` stays
    null (2006, 2016 and 2026 all fit).
  - **Corrections are visible to the app.** There is no warnings array on the
    wire (`docs/api-contract.md`), so every corrected or derived value now
    carries `ocr_confidence` 0.5, below the 0.75 review threshold:
    - digit corrections, the row-from-total solve and the total correction;
    - the year repair;
    - a cell derived from the other two in its row.

    A separator-only fix keeps the model's confidence. Before this, a
    corrected value went out at the model's own (often high) confidence.
  - **Bugs found on the way, fixed:**
    - a value derived earlier in the stage was later treated as a reading;
    - a one-cell repair accepted a row that only multiplied out after rounding;
    - "taken as read" ignored a legible separator (`٥,٩٥` argued for 595).
- **Replay of the laptop readings:** `sahror` **12/12** numbers and the total
  right; `onlyone` 10/12. Expected on the laptop: numbers 22/24, header 8/9
  (the onlyone total remains). Perfect readings still give 100%.

### 2026-11-04 · The invoice total as a fourth equation, measured
- **Laptop, `default_config.yaml`:** exactly as the replay predicted.
  - Numbers **22/24** and header **8/9**.
  - `sahror` is **complete**: every field right. It is the first complete invoice.
  - 14.4 s per invoice, 2.9 GB peak RAM.
- **Left (onlyone):**
  - the total, read as `٥١٤٨١٤٠٠`;
  - row 2, read as `٢٠١ × 6 = ١٢٠٦`, which multiplies out as misread.

  Both are recognition errors no check here can see.

### 2026-11-04 · Every reading approach made comparable
The user asked why Surya + Qwen is not used. Before measuring it, the two Surya
configs had to stop being handicapped:

- **`qwen_detector_config.yaml`** had:
  - binarized the page before Qwen read it (measured to wreck handwriting);
  - declared column fractions measured on form A only, so form B could not work;
  - none of the geometry, margin and crop fixes.
- **`surya_config.yaml`** was in the same state.

Both now copy the default's preprocessing and layout, and a test pins that.

**Detector flow, `assign_to_layout: true`:** each Surya box is clipped to the
classified cells it overlaps. Each piece takes that cell's content kind (digit
prompt and grammar for numbers) and its id. A box across two columns is split at
the rule, and printed captions are dropped. What Surya then changes is only the
crop: tight around the ink instead of the whole cell. A cell where it found no ink
is not read; that is the known weakness, now measurable.

`tools/compare_pipelines.py` runs all four approaches and ranks them. An
optional component that failed to import now says what it needs ("needs
surya-ocr"), not just "unknown".

**Not measured:** Surya cannot run in the cloud environment (no access to
Hugging Face for its models), so both Surya approaches are untested here beyond
config loading and the box-assignment logic.

### 2026-11-04 · All four reading approaches, same invoices (`compare_pipelines.py`)
- **Where:** laptop. Qwen on CUDA; Surya on the CPU (`TORCH_DEVICE=cpu`, since
  the 4 GB GPU holds Qwen). Same preprocessing and layout for all four.
- **Result:**

  | Rank | Approach | Complete | Header | Numbers | Phantom / missed | s / invoice | Peak RAM MB |
  |---|---|---|---|---|---|---|---|
  | 1 | **Grid cells → Qwen3.5-OCR-v4** (`default_config`) | **1/2** | **8/9** | **22/24** | 0 / 0 | **14.5** | **2888** |
  | 2 | Surya boxes → cells → Qwen3.5 (`qwen_detector_config`) | 0/2 | 5/9 | 10/24 | 0 / 0 | 25.6 | 5180 |
  | 3 | Grid cells → Nuqta Q4 (`nuqta_config`) | 0/2 | 5/9 | 8/24 | 0 / 0 | 59.5 | 9365 |
  | 4 | Surya alone (`surya_config`) | 0/2 | 3/9 | 0/24 | 0 / 0 | 63.7 | 4827 |

  Product top-1 was 8/8 for all four, even Surya alone. The evaluation catalog has
  only 9 products, so any reading matches the right one. It does not separate
  approaches on this data.
- **Reading it:**
  - **Surya + Qwen** reads with the same model as rank 1, so the whole gap (22 →
    10 numbers, 8 → 5 header) comes from Surya's crops: tight boxes around the
    ink it found, instead of the whole cell. It is also 1.8× slower and uses
    1.8× the RAM.
  - **Surya alone** does not read handwritten Arabic digits at all (0/24).
  - **Nuqta Q4** in the full pipeline reads numbers far worse than Qwen3.5
    (8 vs. 22), and is 4× slower with 3× the RAM.
- **Decision:** grid cells → Arabic-Qwen3.5-OCR-v4 stays the default. It is first
  on every measure, accuracy, speed and memory, on both invoices. Surya is not
  used for known forms.

  The one role left open for Surya is reading forms the classifier does not
  recognise (`NO_LAYOUT_DETECTED` today). That is untested: no such invoice is
  labelled. Revisit only with one.

### 2026-11-11 · Segmentation, boxes, preprocessing, prompts and combinations, 68 configurations
Full write-up, every number and the scripts: `experiments/REPORT.md`,
`experiments/results.csv`.

- **Data:** `sahror` + `onlyone`, the same 41 labelled cells. Two invoices of one
  transaction; one cell is 2.4 points; no held-out set.
- **Where:** laptop. Qwen3.5, Qwen2.5-VL and Nuqta on CUDA, Surya detection on
  the CPU.
- **Method:** every cell cut under a crop recipe and read with the service's
  prompts (readings cached by crop bytes); then each configuration's readings
  replayed through the real parser and arithmetic repair
  (`experiments/e2e_replay.py`). The replay of the default matches its real run
  field for field.
- **Result, per cell (before arithmetic repair):**

  | What changed | Cell accuracy |
  |---|---|
  | Default: grid cell, gray, pad 10, ×2, Qwen3.5 | 0.37 (15/41) |
  | Full page / table rows to Qwen3.5 | ≤ 0.17 / ≤ 0.25, scored leniently |
  | No enlargement / ×3 / ×4 | 0.27 / 0.34 / 0.32 |
  | Pad 0, 4, 20 / pad 30 | 0.34 to 0.39 / 0.24 |
  | Surya boxes / ink-tight boxes | 0.27 / 0.32 to 0.44 (name CER up to 0.57) |
  | CLAHE, sharpen, NLM / Otsu / adaptive threshold | 0.37 / 0.29 / 0.22 |
  | Model-card prompt / zero-and-comma hint | 0.29 / 0.37 |
  | Nuqta Q4 or Q8 / Qwen2.5-VL, same crops | 0.32 / 0.29 |

- **Result, end to end:** the default is the best of the 37 cell-level Qwen3.5
  configurations (header 8/9, numbers 22/24). Variants two or three cells better
  per cell are worse end to end (16 to 19 of 24 numbers): the arithmetic repair
  was written from the default's misreadings.
- **Result, combinations:** voting, confidence fallback and a disagreement
  referee are at or below the default end to end. Qwen3.5's confidence has an
  AUROC of 0.83, with 9 wrong readings at ≥ 0.90. Qwen3.5 and Nuqta agree on 8
  cells and all 8 are right. The oracle over four models is 0.59.
- **Routing by kind** (numbers Qwen3.5, Arabic text Nuqta Q4), real run of
  `config/routed_config.yaml`: the same header and numbers, name CER 0.18 → 0.11,
  products accepted without review 5/8 → 7/8, one customer name down to 0.71;
  143 s per invoice against 27, 10.2 GB against 3.6. Three name cells of 11,
  p = 0.38.
- **Layout defect:** form B's first item row had two boxes labelled line total;
  the 140 px fragment cut the first digit off 212000. New `bill_layout` option
  `one_cell_per_column` (default off). No end-to-end change on these photos.
- **Decision:** `default_config.yaml` stays the default. `routed_config.yaml` is
  shipped as an option, not recommended until measured on more names.
  `one_cell_per_column` is on only there. Nothing else adopted.
- **Left wrong (onlyone):** the total, and row 2's `200` and `1200` read `٢٠١`
  and `١٢٠٦`. No Qwen3.5 crop reads any of them right.

### 2026-11-14 · Product matching against the real catalog: ranking and acceptance separated

Report: `evaluation/reports/20261114-product-matching.md` (and `.json`), from
`tools/evaluate_matching.py --catalog keywords.xlsx --readings experiments/predictions
--add-missing-products --simulate`. No model runs; it repeats exactly.

- **Data:** `keywords.xlsx`, the real product catalog: 341 names, 335 active (6
  contain `الغاء`). The earlier evaluations used a 9-product catalog, on which
  product top-1 was 8/8 for almost everything and separated nothing.
- **What the catalog is like** (`tools/check_catalog.py keywords.xlsx`): median 6
  words a name; 226 names have another name scoring ≥ 0.75 against them under the
  original matcher; 21 groups differ only in their numbers; in 42 pairs one name's
  words are all contained in the other's; one name is there twice (`قشاط 4 سم صيني` = `قشاط صيني 4سم`).
- **Real readings:** the 344 product-name readings already cached by the
  2026-11-11 experiments (four models, 68 crop recipes), four products. Those four
  are not in the catalog, so they were added to it. They have no look-alikes there,
  which is why real data cannot yet rank matchers.
- **Simulated readings:** every active name written twice, in full and shortened,
  misread at the rates measured on 396 real (label, reading) pairs: average letter
  error 14.4%, concentrated in ب, ش, ت, س, م (roughly a quarter to two fifths of
  them misread) while ا, و, ر, ك, ط, ز are rarely wrong; a word split in two at
  about one word boundary in ten, never two joined.
  **Assumed, not measured:** digits and Latin letters are misread at the average
  rate; shorthand is a random subset of the words. Real writers choose what to drop.
- **Matchers:** `word_overlap` (the original: uniform word overlap over the larger
  word count, accepted at score ≥ 0.75) and `coverage` (new default).

  | Written form | Matcher | Top-1 | In top 5 | Accepted | Wrong accepts | Accepted with the intended product removed |
  |---|---|---|---|---|---|---|
  | Real, 68 crop recipes (344) | word_overlap | 96.5% | 98.3% | 60.8% | 3 | 3 |
  | | coverage | 93.9% | 96.5% | 74.1% | 3 | 3 |
  | Real, the service's own crops (32) | word_overlap | 32/32 | 32/32 | 24 | 0 | 0 |
  | | coverage | 32/32 | 32/32 | 28 | 0 | 0 |
  | Full name (670) | word_overlap | 97.8% | 100% | 65.8% | 5 | 17.8% |
  | | coverage | 97.3% | 100% | 57.5% | 0 | 1.6% |
  | No numbers (336) | word_overlap | 40.8% | 94.6% | 10.7% | 19 | 7.1% |
  | | coverage | 75.0% | 97.6% | 0% | 0 | 0% |
  | Half the words (648) | word_overlap | 39.4% | 76.5% | 0.9% | 6 | 0.9% |
  | | coverage | 67.3% | 94.1% | 1.5% | 7 | 1.9% |
  | Three words (562) | word_overlap | 29.5% | 60.9% | 2.1% | 7 | 1.2% |
  | | coverage | 61.2% | 88.1% | 0.9% | 5 | 0.9% |
  | Two words (648) | word_overlap | 18.2% | 47.4% | 0.3% | 2 | 0.3% |
  | | coverage | 46.0% | 77.0% | 1.5% | 3 | 0.9% |

- **Acceptance was the defect, not ranking.** With the intended product removed
  from the catalog — a product not entered yet — the original matcher accepts a
  sibling for 17.8% of full-name readings, and 41% at half the misreading rate: a
  cleaner reading scores higher against the wrong sibling. `coverage` accepts 1.6%.
  Its price is fewer rows accepted unseen (57.5% against 65.8%).
- **Shorthand is where ranking changes.** Dividing by the larger word count buries
  the long product a short name was written for. Before any misreading, the words
  kept by "half" are contained in exactly one name for 51% of readings, in 2–5 for
  34% and in more than five for 15%: half of shortened names cannot be resolved
  from the text by any matcher.
- **Parameters** (`coverage_weight`, `word_threshold`), swept on the same sets:
  weight 0.3 / 0.5 / 0.7 changes full-name ranking by under half a point and
  half-name top-5 from 86.6% to 91.7%, so 0.7. A word threshold just above 0.5
  instead of 0.5 cuts full-name acceptance from 57.5% to 44.0% for a look-alike
  rate of 0.9% instead of 1.6%, so 0.5. The sweep ran before numbers were ranked
  by closeness (half-name top-5 is 94.1% now), and both values were chosen on the
  sets they are reported on.
- **Tried and dropped:** weighting rare words more (IDF) — worse full-name
  ranking, no gain on shorthand. Hand-set cheaper substitutions between letters of
  the same shape — worse ranking on real and simulated readings. Folding written
  words to a canonical spelling, and crediting a joined word pair per word — both
  wrong, both found on the real readings (CLAUDE.md, Traps).
- **Where `coverage` is worse:** on the 344 real readings it ranks 9 fewer first.
  Of its 21 misses the original also misses 11 (the crop shows another row). Of
  the other 10, 5 put the catalog's own `بنطال كتان جيب كبير` above the added
  label `بنطلون كتان كبير`, which the variant table makes near-identical; 4 are
  readings holding two rows or leaked columns; 1 is a plain loss. It tolerates
  stray extra text less than the original does. On the crops the service
  actually produces, neither matcher misses.
- **Left open:** the 7 wrong accepts on half-names are shortened names that are
  themselves another product's full name (`قشاط محير` / `قشاط مطاط محير`). A full
  name is accepted even when longer names also fit; whether that is right depends
  on how writers shorten, which only real invoices show.
- **Decision:** `coverage` is the default; `word_overlap` stays registered as the
  baseline and rollback. Thresholds stay provisional until 20–30 invoices with
  real shorthand are labelled.
- **Time:** 19 ms per reading against 50 ms, pure Python, on the laptop.

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
