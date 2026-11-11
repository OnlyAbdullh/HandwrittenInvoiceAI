# Which pipeline reads these invoices best

Measured 2026-11-10 and 2026-11-11 on the laptop (RTX 3050 4 GB, Ryzen 5 5600H, 20 GB RAM).

## Result

**The shipped default stays the recommended pipeline:** the printed grid's cells,
each read by Arabic-Qwen3.5-OCR-v4 from a grayscale crop with 10 px of padding,
enlarged twice (`config/default_config.yaml`). Of 68 configurations, none beats
it by more than chance on the 41 labelled cells, and none beats it end to end on
the fields the app needs right.

| Pick | Pipeline | Config | Header | Numbers | Products accepted without review | Name CER | s / invoice | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| Best balance, and simplest | grid cells → Qwen3.5 | `default_config.yaml` | 8/9 | 22/24 | 5/8 | 0.18 | 26.9 | 3.6 GB |
| Highest accuracy, by a margin within noise | grid cells → Qwen3.5 for numbers, Nuqta Q4 for Arabic text | `routed_config.yaml` | 8/9 | 22/24 | 7/8 | 0.11 | 143.2 | 10.2 GB |

Both rows are real runs of `tools/evaluate.py` on both invoices. The routed
pipeline gets the same header fields and numbers right and reads names better,
for 5.3 times the time and 2.8 times the memory. Its gain is 3 name cells of 11
(exact McNemar p = 0.38), and one customer name dropped below the 0.75 review
threshold with it (0.71, against 0.93 with the default). It is implemented and
selectable, not recommended as the default.

## What this evaluation can and cannot show

- **Two invoices, one transaction.** `sahror` (form A) and `onlyone` (form B)
  carry the same four items, amounts, date and total. There are 41 labelled
  cells: 26 numbers, 11 names, 2 dates, 2 invoice numbers.
- **One cell is 2.4 points.** The baseline's 15/41 has a 95% interval of 0.24
  to 0.52. A difference of two or three cells is noise.
- **No held-out set.** Every choice was scored on the cells it was made on,
  except where a leave-one-invoice-out result is given.
- **About 60 comparisons were made.** Two reach p < 0.05 unadjusted (adaptive
  thresholding, full page). Neither would survive a correction for that many tests.
- **The baseline's end-to-end score is partly in-sample.** The arithmetic repair
  rules in `invoice/reconciliation.py` were written from this model's misreadings
  of these two invoices. Crops that read differently lose that fit, which is why
  several variants that look better per cell are worse end to end.
- **Vocabulary is tiny.** The catalog has 9 products, so product top-1 is 8/8 for
  almost every configuration and separates nothing.

The ordering of the large effects (cells over rows over page, no binarization)
is consistent across both invoices and every field type. The ordering among the
close configurations is not established.

## What was measured, and how

- **Models:** Arabic-Qwen3.5-OCR-v4 (transformers, float32), Nuqta-Arabic-OCR-4B
  GGUF Q4_K_M and Q8_0 (llama.cpp), Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3,
  Surya (text detection only; its recognition was measured earlier at 0/24 numbers,
  `evaluation/reports/pipeline_comparison.md`, not re-run).
- **Per cell:** each labelled cell is cut under a crop recipe and read with the
  service's own prompts and constrained decoding. `base` is byte-identical to
  the shipped cropper's output.
- **End to end:** each configuration's readings are replayed through the real
  mapper, parser, catalog matching and arithmetic repair (`e2e_replay.py`) and
  scored as `tools/evaluate.py` scores the wire response. The replay of the
  baseline and of the routed pipeline each match their real run field for field.
- **Metrics** (`results.csv`): cell accuracy (a number or date is right when its
  value is; text when it matches after the pipeline's normalization), exact
  match normalized and raw, CER and WER normalized, CER raw, per-field accuracy,
  row and invoice accuracy, empty rate, seconds; and the end-to-end columns.
- **Raw CER is high for every model that writes Arabic-Indic digits** (0.77 to
  0.90), because the labels are in Western digits and include no caption. It is
  reported so normalization hides nothing; it does not rank models. Qwen2.5-VL
  writes Western digits and scores 0.41 raw while reading fewer values right.
- **Seconds per cell in `results.csv` are approximate**: the first reading of
  each process includes warm-up and other work shared the CPU. The timings to
  rely on are the real runs above.

## Pipelines

Cell accuracy is before arithmetic repair. End-to-end columns are after it.

| ID | Pipeline | Cell accuracy | Numbers | Name CER | End to end: header, numbers |
|---|---|---|---|---|---|
| P1 | Qwen3.5 on the full page (768, 1280, 1536 px) | 0.05 to 0.17 * | 0.08 to 0.23 | n/a | not run |
| P2 | Qwen3.5 on table rows | 0.09 to 0.25 * | 0.13 to 0.33 | n/a | not run |
| P3 / P6 | **Grid cells → Qwen3.5 (baseline)** | **0.37** | **0.46** | 0.18 | **8/9, 22/24** |
| P4 | Surya boxes clipped to cells → Qwen3.5 | 0.27 | 0.31 | 0.24 | 5/9, 13/24 |
| P5 | Surya boxes clipped to cells → Nuqta Q4 | 0.22 | 0.19 | 0.15 | 5/9, 5/24 |
| P7 | Grid cells → Nuqta Q4 (Q8 the same) | 0.32 | 0.31 | 0.11 | 5/9, 8/24 |
| — | Grid cells → Qwen2.5-VL-3B | 0.29 | 0.27 | 0.15 | 6/9, 6/24 |
| P8 | Adaptive row/cell reading | not built | | | |
| P9a | Qwen3.5 and Nuqta, keep the more confident | 0.44 | 0.46 | 0.11 | 7/9, 22/24 |
| P9b | Qwen3.5, re-read by Nuqta Q4 below a confidence threshold | 0.42 | 0.46 | 0.13 | 7/9, 21/24 |
| P9c | **Routed by kind: numbers Qwen3.5, text Nuqta Q4** | 0.44 | 0.46 | 0.11 | 8/9, 22/24 |
| P10a | Vote of Qwen3.5, Nuqta Q4, Qwen2.5-VL | 0.39 | 0.46 | 0.16 | 7/9, 16/24 |
| P10d | Vote of three Qwen3.5 crop views, views chosen on the other invoice | 0.37 | 0.46 | 0.21 | 8/9, 22/24 |
| P10f | Qwen3.5 vs Nuqta, a third model settles disagreements | 0.39 | 0.46 | 0.16 | 7/9, 16/24 |
| — | Oracle: best of four models per cell, in hindsight | 0.59 | 0.54 | 0.09 | upper bound |

\* Rows and pages are scored leniently: a value counts when it appears anywhere
in the transcription, whatever its position.

P8 was not built because its premise failed: rows were worse than cells for
every field type, and widening a cell into its neighbours (`expand_x25`,
`expand_y25`, `expand_y40`, `pad30`) lowered accuracy each time.

## Which stage matters most

**1. Segmentation, by far.** Cells 0.37; rows at most 0.25; full page at most
0.17. The page transcription invents a table (`<table>` markup, repeated
"2006" cells for the empty rows). Against the baseline the full page loses 10
cells and gains 2 (p = 0.04); rows lose 5 and gain none (p = 0.06).

**2. The reading model.** On the same crops Qwen3.5 reads 12/26 numbers, Nuqta
8/26, Qwen2.5-VL 7/26. For names it is the reverse: Nuqta 4/11 exact with CER
0.11, Qwen3.5 1/11 with CER 0.18. End to end the numbers decide it: Nuqta alone
gets 8/24, Qwen2.5-VL 6/24.

**3. Resolution, up to a point.** No enlargement 0.27; ×2 0.37; ×3 0.34;
×4 0.32; a fixed 128 or 256 px height 0.39. Enlarging twice is needed; more
buys nothing measurable. Bicubic instead of Lanczos: 0.34.

**4. Bounding boxes: the grid cell with moderate padding is right.**

| Box | Cell accuracy | Name CER | End to end numbers |
|---|---|---|---|
| Cell + 10 px (baseline) | 0.37 | 0.18 | 22/24 |
| Cell + 0 / 4 / 20 px | 0.34 / 0.34 / 0.39 | 0.17 / 0.18 / 0.18 | 15 / 14 / 16 |
| Cell + 30 px | 0.24 | 0.58 | 9/24 |
| Cell − 6 px (inside the rules) | 0.34 | 0.39 | 14/24 |
| Neighbours included, 25% wider / taller | 0.32 / 0.34 | 0.58 / 0.18 | 11 / 18 |
| Tight around the ink + 8 / 16 / 24 px | 0.32 / 0.44 / 0.34 | 0.30 / 0.51 / 0.57 | 13 / 19 / 18 |
| Surya's boxes, clipped to the cell | 0.27 | 0.24 | 13/24 |

Tight boxes help some number cells on form A (the rules are gone, so the dot
zeros are counted right) and wreck names, where the ink search follows strokes
into the next row. Surya found no text at all in 2 of the 41 cells and cut
form B's figures where they cross a rule.

**5. Preprocessing: none helps.** Plain grayscale 0.37. CLAHE 0.37, sharpening
0.37, non-local-means 0.37, contrast stretch 0.42, colour 0.32, bilateral 0.32,
median 0.34, rule removal 0.32, blur-divide 0.32, Otsu 0.29, adaptive threshold
0.22 (loses 6 cells, gains none, p = 0.03). Contrast stretch is two cells up per
cell and six numbers down end to end (16/24).

**6. Prompts and decoding: no gain.** The model card's own prompt 0.29; short
prompts 0.29; a number prompt explaining the dot-shaped zero and the decimal
comma 0.37 with identical readings on all but two cells; unconstrained decoding
0.37 per cell but one header field lost end to end.

## Combining models

- **Routing by kind is the only combination that does not lose anything end to
  end.** It keeps Qwen3.5's numbers and takes Nuqta's names.
- **Confidence is informative but not safe.** Qwen3.5's confidence ranks a right
  reading above a wrong one 83% of the time (Nuqta 85%, Qwen2.5-VL 75%). Yet 16
  of Qwen3.5's 26 wrong readings have confidence ≥ 0.75 and 9 have ≥ 0.90.
  In the 0.90 to 0.99 band it is right 4 times in 10.
- **Confidence fallback raises cell accuracy and lowers the invoice.** Thresholds
  learned on the other invoice were 0.85 and 0.79. Per cell 0.42; end to end one
  header field and one number lost.
- **Agreement between two models is precise and rare.** Qwen3.5 and Nuqta Q4
  agree on 8 of 41 cells and all 8 are right. They disagree on 33, where Qwen3.5
  is right 7 times and Nuqta 5. Agreement can mark a cell as safe; it cannot
  pick a winner.
- **A model agreeing with itself means little.** Qwen3.5 on two crops of the same
  cell agrees 22 times and is right in 13 of them.
- **Voting across models damages numbers** (16/24 end to end): the two weaker
  models outvote the stronger one.
- **Even perfect selection leaves 41% of cells wrong.** No combination of these
  models reaches the accuracy a better reader would.

## Errors

Baseline, 26 wrong cells of 41 before arithmetic repair (`predictions/error_analysis.json`):

| Category | Cells | Example (label → reading) |
|---|---|---|
| Wrong Arabic characters (letters, dots) | 9 | `حسام المارديني` → `صسام المارديني` |
| Wrong digit | 7 (6 numbers, 1 date) | `200` → `٢٠١`; `214812.05` → `٥١٤٨١٤٠٠` |
| Decimal comma read as a zero | 3 | `5.7` → `٥٠٧` |
| Dot-shaped zeros counted short | 3 | `212000` → `٢١٢٠٠` |
| Separator missing, or a printed sign kept | 2 | `5.95` → `٥٩٥`; `0001901` → `N/ 0001901` |
| Box cut the figure | 1 | `212000` → `١٢٠٠` |
| Hallucinated text | 1 | `باراشوت ملون` → `جا رانيت مولون` |

The `N/` case is a per-cell scoring artifact: the service strips the printed
number sign, and both invoice numbers are right end to end.

Numbers account for 14 of the 26, and 7 of those are the handwriting's own
conventions (a dot for zero, a comma for the decimal point), which is what the
arithmetic repair then fixes: 12/26 numbers right as read, 22/24 after repair.
Nuqta's and Qwen2.5-VL's errors are mostly plain wrong digits (17 and 15 cells),
which arithmetic cannot repair.

**The stage that causes the most remaining error is recognition.** An earlier
check recorded in `docs/experiment-log.md` (`evaluate.py --perfect-readings`,
not re-run here) returned both invoices complete when every cell was read as
its label, so layout, parsing and arithmetic added no error on these two photos.

**Three fields are still wrong end to end, all on form B**, and no Qwen3.5 crop
variant reads any of them right:

- the invoice total, read `٥١٤٨١٤٠٠` (Qwen2.5-VL reads it correctly);
- row 2's unit price, `200` read `٢٠١` (Nuqta Q4 reads it correctly);
- row 2's line total, `1200` read `١٢٠٦`. The row multiplies out as misread
  (201 × 6 = 1206), so no check can see it.

On form B the writing sits on a dotted rule, and a dot zero beside the rule's
own dots is read as a digit.

## A layout defect found on the way

Form B's first item row has two boxes labelled line total on the laptop's
photo: the printed cell (215 px wide) and a fragment the handwriting closed
inside it (140 px). The parser keeps whichever it meets last. The fragment's
crop shows `١٢٠٠٠`, the first digit of 212000 cut off, and the cell benchmark's
label sat on that fragment. Read from a crop widened to the cell's full extent
the same model returns `٢١٢٠٠٠`.

`bill_layout` now has `one_cell_per_column` (off by default): at most one cell
of an item row keeps each handwritten column's role, the one nearest the
column's usual width. The end-to-end result is unchanged with it on, because
the arithmetic already derived that total from 16 × 13250. It is on in
`routed_config.yaml`. Widening every item cell to its column's extent
(`colfix`) fixed that cell too but pulled neighbouring ink into two others
(20/24 end to end), so it was not adopted.

## What was implemented

| Piece | Where | Baseline affected |
|---|---|---|
| Routed pipeline, selectable by config | `config/routed_config.yaml`, using the existing `ocr/recognizers/routed_recognizer.py` | no |
| `one_cell_per_column` layout option | `table_extraction/classifiers/bill_layout_classifier.py` | no, default off |
| Tests | `tests/test_layout_classifier.py`, `tests/test_real_form_layouts.py`, `tests/test_config_files.py` | |

```powershell
python main.py path\to\invoice.jpg                              # baseline, unchanged
python main.py path\to\invoice.jpg config\routed_config.yaml    # routed
$env:INVOICE_AI_CONFIG_PATH = "config/routed_config.yaml"; python -m api.main
```

No new recognizer or flow module was written: the winning arrangement is a
composition of components the repository already had, so it is a configuration.
The wire format and the C# side are untouched.

Test suite: 579 passed, 5 skipped, 1 failed. The failure
(`test_evaluate_prints_arabic_and_symbols_through_a_cp1252_pipe`, a `\r\n`
against `\n` comparison on Windows) fails the same way without these changes.
`pytest` and `httpx` were installed into `D:\OCR\.venv` to run it.

## Did not work, and why

| Tried | Outcome | Why |
|---|---|---|
| Full page to Qwen3.5 | 0.05 to 0.17 | a 0.9 B OCR model fine-tuned on lines; invents table structure, drops digits |
| Row crops | 0.09 to 0.25 | same, and the cell boundaries are lost |
| Surya boxes as crops | 0.27, 2 cells not found | boxes cut figures that cross a rule; short digits missed |
| Adaptive threshold, Otsu | 0.22, 0.29 | binarization breaks thin strokes and the dot zeros |
| Blur-divide | 0.32; 14/24 end to end | washes out the ballpoint, as recorded in `default_config.yaml` |
| 30 px padding, neighbour merging | 0.24 to 0.34 | the neighbour's writing is read too |
| Ink-tight boxes | names CER 0.30 to 0.57 | the ink search follows strokes into adjacent rows |
| Model-card prompt for every kind | 0.29 | loses the per-kind instruction; captions and letters return in number cells |
| Zero/comma hint in the number prompt | no change | the fine-tuned model does not act on the instruction |
| Voting, confidence fallback, disagreement referee | at or below baseline end to end | weaker models outvote the stronger; confidence is not reliable enough |
| Column-wide item boxes (`colfix`) | 20/24 end to end | fixes one cut cell, adds neighbour ink to two |
| Nuqta Q8 instead of Q4 | same accuracy, about twice the time | does not fit the GPU |

Not run: Surya's own recognition (measured earlier at 0/24 numbers), and any
experiment on a third invoice (none is labelled).

## Recommendations

1. **Label more invoices before changing the default.** Ten to twenty, from
   several writers and both forms, would turn most of the "within noise" rows
   above into answers. `tools/export_crops.py` and this folder's scripts take
   new truth files without changes.
2. **Fine-tune the reader on real cells.** The oracle over four models is 0.59,
   so selection cannot fix this. The dominant number errors are two learnable
   conventions (dot zero, decimal comma), and form B's dotted rule.
3. **Keep the arithmetic repair honest.** Re-measure it on invoices it was not
   written against; its 12/26 → 22/24 lift is the largest single contribution to
   the end-to-end result and the least tested.
4. **Turn on `one_cell_per_column` in the default** once it has been seen on more
   photos of both forms. It removes a real defect and changed nothing else here.
5. **Use the routed pipeline only where names matter more than time**, and only
   after measuring it on more names. On a machine with 8 GB of VRAM both models
   fit the GPU and its time cost would shrink; that was not measurable here.
6. **Do not use a model's confidence ≥ 0.75 as "no review needed".** Agreement of
   two different models is a better candidate signal (8 of 8 here) and worth
   testing on more data.

## Reproducing

```powershell
D:\OCR\.venv\Scripts\python.exe experiments\run_all.py          # inference; resumable, cached
D:\OCR\.venv\Scripts\python.exe experiments\combine.py          # combinations, calibration
D:\OCR\.venv\Scripts\python.exe experiments\e2e_replay.py       # whole pipeline on cached readings
D:\OCR\.venv\Scripts\python.exe experiments\score.py            # results.csv
D:\OCR\.venv\Scripts\python.exe experiments\analyze_errors.py qwen35/base nuqta_q4/base nuqta_q8/base qwen25/base
$env:INVOICE_AI_VLM_PYTHON = "D:\OCR\.venv_vlm\Scripts\python.exe"
D:\OCR\.venv\Scripts\python.exe tools\evaluate.py --config config\routed_config.yaml --label routed
```

| File | Holds |
|---|---|
| `results.csv` | every configuration: per-cell metrics, end-to-end columns, real-run time and memory where measured |
| `e2e_replay.csv` | end-to-end result and the wrong fields of each configuration |
| `predictions/<model>/<variant>.jsonl` | every reading with its label, confidence and seconds |
| `predictions/combo/` | the combinations' chosen readings |
| `predictions/cache/` | the reading cache, keyed by crop bytes, kind, prompt and settings |
| `predictions/combination_report.json`, `error_analysis.json` | calibration, agreement, leave-one-out choices, error categories |
| `make_crops.py`, `prompts/`, `combinations.json`, `run_all.py` | the crop recipes, prompt variants, combination plan and step list |
