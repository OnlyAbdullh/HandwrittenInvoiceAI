# Arabic handwritten invoice recognition: research, measurements, decision

This is the record behind the recognizer this project ships. It covers what the
problem is, which models and strategies were considered, what was measured on
real invoices and what was not, the decision, and what would change it.

**Labels used throughout:**

- **Measured**: run on this project's real invoices, on the target laptop (RTX 3050
  4 GB, 24 GB RAM, Ryzen 5 5600H).
- **Self-reported**: a model author's own claim, not reproduced here.
- **Third-party**: someone else's benchmark.
- **Not verified**: nobody here has checked it.

Only measured numbers support a decision.

---

## 1. The actual problem

Read from the code (`orchestration/`, `table_extraction/`, `ocr/`, `invoice/`) and from
the two labelled photographs (`bills/sahror`, form A; `bills/onlyone`, form B):

- **The page is a printed form filled in by hand.** Two suppliers' layouts are
  known:
  - Form A: a 6-column × 10-row item table.
  - Form B: a 4 × 15 bilingual table.

  Header boxes hold the invoice number, date, customer and city. A totals strip
  sits under the table. The grid is printed and ruled, so where each field is
  is a solved geometry problem (`contour_based` + `bill_layout`). What is hard is
  reading the ink inside each cell.
- **The cells are short.**
  - Most are numbers (quantity, unit price, line total, invoice number, total).
  - A few are 1–4 Arabic words: a customer, a city, a product.
  - The rest are a date.

  This is isolated-field recognition, not paragraph or page OCR. Page-level OCR
  models are trained for the opposite case.
- **Numbers dominate the value of an invoice, and the arithmetic checks them.**
  `price × quantity = line total` and `Σ line totals = invoice total` are hard
  constraints. `invoice/reconciliation.py` already uses them to repair misplaced
  decimal separators, which the measured models get wrong far more often than
  the digits themselves.
- **Names are checked against the catalog.** Customers, cities and products are
  matched against the desktop app's own tables (top five candidates). A reading
  only has to be close enough to rank the right entry first.
- **The constraints are hard ones:**
  - Fully offline; invoices never leave the machine.
  - Windows 11 laptops.
  - The development target has a 4 GB GPU; many users have none.

**Consequence for model choice.** The best recognizer here is the one with the
lowest error *per cell crop* of handwritten digits and short Arabic names, at an
acceptable speed on that laptop. Page-level benchmark leadership is weak
evidence for that.

## 2. Research findings

### Benchmarks and datasets

| Source | What it says that matters here | Status |
|---|---|---|
| [KITAB-Bench](https://arxiv.org/abs/2502.14949) (ACL Findings 2025; [code](https://github.com/mbzuai-oryx/KITAB-Bench)) | Multi-domain Arabic OCR benchmark with a handwriting sub-domain. Handwriting is the hardest part for every model tested. Tesseract's CER on handwriting is reported as 1.26. Gemini-2.0-Flash leads overall; AIN-7B is strongest of the open models on historical manuscripts. | Third-party. The handwriting-subset figures quoted by secondary sites did not match the paper, so none are repeated here. |
| [Muharaf](https://arxiv.org/abs/2406.09630) (NeurIPS 2024 D&B) | About 1,600 historical handwritten pages with line polygons. The largest public Arabic handwriting set. | Third-party. Manuscripts, not invoices. |
| KHATT | About 1,000 pages of modern Arabic handwritten paragraphs. The usual HTR benchmark. | Not verified here. Paragraph text, not digits in boxes. |
| [CAMEL-Bench](https://arxiv.org/abs/2410.18976) | Arabic multimodal benchmark (broader than OCR). | Third-party. |
| [When Do VLMs Help Arabic Manuscript OCR?](https://arxiv.org/abs/2608.22366) and [Compact multi-domain Arabic manuscript HTR](https://arxiv.org/abs/2608.19385) | Directly on topic (VLM vs. specialised HTR for Arabic manuscripts). | **Not read**: only titles and reference lists were reachable from this environment. Worth reading before any fine-tuning decision. |

### Candidate models

| Model | Basis / size | Claim | Status | In `OCR_Models`? |
|---|---|---|---|---|
| [sherif1313/Arabic-Qwen3.5-OCR-v4](https://huggingface.co/sherif1313/Arabic-Qwen3.5-OCR-v4) | Qwen3.5-0.8B | Handwritten + printed; about 5% CER on clean print, 5–25% on complex handwriting | Self-reported. **Measured best here** (§4). | yes |
| [sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B](https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2) (v3) | Qwen2.5-VL-3B, 4-bit | Handwriting-specific | Self-reported. Measured second here. Pre-quantized bitsandbytes, so CUDA only. | yes |
| [sherif1313/Arabic-English-handwritten-OCR-v3](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3) | Qwen2.5-VL-3B | 1.78% CER on its own test set | Self-reported. **Not measured: not downloaded.** The strongest untested candidate. | no |
| [Nuqta-Arabic-OCR-4B](https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-4B) ([GGUF](https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF)) | Qwen3.5-4B | KITAB-Bench CER 0.106; first on Misraj-DocOCR | Self-reported, and oriented to printed pages and tashkeel. Measured here, with its model card's own settings. | yes |
| [Qari-OCR](https://arxiv.org/abs/2506.02295) (NAMAA) | Qwen2-VL-2B | High-fidelity Arabic text recognition | Third-party. Measured here: poor on these cells, and slowest. | yes (v0.3) |
| [AIN-7B](https://huggingface.co/MBZUAI/AIN) ([paper](https://arxiv.org/abs/2502.00094)) | Qwen2-VL-7B | Arabic-inclusive LMM; strong historical-manuscript OCR on KITAB-Bench | Third-party. License listed as MIT on mirrors; not verified in the repo. 7B does not fit 4 GB VRAM even at 4-bit with room for the vision tower; CPU-only would be very slow. | no |
| [Baseer](https://huggingface.co/papers/2509.18174) (Misraj) | VLM fine-tuned for Arabic document-to-Markdown | WER 0.25 on its benchmark | Third-party; page-level documents. (The unrelated [AbdoTarek/Baseer-OCR-V1.0](https://huggingface.co/AbdoTarek/Baseer-OCR-V1.0) is a Qwen2-VL-2B legal-document fine-tune.) | no |
| [absir-arabic-ocr-1.6B](https://huggingface.co/g3ida/absir-arabic-ocr-1.6B) | LFM2-VL | Printed Arabic pages | Self-reported; printed only. | no |
| [HATFormer](https://arxiv.org/abs/2410.02179) | TrOCR-based encoder–decoder for Arabic HTR | CER 8.6% on historical Arabic handwriting, 4.2% on a private modern set | Self-reported. Weights' release not confirmed. A fine-tunable small HTR architecture. | no |
| TrOCR fine-tunes (`David-Magdy-TR_OCR_LARGE`, `RayR1-trocr-base-arabic-handwritten`), `Qalam_Net_V2`, `arabic-large-nougat`, Qwen3-VL-2B Persian-Arabic | various | various | Measured here (§4): all worse than the Qwen3.5 fine-tune, most far worse. | yes |
| PaddleOCR-VL, DeepSeek-OCR, dots.ocr, Chandra | general document VLMs | Arabic listed among languages; DeepSeek-OCR strong on printed Arabic in the [MORE benchmark](https://arxiv.org/abs/2607.02956) | Third-party. **No evidence for Arabic handwriting** was found for any of them. | no |

**What the literature supports:**

- **Handwritten Arabic is still unsolved.** Every model is much worse on it than on
  print, and no public benchmark measures *handwritten digits in form cells*.
- **Self-reported CERs come from each author's own test set** and are not comparable
  with each other, nor with this project's invoices.
- **Fine-tuning helps.** VLM fine-tunes on Arabic handwriting beat their base models
  (Qwen3.5-0.8B vs. its OCR fine-tune), and a compact HTR model (HATFormer) can be
  competitive when trained on in-domain data. In-domain data is exactly what this
  project does not yet have in quantity.

So the decision has to come from measurement on this project's own cells. The
literature cannot make it.

## 3. Strategies considered

| | Strategy | Assessment |
|---|---|---|
| A | **One VLM per cell crop** (the grid gives the crop, the cell's kind gives the prompt and decoding constraint) | **Chosen.** It matches the problem (§1): short isolated fields, known kinds. Digit cells get a GBNF/logits constraint, so a number cell cannot come back as prose. It is the measured best (§4). |
| B | Text detection + specialised line recognizer (TrOCR/HATFormer) | `detector_driven` exists. On these forms the grid already gives better boxes than a detector, which misses short isolated digits. The specialised recognizers available were measured **far worse** (CER > 1; §4). It would become competitive only with in-domain fine-tuning. |
| C | Hybrid: a model per field type | **Implemented, off by default** (`routed` recognizer). It is only worth its extra load time and memory if it beats the best single model on held-out invoices. `tools/analyze_benchmark.py` measures exactly that from a benchmark run. No measurement yet shows a gain. |
| D | Whole-page VLM (read the invoice as a document, emit JSON) | Rejected for now. Page-level models are trained on print. A 0.8–4B model on one 4 GB GPU cannot take a full-resolution page at the detail handwritten digits need. Every value would lose its bounding box, which the desktop app's click-to-verify overlay depends on. One hallucinated row is also harder to catch than one bad cell. |
| E | Fine-tune a recognizer on this project's invoices | The right long-term step, **not possible yet**. There are 36 labelled cells from 2 invoices, and both invoices are the same transaction. Fine-tuning on that would only memorise it. §7 says what is needed. |
| F | Multi-stage with fallback (re-read low-confidence cells with a second model; arithmetic and catalog checks) | The checks are already in the pipeline (reconciliation, total check, catalog ranking, 0.75 review threshold). The model fallback is part of the `routed` recognizer, also gated on measured benefit. |

## 4. Measured comparison (baseline vs. alternatives, identical cells)

The data:

- 20 labelled cells from `bills/sahror` (form A).
- Crops cut by the pipeline itself (`tools/export_crops.py`).
- Every model read the same 20 cells, in `--mode pipeline` (the per-kind prompts
  and decoding constraints the service uses).
- Run on the user's laptop (RTX 3050 4 GB).
- CER is computed on the pipeline's own normalization. "Exact" means a CER of 0
  for that cell.

| Rank | Model | CER | Exact | s / cell | Load s | Notes |
|---|---|---|---|---|---|---|
| 1 | **Arabic-Qwen3.5-OCR-v4** (transformers, float32, CUDA) | **0.143** | **0.45** | 0.97 | 11 | **Shipped default** (`default_config.yaml`, `qwen_config.yaml`) |
| 2 | Nuqta-Arabic-OCR-4B Q4_K_M (llama.cpp) | 0.239 | 0.30 | 3.05 | 9 | With its model card's prompt, no repeat penalty, and the projector's own image size. With the earlier settings it scored 0.379. |
| 3 | Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3 | 0.262 | 0.20 | 0.94 | 13 | CUDA only (pre-quantized) |
| 4 | Qwen3-VL-2B Persian-Arabic OCR v1.0 | 0.371 | 0.35 | 1.44 | — | |
| 5 | TrOCR large (David-Magdy) | 1.139 | 0.00 | 0.24 | — | Usable reading on 30% of cells |
| 6 | trocr-base-arabic-handwritten (RayR1) | 1.165 | 0.00 | 0.86 | — | Usable on 25% |
| 7 | Qari-OCR v0.3 | 1.220 | 0.00 | 2.6–3.1 | — | |
| 8 | Qalam_Net_V2 | 1.764 | 0.00 | 0.26 | 10–14 | |
| 9 | arabic-large-nougat | 4.121 | 0.00 | 0.67 | — | Page model: invents text |

"—" means the load time was not recorded in the output that was shared.

**Both invoices** (`sahror` + `onlyone`, 34 cells, leave-one-invoice-out; laptop):

| Model | CER | Exact |
|---|---|---|
| Arabic-Qwen3.5-OCR-v4 | **0.500** | 0.265 |
| Nuqta Q4_K_M | 0.508 | 0.176 |
| Qwen2.5-VL-3B v3 | 0.562 | 0.147 |

The best fallback combination gained 2 cells (unstable threshold). Routing by kind
lost. The oracle bound is 0.357. Details are in [experiment-log.md](experiment-log.md).
That run predates the form-B layout fix and the label fixes.

**Re-run after those fixes** (26 cells; the laptop's full-size `onlyone` photo was
still mis-laid-out, see the experiment log):

| Model | CER | Exact | s/cell |
|---|---|---|---|
| Arabic-Qwen3.5-OCR-v4 | **0.245** | **0.346** | 1.99 |
| Nuqta Q8_0 | 0.321 | 0.269 | 6.22 |
| Nuqta Q4_K_M | 0.368 | 0.231 | 3.83 |
| Qwen3-VL-2B | 0.452 | 0.269 | 3.97 |
| Qwen2.5-VL-3B v3 | 0.562 | 0.192 | 2.12 |

No combination beat Qwen3.5 alone on held-out invoices. Per invoice, Qwen3.5 took
38 s and 3.5 GB of RAM, against Nuqta's 106 s and 9.9 GB.

**All 41 cells, photos read at the working resolution** (latest):

| Model | CER | Exact | Value acc. | s/cell |
|---|---|---|---|---|
| Nuqta Q8_0 | **0.245** | 0.293 | 0.321 | 4.86 |
| Arabic-Qwen3.5-OCR-v4 | 0.276 | 0.293 | **0.393** | **0.92** |
| Nuqta Q4_K_M | 0.279 | 0.293 | 0.321 | 2.70 |
| Qwen2.5-VL-3B v3 | 0.423 | 0.244 | 0.250 | 0.99 |

The top three are within about one cell of each other. Qwen3.5 stays the default:
it reads the most numbers exactly, is 5× faster than Nuqta Q8, and fits the 4 GB
GPU. A whole invoice takes 17 s with it.

**After form B's crops were fixed** (same 41 cells): Qwen3.5-OCR-v4 **0.189** CER
(exact 0.366, 1.05 s/cell), Nuqta Q8 0.259, Nuqta Q4 0.280, Qwen2.5-VL-3B 0.298.
Qwen3.5 is clearly first again.

With every cell read perfectly (`evaluate.py --perfect-readings`), both invoices
come out 100% right. All remaining error is recognition.

**Whole pipelines compared** (`tools/compare_pipelines.py`, same preprocessing and
layout, laptop):

| Approach | Header | Numbers | s / invoice | Peak RAM |
|---|---|---|---|---|
| **Grid cells → Qwen3.5-OCR-v4** | **8/9** | **22/24** | **14.5** | **2.9 GB** |
| Surya boxes → Qwen3.5 | 5/9 | 10/24 | 25.6 | 5.2 GB |
| Grid cells → Nuqta Q4 | 5/9 | 8/24 | 59.5 | 9.4 GB |
| Surya alone | 3/9 | 0/24 | 63.7 | 4.8 GB |

The grid-driven pipeline with Qwen3.5 is first on every measure. Surya's crops cost
Qwen more than half its numbers, and Surya's own recognizer reads no handwritten
digit.

**What these numbers can and cannot say:**

- **The data is small.** 20 cells means one cell is 5 points of exact-match, so ranks
  2–4 are close to noise. Rank 1 versus the rest is not noise: a 0.10 CER gap and
  9 vs. 6 exact cells.
- **Single invoice, single writer.** A second labelled invoice (`onlyone`, 16 cells)
  exists now (§6), but it is the same transaction written on another form,
  probably by the same writer. The results are **not** evidence of
  writer-independent accuracy.
- **CPU speed was not measured.** The speeds above are GPU speeds.

**Verified in this cloud environment, without the real models:**

- **llama.cpp backend.** It works against a real `llama-server` with a tiny stand-in
  Qwen3.5 model. Verified: the image reaches the model, the digit grammar is obeyed,
  the reported confidence is the model's own probability (not a fabricated 1.0),
  the server is private to the service (loopback, a per-launch key), and it never
  outlives the service.
- **Layout on the real photographs.** Form B's first item row is now recovered,
  for 16 labelled cells.
- **The whole evaluation chain**: export, benchmark with several invoices, analysis
  and the invoice-level evaluation. The readings came from random-weight stand-ins,
  so these runs say nothing about real accuracy.

## 5. Decision: the final architecture

```
photo ─► geometric correction (once) ─┬─► OCR photometric branch ──► crop each classified cell
                                      └─► table branch ─► grid (contour_based) ─► bill_layout roles
     cell crop + ContentKind ─► recognizer ─► output validation ─► cell mapper
        (qwen: Arabic-Qwen3.5-OCR-v4, digit-constrained for number/date cells)
        (alternatives: llamacpp Nuqta; routed = per-kind / fallback composite)
     ─► keyword correction ─► invoice parser ─► arithmetic reconciliation ─► wire JSON
        (catalog top-5 per name; 0.75 review threshold applied by the desktop app)
```

**Why this design:**

- **Strategy A, with Arabic-Qwen3.5-OCR-v4, is the measured best:**
  - lowest CER;
  - most exact cells;
  - among the fastest per cell;
  - the smallest model (0.8B), the only one that also has a reasonable CPU path.
- **The other parts are already sound, and are kept:**
  - the grid supplies boxes that match the user's click targets;
  - the content kind supplies the prompt and the character constraint;
  - arithmetic and the catalog catch what the recognizer gets wrong.

**What changed in this round:**

- **Layout.** The item grid is completed when the extractor only partly finds it.
  Form B had lost its first item row.
- **Nuqta.** It is read with its model card's settings, so it now gets a fair
  comparison.
- **Benchmark (`tools/benchmark_recognizers.py`):**
  - scores several invoices in one run;
  - adds WER, number-value accuracy, peak RAM and VRAM, and per-kind and
    per-invoice breakdowns.
- **Combination analysis.** `tools/analyze_benchmark.py` answers whether routing or
  fallback beats the best single model, on held-out invoices.
- **Invoice evaluation (`tools/evaluate.py`):**
  - per-field accuracy and whole-invoice accuracy;
  - model load time;
  - peak RAM including the model server, and VRAM.
- **`routed` recognizer.** Selectable, off by default. It turns a measured win for
  strategy C or F into a configuration change.

## 6. Data and evaluation hygiene

- **Real data only.** `evaluation/ground_truth/*.json` holds transcriptions of real
  photographs, and nothing synthetic is presented as real. Photos (`bills/`), crops
  (`datasets/`) and reports (`evaluation/reports/`) are git-ignored and never leave
  the machine.
- **No leakage.** Every comparison of combinations holds out whole invoices
  (`analyze_benchmark.py` is leave-one-invoice-out), so a choice is never scored
  on the cells it was tuned on. With one invoice it says so and labels the result
  in-sample.
- **Writer independence.** This needs invoices from *other writers*, held out as a
  group. With the current two (same transaction, probably the same writer) it
  cannot be measured, and nothing here claims it. When more are labelled, record
  the writer in each truth file's `notes` and hold out by writer.

## 7. Limitations

- **The data is small:** 36 labelled cells across 2 invoices of one transaction.
  Every accuracy figure is anecdotal until about 20+ invoices from several
  writers are labelled.
- **CPU speed is unmeasured** for every model. On a GPU-less laptop, expect each
  cell to take several times longer. `tools/evaluate.py` now reports it; run it
  there.
- **Form B layout:**
  - its blank-cell gate cannot always tell ink that spilled over from the row
    above from real writing, so phantom rows are possible;
  - crops include the tails of the row above. `item_vertical_shift` exists to A/B
    a fix.
- **Unknown forms.** Only forms A and B are known. Any other form returns
  `UNKNOWN_FORM` / `NO_LAYOUT_DETECTED` rather than a guessed layout.
- **Confidence is uncalibrated, and measured to be a poor accept signal.** On the
  34-cell run, 44–53% of all cells were *wrong* with confidence ≥ 0.75, for every
  model. On the 26-cell re-run it was 27% for Qwen3.5 and 35–50% for the others. Treat OCR confidence only as an ordering for review, never as permission
  to skip it. For numbers, the arithmetic check (`price × qty = total`,
  `Σ = total`) is the better evidence.

## 8. Next steps, in order of expected value

1. **Label more invoices:** ~20 from at least 3–4 writers, with the writer recorded.
   Nothing else improves the accuracy of every later decision as much.
2. **Benchmark `Arabic-English-handwritten-OCR-v3`.** It is the strongest untested
   candidate by self-report and the same family as rank 3; download it into
   `OCR_Models/`, re-run the benchmark, then run `analyze_benchmark.py`.
3. **Run `analyze_benchmark.py` on the full benchmark.** If routing or fallback
   beats the best single model on held-out invoices by more than a few cells,
   enable `routed` in a config; otherwise leave it off.
4. **Measure CPU-only speed** (`INVOICE_AI_LLAMA_GPU_LAYERS=0`, or a machine without
   CUDA) with `tools/evaluate.py`, for the no-GPU users.
5. **Fine-tune once there are a few hundred labelled cells from several writers.**
   Fine-tune Arabic-Qwen3.5-OCR-v4 (LoRA) on cell crops from
   `tools/export_crops.py`, with train/test split by invoice *and* writer.
   Compare against the base model with `benchmark_recognizers.py` on the held-out
   writers only.
6. **Try float16 on CUDA** for the 0.8B model. It halves its VRAM, but has to be
   re-measured: forcing 4-bit was measured to corrupt digits.
