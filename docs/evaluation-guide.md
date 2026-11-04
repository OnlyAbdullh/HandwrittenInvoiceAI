# Evaluating recognizers and the pipeline

There are three tools, each answering one question. All of them run offline,
and photos, crops and reports stay on the machine (`bills/`, `datasets/` and
`evaluation/reports/` are git-ignored).

| Question | Tool | Unit |
|---|---|---|
| Which model reads our cells best? | `tools/benchmark_recognizers.py` | one cell |
| Would combining models beat the best one? | `tools/analyze_benchmark.py` | one cell, held out by invoice |
| How good is the whole invoice the app receives? | `tools/evaluate.py` | one invoice |

Commands are PowerShell, run from the project folder. `D:\OCR\.venv_vlm` stands
for the virtualenv that has torch and transformers (`requirements-vlm.txt`), so
use your own path.

## 1. Labelled invoices

A truth file is a transcription of one real photograph:
`evaluation/ground_truth/<name>.json`, with `image` relative to `bills/`. Copy
`sahror.json` and transcribe the paper:

- values exactly as written;
- dates in ISO format;
- plain numbers.

Then check that the line totals add up to the invoice total. Record who wrote
the invoice in `notes`; it is what makes writer-independent evaluation possible
later.

Never add synthetic or generated images as ground truth.

## 2. Cell benchmark: every model, the same cells

```powershell
python tools\export_crops.py --photo bills\sahror\img.png  --out datasets\sahror
python tools\export_crops.py --photo bills\onlyone\img.png --out datasets\onlyone
& D:\OCR\.venv_vlm\Scripts\python.exe tools\benchmark_recognizers.py --crops datasets\sahror datasets\onlyone --mode pipeline
```

**What the steps do:**

- **`export_crops.py`** cuts the cells exactly as the pipeline does. It labels them
  from the truth file whose `image` is that photo. If the item rows it finds don't
  line up with the truth, it keeps the header labels and leaves the item cells
  unlabelled; it never guesses.
- **Resolution.** A photo larger than the working resolution (2000 px on its long
  side) is shrunk first, exactly as the service does, and the export prints
  `photo WxH, read at wxh`. Use the original photos; there is no need to resize them.
- **`--crops`** takes several folders. Each cell is keyed by its invoice, so the
  scores can be broken down per invoice.
- **`--mode pipeline`** reads each model the way the service would: per-kind
  prompts, digit constraints, and the GGUF model's settings from
  `config/nuqta_config.yaml`. `--mode generic` gives every model the same plain
  prompt.
- **Isolation.** Each model runs in its own process, so a native crash costs only
  that model.

**Output:** `evaluation\reports\model_benchmark.{md,json,html}`, with:

- per model: CER, WER, exact match, number-value accuracy, usable and empty rates,
  mean confidence, seconds per cell, load time, peak RAM and VRAM;
- the same metrics per cell kind and per invoice;
- the HTML page, which shows every crop beside every model's reading.

**Metric definitions:**

- **CER and WER**: computed after the pipeline's own normalization (Arabic-Indic
  digits folded, diacritics stripped, printed captions dropped).
- **Number-value accuracy**: whether a number cell has exactly the right value,
  read as the parser reads it. One wrong digit or a misplaced decimal point is
  wrong.
- **Peak RAM**: the OS high-water mark of the model's process, or of its
  `llama-server`.
- **VRAM**: from `nvidia-smi`, or torch's peak allocation.

A measure that cannot be taken is left empty, never estimated.

## 3. Combination analysis

```powershell
python tools\analyze_benchmark.py evaluation\reports\model_benchmark.json --out evaluation\reports\combination_analysis.md
```

It compares, on the cells every model read:

- each model alone;
- routing by cell kind;
- confidence fallback;
- the oracle: the best model per cell in hindsight, an upper bound no
  combination can reach.

**Leave-one-invoice-out.** Every choice (which model per kind, which threshold) is
made on the other invoices and scored on the held-out one. With only one invoice
it says so, and labels the result as in-sample and optimistic.

**Calibration.** It also shows how each model's confidence behaved: the
exact-match rate per confidence band, and how many wrong readings came with
confidence ≥ 0.75. Those are the ones the review threshold would let through.

**When to switch on a combination.** Only if it beats the best single model on
held-out invoices by more than a couple of cells. Then set it in a config with
the `routed` recognizer:

```yaml
ocr:
  recognizer:
    name: routed
    params:
      recognizers:
        qwen:  {name: qwen, params: {worker_python: "...", worker_script: tools/qwen_worker.py, model_path: "..."}}
        nuqta: {name: llamacpp, params: {model_path: "...", mmproj_path: "..."}}
      default: qwen
      routes: {arabic_text: nuqta}                    # optional: kind -> model
      fallback: {to: nuqta, below_confidence: 0.6}    # optional: threshold from the analysis
```

Kinds are `number`, `date`, `code`, `arabic_text` and `any`. Each listed model is
loaded, so mind the memory: two VLMs will not both fit a 4 GB GPU.

## 4. Invoice evaluation: what the desktop app receives

```powershell
python tools\evaluate.py --config config\default_config.yaml --label qwen
python tools\evaluate.py --config config\nuqta_config.yaml   --label nuqta
```

With the transformers backend, `evaluate.py` starts the model worker from
`INVOICE_AI_VLM_PYTHON` (default `qwen_venv\Scripts\python.exe`). Set it to your VLM
venv first, e.g. `$env:INVOICE_AI_VLM_PYTHON = "D:\OCR\.venv_vlm\Scripts\python.exe"`.

It runs the real pipeline on every labelled invoice and scores the wire response
(`evaluation/metrics.py`):

- **Header fields**, each on its own:
  - the invoice number, date and total must be exact;
  - customer and city are right when the catalog's top candidate is the true
    entry, and the report also shows whether that candidate cleared 0.75.
- **Item rows** are aligned to the truth:
  - phantom rows were invented;
  - missed rows were dropped.
- **Numeric cells**: overall and per column (quantity, unit price, line total).
- **Product names**: whether the catalog's top candidate is correct.
- **Name CER** on the raw reading, before the catalog rescues it.
- **Complete invoices**: everything right. **Complete and nothing to review**:
  everything right, and every catalog match above 0.75 as well.
- **Time and memory:**
  - mean and slowest seconds per invoice;
  - model calls per invoice;
  - pipeline load time (the model is warmed up first, so load time is not charged
    to the first invoice);
  - peak RAM of the process plus its model server or worker;
  - VRAM.

**Output:** `evaluation\reports\<timestamp>-<config>-<label>.{md,json}`.

### CPU vs. GPU

- **llama.cpp backend:** compare with
  `$env:INVOICE_AI_LLAMA_GPU_LAYERS = "0"` (CPU only) against `"auto"`.
- **transformers backend:** it uses CUDA when torch sees a GPU. To measure CPU
  speed, run on a machine without one, or set `$env:CUDA_VISIBLE_DEVICES = ""`
  before starting.

## 5. Protocol for a change meant to improve accuracy

1. Run `evaluate.py` and the benchmark **before** the change, and keep the reports.
2. Make one change.
3. Run them again with a `--label` naming the change.
4. Keep the change only if it helps on the held-out invoices by more than noise.
   With a few dozen cells, one or two cells is noise.
5. Record the result in [experiment-log.md](experiment-log.md), whether it helped
   or not.
