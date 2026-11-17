# Fine-tuning the recognizer

How to turn labelled invoices into a recognizer that reads this project's
handwriting better than the base model, and how to know whether it does.

Everything here is tooling that runs today. What is missing is data: three
invoices are labelled (61 cells), and a fine-tune needs a few hundred cells from
several writers before it can do anything but memorize.

## Where things stand (2026-11-17)

| | Fields right, three labelled invoices |
|---|---|
| Every cell read as its label (`--perfect-readings`) | 61 / 61 |
| Read by the base model (Arabic-Qwen3.5-OCR-v4) | 41 / 61 |

The first row says the layout, the parser and the arithmetic lose nothing on
these photos: every remaining error is a cell the model misread. That is the
gap a fine-tune is for. Most of the misreadings are of two kinds, both
learnable from examples: a zero written as a dot (`١٠٨٠` read `١٠٨`), and the
decimal comma (`٣,٤٠` read `٤٠٤٥٦`).

## The procedure

### 1. Label invoices

One JSON file per photo under `evaluation/ground_truth/`, photos under `bills/<name>/`.
Copy an existing file. The rules that matter for training:

- **Type a decimal number as a string, exactly as written**: `"unit_price": "5,70"`.
  A JSON number cannot hold the trailing zero (`5.70` is 5.7), and a crop showing
  ٥,٧٠ labelled `5,7` teaches the model to drop a zero it can see. Whole numbers
  can stay numbers. Type in 0-9; the digits the writer used are recorded once,
  in `digits` (`arabic_indic`, the default, or `western`).
- **Dates** stay ISO (`"date": "2026-09-25"`); the label is day/month/year. Add
  `"date_written": "2026/9/25"` only when the paper orders it differently.
- **`writer`**: any stable name or number for whoever filled the invoice in, when
  you know it. Invoices by one writer are kept in the same split.
- **List every item.** Rows below the last listed item are used as examples of
  an empty cell.
- Check the arithmetic: every row's quantity × price = total, and the totals
  sum to the invoice total. `pytest tests/test_finetune_dataset.py` checks it
  for every truth file.

To label faster, read the invoice first and correct the result instead of typing
it: `python main.py bills/<name>/photo.jpg config/default_config.yaml`.

Aim for 20-30 invoices from at least 5 writers before the first real run, and
include every printed form you want read.

### 2. Check the layout before trusting any label

    python tools/render_layout.py bills/<name>/photo.jpg --config config/default_config.yaml
    python tools/export_crops.py --photo bills/<name>/photo.jpg --out datasets/<name>
    python tools/evaluate.py --perfect-readings datasets/<name> ...

A crop is the training example. If the box is a row off, the label is pasted on
the wrong picture, and the model is taught that confidently. `--perfect-readings`
must give 100% on an invoice before its cells are worth training on; anything
less is a layout or parsing fault to fix first.

### 3. Build the dataset

    python tools/build_finetune_dataset.py

writes `datasets/finetune/{train,val,test}.jsonl`, the crops, and
`dataset_card.json`. No model runs. Each example carries the service's own
prompt for its kind of cell. Read the warnings it prints: they say when a split
is empty or the data is too thin.

Splits are by invoice (by writer when the truth names one), decided by a hash of
the name, so an invoice never changes split when others are added. **Never move
an invoice between splits by hand**, and never look at test results to choose
between two training runs: that is what the validation split is for.

Rebuild after adding a truth file and after any change to the geometric steps,
the layout profile or the cropper. The crops are a product of that configuration.

### 4. Train

In the VLM's virtualenv (the one the worker runs in):

    D:\OCR\.venv_vlm\Scripts\python.exe tools\finetune_recognizer.py --out OCR_Models\invoice-ft-001

It trains a low-rank adapter beside the frozen base model (`ocr/finetuning/lora.py`),
keeps the epoch with the lowest validation loss, and writes a folder of about
20 MB: `adapter.safetensors`, `finetune.json`, `training_log.json`. The base
model is not copied and must stay where it is.

- It fits the 4 GB GPU. Measured: about 2 seconds per cell per epoch, so 1,000
  cells for 5 epochs is under 3 hours.
- To stop: create an empty file `OCR_Models\invoice-ft-001\STOP`. It finishes the
  epoch and exits. Continue with `--resume`.
- Defaults: 5 epochs, learning rate 1e-4, rank 8, adapters on the language
  model. Try `--adapt both` (the vision tower too) when the validation loss
  stops improving early; try fewer epochs or `--rank 4` when the training loss
  falls and the validation loss rises.

### 5. Measure it against the base model, on invoices it never saw

    python tools/evaluate.py --config config/default_config.yaml --held-out datasets/finetune --label base --reading-cache evaluation/reading_cache
    $env:INVOICE_AI_MODEL_DIR = "OCR_Models/invoice-ft-001"
    python tools/evaluate.py --config config/default_config.yaml --held-out datasets/finetune --label ft-001 --reading-cache evaluation/reading_cache

`--held-out` scores only the test split's invoices. Compare the two reports
field by field. Adopt the fine-tune when it is better on the held-out invoices
by more than a couple of fields, and not worse on any kind of field. Record the
run in `docs/experiment-log.md` either way.

To use it, set `INVOICE_AI_MODEL_DIR` to the folder for the service
(`python -m api.main`). The worker adds the adapter into the base model as it
loads; nothing else changes, and the C# side is unaffected.

## What to expect, and what not to

- **The arithmetic repair was tuned to the base model's mistakes.** A fine-tuned
  model makes different ones. If numbers get better per cell and worse per
  invoice, look at `invoice/reconciliation.py` before blaming the model.
- **Confidence will shift.** The 0.75 review threshold on the desktop side was
  never calibrated; a fine-tuned model can be more confident and no more right.
  Check how many wrong readings sit above 0.9 in the report's `readings`.
- **Not measured yet:** how many cells are enough, whether adapting the vision
  tower helps, and CPU-only reading speed with an adapter (it should equal the
  base model's, since the adapter is merged at load).

## Why the labels look the way they do

- **Digits as written.** The base model transcribes handwritten ٠-٩ as ٠-٩ and
  everything downstream folds them. Training targets in 0-9 would spend the data
  on teaching it to translate. Tried once on 20 cells: starting loss 2.37
  against 1.02, and the tuned model read 15 fields of 41 on the held-out
  invoices against the base model's 30. Too small to be a measurement, large
  enough not to repeat.
- **`EMPTY` for an empty cell**, the word the prompts already ask for. The
  cells used are the ones the pipeline would really send: below the last item,
  not blank enough to be skipped.
- **The prompt is stored with each example**, so a later change to
  `PROMPTS` cannot silently train under one instruction and read under another.
  Rebuild the dataset after changing a prompt.
