# Invoice Extraction Pipeline

The Python half of the Invoice Digitization system. It reads an invoice image and returns
a structured invoice — merchant, number, date, city, total, and the table's line items —
each value carrying a confidence and a ranked list of catalog matches for the user to
confirm.

Two ways in, one pipeline behind both:

| Entry point | Used by |
|---|---|
| `api/main.py` — HTTP, per [docs/api-contract.md](docs/api-contract.md) | The WinUI desktop app |
| `main.py` / `run_batch.py` — CLI | Reproducing a bad extraction offline, tuning config |

Both build the same `PipelineOrchestrator` from the same YAML config, so anything the
desktop app sees can be reproduced from a terminal with one command.

## Key architectural decision: geometric correction runs once

Of all the preprocessing steps, only **perspective correction** and **deskew** change
pixel coordinates. Everything else (channel selection, illumination normalization, CLAHE,
bilateral filtering, thresholding, morphology) is purely photometric.

Because of that, geometric correction is factored out as a single upstream stage
(`preprocessing.geometric_steps` in config) that runs **once**, before the pipeline forks
into the OCR-photometric and table-photometric branches. Both branches — and therefore
every OCR fragment and every table cell they produce — end up in exactly one coordinate
space: that of the geometrically corrected image. That same image is what the UI displays
on both sides (plain on the left, with overlays on the right). This is what guarantees no
bounding box ever needs to be remapped between coordinate spaces — see
`core/domain/geometry.py` for the full reasoning.

## Directory layout

```
api/                HTTP layer: FastAPI app, routes, wire schemas, pipeline pool
config/             AppConfig schema (pydantic) + YAML loader + settings_contract.py
                     (the object-keyed wire shape the desktop settings page sends)
core/
  domain/           Immutable DTOs passed between stages (ImagePayload, OCRResult,
                     TableExtractionResult, StructuredDocument, MatchedElement,
                     PipelineResult, ...)
  interfaces/       Protocols every pluggable family implements
  registry.py       Generic name -> class registry (Strategy + Factory backbone)
  exceptions.py
preprocessing/
  steps/geometric/     perspective_correction, deskew  (run once, produce the display image)
  steps/photometric/   channel_selection, illumination_normalization, clahe,
                        bilateral_filter, thresholding, morphological_cleanup
  pipeline.py           PreprocessingPipeline: runs an ordered step list
  pipeline_builder.py   builds a Pipeline from config + the step registry
ocr/                Engines, detectors, refiners, croppers, recognizers + factories
table_extraction/   Extractor adapters (grid_line, contour_based) + factory
mapping/            CellMapper: OCR fragments + table cells -> StructuredDocument
string_matching/    Normalization, the two similarity algorithms, catalog matching,
                     and the keyword-dictionary matcher adapters + factory
invoice/            The semantic layer: elements -> header fields + line items + warnings
output/             Formatter adapters (invoice_json, ui_overlay_json, csv) + factory
persistence/        ResultStore adapter (file_result_store) for audit/reproducibility
orchestration/      PipelineOrchestrator: the one module that knows the full pipeline shape
tests/              pytest suite, including end-to-end tests of the HTTP contract
```

## How the pieces fit together

`PipelineOrchestrator.run()`:

1. Runs `geometric_pipeline` once → the canonical display image.
2. Runs `ocr_photometric_pipeline` and `table_photometric_pipeline` sequentially on that
   same image.
3. Calls `ocr_engine.recognize()` and `table_extractor.extract()` sequentially.
4. `CellMapper.map()` assigns OCR fragments to table cells (or to free-standing fields).
5. `string_matcher.match()` runs against the shared keyword dictionary for every element,
   table cell or free field alike.
6. `InvoiceParser.parse()` reads those elements as an invoice, matching the merchant,
   city and product names against the catalogs the request carried.
7. Assembles a `PipelineResult` — including a full `config_snapshot` for
   audit/reproducibility — persists it via `ResultStore`, then hands it to
   `output_formatter.format()`.

Everything the orchestrator depends on comes from one `AppConfig` object. The catalogs are
the one thing that does not: they belong to the *request*, not the configuration, so they
are passed to `run()`.

## String matching

Two algorithms, because names fail in two different ways:

| Field | Algorithm | Why |
|---|---|---|
| Merchant, city, governorate | Normalized Levenshtein over the whole string | Read as one run of characters; they go wrong one letter at a time. |
| Product name | Order-independent word matching | The words are usually right but their order is not. "جاكيت صوف أزرق" and "جاكيت أزرق صوف" are the same product; each OCR word takes its best score against the catalog words, and the total is divided by the larger word count so extra words on either side dilute it. |

Both return the **top five** matches ranked highest-first, not a single answer: the
desktop app pre-selects the best one and offers the rest, so confirming a correct read
costs nothing and correcting a wrong one costs a click. Anything scoring below 0.75 is
returned with `requires_manual_review` set and the raw OCR text left as the value — a
wrong confident match silently corrupts an invoice.

Everything is normalized first (`string_matching/normalization.py`): Arabic-Indic digits
folded to ASCII, diacritics and tatweel removed, أ/إ/آ→ا, ى→ي, ة→ه, punctuation stripped,
whitespace collapsed, case folded. Digits survive — they are part of real product names.

**These rules live here alone.** The C# copies in `InvoiceDigitizationApp/Helpers/{TextNormalizer,FuzzyMatch,CatalogMatcher}.cs` are deleted: two implementations of "the same string" made the answer depend on which side computed it.
The C# copies exist because the app re-matches locally when the service returned no
candidates. If one side changes, the other must change with it.

## Adding a new implementation (why this scales across a team)

To add a new OCR engine, for example:

1. Create `ocr/engines/my_engine.py` implementing `OCREngine.recognize()`.
2. Decorate the class with `@engine_registry.register("my_engine")`.
3. Add it to the import list in `ocr/engines/__init__.py`.
4. Set `ocr.engine: my_engine` in the YAML config.

Nothing in `orchestration/`, `ocr/factory.py`, or any other module changes. The same
pattern applies to preprocessing steps, table extractors, string matchers, and output
formatters. To make it selectable from the desktop settings page, add a matching entry to
`InvoiceDigitizationApp/Services/Pipeline/PipelineCatalog.cs`.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt        # core; see requirements-*.txt for the rest

# The HTTP service the desktop app talks to.
python -m api.main              # http://127.0.0.1:8765

# Or one image from the command line.
python main.py path/to/invoice.jpg
python run_batch.py bills/ --config config/qwen_config.yaml
```

### Readers

The shipped configs use the **`layout_driven`** flow: the printed grid is detected, each
labelled cell is cropped, and a **Qwen VLM** reads it with a prompt and decoding mode
chosen for the kind of content (digits, Arabic text, a date). Blank cells are never
sent to the model. Other engines still exist for other flows.

| Reader | Needs | Notes |
|---|---|---|
| `qwen` recognizer (default) | the VLM worker's virtualenv (`requirements-vlm.txt`) and a model directory | Set `INVOICE_AI_VLM_PYTHON` and `INVOICE_AI_MODEL_DIR` (see `docs/settings-config-contract.md` §4). |
| `llamacpp` recognizer | `llama-server` (llama.cpp b7990+) and a GGUF model + its `mmproj` | **Nuqta Arabic OCR 4B**: `config/nuqta_config.yaml`. No second virtualenv, no torch; runs on CPU, GPU, or both. Setup: [docs/llamacpp-nuqta-setup.md](docs/llamacpp-nuqta-setup.md). |
| `routed` recognizer | the recognizers it combines | A model per cell kind, and/or a second model for low-confidence cells. Off by default: enable it only if `tools/analyze_benchmark.py` shows it beats the best single model on held-out invoices. See [docs/evaluation-guide.md](docs/evaluation-guide.md) §3. |
| `stub` | nothing | Deterministic fake output, for testing the plumbing. Never for real invoices. |
| `tesseract` | the Tesseract **binary** plus `ara`/`eng` traineddata | Weak on handwriting. |
| `surya` engine / detector | `requirements-surya.txt` | The detector drives the `detector_driven` flow. |

Heavy components are imported defensively, so a machine without their dependencies still
starts the service; the missing ones simply do not register.

#### Switching between the transformers and GGUF backends

Both are installed side by side and chosen by which configuration the service loads:

```powershell
$env:INVOICE_AI_CONFIG_PATH = "config/nuqta_config.yaml"   # Nuqta GGUF via llama.cpp
python -m api.main

Remove-Item Env:INVOICE_AI_CONFIG_PATH                      # back to default_config.yaml (Qwen, transformers)
python -m api.main
```

With the `llamacpp` recognizer the service starts `llama-server` itself — on loopback, on
a private port, with a per-launch key — keeps the model loaded across requests, restarts
it if it crashes, and stops it on shutdown. `python tools/check_llamacpp_setup.py` checks
the install and the two model files without downloading anything. Everything else —
preprocessing, layout, prompts, validation, the response shape — is the same pipeline.

### Measuring accuracy

Which model, and why: [docs/recognition-research.md](docs/recognition-research.md) — the
research, the measured comparison on real invoices, and the decision. How to measure:
[docs/evaluation-guide.md](docs/evaluation-guide.md). What was tried:
[docs/experiment-log.md](docs/experiment-log.md).

```bash
python tools/evaluate.py --config config/qwen_config.yaml --label my-change
```

scores the pipeline against hand-labelled invoices (`evaluation/ground_truth/*.json`,
image paths relative to `bills/`) and writes a report under `evaluation/reports/`. Run it
before and after any change meant to improve accuracy. Only two invoices are labelled so
far, so treat the numbers as anecdotal and **label more**: copy `sahror.json`, transcribe
the paper (values exactly as written, ISO dates, plain numbers), and check that the line
totals sum to the invoice total. `tools/export_crops.py` will turn labelled invoices
into a cell-crop dataset for fine-tuning the recognizer.

### Choosing a model: comparing everything in `OCR_Models/`

Two steps, because they need different virtualenvs. First cut the cells out of your own
invoice photo exactly as the pipeline would (service venv):

```powershell
python tools\export_crops.py --photo C:\invoices\my_invoice.jpg --out datasets\my_invoice
```

Then run every model in `OCR_Models\` on those cells (the VLM venv, which has torch):

```powershell
qwen_venv\Scripts\python.exe tools\benchmark_recognizers.py --crops datasets\my_invoice
qwen_venv\Scripts\python.exe tools\benchmark_recognizers.py --crops datasets\my_invoice --mode pipeline
```

and open `evaluation\reports\model_benchmark.html`: every cell beside every model's
reading, and a ranked summary with speed, confidence and how often each model's
reading passes the pipeline's own output validation.

- **What it runs.** Chat VLMs (Qwen2/2.5/3-VL, Qwen3.5, GLM-OCR), Florence-2 and
  TrOCR/Nougat run through transformers. Each GGUF quantization (`Q4_K_M`, `Q8_0`, ...)
  runs through llama.cpp as its own entry. Anything else is listed as skipped, with the
  reason.
- **`--mode generic`** gives every model the same plain prompt — the fair comparison.
- **`--mode pipeline`** reads each model the way the service would: the per-kind prompts,
  the production Qwen runtime with its digit constraints, and the GGUF grammar.
- **Scoring.** A photo's crops have no labels, so you judge the readings by eye. To get
  real error rates (CER), type the correct text into `datasets\my_invoice\labels.jsonl`
  (the `text` fields) and run again — or export with `--truth-file` using the
  `evaluation/ground_truth` format.
- **Useful options.** `--only nuqta qari` runs a subset, and `--max-crops 10` gives a
  quick pass. `--gpu-layers` and `--llama-server` apply to GGUF models.
- **Several invoices at once.** `--crops datasets\sahror datasets\onlyone` scores them
  together, broken down per invoice and per cell kind, with WER, number-value accuracy and
  peak RAM/VRAM. Then `python tools\analyze_benchmark.py evaluation\reports\model_benchmark.json`
  says whether routing by kind or a confidence fallback would beat the best single model,
  judged on held-out invoices.

### Configuration

`config/default_config.yaml` is what the service runs when a request sends no
`configuration` of its own. `config/qwen_config.yaml` is the tuned VLM setup, with the
reasoning for every enabled and disabled step written inline — read it before changing
preprocessing.

Environment variables prefixed `INVOICE_AI_` override service-level settings (host, port,
which config file to load, whether to persist results). See `api/service_settings.py`.

## Tests

```bash
pytest
```

Covers the normalization rules, both matching algorithms, the invoice parser, the
configuration contract's translation and validation, and the HTTP contract end to end —
the API tests run the real pipeline over a synthetic image using the `stub` engine, so
they need no OCR installation.
