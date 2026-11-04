# Settings / pipeline configuration contract

The `config` part of `POST /extract`, and the body of `GET /configuration`.
Implemented by `config/settings_contract.py` (`PipelineSettings`); the internal shape
the pipeline runs is `config/schema.py` (`AppConfig`). When one changes, change both,
this document, and the C# settings page.

## 1. Two shapes, one bridge

* **Internal (`AppConfig`, YAML on disk)** — preprocessing steps are ordered lists;
  list position is execution order.
* **Wire (`PipelineSettings`)** — each preprocessing step is a **fixed named key**,
  never an array position. The UI controls only `enabled`, `algorithm` and `params`
  per step. Execution order is fixed by the backend, not by JSON key order.

`PipelineSettings.to_app_config()` and `.from_app_config()` convert between them.

## 2. Preprocessing steps

Phase 1 runs **once** and yields the single corrected image shown on both sides of the
UI. Phases 2–3 are the OCR-photometric branch, in this order.

| Key | Algorithms (`algorithm`) |
|---|---|
| `perspective_correction` | `perspective_correction` |
| `deskew` | `deskew_hough` (`deskew_min_area_rect` — not implemented) |
| `channel_selection` | `channel_selection` |
| `illumination_normalization` | `illumination_normalization_blur_divide` (`…_blackhat` — not implemented) |
| `contrast_enhancement` | `clahe`, `plain_equalization` |
| `denoising` | `median_blur`, `gaussian_blur`, `nlm_denoise`, `bilateral_filter` |
| `thresholding` | `fixed_threshold`, `otsu_threshold`, `adaptive_threshold` |
| `morphological_cleanup` | `morphological_cleanup` (`params.operation`: open, close, erode, dilate) |

An algorithm illegal for its step, or not implemented, is rejected with
`422 INVALID_CONFIGURATION`. Unknown **parameter names** are rejected too, naming the
accepted ones. The table-detection branch's preprocessing is not exposed; the service
default is used.

`perspective_correction.params.margin` (0–0.5, default **0.15**) keeps a band of the
photo around the outline the step found, as a fraction of that outline's size. Before
it existed the step cropped exactly to the outline. On form B (white paper on a white
cloth) the strongest outline is the table's printed border, so the invoice number,
customer, date and most of the totals strip were cut away before anything was read.
A request that omits `margin` gets the default. A settings page that sends `0`
restores the old crop and loses those fields again.

Before `perspective_correction`, the service shrinks any photo whose long side is
larger than `preprocessing.working_long_side` (default 2000 px) to that size. Every
pixel threshold above was tuned at that size; a full-size phone photo read without
the shrink gave a different, broken layout. It is **server-owned** and not part of
the wire shape, so the C# side needs no change. Bounding boxes stay in the space of
the corrected page (`source.width` × `source.height`), which is now at most 2000 px
on its long side, plus the perspective margin.

## 3. Sections the request does not control

`ocr`, `persistence` and `string_matching.dictionary_path` are **taken from the
service's own configuration**. The request's values are accepted for compatibility,
ignored, and logged. They name executables and directories, and a request must never
choose what the machine runs (see `enforce_server_owned`).

Still request-controlled: `preprocessing`, `flow` (name and params), `table_extraction`
(extractor, classifier and their params, including form `profiles`), and
`string_matching.algorithm` / `algorithm_params`. The `output` formatter is forced to
`invoice_json`.

## 4. Machine configuration (YAML, server side)

Machine-specific values come from the environment, never from a committed path:

| Variable | Meaning | Default |
|---|---|---|
| `INVOICE_AI_VLM_PYTHON` | `python.exe` of the VLM worker's venv | `qwen_venv/Scripts/python.exe` |
| `INVOICE_AI_MODEL_DIR` | model directory (architecture auto-detected) | `OCR_Models/Arabic-Qwen3.5-OCR-v4` |
| `INVOICE_AI_CONFIG_PATH` | which YAML the service loads | `config/default_config.yaml` |
| `INVOICE_AI_API_TOKEN` | require `X-Invoice-AI-Token` | unset |
| `INVOICE_AI_MAX_LOADED_MODELS` | OCR model configurations kept in memory | `1` |
| `INVOICE_AI_NUQTA_MODEL` | GGUF language model, `config/nuqta_config.yaml` | `OCR_Models/Nuqta-Arabic-OCR-4B-GGUF/nuqta-ocr-4b-Q4_K_M.gguf` |
| `INVOICE_AI_NUQTA_MMPROJ` | GGUF vision projector, `config/nuqta_config.yaml` | `OCR_Models/Nuqta-Arabic-OCR-4B-GGUF/nuqta-ocr-4b-mmproj-f16.gguf` |
| `INVOICE_AI_LLAMA_SERVER` | `llama-server(.exe)`: full path, its folder, or a name on `PATH` | `llama-server` |
| `INVOICE_AI_LLAMA_GPU_LAYERS` | `auto` (fit to free VRAM), `all`, `0` (CPU), or a layer count | `auto` |

The `ocr` section — including `ocr.recognizer.name: qwen` or `llamacpp` and every
model path — is server-owned (§3), so the GGUF backend needs **no** change on the C#
side: the settings page cannot select it, the machine's configuration does. Setup:
[llamacpp-nuqta-setup.md](llamacpp-nuqta-setup.md).

In YAML, `${NAME:-default}` expands from the environment (`config/loader.py`).
Relative paths are relative to the project root.

## 5. Printed-form profiles

`table_extraction.classifier_params.profiles` lists further printed forms. The page is
labelled by the profile whose column count matches the most detected rows; the
top-level parameters are form A and the fallback. A profile sets `table_row_count`,
`table_col_count`, `column_roles`, `header_roles`, `footer_roles`, and the flags
`printed_header_row` and `footer_from_below`. The profile used is recorded in the
layout's `source`.

Further profile parameters:

- `complete_grid`: rebuild item rows the extractor only partly found.
- `item_vertical_shift`: move item boxes by a fraction of their height. It is for
  A/B tests and off by default.
- `unruled_fields`: fields printed on a dotted line rather than in a box, declared
  where they sit relative to the table outline, as
  `{role, x: [left, right], y: [top, bottom]}`:
  - `x` is a fraction of the table's width, from its left edge;
  - `y` is a fraction of its height, from its top edge (negative is above the table).

Form B's profile (`config/default_config.yaml`) declares its invoice number, customer
and date this way. It also sets `printed_header_row: true` and
`footer_from_below: false`, which are right once the page is kept around the table.

`field_insets` (`{role: {x: [left, right], y: [top, bottom]}}`, fractions of the
classified box, each within 0–1) narrows a box to the part holding the handwriting.
Form B uses it for its totals strip. `item_vertical_shift` (−0.5 to 0.5, a fraction
of the cell height) moves item boxes down onto writing that sits on the row's rule;
form B uses 0.25.

**C# side:** the settings page sends its own `table_extraction` and perspective
parameters, so the desktop app gets the form-B fixes only when its defaults match
`config/default_config.yaml`. That means the perspective `margin`, and the form-B
profile's `printed_header_row`, `footer_from_below`, `unruled_fields`,
`item_vertical_shift` and `field_insets`. Update
`PipelineCatalog.cs` (or wherever the app builds its default configuration)
accordingly.
