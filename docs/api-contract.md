# API contract — `/api/v1`

The wire contract between this service and the desktop app
(`InvoiceDigitizationApp`, separate repo). The Pydantic models in `api/schemas.py`
and the formatter in `output/formatters/invoice_json_formatter.py` are the Python
half; the C# DTOs in `Services/AiServiceClient/Contracts.cs` are the other.
**Change this document first, then both sides.** Unknown JSON fields are ignored by
both, so additive changes are backward compatible.

The service binds to `127.0.0.1:8765` only and makes no outbound calls.

## Changes the C# app must mirror

These are the differences from the previous revision of the contract.

| # | Change | C# action |
|---|---|---|
| 1 | `products[].quantity.value` is a JSON **number**, whole (`16`) or fractional (`2.5`). It was an integer. | Deserialize as `decimal`. |
| 2 | `ocr_confidence` is a real 0–1 **recognition** probability (geometric mean of the model's token probabilities). `0` means *unknown / rejected*, never "low-ish". It used to be a constant `0.5` for every cell. It is the model's own certainty, not a calibrated probability that the reading is correct; both OCR backends (transformers and llama.cpp) compute it the same way, from the raw distribution before any character constraint. | Review-threshold logic is unchanged; expect more fields below 0.75 until the recognizer improves. |
| 3 | The `ocr` and `persistence` sections, and `string_matching.dictionary_path`, of the `config` part are **ignored**. What runs and where files go is decided by the machine's own configuration. They are still accepted so old clients keep working. | Stop sending them, or leave as is. |
| 4 | Requests that carry an `Origin` header are refused with `403 FORBIDDEN_ORIGIN`. If `INVOICE_AI_API_TOKEN` is set, every endpoint except `/health` requires header `X-Invoice-AI-Token` (`401 UNAUTHORIZED`). | `HttpClient` sends no `Origin` by default. Generate a token per install and pass it to the service when launching it. |
| 5 | `.heic`, `.heif` and `.pdf` uploads are accepted when the optional packages are installed (`requirements-optional.txt`); otherwise `400 UNSUPPORTED_FORMAT` naming the package. PDF reads the first page; `source.page_count` reports the total. | Optionally offer these file types. |
| 6 | `422 INVALID_CONFIGURATION` is returned when the configuration cannot be built into a pipeline (unknown component, unknown parameter, unimplemented algorithm). `503 ENGINE_NOT_READY` now means only "model files or dependencies missing, or still loading". | — |

## `GET /health`

```json
{ "status": "ok", "version": "1.0.0", "ocr_engine": "layout+qwen", "engine_ready": true, "languages": ["ar", "en"] }
```

`engine_ready` is true only once the OCR worker has loaded; the app polls it.
`ocr_engine` names the reading strategy: `layout+qwen` (transformers) or `layout+llamacpp`
(GGUF via llama.cpp). The response shape is identical for both.

## `GET /configuration`

The service's own default pipeline configuration, in the wire shape of
[settings-config-contract.md](settings-config-contract.md). The settings page loads
this so the defaults it shows are what actually runs.

## `POST /extract`

`multipart/form-data` with three parts:

| Part | Type | |
|---|---|---|
| `file` | file | `.png .jpg .jpeg` (plus `.heic .heif .pdf` if installed), at most 25 MB |
| `options` | JSON string, optional | per-batch catalogs and flags (below) |
| `config` | JSON string, optional | the settings page's configuration; omitted means service defaults |

### `options`

```json
{
  "known_merchants": [{ "customer_id": 1, "name": "حسام المارديني", "aliases": [] }],
  "known_products":  [{ "product_id": 4, "name": "بوري شبابي زارا" }],
  "known_cities":    [{ "city_id": null, "name": "ريف دمشق", "aliases": [] }],
  "max_candidates": 5,
  "return_debug_images": false
}
```

Bare name strings are accepted in place of the objects. An empty catalog disables
matching for that field.

### 200 response

Exactly these keys:

```json
{
  "processing_ms": 21034,
  "source": { "filename": "invoice.png", "page_count": 1, "page_used": 1, "width": 1346, "height": 1743 },
  "invoice_id":  { "value": "00010", "ocr_confidence": 0.93, "bounding_box": { "x": 132, "y": 88, "w": 285, "h": 143 } },
  "customer_name": { "bounding_box": {}, "ocr_confidence": 0.60, "original_value": "حسام المارد بني",
                     "results": [ { "id": "1", "value": "حسام المارديني", "string_matching_score": 0.87 } ] },
  "date": { "value": "2026-08-07", "ocr_confidence": 0.25, "bounding_box": {} },
  "city": { "note": "same shape as customer_name" },
  "products": [ {
    "product_name": { "note": "matched-field shape" },
    "quantity":    { "value": 16, "ocr_confidence": 0.99, "bounding_box": {} },
    "unit_price":  { "value": 13250.0, "ocr_confidence": 0.88, "bounding_box": {} },
    "total_price": { "value": 212000.0, "ocr_confidence": 0.99, "bounding_box": {} }
  } ],
  "total_invoice_price": { "value": 214812.05, "ocr_confidence": 0.92, "bounding_box": {} },
  "enhanced_image_png": null,
  "ocr_input_image_png": null
}
```

Two field shapes:

* **Value field** — `value`, `ocr_confidence`, `bounding_box`. `value` is `null` when
  unread. `date` is ISO-8601 or `null` (a year outside 2000..next year is `null`, not
  itself). Amounts are JSON numbers.
* **Matched field** — `customer_name`, `city`, `product_name`. **No `value`**:
  `original_value` is always the raw OCR text, and `results` is the ranked catalog
  entries (best first, scores 0–1). Choosing one — against the 0.75 threshold — is the
  desktop app's decision, made in one place (`ExtractionMapper`).

Bounding boxes are origin plus size in the coordinate space of the geometrically
corrected page (`source.width` × `source.height`), the same image the UI displays.
The debug images are downscaled to a maximum width, so boxes do not map onto them 1:1.

### Errors

```json
{ "error": { "code": "INVALID_CONFIGURATION", "message": "...", "detail": "..." } }
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `UNSUPPORTED_FORMAT` | extension not supported, or its optional decoder missing |
| 400 | `EMPTY_FILE`, `CORRUPT_FILE` | |
| 401 | `UNAUTHORIZED` | token required and missing or wrong |
| 403 | `FORBIDDEN_ORIGIN` | request carries a foreign `Origin` header |
| 413 | `FILE_TOO_LARGE` | over 25 MB |
| 422 | `INVALID_OPTIONS`, `INVALID_CONFIGURATION` | |
| 503 | `ENGINE_NOT_READY` | model files or dependencies missing, or still loading |
| 500 | `OCR_FAILED` | an unexpected failure while processing |

## Concurrency

`/extract` is a synchronous handler on a thread pool, so `/health` stays responsive
during a 20–60 s extraction. One worker process serves all requests; reads are
serialized, and a worker that stops answering for `timeout_s` (default 120 s) is
killed and restarted.
