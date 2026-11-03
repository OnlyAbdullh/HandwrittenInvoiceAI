# Nuqta Arabic OCR 4B (GGUF) with llama.cpp — Windows setup

This sets up the second OCR backend: the **Nuqta Arabic OCR 4B** model in GGUF
format, run by **llama.cpp**. It sits beside the existing transformers (Qwen)
backend; neither replaces the other, and switching between them is one
environment variable.

Everything else about the pipeline is unchanged. The page is still corrected,
the printed grid is still detected and classified, and blank cells are still
skipped. Each remaining cell is still read with the same prompt and checked by
the same validation, and the desktop app gets the same response shape.

The service starts `llama-server` itself, keeps it loaded, and stops it when the
service stops. You never start it by hand.

Commands are PowerShell, run from the project folder (the one with `main.py`).

---

## 1. Install llama.cpp

You need a llama.cpp build that knows the Qwen3.5 architecture: **build b7990 or
newer** (February 2026). Any current release qualifies.

**Option A — winget (simplest).** Installs the Vulkan build, which runs on any
modern GPU, including the RTX 3050, and falls back to the CPU:

```powershell
winget install llama.cpp
```

Close and reopen PowerShell so `PATH` picks it up, then:

```powershell
llama-server --version
```

**Option B — the CUDA build (usually fastest on an NVIDIA card).** From
<https://github.com/ggml-org/llama.cpp/releases>, download the
`llama-b<build>-bin-win-cuda-<version>-x64.zip` asset, and — if the NVIDIA CUDA
runtime is not already installed — the matching
`cudart-llama-bin-win-cuda-<version>-x64.zip`. Unzip both into one folder, for
example `C:\tools\llama.cpp`, and tell the service where it is:

```powershell
[Environment]::SetEnvironmentVariable("INVOICE_AI_LLAMA_SERVER", "C:\tools\llama.cpp\llama-server.exe", "User")
```

(`INVOICE_AI_LLAMA_SERVER` may also name the folder; the service finds
`llama-server.exe` inside it.)

## 2. Download the two model files

The model is two files, and both are required:

| File | What it is |
|---|---|
| `nuqta-ocr-4b-Q4_K_M.gguf` | the language model (4-bit) |
| `nuqta-ocr-4b-mmproj-f16.gguf` | the vision projector — what lets the model see the image |

They are at <https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF/tree/main>.

Put them in the project's `OCR_Models` folder. It is git-ignored, so the weights
can never be committed, and it is where the service looks by default:

```powershell
$dir = "OCR_Models\Nuqta-Arabic-OCR-4B-GGUF"
New-Item -ItemType Directory -Force $dir | Out-Null
$base = "https://huggingface.co/SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF/resolve/main"
curl.exe -L -o "$dir\nuqta-ocr-4b-Q4_K_M.gguf"     "$base/nuqta-ocr-4b-Q4_K_M.gguf"
curl.exe -L -o "$dir\nuqta-ocr-4b-mmproj-f16.gguf" "$base/nuqta-ocr-4b-mmproj-f16.gguf"
```

(`curl.exe`, not `curl`: in Windows PowerShell 5, `curl` is an alias for
`Invoke-WebRequest`, which is very slow for files this size. Downloading through
the browser works just as well.)

**Keeping the files somewhere else** — another drive, a shared models folder —
is two environment variables instead:

```powershell
[Environment]::SetEnvironmentVariable("INVOICE_AI_NUQTA_MODEL",  "D:\models\nuqta\nuqta-ocr-4b-Q4_K_M.gguf", "User")
[Environment]::SetEnvironmentVariable("INVOICE_AI_NUQTA_MMPROJ", "D:\models\nuqta\nuqta-ocr-4b-mmproj-f16.gguf", "User")
```

Variables set with `"User"` apply to programs started **after** they were set:
open a new PowerShell window, and restart the desktop app if it launches the
service.

## 3. Check the setup

```powershell
python tools\check_llamacpp_setup.py
```

The check runs without downloading or changing anything. It finds `llama-server`
and checks its build. It checks that both model files exist, are real GGUF files
(not an HTML error page saved under a `.gguf` name, not a half-finished
download), and are the right two files the right way round. It also shows RAM
and GPU.

To also start the server once, exactly as the service will, and read an image:

```powershell
python tools\check_llamacpp_setup.py --start
python tools\check_llamacpp_setup.py --image C:\path\to\a\cell.png --kind number
```

`--kind` is one of `number`, `date`, `code`, `arabic_text` or `any`, and selects
the prompt and character constraint, as the pipeline does per cell.

## 4. Run with Nuqta

The Nuqta pipeline is `config/nuqta_config.yaml`. It is `config/default_config.yaml`
with only the recognizer changed.

**The HTTP service** (what the desktop app talks to):

```powershell
.venv\Scripts\activate
$env:INVOICE_AI_CONFIG_PATH = "config/nuqta_config.yaml"
python -m api.main
```

The first start loads the model; `GET /api/v1/health` reports
`"ocr_engine": "layout+llamacpp"` and `"engine_ready": true` once it is loaded.
To make Nuqta the default for every launch, including when the desktop app
starts the service:

```powershell
[Environment]::SetEnvironmentVariable("INVOICE_AI_CONFIG_PATH", "config/nuqta_config.yaml", "User")
```

**One invoice from the command line:**

```powershell
python main.py C:\path\to\invoice.jpg config\nuqta_config.yaml
```

**A folder of invoices** (the model is loaded once for the whole batch):

```powershell
python run_batch.py C:\path\to\invoices --config config\nuqta_config.yaml
```

**Accuracy, against the labelled invoices:**

```powershell
python tools\evaluate.py --config config\nuqta_config.yaml --label nuqta
```

## 5. Switch back to the transformers (Qwen) backend

Nothing about the existing backend was removed. Point the service back at the
default configuration:

```powershell
Remove-Item Env:INVOICE_AI_CONFIG_PATH                                     # this window
[Environment]::SetEnvironmentVariable("INVOICE_AI_CONFIG_PATH", $null, "User")   # permanently
```

For the command line, pass `config\qwen_config.yaml` (or nothing: `main.py`
defaults to it). Both backends can stay installed side by side.

## 6. GPU, CPU and memory

The defaults suit an RTX 3050 with 4 GB of VRAM, 24 GB of RAM and a Ryzen 5 5600H:

| Setting (`config/nuqta_config.yaml`, `ocr.recognizer.params`) | Default | Meaning |
|---|---|---|
| `gpu_layers` (env `INVOICE_AI_LLAMA_GPU_LAYERS`) | `auto` | llama.cpp puts as many layers on the GPU as fit, keeping 1 GB of VRAM free, and runs the rest on the CPU. A model that does not fit runs slower; it does not crash. |
| `mmproj_offload` | `true` | Run the vision encoder on the GPU. `false` frees its VRAM for language-model layers, at the cost of slower image encoding. |
| `ctx_size` | `4096` | Context per request. One cell needs a few hundred tokens; the model's full context would not fit a laptop. |
| `threads` | llama.cpp's choice | CPU threads. On the 5600H, `6` (one per physical core) is a sensible explicit value. |
| `image_min_tokens` | `auto` | Small cell crops are enlarged to the size the model was trained to read: the minimum the mmproj file declares (about 1 megapixel for Nuqta), or 64 tokens if it declares none. Larger means more accurate and slower; a number forces it. |
| `repetition_penalty` | `1.0` | Off, as Nuqta's model card asks. |
| `prompts` | Nuqta's own | `Transcribe the Arabic text in this image without any diacritics.` — the model card's prompt. Number and date cells still get their digits-only constraint. |

The server runs one request at a time (`--parallel 1`), so memory does not grow
with concurrent invoices.

| You want | Set |
|---|---|
| Fastest, if the model fits in VRAM | `INVOICE_AI_LLAMA_GPU_LAYERS = "all"` |
| GPU, but it runs out of memory | leave `auto`; if needed set `mmproj_offload: false`, or a layer count such as `20` |
| CPU only | `INVOICE_AI_LLAMA_GPU_LAYERS = "0"` |

## 7. Troubleshooting

When the server fails, the service's error includes the last lines of
`llama-server`'s own log, and usually a plain-language hint.

| Message | Cause and fix |
|---|---|
| `'llama-server' was not found on PATH` | Open a new PowerShell window after installing, or set `INVOICE_AI_LLAMA_SERVER` to the full path of `llama-server.exe`. |
| `Model file not found: ...` | The file is not where the configuration says. Check the path in the message; see section 2. |
| `... is not a GGUF file (it starts with b'<!DO')` | The browser saved a web page instead of the model. Download it again. |
| `... ends inside its metadata header. The file is truncated` | The download stopped part way. Download it again. |
| `model_path and mmproj_path are swapped` | The two variables point at each other's file. Swap them. |
| `does not know the model's architecture` / `unknown model architecture` | llama.cpp is older than b7990. `winget upgrade llama.cpp`, or download a current release. |
| `The GPU ran out of memory` | See section 6: leave `gpu_layers` on `auto`, set `mmproj_offload: false`, or use `0` for CPU only. |
| `started without image support` | The projector file did not load as a vision projector. Re-download `nuqta-ocr-4b-mmproj-f16.gguf`. |
| `/health` shows `"engine_ready": false` | The model is still loading (a minute or two on first start), or it failed: the service console says why. `python tools\check_llamacpp_setup.py --start` reproduces the start on its own. |
| `503 ENGINE_NOT_READY` from `/extract` | Same as above: model files or llama.cpp missing, or still loading. |
| `llama-server gave no answer within 120s` | One cell took too long; it comes back empty and flagged, and the rest of the invoice is still read. On a slow CPU, raise `timeout_s`. |
| Warning `returned no token log-probabilities` | This llama.cpp build did not report token probabilities, so confidence is unknown and fields are flagged for review. Update llama.cpp. |
| Very slow on the CPU | Expected for a 4B model; use the GPU (section 6). Each non-blank cell is one model call. |

## 8. How confidence is reported

Each reading's `ocr_confidence` is the geometric mean of the probabilities the
model gave the tokens it generated. They are taken from its raw output
distribution, before the character constraint and the repetition penalty, so a
digit the constraint forced is scored by how much the model itself believed in
it. That is the same definition the transformers backend uses.

It is the model's own certainty, **not** a calibrated probability that the
reading is correct. The desktop app's 0.75 review threshold treats it as a
signal for which fields to check first.

If `llama-server` returns no probabilities, the confidence is unknown. It is
reported as `0`, which the API contract defines as unknown, and the field is
flagged for review. It is never invented.

## 9. What this backend does and does not do

- **Character constraints.** Number and date cells are constrained to digits
  (ASCII, Arabic-Indic and Persian) and their separators by a llama.cpp GBNF
  grammar, as the transformers backend constrains them with its own decoder.
- **Repetition penalty.** Off for Nuqta (its model card); the per-kind default (1.1 on
  free text) applies to a GGUF model whose config does not say.
- **`no_repeat_ngram_size`.** The transformers backend's n-gram ban has no
  llama.cpp equivalent and is **not** applied.
- **Thinking mode.** Qwen3.5 can reason in a `<think>` block. It is switched off
  per request, and any thinking text is removed before validation.
- **Prompts.** `nuqta_config.yaml` uses the model card's own prompt for every cell
  kind. A GGUF model whose config sets no `prompts:` gets the transformers backend's
  per-kind prompts.
