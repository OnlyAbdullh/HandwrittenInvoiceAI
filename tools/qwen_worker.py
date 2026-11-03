# tools/qwen_worker.py
"""The Qwen VLM runtime, and the long-lived worker process that serves it.

Exists as a separate process because Qwen3.5 needs transformers>=5.0 while
surya-ocr 0.17.1 pins <5.0. They cannot share a process, so the pipeline spawns
this script with the other venv's python.exe and talks to it over stdin/stdout.
The same :class:`QwenRuntime` is imported in-process by the ``transformers``
backend of ``ocr/engines/qwen_vlm_engine.py``, so there is one implementation of
"read a crop", not two.

Protocol: one JSON object per line in, one per line out.

    in   {"id": 7, "items": [{"image": "<base64 png>", "prompt": "...", "kind": "number"}, ...]}
    out  {"id": 7, "results": [{"text": "...", "confidence": 0.93, "min_prob": 0.71}
                               | {"error": "..."}, ...]}
    "READY" is printed once the model is loaded.

`results` always has exactly one entry per item, in order.

Decoding is chosen per content kind, because a digit cell and a product name
fail in opposite ways:

* number / code / date -- generation is **constrained** to the characters the
  field can contain, so a numeric cell can no longer come back as an invented
  Arabic sentence, and there is **no** repetition penalty: `no_repeat_ngram_size`
  forbids repeating any token trigram (so "100000" was literally impossible to
  write) and `repetition_penalty` pushes every repeated digit down.
* arabic_text / any -- unconstrained, with mild penalties against the
  degenerate loops a small VLM falls into on a blank crop.

Confidence is the geometric mean of the generated tokens' probabilities under
the *unconstrained* model (raw logits), so forcing a digit does not inflate the
score of a digit the model did not believe in.

Two architectures, auto-detected from the model's own config.json:

  qwen3_5      Arabic-Qwen3.5-OCR-v4. Full precision, ~0.9B params. The model
               card requires both image dimensions to be multiples of 64.
  qwen2_5_vl   Arabic-handwritten-OCR-*-Qwen2.5-VL-3B, shipped pre-quantized.
               Its processor resizes on a 28px patch grid itself, so any resize
               here would be a second, pointless distortion.

    python tools/qwen_worker.py --model OCR_Models/Arabic-Qwen3.5-OCR-v4
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MAX_SIDE = 768
MULTIPLE = 64

ARCH_CLASSES = {
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}
FALLBACK_CLASS = "AutoModelForImageTextToText"

_DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_NUMBER_CHARS = frozenset(_DIGITS + ".,٫٬،- ")
_DATE_CHARS = frozenset(_DIGITS + "/.-٫، ")

# Per content kind: the token budget, which character set (if any) generation is
# constrained to, and the penalties. Budgets are deliberately tight -- a
# quantity is a few tokens, and every token past the answer is time spent
# inventing something.
GENERATION = {
    "number": {"max_new_tokens": 16, "charset": _NUMBER_CHARS},
    # Not constrained: the invoice-number box contains its printed caption
    # ("رقم الفاتورة : 00010"), and a digits-only constraint forces the model to
    # invent digits where the caption is (measured: "٥٠٠١٥" for "00010", at
    # confidence 0.15). Output validation strips the caption instead.
    "code": {"max_new_tokens": 24},
    "date": {"max_new_tokens": 24, "charset": _DATE_CHARS},
    "arabic_text": {"max_new_tokens": 48, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "any": {"max_new_tokens": 64, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
}

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte <-> printable-character map, which byte-level BPE vocabularies use.

    Defined here rather than imported: transformers moved its copy between major
    versions, and this mapping is fixed by the vocabulary format itself.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(c) for c in chars)))


def _vocabulary_bytes(tokenizer) -> tuple[list[bytes | None], set[int]]:
    """The raw UTF-8 bytes of every vocabulary token, and the stop-token ids.

    Qwen's tokenizer is byte-level BPE: an Arabic-Indic digit such as "١" is
    TWO tokens (the lead byte 0xD9, then 0xA1), and neither decodes to a
    character on its own. A constraint that tested each token's decoded text
    therefore banned every Arabic-Indic digit -- forcing ASCII digits on a model
    that reads these invoices in Arabic-Indic, which measurably degraded it.
    Constraints have to be checked on bytes.
    """
    byte_decoder = {char: byte for byte, char in _bytes_to_unicode().items()}
    token_bytes: list[bytes | None] = []
    for token in tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))):
        if token is None or any(ch not in byte_decoder for ch in token):
            token_bytes.append(None)  # special/added token: no byte form
        else:
            token_bytes.append(bytes(byte_decoder[ch] for ch in token))

    stop_ids: set[int] = set()
    for special in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(special, int):
            stop_ids.add(special)
        elif isinstance(special, (list, tuple)):
            stop_ids.update(i for i in special if isinstance(i, int))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(marker_id, int) and marker_id >= 0:
            stop_ids.add(marker_id)
    return token_bytes, stop_ids


class _CharsetConstraint:
    """Restricts generation to text made only of `charset`, checked on bytes.

    The state of a row is its *pending* bytes: the incomplete UTF-8 tail of what
    it has generated so far (b"" between characters, b"\\xd9" halfway through
    an Arabic-Indic digit). A token is allowed when pending + its bytes is a run
    of whole allowed characters followed by, at most, a strict prefix of one
    more. Stop tokens are allowed only between characters. The allowed-id set is
    computed once per pending state and cached; there are only a handful.
    """

    def __init__(self, charset: frozenset, token_bytes: list[bytes | None], stop_ids: set[int]):
        self._encodings = {ch.encode("utf-8") for ch in charset}
        self._prefixes = {enc[:i] for enc in self._encodings for i in range(1, len(enc))}
        self._token_bytes = token_bytes
        self._stop_ids = stop_ids
        self._by_state: dict[bytes, object] = {}

    def _remainder(self, data: bytes) -> bytes | None:
        """Pending bytes left after consuming whole allowed chars, or None if invalid."""
        i = 0
        while i < len(data):
            for enc in self._encodings:
                if data.startswith(enc, i):
                    i += len(enc)
                    break
            else:
                rest = data[i:]
                return rest if rest in self._prefixes else None
        return b""

    def _allowed(self, pending: bytes, device):
        import torch

        key = (pending, str(device))
        cached = self._by_state.get(key)
        if cached is not None:
            return cached

        ids = [
            token_id
            for token_id, data in enumerate(self._token_bytes)
            if data and self._remainder(pending + data) is not None
        ]
        if not pending:
            ids.extend(self._stop_ids)
        allowed = torch.tensor(sorted(set(ids)), device=device)
        self._by_state[key] = allowed
        return allowed

    def _pending(self, generated: list[int]) -> bytes:
        data = b"".join(self._token_bytes[t] or b"" for t in generated)
        return self._remainder(data) or b""

    def processor(self, prompt_len: int):
        import torch
        from transformers import LogitsProcessor

        constraint = self

        class _Processor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, float("-inf"))
                for row in range(input_ids.shape[0]):
                    pending = constraint._pending(input_ids[row, prompt_len:].tolist())
                    mask[row, constraint._allowed(pending, scores.device)] = 0.0
                return scores + mask

        return _Processor()


def inspect_model(model_dir: str) -> dict:
    """Reads the model's own config.json.

    Two things matter and both must come from the file, never from a guess:
    the architecture (wrong class = wrong preprocessing) and whether the
    weights are ALREADY quantized (see `QwenRuntime._load` for why that decides
    everything).
    """
    info = {"arch": "qwen3_5", "quantized": False}
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return info
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return info

    info["arch"] = cfg.get("model_type") or "qwen3_5"
    info["quantized"] = bool(cfg.get("quantization_config"))
    return info


def fit(img, arch: str):
    """Sizes the crop for the target architecture.

    qwen3_5 RESIZES to a multiple of 64 rather than padding to it. Padding is
    the more principled choice -- resizing changes the aspect ratio -- but it
    was measured and it lost: 5/12 cells correct with resize, 3/12 with
    padding, on the identical image. Do not reintroduce padding without a new
    measurement (tools/evaluate.py).
    """
    from PIL import Image

    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    if arch != "qwen3_5":
        return img

    w, h = img.size
    nw = ((w + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    nh = ((h + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    return img.resize((nw, nh), Image.Resampling.LANCZOS) if (nw, nh) != (w, h) else img


class QwenRuntime:
    """A loaded model plus everything needed to read crops with it."""

    def __init__(self, model_dir: str, arch: str | None = None, max_new_tokens: int = 64):
        info = inspect_model(model_dir)
        self.arch = arch or info["arch"]
        self.max_new_tokens_cap = max_new_tokens
        self.model, self.processor = self._load(model_dir, self.arch, info["quantized"])
        self.processor.tokenizer.padding_side = "left"
        # Built on the first constrained read, then reused for every crop.
        self._token_bytes: list[bytes | None] | None = None
        self._stop_ids: set[int] = set()
        self._allowed_ids: dict[frozenset, _CharsetConstraint] = {}

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

    # -- loading ----------------------------------------------------------

    @staticmethod
    def _load(model_dir: str, arch: str, quantized: bool):
        import torch
        import transformers
        from transformers import AutoProcessor

        cls_name = ARCH_CLASSES.get(arch, FALLBACK_CLASS)
        cls = getattr(transformers, cls_name, None) or getattr(transformers, FALLBACK_CLASS, None)
        if cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} exposes neither "
                f"{cls_name} nor {FALLBACK_CLASS}. Qwen3.5 needs >=5.0."
            )

        cuda = torch.cuda.is_available()
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

        if quantized:
            # Pre-quantized (bitsandbytes 4-bit) weights only run on CUDA. Say
            # so plainly instead of failing deep inside the loader.
            if not cuda:
                raise RuntimeError(
                    f"'{model_dir}' ships 4-bit quantized weights, which need a CUDA GPU. "
                    "On a CPU-only machine use the full-precision Arabic-Qwen3.5-OCR-v4."
                )
            # The scheme lives in config.json; passing a fresh BitsAndBytesConfig
            # is ignored at best and a scheme mismatch at worst. {"": 0} pins the
            # whole model to GPU 0 -- "auto" silently spills layers to CPU, and a
            # partially-offloaded quantized model generates noise, not an error.
            kwargs["device_map"] = {"": 0}
            label = "4bit (from the model's own config)"
        else:
            # Never quantize a model that was not shipped quantized: forcing
            # 4-bit on Arabic-Qwen3.5-OCR-v4 was measured to turn repairable
            # digit errors ("017,68") into unrepairable ones ("517,86,42").
            # float32, the precision the 5/12 baseline was measured at.
            kwargs["dtype"] = torch.float32
            if cuda:
                kwargs["device_map"] = {"": 0}
            label = "float32 (unquantized)"

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = cls.from_pretrained(model_dir, **kwargs).eval()

        print(
            f"[worker] arch={arch} class={cls.__name__} cuda={cuda} weights={label}",
            file=sys.stderr,
            flush=True,
        )
        return model, processor

    # -- constrained decoding ---------------------------------------------

    def _constraint(self, charset: frozenset | None, prompt_len: int):
        if charset is None:
            return None

        from transformers import LogitsProcessorList

        if self._token_bytes is None:
            self._token_bytes, self._stop_ids = _vocabulary_bytes(self.processor.tokenizer)
        constraint = self._allowed_ids.get(charset)
        if constraint is None:
            constraint = _CharsetConstraint(charset, self._token_bytes, self._stop_ids)
            self._allowed_ids[charset] = constraint
        return LogitsProcessorList([constraint.processor(prompt_len)])

    # -- reading ----------------------------------------------------------

    def _inputs(self, images: list, prompts: list[str]):
        texts, vision = [], []
        for img, prompt in zip(images, prompts):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            vision.extend(self._process_vision_info(messages)[0]
                          if self._process_vision_info else [img])
        return self.processor(text=texts, images=vision, padding=True,
                              return_tensors="pt").to(self.model.device)

    def _generate(self, images: list, prompts: list[str], kind: str) -> list[dict]:
        import torch

        settings = GENERATION.get(kind, GENERATION["any"])
        inputs = self._inputs(images, prompts)
        tokenizer = self.processor.tokenizer

        generate_kwargs = {
            "max_new_tokens": min(settings["max_new_tokens"], self.max_new_tokens_cap),
            "do_sample": False,  # same crop must give the same answer
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_logits": True,
        }
        for key in ("repetition_penalty", "no_repeat_ngram_size"):
            if key in settings:
                generate_kwargs[key] = settings[key]
        constraint = self._constraint(settings.get("charset"), inputs.input_ids.shape[1])
        if constraint is not None:
            generate_kwargs["logits_processor"] = constraint

        with torch.inference_mode():
            out = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs.input_ids.shape[1]
        generated = out.sequences[:, prompt_len:]

        # Raw (unconstrained, unpenalized) logits -> the model's own belief in
        # each token it emitted.
        logits = torch.stack(out.logits, dim=1).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(-1, generated.unsqueeze(-1)).squeeze(-1)

        stop_ids = {tokenizer.pad_token_id}
        eos = tokenizer.eos_token_id
        stop_ids.update(eos if isinstance(eos, (list, tuple)) else [eos])

        results = []
        for row in range(generated.shape[0]):
            tokens = generated[row].tolist()
            length = next((i for i, t in enumerate(tokens) if t in stop_ids), len(tokens))
            text = tokenizer.decode(tokens[:length], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
            if length == 0:
                results.append({"text": text, "confidence": None, "min_prob": None})
                continue
            row_lp = chosen[row, :length]
            results.append({
                "text": text,
                "confidence": round(float(torch.exp(row_lp.mean())), 4),
                "min_prob": round(float(torch.exp(row_lp.min())), 4),
            })
        return results

    def read_batch(self, items: list[dict]) -> list[dict]:
        """Read `items` ({"image": PIL, "prompt", "kind"}), one result each, in order.

        Items are grouped by kind (each kind decodes differently) and each group
        runs as one batched `generate`. A group that runs out of memory is
        retried one item at a time rather than failing the page.
        """
        import torch

        results: list[dict | None] = [None] * len(items)
        groups: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault(item.get("kind") or "any", []).append(index)

        for kind, indices in groups.items():
            images = [fit(items[i]["image"], self.arch) for i in indices]
            prompts = [items[i]["prompt"] for i in indices]
            try:
                batch = self._generate(images, prompts, kind)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch = []
                for img, prompt in zip(images, prompts):
                    try:
                        batch.extend(self._generate([img], [prompt], kind))
                    except Exception as exc:  # noqa: BLE001
                        batch.append({"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                batch = [{"error": str(exc)} for _ in indices]

            for i, result in zip(indices, batch):
                results[i] = result

        return results


def _decode_image(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def main() -> None:
    # Windows defaults stdio to cp1252/cp1256, which cannot represent Arabic.
    # Both ends of the pipe agree on UTF-8 explicitly.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--arch", default=None, help="override auto-detection: " + ", ".join(ARCH_CLASSES))
    args = ap.parse_args()

    try:
        runtime = QwenRuntime(args.model, arch=args.arch, max_new_tokens=args.max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        sys.exit(1)

    # The parent blocks on this line before sending any work.
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            req = json.loads(line)
            request_id = req.get("id")
            items = [
                {"image": _decode_image(it["image"]), "prompt": it["prompt"], "kind": it.get("kind")}
                for it in req["items"]
            ]
            response = {"id": request_id, "results": runtime.read_batch(items)}
        except Exception as exc:  # noqa: BLE001
            # Never die on one bad request: the parent needs the rest of the page.
            response = {"id": request_id, "error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
