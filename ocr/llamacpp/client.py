"""Reading one crop through llama-server's chat endpoint.

Each crop becomes one ``/v1/chat/completions`` request carrying the image as a
base64 PNG ``image_url`` -- the multimodal path, decoded by the ``--mmproj``
projector -- followed by the prompt for the cell's content kind. The prompts and
the per-kind decoding policy are the ones the transformers backend uses
(:data:`ocr.engines.qwen_vlm_engine.PROMPTS`,
:mod:`ocr.decoding_policy`), so switching runtime does not change what the model
is asked.

How each part of that policy is carried over, and what is not:

* **Character constraints** (number, date) -- a GBNF grammar listing the allowed
  characters. llama.cpp applies it during sampling, so a numeric cell cannot
  come back as words. Grammars match UTF-8 characters, not byte-level tokens, so
  Arabic-Indic digits stay allowed without the byte bookkeeping the
  transformers backend needs.
* **Repetition penalty** -- sent as ``repeat_penalty``, unless the configuration
  overrides it (Nuqta's model card asks for it off: 1.0).
* **no_repeat_ngram_size** -- llama.cpp has no equivalent; it is *not* applied,
  rather than approximated with a different sampler.
* **Greedy decoding** -- ``temperature: 0``; the same crop gives the same text.

Confidence is computed exactly as the transformers backend computes it: the
geometric mean of the generated tokens' probabilities, each taken from a
softmax over the model's raw logits -- before the grammar and the repetition
penalty, so a digit the grammar forced is scored by how much the model itself
believed in it. llama-server reports those numbers when asked for
``logprobs`` with ``post_sampling_probs`` off (its default) and
``top_logprobs >= 1``; with ``top_logprobs: 0`` it reports a fixed 1.0 for
every token, which would be a fabricated certainty.

It is the model's own token probability, not a calibrated probability that
the reading is correct. When the server returns no log-probabilities the
reading's confidence is None -- unknown -- never a number made up to fill the
gap.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from typing import Mapping, Sequence

from PIL import Image

from core.domain.ocr import Recognition
from ocr.decoding_policy import MAX_IMAGE_SIDE, generation_settings
from ocr.llamacpp.server import LlamaServer

logger = logging.getLogger(__name__)

CHAT_PATH = "/v1/chat/completions"

# Qwen3.5 is a hybrid reasoning family. A thinking block is wasted time on a
# single cell and must never reach the value, whether or not the template
# honoured the request to switch thinking off.
_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """The answer with any thinking block removed.

    Covers both shapes: a whole ``<think>...</think>`` block, and the closing
    tag alone, which is what remains when the chat template opened the block
    in the prompt.
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def charset_grammar(charset: frozenset[str]) -> str:
    """A GBNF grammar accepting any run of `charset`, including the empty one.

    Each character is a quoted literal -- JSON string escaping is valid GBNF
    string escaping -- which sidesteps the character-class metacharacters
    ("-", "]", "^") that a hand-built [...] class would have to escape.
    """
    alternatives = " | ".join(json.dumps(ch, ensure_ascii=False) for ch in sorted(charset))
    return f"root ::= char*\nchar ::= {alternatives}\n"


def encode_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> str:
    """The crop as a PNG data URI, no larger than `max_side` on its long edge.

    PNG, not JPEG: a handwritten stroke is the high-frequency detail JPEG
    discards first. The projector does its own patch-grid resizing, so beyond
    this cap nothing is resampled here.
    """
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def confidence_from_logprobs(entries: Sequence[dict]) -> float | None:
    """Geometric mean of the generated tokens' probabilities, or None.

    None when there is nothing to average (no token was generated). A token
    whose log-probability is missing or -inf had, under the model, probability
    zero; the mean is then zero, which is what the model said.
    """
    if not entries:
        return None
    total = 0.0
    for entry in entries:
        value = entry.get("logprob")
        if value is None or not math.isfinite(float(value)):
            return 0.0
        total += float(value)
    return round(math.exp(total / len(entries)), 4)


class LlamaCppVisionBackend:
    """Reads (image, content kind) pairs through a :class:`LlamaServer`."""

    name = "llamacpp"

    def __init__(
        self,
        server: LlamaServer,
        prompts: Mapping[str, str],
        max_new_tokens: int = 64,
        max_image_side: int = MAX_IMAGE_SIDE,
        constrained_decoding: bool = True,
        disable_thinking: bool = True,
        timeout_s: float = 120.0,
        repetition_penalty: float | None = None,
    ):
        self.server = server
        self.prompts = dict(prompts)
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.constrained_decoding = constrained_decoding
        self.disable_thinking = disable_thinking
        self.timeout_s = timeout_s
        # None: the per-kind policy. A number: that, for every kind.
        self.repetition_penalty = repetition_penalty

        self._grammars: dict[frozenset, str] = {}
        self._warned_no_logprobs = False

    # -- lifecycle --------------------------------------------------------

    def is_ready(self) -> bool:
        return self.server.is_ready()

    def warmup(self) -> None:
        self.server.start()

    def close(self) -> None:
        self.server.close()

    # -- reading ----------------------------------------------------------

    def read_batch(self, items: Sequence[tuple[Image.Image, str]]) -> list[Recognition]:
        """One reading per item, in order. Raises on the first failure.

        Items are read one after another: the server has one slot, so sending
        them together would only queue them inside it.
        """
        return [self.read(image, kind) for image, kind in items]

    def read(self, image: Image.Image, kind: str | None = None) -> Recognition:
        body = self.server.post_json(CHAT_PATH, self.request_for(image, kind), self.timeout_s)
        return self.recognition_from(body)

    def request_for(self, image: Image.Image, kind: str | None) -> dict:
        kind = kind or "any"
        settings = generation_settings(kind)
        prompt = self.prompts.get(kind, self.prompts["any"])

        payload: dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image(image, self.max_image_side)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": min(settings["max_new_tokens"], self.max_new_tokens),
            "temperature": 0.0,
            "stream": False,
            # The scored probability of each generated token. top_logprobs must
            # be at least 1 -- see the module docstring -- and 1 keeps the
            # answer small.
            "logprobs": True,
            "top_logprobs": 1,
        }

        charset = settings.get("charset")
        if self.constrained_decoding and charset:
            payload["grammar"] = self._grammar(charset)

        penalty = (
            self.repetition_penalty if self.repetition_penalty is not None else settings.get("repetition_penalty")
        )
        if penalty is not None:
            payload["repeat_penalty"] = penalty

        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        return payload

    def _grammar(self, charset: frozenset) -> str:
        grammar = self._grammars.get(charset)
        if grammar is None:
            grammar = self._grammars[charset] = charset_grammar(charset)
        return grammar

    def recognition_from(self, body: dict) -> Recognition:
        try:
            choice = body["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"llama-server answered without a completion: {str(body)[:300]}") from exc

        text = strip_reasoning(message.get("content") or "")

        logprobs = choice.get("logprobs")
        entries = logprobs.get("content") if isinstance(logprobs, dict) else None
        if entries is None:
            if not self._warned_no_logprobs:
                logger.warning(
                    "llama-server returned no token log-probabilities; readings from it carry no "
                    "confidence (reported as unknown) and will be flagged for review."
                )
                self._warned_no_logprobs = True
            return Recognition(text=text, confidence=None)

        return Recognition(text=text, confidence=confidence_from_logprobs(entries))
