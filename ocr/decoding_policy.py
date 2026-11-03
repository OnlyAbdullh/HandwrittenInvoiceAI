"""How a cell of each content kind is decoded, shared by every VLM backend.

One table, two consumers:

* ``tools/qwen_worker.py`` -- the transformers runtime, which enforces a
  character set with its own byte-level ``LogitsProcessor``.
* ``ocr/llamacpp/`` -- the llama.cpp runtime, which enforces the same character
  set with a GBNF grammar.

Before this module the table lived inside the worker script. A second backend
would have needed a second copy, and two copies of "which characters may a
quantity cell contain" drift exactly the way the two normalizers did.

Standard library only: the worker imports this from a virtualenv that has
torch and transformers but none of the service's own dependencies.

A digit cell and a product name fail in opposite ways, which is why the policy
is per kind:

* number / date -- generation is **constrained** to the characters the field
  can contain, so a numeric cell can no longer come back as an invented Arabic
  sentence, and there is **no** repetition penalty: it pushes every repeated
  digit down, and "100000" is a perfectly ordinary amount.
* code -- deliberately unconstrained. The invoice-number box contains its
  printed caption ("رقم الفاتورة : 00010"), and a digits-only constraint forces
  the model to invent digits where the caption is (measured: "٥٠٠١٥" for
  "00010", at confidence 0.15). Output validation strips the caption instead.
* arabic_text / any -- unconstrained, with mild penalties against the
  degenerate loops a small VLM falls into on a blank crop.
"""

from __future__ import annotations

DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
NUMBER_CHARS = frozenset(DIGITS + ".,٫٬،- ")
DATE_CHARS = frozenset(DIGITS + "/.-٫، ")

# Longest side a crop is sent at. Larger costs vision tokens -- time on a CPU --
# and a single invoice cell has nothing to gain from more pixels than this.
MAX_IMAGE_SIDE = 768

# Per content kind: the token budget, which character set (if any) generation is
# constrained to, and the penalties. Budgets are deliberately tight -- a
# quantity is a few tokens, and every token past the answer is time spent
# inventing something.
#
# `no_repeat_ngram_size` is a transformers feature with no llama.cpp
# equivalent; the llama.cpp backend applies `repetition_penalty` only and says
# so rather than approximating the other one with a different sampler.
GENERATION: dict[str, dict] = {
    "number": {"max_new_tokens": 16, "charset": NUMBER_CHARS},
    "code": {"max_new_tokens": 24},
    "date": {"max_new_tokens": 24, "charset": DATE_CHARS},
    "arabic_text": {"max_new_tokens": 48, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "any": {"max_new_tokens": 64, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
}


def generation_settings(kind: str | None) -> dict:
    """The decoding settings for `kind`, falling back to the generic ones."""
    return GENERATION.get(kind or "any", GENERATION["any"])
