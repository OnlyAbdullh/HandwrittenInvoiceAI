"""Byte-level constrained decoding, on a fake byte-level vocabulary.

Qwen tokenizes an Arabic-Indic digit as two byte tokens (0xD9, then e.g. 0xA1),
neither of which is a character. The first version of the constraint tested
each token's decoded text, banned both halves, and so forced ASCII digits on a
model that reads these invoices in Arabic-Indic -- numeric accuracy dropped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tools.qwen_worker import _NUMBER_CHARS, _CharsetConstraint  # noqa: E402

# A tiny vocabulary shaped like Qwen's: ASCII digits whole, Arabic-Indic digits
# as lead byte + continuation byte, plus tokens that must stay banned.
VOCAB = [
    b"1", b"2", b",",          # 0-2 allowed whole
    b"\xd9",                   # 3 lead byte of U+0660..U+066C
    b"\xa1", b"\xa2",          # 4-5 continuations -> "١", "٢"
    b"\x84",                   # 6 continuation -> "ل" (a letter: never allowed)
    b"\xd9\xa5",               # 7 "٥" as one token
    b"abc",                    # 8 letters
    None,                      # 9 special token with no byte form
]
STOP = {9}


@pytest.fixture
def constraint():
    return _CharsetConstraint(_NUMBER_CHARS, VOCAB, STOP)


def allowed(constraint, pending: bytes) -> set[int]:
    return set(constraint._allowed(pending, "cpu").tolist())


def test_between_characters_digits_lead_bytes_and_stop_are_allowed(constraint):
    assert allowed(constraint, b"") == {0, 1, 2, 3, 7, 9}


def test_after_a_lead_byte_only_digit_continuations_are_allowed(constraint):
    # Not "\x84": D9 84 is the letter lam. Not stop: the character is unfinished.
    assert allowed(constraint, b"\xd9") == {4, 5}


def test_pending_state_is_read_from_the_generated_tokens(constraint):
    assert constraint._pending([3]) == b"\xd9"
    assert constraint._pending([3, 4]) == b""
    assert constraint._pending([0, 7, 2]) == b""


def test_processor_masks_per_row(constraint):
    processor = constraint.processor(prompt_len=1)
    input_ids = torch.tensor([[99, 3], [99, 0]])  # row 0 mid-digit, row 1 between chars
    scores = torch.zeros(2, len(VOCAB))

    out = processor(input_ids, scores)

    assert torch.isfinite(out[0]).nonzero().flatten().tolist() == [4, 5]
    assert set(torch.isfinite(out[1]).nonzero().flatten().tolist()) == {0, 1, 2, 3, 7, 9}
