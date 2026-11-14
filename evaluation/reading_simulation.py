"""Simulated readings of catalog names, for when labelled invoices are few.

Two labelled invoices cannot rank product matchers: they carry four products,
none of which has a look-alike in the catalog. Until more are labelled, the
only way to exercise a matcher against the catalog's real siblings is to write
each catalog name the way a recognizer would have read it.

The errors are not invented. :class:`ErrorChannel` is estimated from real
(label, reading) pairs -- the recognizers' own output on this project's
invoices -- and reproduces what was measured: which letters are misread and
as what, how often a letter is dropped or inserted, how often a word is split.

What it cannot know, and assumes:

* a letter seen fewer than `MIN_OBSERVATIONS` times in the labels is misread at
  the average rate, as a random letter;
* digits and Latin letters are misread at that same average rate (the labelled
  name cells contain none);
* shorthand is a random subset of the name's words (:func:`shorten`). Real
  writers choose which words to drop.

So these numbers compare matchers; they do not predict accuracy on invoices.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable

from string_matching.normalization import normalize_text

MIN_OBSERVATIONS = 40

# A reading further than this from its label is another row's text or an
# invented sentence, not a misreading of this one, and teaches nothing about
# which letters are confused.
MAX_PAIR_ERROR_RATE = 0.6

_ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
_LATIN = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"

SHORTHAND_MODES = ("full", "no_numbers", "half", "three_words", "two_words")


def _alignment(label: str, reading: str) -> tuple[int, list[tuple[str, str]]]:
    """Edit distance and one optimal list of (label char, read char) pairs.

    An empty first member is an insertion, an empty second a deletion.
    """
    rows, cols = len(label), len(reading)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        table[i][0] = i
    for j in range(cols + 1):
        table[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + (label[i - 1] != reading[j - 1]),
            )

    pairs: list[tuple[str, str]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (label[i - 1] != reading[j - 1]):
            pairs.append((label[i - 1], reading[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            pairs.append((label[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", reading[j - 1]))
            j -= 1
    return table[rows][cols], pairs[::-1]


class ErrorChannel:
    """Character-level misreading statistics, and a sampler built on them."""

    def __init__(self, pairs: Iterable[tuple[str, str]]):
        self.seen: Counter[str] = Counter()
        self.misread: dict[str, Counter[str]] = defaultdict(Counter)
        self.inserted: Counter[str] = Counter()
        self.pairs_used = 0

        for label, reading in pairs:
            truth, read = normalize_text(label), normalize_text(reading)
            if not truth or not read:
                continue
            distance, aligned = _alignment(truth, read)
            if distance / len(truth) > MAX_PAIR_ERROR_RATE:
                continue

            self.pairs_used += 1
            for written, got in aligned:
                if written:
                    self.seen[written] += 1
                    if written != got:
                        self.misread[written][got] += 1
                else:
                    self.inserted[got] += 1

        letters = sum(n for ch, n in self.seen.items() if ch != " ")
        errors = sum(sum(c.values()) for ch, c in self.misread.items() if ch != " ")
        dropped = sum(c[""] for ch, c in self.misread.items() if ch != " ")
        total = sum(self.seen.values())

        self.average_error = errors / letters if letters else 0.0
        self.drop_share = dropped / errors if errors else 0.0
        self.insert_rate = (
            sum(n for ch, n in self.inserted.items() if ch != " ") / total if total else 0.0
        )
        self.split_rate = self.inserted[" "] / total if total else 0.0

    def error_rate(self, char: str) -> float:
        if self.seen[char] >= MIN_OBSERVATIONS:
            return sum(self.misread[char].values()) / self.seen[char]
        return self.average_error

    def corrupt(self, rng: random.Random, text: str, scale: float = 1.0) -> str:
        """`text` as a recognizer might have read it. `scale` 0 returns it unchanged."""
        if scale <= 0:
            return text

        inserts = [(ch, n) for ch, n in self.inserted.items() if ch != " "]
        out: list[str] = []

        for char in text:
            if char == " ":
                out.append(char)
                continue

            if rng.random() < self.error_rate(char) * scale:
                if self.seen[char] >= MIN_OBSERVATIONS:
                    out.append(_weighted(rng, list(self.misread[char].items())))
                elif rng.random() >= self.drop_share:
                    pool = _DIGITS if char.isdigit() else _LATIN if "a" <= char <= "z" else _ARABIC
                    out.append(rng.choice(pool))
            else:
                out.append(char)

            draw = rng.random()
            if inserts and draw < self.insert_rate * scale:
                out.append(_weighted(rng, inserts))
            elif draw < (self.insert_rate + self.split_rate) * scale:
                out.append(" ")

        return "".join(out)


def _weighted(rng: random.Random, weighted: list[tuple[str, int]]) -> str:
    draw = rng.uniform(0, sum(weight for _, weight in weighted))
    for value, weight in weighted:
        draw -= weight
        if draw <= 0:
            return value
    return weighted[-1][0]


def shorten(rng: random.Random, words: list[str], mode: str) -> list[str] | None:
    """A shortened form of a name's words, or None when `mode` does not apply.

    Word order is kept. `no_numbers` applies only to names that have numbers;
    the subset modes only to names long enough to lose a word.
    """
    if mode == "full":
        return words
    if mode == "no_numbers":
        kept = [word for word in words if not word.isdigit()]
        return kept if kept and len(kept) < len(words) else None

    if mode == "half":
        keep = math.ceil(len(words) / 2) if len(words) >= 3 else 0
    elif mode == "three_words":
        keep = 3 if len(words) >= 4 else 0
    elif mode == "two_words":
        keep = 2 if len(words) >= 3 else 0
    else:
        raise ValueError(f"Unknown shorthand mode '{mode}'. Known: {SHORTHAND_MODES}.")

    if not keep:
        return None
    positions = sorted(rng.sample(range(len(words)), keep))
    return [words[i] for i in positions]
