"""Turning labelled cell crops into a training set for the recognizer.

The rules, kept apart from the tool that applies them
(`tools/build_finetune_dataset.py`) so they can be tested without a photo:

* **The target is what the crop shows**: the digits the writer used
  (٠١٢٣٤٥٦٧٨٩ on these invoices; the truth file's `digits` says), the separator
  as written, and `EMPTY` -- the word the service's prompts ask for -- for a
  cell that holds no writing. The base model already transcribes handwritten
  Arabic-Indic digits as Arabic-Indic digits, and everything downstream folds
  them; a target in the other script would spend the training data on teaching
  the model to translate, not to read. (Tried once, on 20 cells: the tuned
  model switched to 0-9 and read 15 fields of 41 on the held-out invoices
  against the base model's 30. Too small to be a measurement, large enough to
  not repeat.) Printed numbers -- an invoice number -- stay as printed.
* **The prompt is the service's own** for that kind of cell. A recognizer
  trained under one instruction and asked under another is being tested on
  something it was not trained for.
* **Invoices are never divided between splits**, and neither are the invoices
  of one writer. Cells of one invoice share a pen, a hand, a photo and often a
  product; a model tested on cells whose neighbours it trained on scores as if
  it had seen the test. The split is by *group*: the writer when the truth
  names one, otherwise the invoice.
* **A group's split never changes when invoices are added.** It is decided by a
  hash of the group's name, not by position in a shuffled list, so the test
  invoices of last month's run are still test invoices this month and two
  models trained a month apart can be compared.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

EMPTY_TARGET = "EMPTY"
SPLITS = ("train", "val", "test")

# Below this many labelled cells a fine-tune mostly memorizes; the figure is
# docs/recognition-research.md's "a few hundred labelled cells from several
# writers", stated as the smallest number it could mean.
MIN_CELLS_WORTH_TRAINING_ON = 300
MIN_GROUPS_WORTH_TRAINING_ON = 10


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        parts = (self.train, self.val, self.test)
        if any(part < 0 for part in parts) or abs(sum(parts) - 1.0) > 1e-6 or self.train <= 0:
            raise ValueError(f"split fractions must be non-negative, sum to 1 and leave a train part; got {parts}.")


def split_of(group: str, fractions: SplitFractions = SplitFractions(), seed: str = "") -> str:
    """Which split every cell of `group` belongs to. The same answer forever."""
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    if position < fractions.train:
        return "train"
    if position < fractions.train + fractions.val:
        return "val"
    return "test"


def group_of(writer: str | None, invoice: str) -> str:
    """The unit that must stay in one split: a writer, else one invoice."""
    return f"writer:{writer.strip()}" if writer and writer.strip() else f"invoice:{invoice}"


_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
# The kinds of cell whose digits are handwriting. A code (the invoice number)
# is printed, in whatever digits the form's printer used: left as labelled.
_HANDWRITTEN_DIGIT_KINDS = frozenset({"number", "date"})


def training_target(row: dict, digits: str = "arabic_indic") -> str | None:
    """What the recognizer should answer for an exported crop; None if unlabelled.

    `digits` is the script the invoice's writer used (the truth's `digits`).
    """
    if row.get("empty"):
        return EMPTY_TARGET
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if digits == "arabic_indic" and row.get("kind") in _HANDWRITTEN_DIGIT_KINDS:
        text = text.translate(_TO_ARABIC_INDIC)
    return text


def to_record(row: dict, prompt: str, group: str, split: str, digits: str = "arabic_indic") -> dict | None:
    """One training example from one exported crop, or None if it has no label."""
    target = training_target(row, digits)
    if target is None:
        return None
    return {
        "image": row["image"],
        "prompt": prompt,
        "text": target,
        "kind": row["kind"],
        "role": row["role"],
        "source": row["source"],
        "cell": row["cell"],
        "group": group,
        "split": split,
    }


def describe(records: list[dict], skipped: dict[str, str] | None = None) -> dict:
    """The dataset's card: what is in each split, and what is wrong with it."""
    by_split = {split: [r for r in records if r["split"] == split] for split in SPLITS}
    groups = {split: sorted({r["group"] for r in rows}) for split, rows in by_split.items()}
    all_groups = sorted({r["group"] for r in records})

    warnings: list[str] = []
    if len(records) < MIN_CELLS_WORTH_TRAINING_ON:
        warnings.append(
            f"{len(records)} labelled cells: fewer than {MIN_CELLS_WORTH_TRAINING_ON}. "
            "A model trained on this will memorize it; use it to test the tooling, not to ship."
        )
    if len(all_groups) < MIN_GROUPS_WORTH_TRAINING_ON:
        warnings.append(
            f"{len(all_groups)} group(s) (writers or invoices): fewer than {MIN_GROUPS_WORTH_TRAINING_ON}. "
            "Held-out results on so few say little about a new writer."
        )
    for split in ("val", "test"):
        if not by_split[split]:
            warnings.append(
                f"The {split} split is empty: no group's hash fell in it. Label more invoices; "
                "do not move a training invoice into it by hand after training on it."
            )
    if not any(r["text"] == EMPTY_TARGET for r in records):
        warnings.append(
            "No empty cells: nothing here teaches the model to answer EMPTY for a cell "
            "without writing. They come from invoices whose rows pair with their truth."
        )

    return {
        "cells": len(records),
        "groups": len(all_groups),
        "splits": {
            split: {
                "cells": len(rows),
                "groups": groups[split],
                # The invoices (truth `image` paths) in this split:
                # `tools/evaluate.py --held-out` scores the test ones.
                "sources": sorted({r["source"] for r in rows}),
                "by_kind": dict(sorted(Counter(r["kind"] for r in rows).items())),
                "empty_cells": sum(r["text"] == EMPTY_TARGET for r in rows),
            }
            for split, rows in by_split.items()
        },
        "skipped": dict(skipped or {}),
        "warnings": warnings,
    }
