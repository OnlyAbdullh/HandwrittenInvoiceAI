"""Fine-tune the recognizer on this project's own labelled cells.

Runs in the VLM's virtualenv (the one `tools/qwen_worker.py` runs in), on the
dataset `tools/build_finetune_dataset.py` wrote:

    qwen_venv/Scripts/python tools/finetune_recognizer.py --out OCR_Models/invoice-ft-001
    qwen_venv/Scripts/python tools/finetune_recognizer.py --out OCR_Models/invoice-ft-001 --resume

What it does, and what it deliberately keeps equal to the service:

* **The same conversation.** Each example is the crop and the service's prompt
  for that kind of cell (stored in the dataset), sized by the worker's own
  `fit`, in the worker's own message shape, answered with the label. Only the
  answer is learned: the loss covers the label's tokens and the end-of-turn
  marker, nothing of the prompt.
* **A low-rank adapter, not the model** (`ocr/finetuning/lora.py`): the base
  weights stay frozen, a few million numbers train. That is what fits a 4 GB
  GPU and what a few hundred cells can support.
* **The output is a small folder**: `adapter.safetensors` and `finetune.json`.
  Point the service at it like any model --
  `INVOICE_AI_MODEL_DIR=OCR_Models/invoice-ft-001` -- and the worker adds the
  adapter into the base model as it loads. The base model must stay where it is.

It can be stopped and continued. A checkpoint is written after every epoch;
create an empty file named `STOP` in the output folder to stop after the
current epoch, and pass `--resume` to go on from the last checkpoint.

The validation split decides which epoch is kept (the lowest validation loss).
The test split is never read here. Measure a finished model on whole held-out
invoices, against the base model, before believing it:

    python tools/evaluate.py --config config/default_config.yaml --held-out datasets/finetune --label base
    INVOICE_AI_MODEL_DIR=OCR_Models/invoice-ft-001 python tools/evaluate.py \\
        --config config/default_config.yaml --held-out datasets/finetune --label ft-001

docs/fine-tuning.md has the whole procedure and what to expect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr.finetuning import lora  # noqa: E402
from qwen_worker import ARCH_CLASSES, FALLBACK_CLASS, chat_messages, fit, inspect_model  # noqa: E402

CHECKPOINT_FILE = "checkpoint.pt"
LOG_FILE = "training_log.json"
STOP_FILE = "STOP"
END_OF_TURN = "<|im_end|>"

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def load_split(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same_reading(predicted: str, expected: str) -> bool:
    """Whether a reading is the label, ignoring digit script and spacing."""
    def plain(text: str) -> str:
        return "".join(text.translate(_DIGITS).split())
    return plain(predicted) == plain(expected)


def relative_to_project(path: Path) -> str:
    """How a base model is recorded: relative to the project when it is inside
    it, so an adapter folder still finds it after the project moves."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


class Trainer:
    def __init__(self, args: argparse.Namespace):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.torch = torch
        self.args = args
        self.data_dir = Path(args.data)
        self.out_dir = Path(args.out)
        self.base_dir = Path(args.base)

        info = inspect_model(str(self.base_dir))
        if info["quantized"]:
            raise SystemExit(f"'{self.base_dir}' is quantized; fine-tune the full-precision model.")
        self.arch = info["arch"]

        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        # Half precision on the GPU is what leaves room to train in 4 GB; the
        # adapters themselves stay float32 (lora.LoRALinear).
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        cls_name = ARCH_CLASSES.get(self.arch, FALLBACK_CLASS)
        cls = getattr(transformers, cls_name, None) or getattr(transformers, FALLBACK_CLASS)
        self.processor = AutoProcessor.from_pretrained(str(self.base_dir), trust_remote_code=True)
        self.model = cls.from_pretrained(
            str(self.base_dir), trust_remote_code=True, low_cpu_mem_usage=True, dtype=self.dtype
        ).to(self.device)

        self.pattern = lora.TARGETS[args.adapt]
        self.layers = lora.inject(self.model, self.pattern, args.rank, args.alpha, args.dropout)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in self.parameters)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Base {self.base_dir} ({self.arch}) on {self.device}; adapting {len(self.layers)} layer(s) "
              f"({args.adapt}): {trainable / 1e6:.2f}M of {total / 1e6:.0f}M parameters train.")

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

        tokenizer = self.processor.tokenizer
        self.end_of_turn = END_OF_TURN if tokenizer.convert_tokens_to_ids(END_OF_TURN) not in (None, tokenizer.unk_token_id) \
            else (tokenizer.eos_token or "")

    # -- examples -----------------------------------------------------------

    def _image(self, record: dict):
        from PIL import Image

        return fit(Image.open(self.data_dir / record["image"]).convert("RGB"), self.arch)

    def _encode(self, record: dict, with_answer: bool):
        """The model's inputs for one example, and how many tokens are prompt."""
        image = self._image(record)
        messages = chat_messages(image, record["prompt"])
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision = self._process_vision_info(messages)[0] if self._process_vision_info else [image]

        prompt_ids = self.processor(text=[prompt], images=vision, return_tensors="pt")
        if not with_answer:
            return prompt_ids.to(self.device), prompt_ids.input_ids.shape[1]
        full = self.processor(text=[prompt + record["text"] + self.end_of_turn], images=vision, return_tensors="pt")
        return full.to(self.device), prompt_ids.input_ids.shape[1]

    def loss_of(self, record: dict):
        inputs, prompt_length = self._encode(record, with_answer=True)
        labels = inputs.input_ids.clone()
        labels[:, :prompt_length] = -100  # only the answer is learned
        with self.torch.autocast(self.device, dtype=self.torch.bfloat16, enabled=self.device == "cuda"):
            return self.model(**inputs, labels=labels).loss

    def read(self, record: dict) -> str:
        """The model's own greedy reading of one example (no character constraint)."""
        inputs, prompt_length = self._encode(record, with_answer=False)
        tokenizer = self.processor.tokenizer
        with self.torch.inference_mode(), self.torch.autocast(
            self.device, dtype=self.torch.bfloat16, enabled=self.device == "cuda"
        ):
            out = self.model.generate(
                **inputs, max_new_tokens=48, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, prompt_length:], skip_special_tokens=True).strip()

    # -- measuring ----------------------------------------------------------

    def mean_loss(self, records: list[dict]) -> float | None:
        if not records:
            return None
        self.model.eval()
        with self.torch.no_grad():
            losses = [float(self.loss_of(record)) for record in records]
        return sum(losses) / len(losses)

    def exact_matches(self, records: list[dict]) -> dict | None:
        """Cells read exactly right, overall and per kind -- the model alone,
        without the service's character constraints or arithmetic."""
        if not records:
            return None
        self.model.eval()
        by_kind: dict[str, list[int]] = {}
        for record in records:
            tally = by_kind.setdefault(record["kind"], [0, 0])
            tally[0] += int(same_reading(self.read(record), record["text"]))
            tally[1] += 1
        right, total = (sum(t[i] for t in by_kind.values()) for i in (0, 1))
        return {"right": right, "cells": total, "by_kind": {k: f"{r}/{n}" for k, (r, n) in sorted(by_kind.items())}}

    # -- saving -------------------------------------------------------------

    def manifest(self, log: dict) -> dict:
        args = self.args
        return {
            "base_model": relative_to_project(self.base_dir),
            "rank": args.rank,
            "alpha": args.alpha,
            "adapt": args.adapt,
            "layers": len(self.layers),
            "dataset": str(self.data_dir),
            "trained_on_cells": log["train_cells"],
            "kept_epoch": log.get("kept_epoch"),
            "learning_rate": args.lr,
            "epochs_run": len(log["epochs"]),
        }

    def save_checkpoint(self, optimizer, epoch: int, log: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.out_dir / (CHECKPOINT_FILE + ".tmp")
        self.torch.save(
            {"adapter": lora.adapter_state(self.model), "optimizer": optimizer.state_dict(), "epoch": epoch, "log": log},
            temporary,
        )
        temporary.replace(self.out_dir / CHECKPOINT_FILE)
        (self.out_dir / LOG_FILE).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- the loop -----------------------------------------------------------

    def run(self) -> int:
        torch, args = self.torch, self.args
        train, val = load_split(self.data_dir, "train"), load_split(self.data_dir, "val")
        if not train:
            raise SystemExit(f"No training examples in {self.data_dir / 'train.jsonl'}. "
                             "Run tools/build_finetune_dataset.py first.")
        if args.limit:
            train, val = train[: args.limit], val[: args.limit]

        optimizer = torch.optim.AdamW(self.parameters, lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(train) / args.accumulate)
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup = max(1, int(0.06 * total_steps))

        def learning_rate(step: int) -> float:
            if step < warmup:
                return args.lr * (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        log: dict = {
            "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "train_cells": len(train), "val_cells": len(val), "epochs": [],
            "kept_epoch": None, "best_val_loss": None,
        }
        first_epoch = 0
        checkpoint_path = self.out_dir / CHECKPOINT_FILE
        if args.resume and checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            lora.load_adapter_state(self.model, checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            log, first_epoch = checkpoint["log"], checkpoint["epoch"] + 1
            print(f"Resuming after epoch {first_epoch} of {args.epochs}.")
        elif (self.out_dir / lora.MANIFEST_FILE).exists() and not args.overwrite:
            raise SystemExit(f"{self.out_dir} already holds a fine-tune. Pass --resume to continue it, "
                             "--overwrite to replace it, or another --out.")
        else:
            if not val:
                print("No validation examples: the last epoch is kept, and nothing says whether it overfits.")
            log["before"] = {"val_loss": self.mean_loss(val), "val_exact": self.exact_matches(val)}
            print(f"Before training: validation {log['before']}")

        (self.out_dir / STOP_FILE).unlink(missing_ok=True)
        order = random.Random(args.seed)
        step = first_epoch * steps_per_epoch
        for epoch in range(first_epoch, args.epochs):
            started = time.perf_counter()
            self.model.train()
            examples = train[:]
            order.shuffle(examples)
            running, optimizer_steps = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for index, record in enumerate(examples, 1):
                loss = self.loss_of(record)
                (loss / args.accumulate).backward()
                running += float(loss)
                if index % args.accumulate == 0 or index == len(examples):
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(step)
                    torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    optimizer_steps += 1

            val_loss = self.mean_loss(val)
            entry = {
                "epoch": epoch + 1,
                "train_loss": round(running / len(examples), 4),
                "val_loss": None if val_loss is None else round(val_loss, 4),
                "seconds": round(time.perf_counter() - started, 1),
            }
            log["epochs"].append(entry)
            print(f"Epoch {entry['epoch']}/{args.epochs}: train loss {entry['train_loss']}, "
                  f"validation loss {entry['val_loss']}, {entry['seconds']}s")

            # The epoch kept is the one that reads unseen invoices best, not
            # the last: with little data the last one has memorized.
            improved = val_loss is None or log["best_val_loss"] is None or val_loss < log["best_val_loss"]
            if improved:
                log["best_val_loss"], log["kept_epoch"] = val_loss, epoch + 1
                lora.save_adapter(self.out_dir, lora.adapter_state(self.model), self.manifest(log))
            self.save_checkpoint(optimizer, epoch, log)

            if (self.out_dir / STOP_FILE).exists():
                print(f"Stopped on request after epoch {epoch + 1}. Continue with --resume.")
                return 0

        # Report the adapter that was kept, not the last one trained.
        kept_state, _ = lora.load_adapter(self.out_dir)
        lora.load_adapter_state(self.model, kept_state)
        log["after"] = {"val_loss": self.mean_loss(val), "val_exact": self.exact_matches(val)}
        (self.out_dir / LOG_FILE).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)

        print(f"\nKept epoch {log['kept_epoch']}. Validation before: {log.get('before')}")
        print(f"Validation after:  {log['after']}")
        print(f"Fine-tuned model: {self.out_dir}  (adapter only; needs the base model at {self.base_dir})")
        print("Measure it on whole held-out invoices before using it:")
        print(f"  python tools/evaluate.py --config config/default_config.yaml --held-out {self.data_dir} --label base")
        print(f"  INVOICE_AI_MODEL_DIR={self.out_dir.as_posix()} python tools/evaluate.py "
              f"--config config/default_config.yaml --held-out {self.data_dir} --label {self.out_dir.name}")
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/finetune", help="the folder tools/build_finetune_dataset.py wrote")
    ap.add_argument("--base", default=os.environ.get("INVOICE_AI_BASE_MODEL_DIR", "OCR_Models/Arabic-Qwen3.5-OCR-v4"),
                    help="the full-precision model to tune")
    ap.add_argument("--out", required=True, help="folder for the fine-tuned model (the adapter)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8, help="adapter rank: more learns more and memorizes sooner")
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--accumulate", type=int, default=8,
                    help="examples per optimizer step (one is read at a time, to fit a small GPU)")
    ap.add_argument("--adapt", choices=sorted(lora.TARGETS), default="language",
                    help="which part of the model gets adapters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N examples (to try the tooling)")
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even when a GPU is present")
    ap.add_argument("--resume", action="store_true", help="continue from the output folder's last checkpoint")
    ap.add_argument("--overwrite", action="store_true", help="replace a fine-tune already in --out")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        raise SystemExit("torch is not installed here. Run this with the VLM virtualenv's python "
                         "(the one built from requirements-vlm.txt).")
    return Trainer(args).run()


if __name__ == "__main__":
    sys.exit(main())
