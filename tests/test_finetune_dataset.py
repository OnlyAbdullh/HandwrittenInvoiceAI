"""From labelled invoices to a training set: what a label is, and who is held out.

No model runs here. The training loop itself (tools/finetune_recognizer.py)
needs the weights and a GPU; what is pinned is everything that decides *what
it is trained on* -- a wrong label or a leaked invoice is invisible in a loss
curve.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import pytest
from pydantic import ValidationError

from evaluation.finetune_dataset import (
    EMPTY_TARGET,
    SplitFractions,
    describe,
    group_of,
    split_of,
    to_record,
    training_target,
)
from evaluation.schema import InvoiceTruth, LineItemTruth, parse_written_number, plain_number

ROOT = Path(__file__).resolve().parent.parent


def _load_tool(name: str):
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)  # the tools import one another by name
    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestNumbersAsWritten:
    """`5.70` in JSON is the number 5.7. The crop shows ٥,٧٠, and on handwriting
    where a zero is a dot, a label that drops a zero teaches the commonest
    misreading there is."""

    def test_a_number_typed_as_a_string_keeps_its_written_form_and_its_value(self):
        item = LineItemTruth(product_name="x", quantity=192, unit_price="5,70", total_price="1094,4")

        assert (item.unit_price, item.total_price) == (5.7, 1094.4)
        assert item.text_of("unit_price") == "5,70"
        assert item.text_of("total_price") == "1094,4"

    def test_a_number_given_as_a_number_gets_its_shortest_spelling(self):
        item = LineItemTruth(product_name="x", quantity=16, unit_price=13250, total_price=212000.0)

        assert item.text_of("quantity") == "16"
        assert item.text_of("total_price") == "212000"
        assert item.written == {}

    def test_digits_typed_from_an_arabic_keyboard_are_the_same_number(self):
        item = LineItemTruth(product_name="x", unit_price="٥٫٧٠")

        assert item.unit_price == 5.7
        assert item.text_of("unit_price") == "5,70"

    def test_the_invoice_total_too(self):
        truth = InvoiceTruth(image="a/img.png", total="214812,05")

        assert truth.total == 214812.05
        assert truth.total_text == "214812,05"
        assert InvoiceTruth(image="a/img.png", total=5389.8).total_text == "5389.8"

    @pytest.mark.parametrize("text", ["1,080.50", "12 000", "abc", "", "5,", "1..2"])
    def test_what_is_not_one_number_is_refused_not_guessed(self, text):
        with pytest.raises(ValueError):
            parse_written_number(text)
        with pytest.raises(ValidationError):
            LineItemTruth(product_name="x", unit_price=text)

    def test_amounts_keep_their_cents(self):
        assert plain_number(214812.05) == "214812.05"
        assert plain_number(None) is None

    def test_the_date_is_labelled_as_these_forms_write_it_unless_told_otherwise(self):
        assert InvoiceTruth(image="a", date="2026-08-07").date_text == "7/8/2026"
        assert InvoiceTruth(image="a", date="2026-08-07", date_written="٢٠٢٦-٨-٧").date_text == "2026-8-7"
        assert InvoiceTruth(image="a").date_text is None

    def test_the_real_truth_files_still_load_and_still_add_up(self):
        for path in sorted((ROOT / "evaluation" / "ground_truth").glob("*.json")):
            truth = InvoiceTruth.model_validate_json(path.read_text(encoding="utf-8"))
            for item in truth.items:
                assert abs(item.quantity * item.unit_price - item.total_price) < 0.005, path.name
            assert abs(sum(i.total_price for i in truth.items) - truth.total) < 0.005, path.name


class TestTargets:
    def row(self, text, kind="number", **extra):
        return {"image": "images/a_r1c0.png", "text": text, "kind": kind, "role": "unit_price",
                "source": "a/img.png", "cell": "r1c0", **extra}

    def test_handwritten_digits_are_spelled_as_the_writer_wrote_them(self):
        assert training_target(self.row("3,40")) == "٣,٤٠"
        assert training_target(self.row("25/9/2026", kind="date")) == "٢٥/٩/٢٠٢٦"

    def test_a_writer_of_western_digits_is_left_in_western_digits(self):
        assert training_target(self.row("3,40"), digits="western") == "3,40"

    def test_a_printed_invoice_number_stays_as_printed(self):
        assert training_target(self.row("0001902", kind="code")) == "0001902"

    def test_a_name_with_a_size_in_it_is_not_touched(self):
        assert training_target(self.row("بنطال جينز 42-52", kind="arabic_text")) == "بنطال جينز 42-52"

    def test_an_empty_cell_is_answered_with_the_word_the_prompts_ask_for(self):
        assert training_target(self.row("", empty=True)) == EMPTY_TARGET

    @pytest.mark.parametrize("text", [None, "", "   "])
    def test_an_unlabelled_crop_is_not_an_example(self, text):
        assert training_target(self.row(text)) is None
        assert to_record(self.row(text), "prompt", "invoice:a", "train") is None

    def test_an_example_carries_its_prompt_and_where_it_came_from(self):
        record = to_record(self.row("5"), "اكتب الرقم", "invoice:a", "val")

        assert record == {
            "image": "images/a_r1c0.png", "prompt": "اكتب الرقم", "text": "٥", "kind": "number",
            "role": "unit_price", "source": "a/img.png", "cell": "r1c0", "group": "invoice:a", "split": "val",
        }


class TestSplit:
    def test_an_invoices_split_never_changes(self):
        """Decided by the group's name alone, so adding invoices cannot move a
        test invoice into training between two runs."""
        assert split_of("invoice:abd") == split_of("invoice:abd")
        first = {name: split_of(name) for name in ("invoice:a", "invoice:b", "invoice:c")}
        assert {name: split_of(name) for name in first} == first

    def test_the_shares_are_roughly_the_ones_asked_for(self):
        splits = [split_of(f"invoice:{i}", SplitFractions(0.8, 0.1, 0.1)) for i in range(4000)]
        assert 0.77 < splits.count("train") / 4000 < 0.83
        assert 0.08 < splits.count("val") / 4000 < 0.12
        assert 0.08 < splits.count("test") / 4000 < 0.12

    def test_another_seed_holds_out_other_invoices(self):
        names = [f"invoice:{i}" for i in range(200)]
        assert [split_of(n, seed="a") for n in names] != [split_of(n, seed="b") for n in names]

    def test_a_writers_invoices_stay_together(self):
        assert group_of("Abu Jamal", "inv-17") == group_of(" Abu Jamal ", "inv-90") == "writer:Abu Jamal"
        assert group_of(None, "inv-17") == "invoice:inv-17"
        assert group_of("  ", "inv-17") == "invoice:inv-17"

    @pytest.mark.parametrize("parts", [(0.5, 0.1, 0.1), (0.0, 0.5, 0.5), (1.2, -0.1, -0.1)])
    def test_fractions_that_do_not_make_a_split_are_refused(self, parts):
        with pytest.raises(ValueError):
            SplitFractions(*parts)


class TestCard:
    def records(self, n=4, split="train", text="٥", source="a/img.png", group="invoice:a"):
        return [{"image": f"images/{i}.png", "prompt": "p", "text": text, "kind": "number", "role": "quantity",
                 "source": source, "cell": f"r{i}c0", "group": group, "split": split} for i in range(n)]

    def test_it_says_plainly_when_there_is_too_little_to_train_on(self):
        card = describe(self.records())

        warnings = " ".join(card["warnings"])
        assert "fewer than 300" in warnings and "fewer than 10" in warnings
        assert "val split is empty" in warnings and "test split is empty" in warnings
        assert "No empty cells" in warnings

    def test_it_lists_the_invoices_of_each_split(self):
        records = self.records(split="train") + self.records(split="test", source="b/img.png", group="invoice:b")
        card = describe(records, skipped={"c/img.png": "cannot read"})

        assert card["splits"]["test"]["sources"] == ["b/img.png"]
        assert card["splits"]["train"]["cells"] == 4 and card["groups"] == 2
        assert card["skipped"] == {"c/img.png": "cannot read"}

    def test_empty_cells_are_counted(self):
        card = describe(self.records() + self.records(n=2, text=EMPTY_TARGET))
        assert card["splits"]["train"]["empty_cells"] == 2
        assert not any("No empty cells" in w for w in card["warnings"])


class TestBuildingFromAPhoto:
    """The whole tool on a synthetic form A: ten written rows, a truth of four
    items. The rows below the fourth hold ink the pipeline would send to the
    recognizer and no item; they are the examples of an empty cell."""

    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory):
        from tests.test_llamacpp_pipeline import form_a_photo

        root = tmp_path_factory.mktemp("finetune")
        (root / "truth").mkdir()
        (root / "truth" / "synthetic.json").write_text(json.dumps({
            "image": "synthetic/img.png", "invoice_number": "00010", "customer_name": "حسام المارديني",
            "city": "ريف دمشق", "total": "48,50",
            "items": [{"product_name": "بوري شبابي زارا", "quantity": 1, "unit_price": "12,00", "total_price": 12}] * 4,
        }, ensure_ascii=False), encoding="utf-8")
        (root / "bills" / "synthetic").mkdir(parents=True)
        cv2.imwrite(str(root / "bills" / "synthetic" / "img.png"), form_a_photo())

        tool = _load_tool("build_finetune_dataset")
        # Everything held out for training: which split a single invoice's
        # hash falls in is not what is under test.
        card = tool.build("config/default_config.yaml", root / "truth", root / "bills", root / "out",
                          SplitFractions(1.0, 0.0, 0.0), seed="")
        records = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        return root / "out", card, records

    def test_every_example_has_its_crop_and_the_services_prompt(self, built):
        from ocr.engines.qwen_vlm_engine import PROMPTS

        out, _, records = built
        assert records
        for record in records:
            assert (out / record["image"]).is_file()
            assert record["prompt"] == PROMPTS[record["kind"]]
            assert record["group"] == "invoice:synthetic" and record["split"] == "train"

    def test_labels_are_the_written_forms_in_the_writers_digits(self, built):
        _, _, records = built
        by_role = {}
        for record in records:
            if record["text"] != EMPTY_TARGET:
                by_role.setdefault(record["role"], set()).add(record["text"])

        assert by_role["unit_price"] == {"١٢,٠٠"}  # the zeros the truth typed are kept
        assert by_role["total_in_figures"] == {"٤٨,٥٠"}
        assert by_role["invoice_number"] == {"00010"}  # printed: as printed
        assert by_role["product_name"] == {"بوري شبابي زارا"}

    def test_rows_below_the_last_item_are_examples_of_an_empty_cell(self, built):
        _, card, records = built
        empty = [r for r in records if r["text"] == EMPTY_TARGET]

        assert empty and card["splits"]["train"]["empty_cells"] == len(empty)
        assert {r["role"] for r in empty} <= {"quantity", "unit_price", "line_total", "product_name"}
        # The four items' own cells are never among them.
        quantities = [r for r in records if r["role"] == "quantity" and r["text"] != EMPTY_TARGET]
        assert len(quantities) == 4

    def test_the_other_splits_are_written_even_when_empty(self, built):
        out, card, _ = built
        assert (out / "val.jsonl").read_text(encoding="utf-8") == ""
        assert (out / "test.jsonl").read_text(encoding="utf-8") == ""
        assert json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))["cells"] == card["cells"]

    def test_evaluate_scores_only_the_held_out_invoices(self, built, tmp_path):
        evaluate = _load_tool("evaluate")
        (tmp_path / "dataset_card.json").write_text(json.dumps(
            {"splits": {"test": {"sources": ["onlyone/img.png"]}, "train": {"sources": ["abd/photo.jpg"]}}}
        ), encoding="utf-8")

        assert evaluate.held_out_images(tmp_path) == {"onlyone/img.png"}
