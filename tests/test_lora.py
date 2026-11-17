"""The fine-tune's adapter: what trains, and that loading it is the same model.

A tuned model reaches the service as two small matrices per layer, added into
the base weights when the worker loads. If that sum is not exactly what was
trained, every reading of the tuned model is of some other model -- and nothing
would say so.
"""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from torch import nn  # noqa: E402

from ocr.finetuning import lora  # noqa: E402


class Tiny(nn.Module):
    """Two "attention" layers and a head, named as a real model names them."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleList([
            nn.ModuleDict({"q_proj": nn.Linear(6, 6), "mlp": nn.Linear(6, 6)}) for _ in range(2)
        ])})
        self.lm_head = nn.Linear(6, 10)

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer["mlp"](torch.tanh(layer["q_proj"](x)))
        return self.lm_head(x)


PATTERN = r"model\.layers\.\d+\.q_proj"


def tiny() -> Tiny:
    torch.manual_seed(0)
    return Tiny()


def test_only_the_adapters_train():
    model = tiny()
    names = lora.inject(model, PATTERN, rank=2, alpha=4)

    assert names == ["model.layers.0.q_proj", "model.layers.1.q_proj"]
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {f"{n}.{part}" for n in names for part in ("lora_a", "lora_b")}


def test_an_untrained_adapter_changes_nothing():
    """B starts at zero: training begins from exactly the base model."""
    model, x = tiny(), torch.randn(3, 6)
    before = model(x)
    lora.inject(model, PATTERN, rank=2, alpha=4)

    assert torch.allclose(model(x), before, atol=1e-6)


def test_a_pattern_that_matches_nothing_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        lora.inject(tiny(), r"model\.blocks\.\d+\.attn", rank=2, alpha=4)


def trained() -> tuple[Tiny, dict]:
    """A model whose adapters have moved, and their saved state."""
    model = tiny()
    lora.inject(model, PATTERN, rank=2, alpha=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, lora.LoRALinear):
                module.lora_b.normal_(std=0.5)
    model.eval()
    return model, lora.adapter_state(model)


def test_adding_the_adapter_into_the_base_weights_is_the_trained_model():
    tuned, state = trained()
    x = torch.randn(5, 6)

    plain = tiny()  # the base model, as the worker loads it
    assert not torch.allclose(plain(x), tuned(x), atol=1e-4)  # the adapter does something
    assert lora.merge_into(plain, state, scale=4 / 2) == 2

    assert torch.allclose(plain(x), tuned(x), atol=1e-5)


def test_a_saved_adapter_loads_back_as_the_same_model(tmp_path):
    tuned, state = trained()
    lora.save_adapter(tmp_path / "ft", state, {"base_model": "OCR_Models/base", "rank": 2, "alpha": 4})
    x = torch.randn(5, 6)

    assert lora.is_adapter_dir(tmp_path / "ft") and not lora.is_adapter_dir(tmp_path)
    loaded, scale = lora.load_adapter(tmp_path / "ft")
    plain = tiny()
    lora.merge_into(plain, loaded, scale)

    assert scale == 2.0
    assert torch.allclose(plain(x), tuned(x), atol=1e-5)


def test_a_checkpoint_restores_the_adapters():
    tuned, state = trained()
    fresh = tiny()
    lora.inject(fresh, PATTERN, rank=2, alpha=4)
    lora.load_adapter_state(fresh, state)
    x = torch.randn(5, 6)

    assert torch.allclose(fresh.eval()(x), tuned(x), atol=1e-6)


def test_an_adapter_of_another_model_is_refused():
    _, state = trained()

    class Wider(Tiny):
        def __init__(self):
            super().__init__()
            self.model["layers"][0]["q_proj"] = nn.Linear(6, 8)

    with pytest.raises(ValueError, match="different base model"):
        lora.merge_into(Wider(), state, scale=2.0)
    with pytest.raises(ValueError, match="does not have"):
        lora.merge_into(nn.Linear(6, 6), state, scale=2.0)


def test_the_base_model_is_found_relative_to_the_project(tmp_path):
    _, state = trained()
    lora.save_adapter(tmp_path / "ft", state, {"base_model": "OCR_Models/base", "rank": 2, "alpha": 4})

    assert lora.base_model_dir(tmp_path / "ft", tmp_path / "project") == tmp_path / "project" / "OCR_Models" / "base"


def test_a_folder_in_another_format_is_refused(tmp_path):
    (tmp_path / lora.MANIFEST_FILE).write_text('{"format": "something-else/9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a recognizer adapter"):
        lora.read_manifest(tmp_path)


class TestWhichLayersOfTheRealModel:
    """Arabic-Qwen3.5-OCR-v4's layer names (read off the loaded model): the
    patterns must select its attention and MLP projections, and never the
    output head -- a quarter of a billion parameters no adapter is meant for."""

    LANGUAGE = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.23.mlp.down_proj",
    ]
    VISION = [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.11.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
    ]
    NEITHER = ["lm_head", "model.language_model.layers.0.linear_attn.in_proj_a"]

    @pytest.mark.parametrize("which,chosen,left", [
        ("language", LANGUAGE, VISION + NEITHER),
        ("vision", VISION, LANGUAGE + NEITHER),
        ("both", LANGUAGE + VISION, NEITHER),
    ])
    def test_each_choice_selects_its_layers_only(self, which, chosen, left):
        pattern = re.compile(lora.TARGETS[which])
        assert all(pattern.fullmatch(name) for name in chosen)
        assert not any(pattern.fullmatch(name) for name in left)
