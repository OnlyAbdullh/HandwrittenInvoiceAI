"""Low-rank adapters (LoRA) for the recognizer, and the adapter file format.

Fine-tuning every weight of the model is neither possible on the hardware this
project runs on (a 4 GB GPU holds the model and little else) nor wanted with a
few hundred labelled cells. A low-rank adapter leaves each chosen linear layer's
weight W frozen and learns two small matrices beside it, so the layer computes

    W x + (alpha / rank) * B (A x)

with A (rank x in) and B (out x rank). B starts at zero, so training starts
from exactly the base model. A few million numbers are trained instead of 850
million.

**A tuned model is a folder holding only the adapter**, a few megabytes:

    adapter.safetensors   A and B for every adapted layer
    finetune.json         the base model it belongs to, rank, alpha, how it was trained

`tools/qwen_worker.py` loads the base model and adds each adapter into its
layer's weight (`merge_into`): W becomes W + (alpha / rank) * B A, in the
float32 the service reads at. After that the model is an ordinary one -- the
same layers, the same speed -- and nothing else in the pipeline knows it was
tuned. Keeping the adapter apart from the base costs no second copy of a
1.7 GB model per experiment, and adding it in float32 loses nothing to the
rounding that writing merged weights back at half precision would.

Written here rather than taken from a library so that training and loading
depend on torch alone, in a virtualenv that must stay installable offline.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from torch import nn

ADAPTER_FILE = "adapter.safetensors"
MANIFEST_FILE = "finetune.json"
FORMAT = "invoice-recognizer-lora/1"

# Which linear layers get an adapter, as a regular expression over module
# names. The language model's attention and MLP projections: what turns the
# picture into text. (Arabic-Qwen3.5-OCR-v4 mixes two attention kinds, hence
# both sets of names.)
LANGUAGE_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)
# The vision tower: what sees the strokes. Worth adapting when the handwriting
# itself -- not the vocabulary -- is what the model gets wrong.
VISION_TARGETS = (
    r"model\.visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
    r"|merger\.(linear_fc1|linear_fc2))"
)
TARGETS = {
    "language": LANGUAGE_TARGETS,
    "vision": VISION_TARGETS,
    "both": f"(?:{LANGUAGE_TARGETS})|(?:{VISION_TARGETS})",
}


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update beside it."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be at least 1, got {rank}.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        # float32 whatever the base is stored in: an optimizer's small steps
        # vanish in half precision.
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.dropout(x).to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return out + (update * self.scale).to(out.dtype)


def target_names(model: nn.Module, pattern: str) -> list[str]:
    """Names of the linear layers `pattern` selects, in the model's own order."""
    chosen = re.compile(pattern)
    return [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and chosen.fullmatch(name)
    ]


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attribute


def inject(model: nn.Module, pattern: str, rank: int, alpha: float, dropout: float = 0.0) -> list[str]:
    """Freeze `model` and put an adapter beside every layer `pattern` selects.

    Returns the adapted layers' names. Afterwards the adapters are the only
    parameters that train.
    """
    names = target_names(model, pattern)
    if not names:
        raise ValueError(f"No linear layer of the model matches {pattern!r}; nothing would be trained.")
    model.requires_grad_(False)
    for name in names:
        parent, attribute = _parent_and_attribute(model, name)
        setattr(parent, attribute, LoRALinear(getattr(parent, attribute), rank, alpha, dropout))
    return names


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every adapter's two matrices, keyed `<layer name>.lora_a` / `.lora_b`."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_a"] = module.lora_a.detach().to("cpu", torch.float32).contiguous()
            state[f"{name}.lora_b"] = module.lora_b.detach().to("cpu", torch.float32).contiguous()
    return state


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Put saved matrices back into a model `inject` has already prepared."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_a.copy_(state[f"{name}.lora_a"])
                module.lora_b.copy_(state[f"{name}.lora_b"])


def merge_into(model: nn.Module, state: dict[str, torch.Tensor], scale: float) -> int:
    """Add each adapter into its layer's weight: W += scale * B A.

    `model` is a plain model (no `LoRALinear` in it). Returns how many layers
    were changed, and refuses an adapter that names a layer the model does not
    have or has in another shape -- the adapter of a different base model.
    """
    layers = sorted({key.rsplit(".", 1)[0] for key in state})
    for name in layers:
        try:
            layer = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"The adapter names a layer this model does not have: {name}.") from exc
        a, b = state[f"{name}.lora_a"], state[f"{name}.lora_b"]
        if not isinstance(layer, nn.Linear) or layer.weight.shape != (b.shape[0], a.shape[1]):
            raise ValueError(f"The adapter does not fit layer {name}: it was trained on a different base model.")
        with torch.no_grad():
            update = (b.to(torch.float32) @ a.to(torch.float32)) * scale
            layer.weight.add_(update.to(layer.weight.device, layer.weight.dtype))
    return len(layers)


# -- the adapter folder ---------------------------------------------------------


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / MANIFEST_FILE).is_file()


def read_manifest(path: str | Path) -> dict:
    manifest = json.loads((Path(path) / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{Path(path) / MANIFEST_FILE} is not a recognizer adapter this version reads "
            f"(format {manifest.get('format')!r}, expected {FORMAT!r})."
        )
    return manifest


def base_model_dir(adapter_dir: str | Path, project_root: str | Path) -> Path:
    """The base model an adapter folder belongs to.

    Stored relative to the project when it lives inside it, so the folder can
    move with the project to another machine.
    """
    base = Path(read_manifest(adapter_dir)["base_model"])
    return base if base.is_absolute() else Path(project_root) / base


def save_adapter(path: str | Path, state: dict[str, torch.Tensor], manifest: dict) -> None:
    from safetensors.torch import save_file

    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(state, str(folder / ADAPTER_FILE))
    (folder / MANIFEST_FILE).write_text(
        json.dumps({"format": FORMAT, **manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_adapter(path: str | Path) -> tuple[dict[str, torch.Tensor], float]:
    """An adapter folder's matrices and the scale they are added with."""
    from safetensors.torch import load_file

    manifest = read_manifest(path)
    return load_file(str(Path(path) / ADAPTER_FILE)), float(manifest["alpha"]) / int(manifest["rank"])
