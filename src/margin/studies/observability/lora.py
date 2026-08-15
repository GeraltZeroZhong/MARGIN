"""Manual low-rank adapters for the frozen ESM2 exploratory probe."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Freeze an existing projection and add a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            **factory_kwargs,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            **factory_kwargs,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.lora_b(self.lora_a(values)) * self.scale


def inject_esm2_lora(
    model: nn.Module, layers: Iterable[int], rank: int, alpha: float
) -> dict[str, LoRALinear]:
    """Attach LoRA to query/value projections of the declared ESM2 layers."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = model.esm.encoder.layer
    adapters: dict[str, LoRALinear] = {}
    for layer_index in layers:
        if layer_index < 0 or layer_index >= len(blocks):
            raise ValueError(f"ESM2 LoRA layer is out of range: {layer_index}")
        attention = blocks[layer_index].attention.self
        for projection in ("query", "value"):
            base = getattr(attention, projection)
            adapter = LoRALinear(base, rank, alpha)
            setattr(attention, projection, adapter)
            adapters[f"layer_{layer_index}.{projection}"] = adapter
    return adapters


def adapter_state(adapters: dict[str, LoRALinear]) -> dict[str, dict[str, torch.Tensor]]:
    """Return only small trainable tensors, excluding frozen base projections."""

    return {
        name: {
            "lora_a": module.lora_a.weight.detach().cpu().clone(),
            "lora_b": module.lora_b.weight.detach().cpu().clone(),
        }
        for name, module in adapters.items()
    }


def load_adapter_state(
    adapters: dict[str, LoRALinear], state: dict[str, dict[str, torch.Tensor]]
) -> None:
    """Restore adapter-only tensors into an already injected model."""

    if set(adapters) != set(state):
        raise ValueError("LoRA checkpoint adapter set does not match the configured layers")
    for name, module in adapters.items():
        module.lora_a.weight.data.copy_(state[name]["lora_a"])
        module.lora_b.weight.data.copy_(state[name]["lora_b"])
