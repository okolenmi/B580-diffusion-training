"""Bake a saved LoRA into the weights of a network, so training starts from the merged model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MergeReport:
    merged: int
    unused: tuple[str, ...]  # file keys no layer of the network claimed


def merge_adapter(module: nn.Module, scheme, state: Mapping[str, Tensor], strength: float = 1.0) -> MergeReport:
    """``W += strength * alpha / rank * (up @ down)`` for every Linear or Conv2d whose path ``scheme`` finds in ``state``
    (the file's own alpha; rank when it has none). The sum is made in float32 and rounded once to the weight's dtype."""
    used: set[str] = set()
    merged = 0
    with torch.no_grad():
        for path, layer in module.named_modules():
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                continue
            base = scheme.key(path)
            down, up = state.get(f"{base}.lora_down.weight"), state.get(f"{base}.lora_up.weight")
            if down is None or up is None:
                continue
            rank = down.shape[0]
            alpha = state.get(f"{base}.alpha")
            scaling = strength * (float(alpha.reshape(-1)[0]) if alpha is not None else float(rank)) / rank
            weight = layer.weight
            delta = (up.reshape(up.shape[0], rank).float() @ down.reshape(rank, -1).float()).to(weight.device).view_as(weight)
            weight.copy_((weight.float() + delta * scaling).to(weight.dtype))
            used.update(k for k in (f"{base}.lora_down.weight", f"{base}.lora_up.weight", f"{base}.alpha") if k in state)
            merged += 1
    return MergeReport(merged, tuple(sorted(set(state) - used)))
