"""Adapter file layouts: which keys a saved adapter uses, and how to read one back into injected layers.

A scheme refuses what it cannot write correctly instead of producing a file that loads as "no effect".
Both schemes store ``alpha`` as the effective value (scaling * rank, region weight and policy included), so a file
always means ``alpha / rank`` on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .inject import AdaptedPart
from .layers import AdaptedLayer


@dataclass(frozen=True)
class LoadReport:
    matched: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


class KeyScheme(Protocol):
    name: str

    def can_write(self, kind: str) -> bool: ...

    def export(self, part: AdaptedPart) -> dict[str, Tensor]: ...

    def load(self, part: AdaptedPart, state: Mapping[str, Tensor]) -> LoadReport: ...


def _effective_alpha(layer: AdaptedLayer) -> Tensor:
    return torch.tensor([layer.scaling * layer.rank], dtype=torch.float32)


def _adopt(layer: AdaptedLayer, alpha: float) -> None:
    layer.policy, layer.alpha, layer.training_weight = "classic", alpha, 1.0


class KohyaScheme:
    """``<prefix><path with dots as underscores>.lora_down.weight / .lora_up.weight / .alpha``. LoRA only."""

    def __init__(self, prefix: str, name: str = "kohya"):
        self.prefix, self.name = prefix, name

    def can_write(self, kind: str) -> bool:
        return kind == "lora"

    def key(self, path: str) -> str:
        return self.prefix + path.replace(".", "_")

    def export(self, part: AdaptedPart) -> dict[str, Tensor]:
        if not self.can_write(part.kind):
            raise ValueError(f"the {self.name} format cannot hold {part.kind} adapters; use the native format")
        out: dict[str, Tensor] = {}
        for path, layer in part.layers.items():
            base = self.key(path)
            out[f"{base}.lora_down.weight"] = layer.lora_A.detach().cpu().contiguous()
            out[f"{base}.lora_up.weight"] = layer.lora_B.detach().cpu().contiguous()
            out[f"{base}.alpha"] = _effective_alpha(layer)
        return out

    def load(self, part: AdaptedPart, state: Mapping[str, Tensor]) -> LoadReport:
        used: set[str] = set()
        missing: list[str] = []
        for path, layer in part.layers.items():
            base = self.key(path)
            down, up = f"{base}.lora_down.weight", f"{base}.lora_up.weight"
            if down not in state or up not in state:
                missing.append(path)
                continue
            layer.load_adapter_state({"lora_A": state[down], "lora_B": state[up]})
            used.update((down, up))
            if f"{base}.alpha" in state:
                _adopt(layer, float(state[f"{base}.alpha"].reshape(-1)[0]))
                used.add(f"{base}.alpha")
        return LoadReport(len(part.layers) - len(missing), tuple(missing), tuple(sorted(set(state) - used)))


class NativeScheme:
    """``<path>.lora_A / .lora_B / .magnitude / .alpha`` under this project's own names; holds LoRA and DoRA."""

    name = "native"

    def can_write(self, kind: str) -> bool:
        return True

    def export(self, part: AdaptedPart) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for path, layer in part.layers.items():
            for name, tensor in layer.adapter_state().items():
                out[f"{path}.{name}"] = tensor.detach().cpu().contiguous()
            out[f"{path}.alpha"] = _effective_alpha(layer)
        return out

    def load(self, part: AdaptedPart, state: Mapping[str, Tensor]) -> LoadReport:
        used: set[str] = set()
        missing: list[str] = []
        for path, layer in part.layers.items():
            needed = {n: f"{path}.{n}" for n in layer.adapter_state() if n != "magnitude" or f"{path}.magnitude" in state}
            if any(k not in state for k in needed.values()):
                missing.append(path)
                continue
            layer.load_adapter_state({n: state[k] for n, k in needed.items()})
            used.update(needed.values())
            if f"{path}.alpha" in state:
                _adopt(layer, float(state[f"{path}.alpha"].reshape(-1)[0]))
                used.add(f"{path}.alpha")
        return LoadReport(len(part.layers) - len(missing), tuple(missing), tuple(sorted(set(state) - used)))


def save_adapter(path: str, tensors: Mapping[str, Tensor], metadata: Mapping[str, str]) -> None:
    save_file(dict(tensors), path, metadata={k: str(v) for k, v in metadata.items()})


def read_adapter(path: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    with safe_open(path, framework="pt") as handle:
        return {k: handle.get_tensor(k) for k in handle.keys()}, dict(handle.metadata() or {})


def load_checked(scheme: KeyScheme, part: AdaptedPart, state: Mapping[str, Tensor]) -> LoadReport:
    """Load and refuse a file that matched no layer at all (wrong family or key scheme)."""
    report = scheme.load(part, state)
    if report.matched == 0:
        raise ValueError(f"no layer matched the {scheme.name} keys in this file ({len(state)} keys); wrong format or model?")
    return report
