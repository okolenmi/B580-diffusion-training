"""Wrapping the planned modules of a model with adapter layers, and handing the result to training and saving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from torch import Tensor, nn

from ...api.adapters import AdapterPlan, ModuleDescriptor
from .gate import TimestepGate
from .layers import AdaptedConv2d, AdaptedLayer, AdaptedLinear


def descriptors_from_model(model: nn.Module) -> list[ModuleDescriptor]:
    """Every Linear and Conv2d in the model, in module order, described by path and shape."""
    out: list[ModuleDescriptor] = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out.append(ModuleDescriptor(path, "linear", module.in_features, module.out_features))
        elif isinstance(module, nn.Conv2d):
            out.append(ModuleDescriptor(path, "conv2d", module.in_channels, module.out_channels,
                                        tuple(module.kernel_size), module.groups))
    return out


@dataclass
class AdaptedPart:
    """A model with adapters injected. Only the adapter factors (and DoRA magnitudes) train."""

    model: nn.Module
    plan: AdapterPlan
    layers: dict[str, AdaptedLayer]
    gate: TimestepGate
    kind: str
    policy: str
    _lr_mult: dict[str, float] = field(default_factory=dict)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for layer in self.layers.values() for p in layer.trainable_parameters()]

    def param_groups(self, lr: float | None = None) -> list[dict]:
        """One optimizer group per (module group, learning-rate multiplier). ``lr`` is scaled by the multiplier."""
        buckets: dict[tuple[str, float], list[nn.Parameter]] = {}
        for planned in self.plan.modules:
            key = (planned.group, planned.lr_mult)
            buckets.setdefault(key, []).extend(self.layers[planned.path].trainable_parameters())
        groups = []
        for (name, mult), params in buckets.items():
            entry = {"name": f"{name}" if mult == 1.0 else f"{name}@{mult:g}", "group": name, "lr_mult": mult, "params": params}
            if lr is not None:
                entry["lr"] = lr * mult
            groups.append(entry)
        return groups

    def state(self) -> dict[str, dict[str, Tensor]]:
        return {path: layer.adapter_state() for path, layer in self.layers.items()}

    def merge(self) -> None:
        for layer in self.layers.values():
            layer.merge()


def inject(model: nn.Module, plan: AdapterPlan, *, kind: str = "lora", policy: str = "classic",
           dropout: float = 0.0, gate: TimestepGate | None = None) -> AdaptedPart:
    """Freeze ``model`` and wrap every planned module. The model is modified in place."""
    gate = gate or TimestepGate()
    model.requires_grad_(False)
    layers: dict[str, AdaptedLayer] = {}
    for planned in plan.modules:
        parent = model
        *parents, leaf = planned.path.split(".")
        for name in parents:
            parent = getattr(parent, name)
        original = getattr(parent, leaf)
        cls = AdaptedLinear if isinstance(original, nn.Linear) else AdaptedConv2d if isinstance(original, nn.Conv2d) else None
        if cls is None:
            raise TypeError(f"{planned.path} is a {type(original).__name__}; adapters wrap Linear and Conv2d only")
        layer = cls(original, kind=kind, rank=planned.rank, alpha=planned.alpha, policy=policy, dropout=dropout,
                    weight=planned.weight, gate=gate)
        setattr(parent, leaf, layer)
        layers[planned.path] = layer
    return AdaptedPart(model, plan, layers, gate, kind, policy)
