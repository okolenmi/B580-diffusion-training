"""Choosing what an adapter trains, planned from module descriptions alone (no torch, no weights).

A selection has three ingredients: *groups* (kinds of layer, defined by the family), *regions* (where in the network,
as module-path prefixes with a weight or off) and *patterns* (regular expressions that add or remove modules). The plan
lists exactly the modules that will be wrapped, so a preview computed from a file header equals what injection builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from collections.abc import Iterable, Mapping, Sequence

CUSTOM_GROUP = "custom"


@dataclass(frozen=True)
class ModuleGroup:
    name: str
    label: str
    default_on: bool = True
    doc: str = ""


@dataclass(frozen=True)
class GroupOverride:
    rank: int | None = None
    alpha: float | None = None
    lr_mult: float = 1.0


@dataclass(frozen=True)
class TrainableSelection:
    """``groups=None`` means the layout's defaults. ``regions`` maps a module-path prefix to a weight multiplier, or
    ``None`` for off; the longest matching prefix decides. ``include`` and ``exclude`` are ``re.fullmatch`` patterns on
    the module path; exclusion wins."""

    groups: tuple[str, ...] | None = None
    regions: Mapping[str, float | None] = field(default_factory=dict, hash=False)
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    per_group: Mapping[str, GroupOverride] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class GateWindow:
    """The timestep range an adapter is trained on; outside it the adapter fades out. ``width`` (timestep units)
    sets how soft the edge is."""

    low: float
    high: float
    width: float

    def gate(self, t):
        """Sigmoid of the signed distance to the nearest edge: positive depth inside, negative distance outside.
        Works on any tensor type (only ``float``, ``minimum`` and ``sigmoid`` are used)."""
        t = t.float()
        return ((t - self.low).minimum(self.high - t) / max(self.width, 1e-6)).sigmoid()


@dataclass(frozen=True)
class ModuleDescriptor:
    """What the planner needs to know about a layer: its path, its kind and its shape."""

    path: str
    kind: str  # "linear" or "conv2d"
    in_features: int
    out_features: int
    kernel: tuple[int, int] = (1, 1)
    groups: int = 1


@dataclass(frozen=True)
class PlannedModule:
    path: str
    group: str
    kind: str
    rank: int
    alpha: float
    weight: float
    lr_mult: float
    params: int


@dataclass(frozen=True)
class AdapterPlan:
    modules: tuple[PlannedModule, ...]
    warnings: tuple[str, ...] = ()

    @property
    def param_count(self) -> int:
        return sum(m.params for m in self.modules)

    @property
    def nbytes(self) -> int:
        return self.param_count * 4

    def by_group(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for m in self.modules:
            entry = out.setdefault(m.group, {"modules": 0, "params": 0})
            entry["modules"] += 1
            entry["params"] += m.params
        return out

    def to_json(self) -> dict:
        return {"modules": len(self.modules), "params": self.param_count, "bytes": self.nbytes,
                "groups": self.by_group(), "warnings": list(self.warnings)}


class AdapterLayout(Protocol):
    """A family's answers about where adapters can go."""

    def groups(self) -> tuple[ModuleGroup, ...]: ...

    def classify(self, path: str, kind: str) -> str | None: ...


def region_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".")


def _factor_params(m: ModuleDescriptor, rank: int) -> int:
    if m.kind == "conv2d":
        return rank * (m.in_features // m.groups) * m.kernel[0] * m.kernel[1] + m.out_features * rank
    return rank * m.in_features + m.out_features * rank


def plan_adapter(layout: AdapterLayout, modules: Iterable[ModuleDescriptor], selection: TrainableSelection,
                 rank: int, alpha: float) -> AdapterPlan:
    """The modules a selection trains, with per-module rank, alpha, region weight and learning-rate multiplier."""
    known = {g.name: g for g in layout.groups()}
    chosen = set(selection.groups) if selection.groups is not None else {n for n, g in known.items() if g.default_on}
    unknown = (chosen | set(selection.per_group)) - set(known)
    if unknown:
        raise ValueError(f"unknown group(s) {sorted(unknown)}; this family has {sorted(known)}")
    include = [re.compile(p) for p in selection.include]
    exclude = [re.compile(p) for p in selection.exclude]
    planned: list[PlannedModule] = []
    for m in modules:
        group = layout.classify(m.path, m.kind)
        added = any(p.fullmatch(m.path) for p in include)
        if group is None:
            if not added:
                continue
            group = CUSTOM_GROUP
        elif group not in chosen and not added:
            continue
        if any(p.fullmatch(m.path) for p in exclude):
            continue
        weight = 1.0
        match = max((r for r in selection.regions if region_matches(m.path, r)), key=len, default=None)
        if match is not None:
            if selection.regions[match] is None or selection.regions[match] <= 0:
                continue
            weight = float(selection.regions[match])
        over = selection.per_group.get(group, GroupOverride())
        m_rank = over.rank or rank
        planned.append(PlannedModule(m.path, group, m.kind, m_rank, over.alpha if over.alpha is not None else alpha,
                                     weight, over.lr_mult, _factor_params(m, m_rank)))
    unused = [r for r in selection.regions if not any(region_matches(p.path, r) for p in planned)
              and selection.regions[r] is not None]
    warnings = tuple(f"region {r!r} matched no planned module" for r in unused)
    return AdapterPlan(tuple(planned), warnings)


def descriptors_from_shapes(shapes: Mapping[str, Sequence[int]], prefix: str = "") -> list[ModuleDescriptor]:
    """Module descriptions from weight shapes: ``<path>.weight`` of rank 2 is a linear layer, of rank 4 a convolution.
    Works on a safetensors header index or on a module's state dict."""
    out: list[ModuleDescriptor] = []
    for key, shape in shapes.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        path = key[len(prefix):-len(".weight")]
        if len(shape) == 2:
            out.append(ModuleDescriptor(path, "linear", shape[1], shape[0]))
        elif len(shape) == 4:
            out.append(ModuleDescriptor(path, "conv2d", shape[1], shape[0], (shape[2], shape[3])))
    return out
