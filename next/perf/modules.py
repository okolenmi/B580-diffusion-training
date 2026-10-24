"""Per-module device time, forward, recompute and backward, from events around module calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .clock import DeviceClock
from .cost import Cost, CostModel, element_bytes

KINDS = {"Linear": "linear", "AdaptedLinear": "linear", "Int8Linear": "linear", "Conv2d": "conv", "AdaptedConv2d": "conv",
         "GroupNorm": "norm", "LayerNorm": "norm", "CrossAttention": "attention", "FeedForward": "feedforward",
         "BasicTransformerBlock": "block", "SpatialTransformer": "block", "ResBlock": "block"}


def region_of(path: str) -> str:
    head = path.split(".")[0]
    return {"input_blocks": "down", "middle_block": "mid", "output_blocks": "up"}.get(head, "other")


def role_of(path: str) -> str:
    parts = [p for p in path.split(".") if not p.isdigit()]
    return ".".join(parts[-2:]) if len(parts) > 1 else path


@dataclass
class ModuleRecord:
    path: str
    kind: str
    parent: str | None = None
    calls: int = 0
    recompute_calls: int = 0
    forward_ms: float = 0.0
    recompute_ms: float = 0.0
    backward_ms: float = 0.0
    self_forward_ms: float = 0.0
    self_recompute_ms: float = 0.0
    self_backward_ms: float = 0.0
    cost: Cost | None = None
    int8: bool = False
    shapes: list[tuple] = field(default_factory=list)

    @property
    def region(self) -> str:
        return region_of(self.path)

    @property
    def role(self) -> str:
        return role_of(self.path)


def selected_kind(module: nn.Module) -> str | None:
    return KINDS.get(type(module).__name__)


class ModuleTimer:
    """Attaches hooks to the selected modules of ``model`` while it is active. Events are recorded without waiting; ``records()``
    synchronises once and turns them into per-module totals over every call seen while attached."""

    def __init__(self, model: nn.Module, clock: DeviceClock, cost_model: CostModel | None = None):
        self.model, self.clock, self.cost_model = model, clock, cost_model or CostModel()
        self._records: dict[str, ModuleRecord] = {}
        self._handles: list[Any] = []
        self._open: dict[int, Any] = {}
        self._events: list[tuple[str, str, Any, Any]] = []  # path, tag, start, end
        self.skipped = 0

    def attach(self) -> "ModuleTimer":
        paths: dict[str, str] = {}
        for path, module in self.model.named_modules():
            kind = selected_kind(module)
            if kind is None:
                continue
            paths[path] = kind
            record = self._records[path] = ModuleRecord(path, kind, int8=bool(getattr(module, "is_int8", False)) or hasattr(module, "weight_int8"))
            self._handles.append(module.register_forward_pre_hook(self._forward_start(record)))
            self._handles.append(module.register_forward_hook(self._forward_end(record, module)))
            try:
                self._handles.append(module.register_full_backward_pre_hook(self._backward_start(record)))
                self._handles.append(module.register_full_backward_hook(self._backward_end(record)))
            except RuntimeError:
                pass
        for path in paths:  # nearest selected ancestor
            parts = path.split(".")
            for cut in range(len(parts) - 1, 0, -1):
                ancestor = ".".join(parts[:cut])
                if ancestor in paths:
                    self._records[path].parent = ancestor
                    break
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @staticmethod
    def _in_backward() -> bool:
        try:
            return torch._C._current_graph_task_id() != -1
        except AttributeError:
            return False

    def _forward_start(self, record: ModuleRecord):
        def hook(module, args):
            if not self.clock.capturing():
                self._open.setdefault(id(module), []).append(self.clock.stamp())
        return hook

    def _forward_end(self, record: ModuleRecord, module: nn.Module):
        def hook(mod, args, output):
            if self.clock.capturing():  # a graph capture runs the module once without running its kernels
                return
            end = self.clock.stamp()
            start = self._open[id(mod)].pop()
            tag = "recompute" if self._in_backward() else "forward"
            self._events.append((record.path, tag, start, end))
            if tag == "forward":
                record.calls += 1
            else:
                record.recompute_calls += 1
            if record.cost is None and isinstance(output, torch.Tensor):
                shapes = [tuple(a.shape) if isinstance(a, torch.Tensor) else () for a in args]
                dtype = args[0].dtype if args and isinstance(args[0], torch.Tensor) else output.dtype
                record.shapes = shapes
                record.cost = self.cost_model.cost(record.kind, mod, shapes, tuple(output.shape), element_bytes(dtype))
        return hook

    def _backward_start(self, record: ModuleRecord):
        def hook(module, grad_output):
            if not self.clock.capturing():
                self._open.setdefault(("b", id(module)), []).append(self.clock.stamp())
        return hook

    def _backward_end(self, record: ModuleRecord):
        def hook(module, grad_input, grad_output):
            stack = self._open.get(("b", id(module)))
            if stack and not self.clock.capturing():
                self._events.append((record.path, "backward", stack.pop(), self.clock.stamp()))
        return hook

    def records(self) -> list[ModuleRecord]:
        self.clock.synchronize()
        for path, tag, start, end in self._events:
            record = self._records[path]
            try:
                ms = self.clock.elapsed_ms(start, end)
            except RuntimeError:  # an event recorded into a graph under capture that capturing() did not see
                self.skipped += 1
                continue
            if tag == "forward":
                record.forward_ms += ms
            elif tag == "recompute":
                record.recompute_ms += ms
            else:
                record.backward_ms += ms
        self._events = []
        for record in self._records.values():
            record.self_forward_ms, record.self_recompute_ms, record.self_backward_ms = record.forward_ms, record.recompute_ms, record.backward_ms
        for record in self._records.values():
            if record.parent is not None:
                parent = self._records[record.parent]
                parent.self_forward_ms -= record.forward_ms
                parent.self_recompute_ms -= record.recompute_ms
                parent.self_backward_ms -= record.backward_ms
        for record in self._records.values():
            record.self_forward_ms = max(record.self_forward_ms, 0.0)
            record.self_recompute_ms = max(record.self_recompute_ms, 0.0)
            record.self_backward_ms = max(record.self_backward_ms, 0.0)
        return [r for r in self._records.values() if r.calls or r.backward_ms]
