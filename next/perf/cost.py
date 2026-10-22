"""FLOPs and bytes of a module's own work (nested timed modules are rated separately), from the shapes it saw."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Cost:
    flops: float
    bytes: float
    low_precision: bool = False  # the matmul runs on the int8 path


def _numel(shape) -> int:
    return int(math.prod(shape)) if shape else 1


class CostModel:
    """Analytic cost per module kind. A kind it cannot model returns None, and its time is reported as unrated."""

    def cost(self, kind: str, module: nn.Module, in_shapes: list[tuple], out_shape: tuple, element_bytes: int) -> Cost | None:
        handler = getattr(self, f"_{kind}", None)
        if handler is None or not in_shapes:
            return None
        try:
            return handler(module, in_shapes, out_shape, element_bytes)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _linear(module, shapes, out, e) -> Cost:
        rows, k, n = _numel(shapes[0][:-1]), shapes[0][-1], out[-1]
        int8 = bool(getattr(module, "is_int8", False)) or hasattr(module, "weight_int8")
        weight_bytes = n * k * (1 if int8 else e)
        return Cost(2.0 * rows * n * k, (rows * k + rows * n) * e + weight_bytes, int8)

    @staticmethod
    def _conv(module, shapes, out, e) -> Cost:
        batch, cout, h, w = out
        cin = shapes[0][1]
        kh, kw = module.kernel_size
        groups = getattr(module, "groups", 1)
        weight = cout * (cin // groups) * kh * kw
        return Cost(2.0 * batch * cout * h * w * (cin // groups) * kh * kw, (_numel(shapes[0]) + _numel(out) + weight) * e)

    @staticmethod
    def _norm(module, shapes, out, e) -> Cost:
        return Cost(0.0, 2.0 * _numel(shapes[0]) * e)

    @staticmethod
    def _attention(module, shapes, out, e) -> Cost:
        batch, lq = shapes[0][0], shapes[0][1]
        context = shapes[1] if len(shapes) > 1 and shapes[1] else shapes[0]
        lk = context[1]
        inner = module.to_q.out_features
        heads = module.heads
        flops = 4.0 * batch * heads * lq * lk * (inner // heads)
        return Cost(flops, (2 * batch * lq * inner + 2 * batch * lk * inner) * e)

    @staticmethod
    def _feedforward(module, shapes, out, e) -> Cost:
        rows = _numel(shapes[0][:-1])
        first = module.net[0]
        width = first.proj.out_features if hasattr(first, "proj") else first[0].out_features
        return Cost(0.0, 3.0 * rows * width * e)


def element_bytes(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()
