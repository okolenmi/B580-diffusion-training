"""The arithmetic behind int8 frozen weights, behind one interface.

A weight is stored rotated (blocks of ``GROUP`` along the input dimension, a Hadamard matrix per block) and quantized to
int8 with one fp32 scale per output row; the activation is rotated and quantized per row at call time. The rotation spreads
outliers so a single scale per row loses little. Two implementations: the ``xpu-kernels`` package (fused SYCL kernels) and a
plain torch one that is exact but slow, used on other devices and as the reference in checks.
"""

from __future__ import annotations

import importlib
import math

import torch
from torch import Tensor

GROUP = 256  # input features must be a multiple of this; the rotation works on blocks of it
MIN_VERSION = (0, 3)  # xpu-kernels with the fused weight dequantizer
ROW_MULTIPLE = 16  # output features must be a multiple of this (the dequant epilogue reads 16 columns at a time)


def shape_problem(out_features: int, in_features: int) -> str | None:
    if in_features % GROUP:
        return f"{in_features} input features is not a multiple of {GROUP}"
    if out_features % ROW_MULTIPLE:
        return f"{out_features} output features is not a multiple of {ROW_MULTIPLE}"
    return None


class Int8Backend:
    """Prepares weights and runs the quantized linear. ``prepare`` and ``dequant`` are inverse up to int8 rounding."""

    name = "?"

    def problem(self) -> str | None:
        """Why this backend cannot run in this process, or None."""
        return None

    def call_problem(self, x: Tensor, w8: Tensor) -> str | None:
        """Why this one call needs the dense path (shape, device), or None."""
        return None

    def prepare(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        """``[N, K]`` float weight -> (int8 ``[N, K]``, fp32 ``[N]``), rotated."""
        raise NotImplementedError

    def linear(self, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
        """``x @ W.T + bias`` in bf16 for ``x`` of shape ``[.., K]``."""
        raise NotImplementedError

    def dequant(self, w8: Tensor, scale: Tensor) -> Tensor:
        """The bf16 ``[N, K]`` weight in its original (unrotated) orientation."""
        raise NotImplementedError

    def dx_int8(self, dy: Tensor, w8: Tensor, scale: Tensor) -> Tensor:
        """``dy @ W`` with the weight left in int8 and ``dy`` quantized per row instead. ``W = rotate(w8 * scale)`` and the scale
        sits on the contraction index, so ``dy @ W = rotate((dy * scale) @ w8)``. No dense weight exists, at the price of one
        quantizer's worth of error in the gradient."""
        raise NotImplementedError


class TorchBackend(Int8Backend):
    """Pure torch. The integer product runs in float64 off the XPU, which is exact for any K here."""

    name = "torch"

    def __init__(self):
        self._hadamard: dict[torch.device, Tensor] = {}

    def _matrix(self, device: torch.device) -> Tensor:
        if device not in self._hadamard:
            h = torch.ones(1, 1, dtype=torch.float64)
            while h.shape[0] < GROUP:
                h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
            self._hadamard[device] = (h / math.sqrt(GROUP)).to(device=device, dtype=torch.float32)
        return self._hadamard[device]

    def _rotated(self, x: Tensor) -> Tensor:
        blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP)
        return (blocks @ self._matrix(x.device)).reshape(x.shape)

    def prepare(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        rotated = self._rotated(weight.detach().to(torch.bfloat16))
        scale = (rotated.abs().amax(dim=1).clamp_min(1e-30) / 127.0).float()
        return torch.round(rotated / scale[:, None]).clamp_(-127, 127).to(torch.int8), scale

    def linear(self, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
        rotated = self._rotated(x.to(torch.bfloat16)).reshape(-1, x.shape[-1])
        row = (rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0).float()
        q = torch.round(rotated / row).clamp_(-127, 127)
        if x.device.type == "xpu":
            acc = torch._int_mm(q.to(torch.int8), w8.t()).float()
        else:
            acc = (q.double() @ w8.double().t()).float()
        out = acc * row * scale[None, :]
        if bias is not None:
            out = out + bias.float()
        return out.to(torch.bfloat16).reshape(*x.shape[:-1], w8.shape[0])

    def dx_int8(self, dy: Tensor, w8: Tensor, scale: Tensor) -> Tensor:
        scaled = (dy.to(torch.bfloat16) * scale.to(torch.bfloat16)).reshape(-1, dy.shape[-1]).float()
        row = (scaled.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
        q = torch.round(scaled / row).clamp_(-127, 127)
        if dy.device.type == "xpu":
            acc = torch._int_mm(q.to(torch.int8), w8).float()
        else:
            acc = (q.double() @ w8.double()).float()
        return self._rotated((acc * row).to(torch.bfloat16)).to(torch.bfloat16).reshape(*dy.shape[:-1], w8.shape[1])

    def dequant(self, w8: Tensor, scale: Tensor) -> Tensor:
        return self._rotated((w8.float() * scale[:, None]).to(torch.bfloat16)).to(torch.bfloat16)


class XpuKernelsBackend(Int8Backend):
    """The ``xpu-kernels`` package (``pip install xpu-kernels``, Intel Arc, oneAPI 2025.3 runtime)."""

    name = "xpu_kernels"

    def __init__(self, module=None):
        self._module = module
        self._two_pass = None  # xpu_kernels.triton.dequant when it is installed and has verified itself on this machine

    def _kernels(self):
        if self._module is None:
            self._module = importlib.import_module("xpu_kernels")
        return self._module

    def problem(self) -> str | None:
        try:
            kernels = self._kernels()
        except ImportError:
            return "the xpu-kernels package is not installed (pip install xpu-kernels)"
        version = tuple(int(part) for part in str(getattr(kernels, "__version__", "0")).split(".")[:2] if part.isdigit())
        if version < MIN_VERSION:
            return f"xpu-kernels {kernels.__version__} is too old: the backward needs the fused weight dequantizer of {'.'.join(map(str, MIN_VERSION))}"
        if not kernels.available():
            return "xpu-kernels is installed but its library did not load (it needs the oneAPI 2025.3 runtime that torch ships)"
        self._resolve_two_pass()
        return None

    def _resolve_two_pass(self) -> None:
        """The two-pass weight dequantizer (a Triton scale pass and one matrix product) is faster than the single SYCL kernel; it checks
        itself against it on first use, which cannot happen inside a graph capture, so it is settled here, before any."""
        try:
            module = importlib.import_module("xpu_kernels.triton.dequant")
            probe = torch.zeros(16, GROUP, dtype=torch.int8, device="xpu")
            self._two_pass = module if module.usable(probe) else None
            if self._two_pass is not None:
                module.prepare(probe.device)
        except Exception:  # noqa: BLE001 - an older package, no Triton, no XPU: the SYCL kernel stays
            self._two_pass = None

    def call_problem(self, x: Tensor, w8: Tensor) -> str | None:
        if x.device.type != "xpu" or w8.device != x.device:
            return f"the activation is on {x.device} and the weight on {w8.device}; the kernels need both on one XPU"
        rows = x.numel() // max(x.shape[-1], 1)
        if rows <= ROW_MULTIPLE:
            return f"{rows} rows is too few for the integer matmul"
        return None

    def prepare(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        return self._kernels().prepare_weight(weight)

    def linear(self, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
        return self._kernels().linear(x.to(torch.bfloat16), w8, scale, None if bias is None else bias.to(torch.bfloat16))

    def dequant(self, w8: Tensor, scale: Tensor) -> Tensor:
        if self._two_pass is not None and self._two_pass.usable(w8):
            return self._two_pass.dequant_weight(w8, scale)
        return self._kernels().dequant_weight(w8, scale)

    def dx_int8(self, dy: Tensor, w8: Tensor, scale: Tensor) -> Tensor:
        return self._kernels().linear_dx_int8(dy.to(torch.bfloat16), w8, scale)


BACKENDS = ("auto", "xpu_kernels", "torch")


def make_backend(choice: str, device: torch.device | str) -> Int8Backend:
    """``auto`` takes the kernels on an XPU when they load and the torch reference otherwise; naming a backend that cannot run raises."""
    if choice not in BACKENDS:
        raise ValueError(f"unknown int8 backend {choice!r}; choose from {BACKENDS}")
    if choice == "torch":
        return TorchBackend()
    kernels = XpuKernelsBackend()
    problem = kernels.problem() if (choice == "xpu_kernels" or torch.device(device).type == "xpu") else "not an XPU"
    if problem is None:
        return kernels
    if choice == "xpu_kernels":
        raise RuntimeError(f"int8 backend xpu_kernels: {problem}")
    return TorchBackend()
