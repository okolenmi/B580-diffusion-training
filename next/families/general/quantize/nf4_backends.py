"""The arithmetic behind NF4 frozen weights: a 16-level normal-quantile codebook, blocks of 64 weights sharing one absmax, two indices per byte.

Two implementations of the dequantizer: the ``xpu-kernels`` fused kernel (one pass, no temporaries, bit-identical to the chain, safe inside a
graph capture) and the torch chain, which is the reference, the fallback per call and the version for other devices. Quantizing is done once
per layer, in torch.

Double quantization (optional) stores the per-block absmax as int8 around its mean, one fp32 scale per 256 blocks. It saves about 0.05 byte per
weight and costs two kernels per layer call, so it is off unless asked for.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

BLOCK = 64  # weights per absmax; the fused kernel's block size
DQ_BLOCK = 256  # absmaxes per second-level scale
TINY = torch.finfo(torch.float32).tiny
_CODE_OFFSET = 0.9677083  # bitsandbytes' create_normal_map


def codebook(device=None, dtype=torch.float32) -> Tensor:
    """The 16 NF4 levels, sorted, from -1 to 1, with an exact zero (bitsandbytes' ``create_normal_map``, in float64 first)."""
    negative = torch.special.ndtri(torch.linspace(_CODE_OFFSET, 0.5, 9, dtype=torch.float64)[:-1])
    positive = -torch.special.ndtri(torch.linspace(_CODE_OFFSET, 0.5, 8, dtype=torch.float64)[:-1])
    values = torch.cat([negative, torch.zeros(1, dtype=torch.float64), positive]).sort().values
    return (values / values.max()).to(device=device, dtype=dtype)


def shape_problem(weight_shape) -> str | None:
    count = 1
    for n in weight_shape:
        count *= n
    return None if count % BLOCK == 0 else f"{count} weights is not a multiple of {BLOCK}"


class Nf4Codec:
    """Quantize and dequantize with plain torch operations; exact, any device."""

    def __init__(self):
        self._books: dict[str, Tensor] = {}

    def book(self, device) -> Tensor:
        key = str(device)
        if key not in self._books:
            self._books[key] = codebook(device)
        return self._books[key]

    def quantize(self, weight: Tensor, double_quant: bool) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """``weight`` of any shape (a multiple of 64 elements) -> packed uint8 ``[n/2]``, absmax (fp32 ``[n/64]``, or int8 with
        double quantization), and for double quantization the per-256 fp32 scales and the fp32 mean."""
        book = self.book(weight.device)
        blocks = weight.detach().float().reshape(-1, BLOCK)
        absmax = blocks.abs().amax(dim=1)
        normalized = blocks / absmax.clamp_min(TINY).unsqueeze(1)
        index = torch.bucketize(normalized, (book[1:] + book[:-1]) / 2, out_int32=True).reshape(-1, 2)
        packed = (index[:, 0] | (index[:, 1] << 4)).to(torch.uint8)
        if not double_quant:
            return packed, absmax, None, None
        offset = absmax.mean()
        padded = torch.nn.functional.pad(absmax - offset, (0, -absmax.numel() % DQ_BLOCK)).view(-1, DQ_BLOCK)
        scale = (padded.abs().amax(dim=1) / 127).clamp_min(TINY)
        return packed, (padded / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8).flatten(), scale, offset

    @staticmethod
    def absmax(stored: Tensor, scale: Tensor | None, offset: Tensor | None, blocks: int) -> Tensor:
        """The fp32 absmax per block (a flat tensor that may be longer than ``blocks``)."""
        if scale is None:
            return stored
        return torch.addcmul(offset, stored.view(-1, DQ_BLOCK).float(), scale.unsqueeze(1)).flatten()[:blocks]

    def dequant(self, packed: Tensor, absmax: Tensor, shape, dtype: torch.dtype) -> Tensor:
        book = self.book(packed.device)
        index = torch.stack([packed & 0x0F, packed >> 4], dim=1).flatten()
        codes = book[index.long()].view(-1, BLOCK)
        return (codes * absmax[: codes.shape[0]].unsqueeze(1)).to(dtype).view(shape)


class Nf4Backend:
    name = "?"

    def problem(self) -> str | None:
        return None

    def quantize(self, weight: Tensor, double_quant: bool):
        return self.codec.quantize(weight, double_quant)

    def dequant(self, packed: Tensor, absmax: Tensor, scale: Tensor | None, offset: Tensor | None, shape, dtype: torch.dtype) -> Tensor:
        raise NotImplementedError


class TorchNf4Backend(Nf4Backend):
    name = "torch"

    def __init__(self):
        self.codec = Nf4Codec()

    def dequant(self, packed, absmax, scale, offset, shape, dtype):
        count = packed.numel() * 2
        return self.codec.dequant(packed, self.codec.absmax(absmax, scale, offset, count // BLOCK), shape, dtype)


class XpuKernelsNf4Backend(TorchNf4Backend):
    """The fused dequantizer of ``xpu-kernels`` where it can run (an accelerator tensor, bf16 or fp32 output); the torch chain otherwise."""

    name = "xpu_kernels"

    def __init__(self):
        super().__init__()
        self._module = None

    def problem(self) -> str | None:
        try:
            self._module = importlib.import_module("xpu_kernels.triton.nf4")
        except ImportError as error:
            return f"the xpu-kernels package with its NF4 kernel is not importable ({error}); pip install -U xpu-kernels"
        return None

    def dequant(self, packed, absmax, scale, offset, shape, dtype):
        if self._module is not None and dtype in (torch.bfloat16, torch.float32) and self._module.usable(packed):
            count = packed.numel() * 2
            flat = self._module.dequant_nf4(packed, self.codec.absmax(absmax, scale, offset, count // BLOCK), self.codec.book(packed.device), dtype=dtype)
            return flat.view(shape)
        return super().dequant(packed, absmax, scale, offset, shape, dtype)


def make_nf4_backend(choice: str, device) -> Nf4Backend:
    """``auto`` is the fused kernel on an XPU when the package loads, otherwise torch; ``xpu_kernels`` insists on it."""
    if choice not in ("auto", "xpu_kernels", "torch"):
        raise ValueError(f"unknown NF4 backend {choice!r}; choose auto, xpu_kernels or torch")
    if choice == "torch" or (choice == "auto" and torch.device(device).type != "xpu"):
        return TorchNf4Backend()
    backend = XpuKernelsNf4Backend()
    if choice == "auto" and backend.problem() is not None:
        return TorchNf4Backend()
    return backend
