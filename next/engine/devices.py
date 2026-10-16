"""Turning setting values (``xpu``, ``bf16``, ``auto``) into torch devices and dtypes."""

from __future__ import annotations

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_device(name: str) -> str:
    """``auto`` picks an Intel GPU when present, then CUDA, then the CPU."""
    if name != "auto":
        return name
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name: str, default: torch.dtype) -> torch.dtype:
    if name == "auto":
        return default
    if name not in DTYPES:
        raise ValueError(f"unknown precision {name!r}; choose from auto, {', '.join(DTYPES)}")
    return DTYPES[name]


class Accelerator:
    """Synchronize and memory statistics for whichever backend the device uses (all no-ops on the CPU)."""

    def __init__(self, device: str):
        self.device = device
        self.module = getattr(torch, device.split(":")[0], None) if device != "cpu" else None

    def synchronize(self) -> None:
        if self.module is not None and hasattr(self.module, "synchronize"):
            self.module.synchronize()

    def empty_cache(self) -> None:
        if self.module is not None and hasattr(self.module, "empty_cache"):
            self.module.empty_cache()

    def peak_mb(self) -> float | None:
        """Peak of live tensors; understates what the card holds (see ``memory``)."""
        if self.module is not None and hasattr(self.module, "max_memory_allocated"):
            return self.module.max_memory_allocated() / 2**20
        return None

    def reset_peak(self) -> None:
        if self.module is not None and hasattr(self.module, "reset_peak_memory_stats"):
            self.module.reset_peak_memory_stats()

    def memory(self) -> dict[str, float]:
        """Megabytes: peak live tensors, peak held by the allocator, and what the driver reports in use.

        The allocator keeps freed blocks and the driver adds kernel workspaces and the context, so
        only ``device_used`` predicts an out-of-memory error.
        Call it before ``empty_cache``, while the step's blocks are still held. Missing values are omitted.
        """
        out: dict[str, float] = {}
        if self.module is None:
            return out
        for key, name in (("peak_allocated", "max_memory_allocated"), ("peak_reserved", "max_memory_reserved")):
            if hasattr(self.module, name):
                out[key] = round(getattr(self.module, name)() / 2**20, 1)
        if hasattr(self.module, "mem_get_info"):
            try:
                free, total = self.module.mem_get_info()
                out["device_used"] = round((total - free) / 2**20, 1)
                out["device_total"] = round(total / 2**20, 1)
            except Exception:  # noqa: BLE001 - older builds lack the call for some devices
                pass
        return out
