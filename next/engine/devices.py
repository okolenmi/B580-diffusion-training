"""Turning setting values (``xpu``, ``bf16``, ``auto``) into torch devices and dtypes."""

from __future__ import annotations

import torch

from .driver_memory import DriverMemory

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
        self._driver = DriverMemory()

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

    def cap_allocator(self, mb: float) -> bool:
        """Limit what the caching allocator may hold to ``mb``. At the limit it gives back its free blocks before it
        fails, which is what keeps cached memory from starving the driver's own allocations. False when unsupported."""
        if self.module is None or not hasattr(self.module, "set_per_process_memory_fraction") or not hasattr(self.module, "mem_get_info"):
            return False
        try:
            total = self.module.mem_get_info()[1] / 2**20
            self.module.set_per_process_memory_fraction(min(max(mb / total, 0.01), 1.0))
        except Exception:  # noqa: BLE001 - a build that cannot cap simply runs uncapped
            return False
        return True

    def memory(self) -> dict[str, float]:
        """Megabytes, by what each number counts (missing values are omitted):

        ``allocated``/``peak_allocated``: live tensors now and since the last reset. ``reserved``: held by the caching
        allocator, ``active`` of it in blocks in use and ``split_free`` in free pieces between them (those stay
        while a neighbour lives; only whole free segments can be returned). ``process``: everything this process holds
        on the card, the allocator included. ``device_used``: the driver's view of the whole card, other programs
        included; it is what runs out, so it is what predicts an out-of-memory error.
        Call it before ``empty_cache``, while the step's blocks are still held.
        """
        out: dict[str, float] = {}
        if self.module is None:
            return out
        for key, name in (("peak_allocated", "max_memory_allocated"), ("peak_reserved", "max_memory_reserved"),
                          ("reserved", "memory_reserved"), ("allocated", "memory_allocated")):
            if hasattr(self.module, name):
                out[key] = round(getattr(self.module, name)() / 2**20, 1)
        if hasattr(self.module, "memory_stats"):
            try:
                stats = self.module.memory_stats()
                for key, name in (("active", "active_bytes.all.current"), ("split_free", "inactive_split_bytes.all.current")):
                    if name in stats:
                        out[key] = round(stats[name] / 2**20, 1)
                for key, name in (("segments", "segment.all.current"), ("alloc_retries", "num_alloc_retries")):
                    if name in stats:
                        out[key] = float(stats[name])
            except Exception:  # noqa: BLE001 - statistics are optional detail
                pass
        if hasattr(self.module, "mem_get_info"):
            try:
                free, total = self.module.mem_get_info()
                out["device_used"] = round((total - free) / 2**20, 1)
                out["device_total"] = round(total / 2**20, 1)
            except Exception:  # noqa: BLE001 - older builds lack the call for some devices
                pass
            process = self._driver.resident_mb()
            if process is not None and "device_used" in out:
                out["process"] = min(process, out["device_used"])
        return out
