"""DeviceContext: backend-specific device operations (empty_cache/synchronize/
memory_stats), selected once via for_device() instead of core.comfy_setup's
hasattr(torch, "xpu")/is_available() checks duplicated at every call site.
See docs/training_pipeline_design.md section 1.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DeviceContext(ABC):

    @abstractmethod
    def empty_cache(self) -> None:
        """Return unused cached memory to the driver. Best-effort -- can't
        free memory still genuinely referenced by something."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all pending device work (including async transfers)
        completes -- required before timing anything on an async backend."""

    @abstractmethod
    def memory_stats(self) -> dict[str, float] | None:
        """{'allocated_mb', 'reserved_mb'} or None if this backend has no
        such concept (CPU)."""

    @staticmethod
    def for_device(device) -> "DeviceContext":
        """Factory, called once at pipeline-construction time -- not a
        lookup repeated at every call site. The returned object is then
        passed around explicitly like anything else this design injects."""
        d = str(device)
        if d.startswith("xpu"):
            return _XPUDeviceContext()
        if d.startswith("cuda"):
            return _CUDADeviceContext()
        return _NullDeviceContext()


class _XPUDeviceContext(DeviceContext):
    """Matches core.comfy_setup.xpu_empty_cache/xpu_synchronize/
    xpu_memory_stats exactly -- same hasattr(torch, "xpu") and
    torch.xpu.is_available() guard, moved from three free functions into
    one object's three methods."""

    def empty_cache(self) -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def synchronize(self) -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    def memory_stats(self) -> dict[str, float] | None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return {
                "allocated_mb": torch.xpu.memory_allocated() / (1024 ** 2),
                "reserved_mb": torch.xpu.memory_reserved() / (1024 ** 2),
            }
        return None


class _CUDADeviceContext(DeviceContext):
    """Same three operations for CUDA. core.comfy_setup has no CUDA
    equivalent (this project targets XPU) -- added here because
    for_device() dispatches on the device string prefix and "cuda" is a
    real, reachable one; not exercised by any current caller."""

    def empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def memory_stats(self) -> dict[str, float] | None:
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            }
        return None


class _NullDeviceContext(DeviceContext):
    """CPU, or any backend without a cache/sync/stats concept. Every
    method is a correct, cheap no-op -- callers don't need an "if device
    supports this" branch of their own."""

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def memory_stats(self) -> dict[str, float] | None:
        return None
