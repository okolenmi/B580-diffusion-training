"""DeviceContext: backend-specific device operations (empty_cache/synchronize/
memory_stats), selected once via for_device() instead of core.comfy_setup's
hasattr(torch, "xpu")/is_available() checks duplicated at every call site.
See docs/training_pipeline_design.md section 1.5.

**memory_stats() extended, docs/optimizer_execution_redesign_plan.md Phase
0.** Previously just {'allocated_mb', 'reserved_mb'} -- enough to see *that*
there's a reserved/allocated gap, nothing about *why*. torch.xpu.memory_stats()
(confirmed real, documented -- docs.pytorch.org/docs/stable/generated/
torch.xpu.memory.memory_stats.html -- mirrors torch.cuda's own allocator
stats dict key-for-key) exposes the actual allocator internals: segment
count, active vs. reserved-but-inactive bytes, and critically
num_alloc_retries -- a direct, unambiguous fragmentation signal (a failed
allocation that forced a cache flush + retry), not an inference from a
snapshot gap. All new keys are additive; every existing caller of the old
two keys is unaffected.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch


def allocator_conf_env() -> str:
    """Which allocator-config env var (if any) is actually set, checked in
    the order PyTorch itself resolves them (generalized name first, then
    the older per-backend/CUDA-specific names still supported for back
    compat) -- see docs/optimizer_execution_redesign_plan.md Phase 1 for
    why this specifically is being tested (PYTORCH_ALLOC_CONF is the
    newer, backend-agnostic name; current PyTorch warns that
    PYTORCH_CUDA_ALLOC_CONF is deprecated in its favor). Printed once per
    run so a person testing this env var gets direct confirmation it was
    actually read, in the same place as the numbers it's meant to
    affect, rather than needing to remember what they exported. Not
    underscore-prefixed -- deliberately part of this module's public
    surface, reused directly by nodes/train/supervised.py's own
    one-time startup prints rather than each call site re-implementing
    the same env var precedence check."""
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_XPU_ALLOC_CONF"):
        val = os.environ.get(key)
        if val:
            return f"{key}={val}"
    return "(none set)"


def _torch_memory_snapshot(memory_stats_fn, current_allocated_fn, current_reserved_fn) -> dict[str, float]:
    """Single source of truth for one backend's full memory snapshot --
    reads everything (current, peak, active, requested, retries,
    segments) from one memory_stats() call rather than several separate
    API calls, since the documented key names
    (docs.pytorch.org/docs/stable/generated/torch.xpu.memory.memory_stats.html,
    identical for torch.cuda) already carry both 'current' and 'peak'
    per stat. Falls back to the plain memory_allocated()/memory_reserved()
    scalars only if memory_stats() itself raises (older torch build,
    unexpected backend quirk) -- a version gap degrades to the original
    two-number behavior instead of crashing profiling entirely."""
    try:
        stats = memory_stats_fn()
        return {
            "allocated_mb": stats.get("allocated_bytes.all.current", 0) / (1024 ** 2),
            "reserved_mb": stats.get("reserved_bytes.all.current", 0) / (1024 ** 2),
            "peak_allocated_mb": stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2),
            "peak_reserved_mb": stats.get("reserved_bytes.all.peak", 0) / (1024 ** 2),
            "active_mb": stats.get("active_bytes.all.current", 0) / (1024 ** 2),
            "requested_mb": stats.get("requested_bytes.all.current", 0) / (1024 ** 2),
            "num_segments": float(stats.get("segment.all.current", 0)),
            "num_alloc_retries": float(stats.get("num_alloc_retries", 0)),
            "num_ooms": float(stats.get("num_ooms", 0)),
        }
    except Exception:
        try:
            allocated = current_allocated_fn() / (1024 ** 2)
            reserved = current_reserved_fn() / (1024 ** 2)
        except Exception:
            return {
                "allocated_mb": 0.0, "reserved_mb": 0.0,
                "peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0,
                "active_mb": 0.0, "requested_mb": 0.0,
                "num_segments": 0.0, "num_alloc_retries": 0.0, "num_ooms": 0.0,
            }
        return {
            "allocated_mb": allocated, "reserved_mb": reserved,
            "peak_allocated_mb": allocated, "peak_reserved_mb": reserved,
            "active_mb": 0.0, "requested_mb": 0.0,
            "num_segments": 0.0, "num_alloc_retries": 0.0, "num_ooms": 0.0,
        }


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
        """{'allocated_mb', 'reserved_mb', 'peak_allocated_mb',
        'peak_reserved_mb', 'active_mb', 'requested_mb', 'num_segments',
        'num_alloc_retries', 'num_ooms'} or None if this backend has no
        such concept (CPU). 'active_mb' vs 'reserved_mb': the gap between
        them is memory the allocator holds but isn't currently handing
        out to anything -- the caching/fragmentation overhead itself, as
        a direct number instead of an inference. 'requested_mb' vs
        'allocated_mb': the gap is allocator *rounding* overhead, a
        different and usually much smaller effect than fragmentation.
        'num_alloc_retries' > 0 (or increasing across two reports) is
        real, hard evidence of fragmentation forcing a cache flush --
        the strongest single signal in this dict, worth reading before
        the others."""

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
    """empty_cache/synchronize match core.comfy_setup.xpu_empty_cache/
    xpu_synchronize exactly -- same hasattr(torch, "xpu") and
    torch.xpu.is_available() guard, moved from free functions into one
    object's methods. memory_stats() no longer matches
    core.comfy_setup.xpu_memory_stats() (that one's still the original
    two-key {allocated_mb, reserved_mb} version, and as of grepping this
    codebase directly, isn't actually called from anywhere anymore --
    left as-is, out of scope here, not on this project's current
    nodes/-pipeline path); this class's own version is the one
    SupervisedLoRATrainerNode's profile output actually uses, extended
    per docs/optimizer_execution_redesign_plan.md Phase 0 -- see this
    module's own top docstring."""

    def empty_cache(self) -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def synchronize(self) -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    def memory_stats(self) -> dict[str, float] | None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return _torch_memory_snapshot(
                torch.xpu.memory_stats, torch.xpu.memory_allocated, torch.xpu.memory_reserved)
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
            return _torch_memory_snapshot(
                torch.cuda.memory_stats, torch.cuda.memory_allocated, torch.cuda.memory_reserved)
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
