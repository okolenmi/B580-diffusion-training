"""ComfyUI setup and device utilities."""

import os
import sys
import time
from pathlib import Path

import torch

# Ensure paths is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paths import get_comfy_dir


def setup_comfy():
    """Ensure ComfyUI is importable.

    Uses paths.get_comfy_dir() which respects:
    - COMFY_DIR environment variable
    - Explicit set_comfy_dir() call
    - Auto-detection from current directory
    """
    comfy_dir = get_comfy_dir()
    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))

    try:
        import comfy  # noqa: F401
        return
    except ImportError:
        pass

    if not (comfy_dir / "comfy").exists():
        print(f"Warning: ComfyUI not found at {comfy_dir}")


def setup_device(device_name: str = "auto") -> torch.device:
    """Detect and return the best available torch device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")
    return torch.device(device_name)


def xpu_empty_cache():
    """Clear XPU cache if available."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def xpu_synchronize():
    """Block until all pending XPU work (including async transfers) completes."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def xpu_memory_stats() -> dict | None:
    """(allocated_mb, reserved_mb) or None if there's no XPU. Two separate
    numbers on purpose: allocated is memory actually holding live tensors
    right now; reserved is PyTorch's caching allocator's total pool
    (>= allocated, can stay high after tensors are freed since the
    allocator keeps freed blocks around for reuse rather than returning
    them to the driver -- see xpu_empty_cache). Growing *allocated* means
    something's genuinely still referencing more memory over time (a real
    leak to chase); reserved growing while allocated stays flat is just
    the allocator's own bookkeeping, not a leak."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # Deliberately only the two torch.xpu.memory_stats() keys: nodes/'
        # _XPUDeviceContext.memory_stats() is documented (and smoke-tested)
        # as a superset of exactly this dict, reading the same underlying
        # call -- a key from a different API (e.g. mem_get_info) doesn't
        # belong here. Driver-level usage is vram_snapshot's job instead.
        return {
            "allocated_mb": torch.xpu.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.xpu.memory_reserved() / (1024 ** 2),
        }
    return None


_VRAM_DEBUG = os.environ.get("TRAIN_VRAM_DEBUG", "0") == "1"
_T0 = time.monotonic()


def vram_snapshot(label: str):
    """Print allocated vs reserved XPU memory (MB) if TRAIN_VRAM_DEBUG=1.

    'allocated' = memory actively backing live tensors right now.
    'reserved'  = memory the caching allocator holds from the driver (>=
    allocated; the gap is the allocator's own free-block pool, kept around
    to avoid re-requesting from the driver on every alloc -- this is
    normal and doesn't by itself mean anything is leaked).

    A completed op's allocated-vs-reserved *gap* growing steadily across
    repeated calls to the same code path (not just staying elevated once)
    is the actual leak signature to look for; a one-time step up that then
    stays flat across further calls is ordinary allocator high-water-mark
    behavior, not a leak. No-op (zero overhead) unless the env var is set.
    """
    if not _VRAM_DEBUG or not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    try:
        alloc = torch.xpu.memory_allocated() / (1024 ** 2)
        reserved = torch.xpu.memory_reserved() / (1024 ** 2)
        # driver_used: what the GPU driver reports consumed device-wide
        # (this process + everything else -- desktop baseline included).
        # Expected to reconcile as reserved + ~0.5-1.5GB (other apps +
        # runtime/context overhead); the 2026-09-28 known-issues
        # investigation confirmed that reconciliation holds on this
        # hardware at every phase. A driver_used that stays *far* above
        # reserved + baseline would mean memory held outside torch's
        # allocator -- that gap is the thing to chase.
        try:
            _free, _total = torch.xpu.mem_get_info()
            driver = ( (_total - _free) / (1024 ** 2) )
            print(f"    [vram +{time.monotonic() - _T0:6.1f}s] {label}: allocated={alloc:.1f}MB reserved={reserved:.1f}MB driver_used={driver:.1f}MB", flush=True)
        except Exception:
            print(f"    [vram +{time.monotonic() - _T0:6.1f}s] {label}: allocated={alloc:.1f}MB reserved={reserved:.1f}MB", flush=True)
        if os.environ.get("TRAIN_VRAM_TENSOR_CENSUS") == "1":
            # Diagnostic: sum every live XPU tensor reachable via gc. Compare
            # against allocated= above. If census >> allocated, memory is
            # held in torch tensors that memory_allocated() isn't counting
            # (an accounting bug to report, not a leak); if census ~= allocated,
            # the missing device memory (see known-issues) lives outside torch.
            try:
                import gc as _gc
                _tot = 0
                _n = 0
                for _o in _gc.get_objects():
                    try:
                        if torch.is_tensor(_o) and _o.device.type == "xpu":
                            _tot += _o.numel() * _o.element_size()
                            _n += 1
                    except Exception:
                        continue
                print(f"    [vram +{time.monotonic() - _T0:6.1f}s] {label}: xpu tensor census={_tot / (1024 ** 2):.1f}MB across {_n} tensors", flush=True)
            except Exception as e:
                print(f"    [vram +{time.monotonic() - _T0:6.1f}s] {label}: census failed ({e})", flush=True)
    except Exception as e:
        print(f"    [vram +{time.monotonic() - _T0:6.1f}s] {label}: snapshot failed ({e})", flush=True)
