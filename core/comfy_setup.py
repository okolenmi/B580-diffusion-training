"""ComfyUI setup and device utilities."""

import os
import sys
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


_VRAM_DEBUG = os.environ.get("TRAIN_VRAM_DEBUG", "0") == "1"


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
        print(f"    [vram] {label}: allocated={alloc:.1f}MB reserved={reserved:.1f}MB", flush=True)
    except Exception as e:
        print(f"    [vram] {label}: snapshot failed ({e})", flush=True)
