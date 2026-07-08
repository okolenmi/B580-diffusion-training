"""ComfyUI setup and device utilities."""

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
