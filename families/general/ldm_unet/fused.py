"""Optional fused kernels for the UNet's feed-forward, from the ``xpu-kernels`` package.

The package verifies each kernel against the eager chain on this machine before it may run, so a closed gate is a reason,
not an error: the model keeps its eager path and the reason is reported.
"""

from __future__ import annotations

import importlib

import torch

from .attention import GEGLU


class FusedGeglu:
    @staticmethod
    def enable(device: torch.device) -> str:
        """Route every GEGLU through the fused kernel. Returns "" when it is on, otherwise why it stays off."""
        try:
            module = importlib.import_module("xpu_kernels.triton.geglu")
        except ImportError as error:
            return f"xpu-kernels is not installed ({error})"
        if not hasattr(module, "geglu"):
            return "the installed xpu-kernels has no fused GEGLU"
        probe = torch.zeros(2, 8, dtype=torch.bfloat16, device=device)
        if not module.usable(probe):  # runs the self-test on first use, which must happen before a graph is captured
            return module.gates()["geglu"].describe()
        GEGLU.fused = module.geglu
        return ""

    @staticmethod
    def active() -> bool:
        return GEGLU.fused is not None

    @staticmethod
    def disable() -> None:
        GEGLU.fused = None
