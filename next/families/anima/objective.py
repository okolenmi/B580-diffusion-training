"""What Anima's network predicts and how a training sample is noised: rectified flow, velocity target."""

from __future__ import annotations

import torch
from torch import Tensor

from ..general.schedules import flow

SHIFT = 3.0   # the value ComfyUI samples Anima with; training on the same shift keeps the two consistent


class AnimaObjective:
    """``x_s = (1 - s) * x0 + s * noise`` for flow positions ``s`` in (0, 1); the target is ``noise - x0``.

    ``shift`` pushes the drawn positions toward noise, ``scale`` is the spread of the sigmoid / logit-normal samplers. The
    ``sigma`` a sample reports is the noise-to-signal ratio ``s / (1 - s)``, the scale loss weightings are written for.
    """

    OPTIONS = ("shift", "scale")
    timestep_modes = flow.SAMPLERS
    steps = 1000

    def __init__(self, shift: float | None = None, scale: float | None = None):
        self.shift = SHIFT if shift is None else float(shift)
        self.scale = 1.0 if scale is None else float(scale)
        if self.shift <= 0 or self.scale <= 0:
            raise ValueError("shift and scale must be positive")

    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def sample_t(self, n: int, mode: str, low: float, high: float, generator: torch.Generator | None = None) -> Tensor:
        """Flow positions after the shift, cut to ``[low, high]``."""
        return flow.sample_sigma(n, mode, shift=self.shift, scale=self.scale, generator=generator, low=low, high=high)

    def noisy(self, x0: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(network input, velocity target, noise-to-signal ratio) for flow positions ``t``."""
        s = t.float().clamp(0.0, 1.0 - 1e-5)
        x_s, target = flow.noisy(x0, noise, s)
        return x_s, target, s / (1.0 - s)

    def to_x0(self, prediction: Tensor, x_t: Tensor, sigma: Tensor) -> Tensor:
        return flow.to_x0(prediction, x_t, sigma / (1.0 + sigma))
