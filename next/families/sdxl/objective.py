"""What SDXL's network predicts and how a training sample is noised."""

from __future__ import annotations

import torch
from torch import Tensor

from ..general.schedules.ddpm import (PARAMETERIZATIONS, DiffusionProcess, DiscreteSchedule, ZeroTerminalSNRSchedule,
                                      scale_input)
from ..general.schedules.timesteps import MODES as TIMESTEP_MODES, sample_timesteps


class SDXLObjective:
    """Scaled-linear schedule with eps or v prediction; ``zero_terminal_snr`` rescales the schedule (v only)."""

    parameterizations = tuple(PARAMETERIZATIONS)
    timestep_modes = TIMESTEP_MODES
    steps = 1000

    def __init__(self, parameterization: str = "eps", zero_terminal_snr: bool = False):
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"unknown parameterization {parameterization!r}; choose from {self.parameterizations}")
        schedule = ZeroTerminalSNRSchedule() if zero_terminal_snr else DiscreteSchedule()
        self.process = DiffusionProcess(schedule, PARAMETERIZATIONS[parameterization]())

    @property
    def t_range(self) -> tuple[int, int]:
        return 0, self.steps - 1

    def sample_t(self, n: int, mode: str, low: int, high: int, generator: torch.Generator | None = None) -> Tensor:
        return sample_timesteps(n, mode, low, high, generator)

    def noisy(self, x0: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(network input, regression target, sigma) for integer timesteps ``t``."""
        x_t, target, sigma = self.process.noisy(x0, noise, t)
        return scale_input(x_t, sigma.flatten()), target, sigma

    def to_x0(self, prediction: Tensor, x_t: Tensor, sigma: Tensor) -> Tensor:
        return self.process.parameterization.to_x0(prediction, x_t, sigma)
