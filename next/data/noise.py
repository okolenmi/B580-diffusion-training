"""Turning clean latents into a training sample: draw timesteps and noise, let the family's objective
build the network input and the regression target."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..families.api.objective import Objective


@dataclass(frozen=True)
class TimestepPolicy:
    """Where timesteps come from. ``values`` (if given) overrides the distribution: a fixed list to choose from."""

    mode: str = "uniform"
    low: float | None = None
    high: float | None = None
    values: tuple[float, ...] = ()


class Noiser:
    def __init__(self, objective: Objective, policy: TimestepPolicy = TimestepPolicy()):
        if policy.mode not in objective.timestep_modes:
            raise ValueError(f"timestep mode {policy.mode!r} is not offered by this objective; "
                             f"choose from {tuple(objective.timestep_modes)}")
        low, high = objective.t_range
        self._integer = isinstance(low, int)
        cast = int if self._integer else float
        self.low = low if policy.low is None else cast(policy.low)
        self.high = high if policy.high is None else cast(policy.high)
        if not low <= self.low <= self.high <= high:
            raise ValueError(f"timestep range [{self.low}, {self.high}] must lie inside [{low}, {high}]")
        bad = [v for v in policy.values if not low <= v <= high]
        if bad:
            raise ValueError(f"timestep values {bad} lie outside [{low}, {high}]")
        self.objective, self.policy = objective, policy

    def __call__(self, x0: Tensor, generator: torch.Generator) -> dict[str, Tensor]:
        """``x_t`` (network input), ``t``, ``target``, ``sigma`` and ``t_frac`` (t as a fraction of the objective's range)."""
        n = x0.shape[0]
        if self.policy.values:
            pick = torch.randint(0, len(self.policy.values), (n,), generator=generator)
            t = torch.tensor(self.policy.values)[pick]
            t = t.round().long() if self._integer else t.float()
        else:
            t = self.objective.sample_t(n, self.policy.mode, self.low, self.high, generator)
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
        x_t, target, sigma = self.objective.noisy(x0, noise, t)
        t_frac = (t.float() - self.low) / max(float(self.high - self.low), 1e-12)
        return {"x_t": x_t, "t": t, "target": target, "sigma": sigma, "t_frac": t_frac}
