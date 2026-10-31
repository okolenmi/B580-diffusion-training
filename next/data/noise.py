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
    regions: int = 1  # timesteps per image in one step (see ``Noiser``)
    layout: str = "even"  # "even": equally spaced around the range from a random phase; "strata": one random draw in each equal part of it


class Noiser:
    def __init__(self, objective: Objective, policy: TimestepPolicy | None = None, balance=None):
        policy = policy or TimestepPolicy()
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
        if policy.regions < 1 or policy.layout not in ("even", "strata"):
            raise ValueError("regions must be at least 1 and the layout 'even' or 'strata'")
        self.regions = policy.regions
        self.objective, self.policy, self.balance = objective, policy, balance  # balance: steers the draw towards lagging thirds

    def t_at(self, fraction: Tensor) -> Tensor:
        """Timesteps at the given fractions of this noiser's range (rounded when the objective counts in integers)."""
        t = self.low + fraction * (self.high - self.low)
        return t.round().long() if self._integer else t.float()

    def spread(self, n: int, generator: torch.Generator) -> Tensor:
        """Fractions of the range for ``n`` samples laid out region by region (``n / regions`` images, each repeated once per
        region): every image gets one fraction per region. "even": ``1 / regions`` apart around the range from a random common
        phase, so one step covers the range evenly instead of one random spot. "strata": region r is always the r-th equal part
        of the range, so a region means the same part of the range in every step."""
        images = n // self.regions
        draw = torch.rand(images, generator=generator)
        if self.policy.layout == "strata":
            return torch.cat([(draw + r) / self.regions for r in range(self.regions)])
        return torch.cat([(draw + r / self.regions) % 1.0 for r in range(self.regions)])

    def __call__(self, x0: Tensor, generator: torch.Generator) -> dict[str, Tensor]:
        """``x_t`` (network input), ``t``, ``target``, ``sigma`` and ``t_frac`` (t as a fraction of the objective's range).

        With ``regions`` above 1 a batch is ``regions`` copies of its images laid out region by region (the batcher makes the
        copies); ``region`` tells which one each sample belongs to. A batch that cannot be split that way is drawn as usual."""
        n = x0.shape[0]
        regional = self.regions > 1 and n % self.regions == 0 and not self.policy.values and self.balance is None
        region = torch.arange(self.regions).repeat_interleave(n // self.regions) if regional else torch.zeros(n, dtype=torch.long)
        if regional:
            t = self.t_at(self.spread(n, generator))
        elif self.policy.values:
            pick = torch.randint(0, len(self.policy.values), (n,), generator=generator)
            t = torch.tensor(self.policy.values)[pick]
            t = t.round().long() if self._integer else t.float()
        elif self.balance is not None and (shares := self.balance.third_probabilities()) is not None:
            third = torch.multinomial(torch.tensor(shares), n, replacement=True, generator=generator)
            t = self.t_at((third + torch.rand(n, generator=generator)) / 3)
        else:
            t = self.objective.sample_t(n, self.policy.mode, self.low, self.high, generator)
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
        x_t, target, sigma = self.objective.noisy(x0, noise, t)
        t_frac = (t.float() - self.low) / max(float(self.high - self.low), 1e-12)
        return {"x_t": x_t, "t": t, "target": target, "sigma": sigma, "t_frac": t_frac, "region": region}
