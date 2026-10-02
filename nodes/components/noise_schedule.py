"""Free-function access to the diffusion objects, for call sites that have
no injection seam.

`nodes/components/diffusion.py` holds the real implementation, as
constructed objects: a `NoiseSchedule`, a `Parameterization`. That is the
shape the training loop wants, because the three pieces travel together and
cannot be mismatched.

`manager/` predates that and reaches for these as module-level functions.
Rather than keep a second copy of the maths alive in the tree -- which is
what `core/noise_schedule.py` was, and which existed only to be migrated
*to* `diffusion.py` -- these are thin adapters over the objects. Same
call signatures, one implementation.

Equivalence is not assumed: `smoke_test_diffusion_equivalence.py` proves
the objects compute what `core/noise_schedule` computed, and this module
adds no arithmetic of its own, so the chain is
adapter -> object == retired free function.

`T_MODES` used to be a *copy* of the list here, duplicated into
`nodes/dataset/timestep_modes.py`, with a comment explaining the two
could not be kept in step because one side could not import the other.
That reason is gone: both live in `nodes/`. It is now imported, not copied.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..dataset.timestep_modes import T_MODES
from .diffusion import (
    DiscreteLinearNoiseSchedule,
    EpsParameterization,
    VPredParameterization,
)

__all__ = [
    "T_MODES", "get_alpha_sigma", "sample_timestep",
    "eps_to_x0", "eps_to_vpred", "vpred_to_x0", "vpred_to_eps",
]

# One schedule for the process, built once. The object's per-device cache
# is why this is safe where `core`'s module globals were not: two callers
# on two devices index the same instance without stepping on each other.
_SCHEDULE = DiscreteLinearNoiseSchedule()

_EPS = EpsParameterization()
_VPRED = VPredParameterization()


def get_alpha_sigma(t: Any):
    """(alpha, sigma) for a timestep index. Accepts an int or a Tensor."""
    return _SCHEDULE.alpha_sigma(t)


def sample_timestep(rng: random.Random, mode: str, t_low: int, t_high: int) -> int:
    """Sample a timestep in [t_low, t_high] per one of ``T_MODES``.

      uniform -- equal probability across the range (default)
      low     -- Beta(1, 3): biased toward low t (late denoising, fine detail)
      mid     -- Beta(2, 2): biased toward the middle
      high    -- Beta(3, 1): biased toward high t (coarse structure)
      logit   -- logit-normal: middle-weighted with heavier tails
    """
    lo, hi = t_low, t_high
    if mode == "uniform":
        return rng.randint(lo, hi)
    if mode == "logit":
        u = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-u))
        return max(lo, min(hi, int(round(lo + p * (hi - lo)))))
    a, b = {"low": (1.0, 3.0), "mid": (2.0, 2.0), "high": (3.0, 1.0)}.get(
        mode, (1.0, 1.0)
    )
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    return max(lo, min(hi, int(round(lo + x / (x + y) * (hi - lo)))))


def eps_to_x0(eps, x_t, alpha, sigma):
    return _EPS.to_x0(eps, x_t, alpha, sigma)


def eps_to_vpred(eps, x_t, alpha, sigma):
    return _EPS.convert_to(eps, x_t, alpha, sigma, _VPRED)


def vpred_to_x0(v, x_t, alpha, sigma):
    return _VPRED.to_x0(v, x_t, alpha, sigma)


def vpred_to_eps(v, x_t, alpha, sigma):
    return _VPRED.convert_to(v, x_t, alpha, sigma, _EPS)