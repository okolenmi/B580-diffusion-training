"""Weights for the regions of a step from how the loss of their part of the timestep range is doing.

The loss of one step is mostly luck (the image and the noise drawn), so a region's success is never read from one number. The
range is cut into ``bins`` parts and every sample counts for the two nearest bins in proportion to how close it is (a timestep
between bins 10 and 11 feeds both). Each bin keeps two decaying averages of its loss, a slow one (``window`` steps) and a fast one
(``recent`` steps).

``trend`` mode: a bin has succeeded when its fast average sits below its slow one, ``(slow - fast) / slow``: the loss there has
recently been going down. ``current`` mode: the region's own loss is below the slow average of its bins, ``(slow - loss) / slow``:
a better fit than usual at this timestep, whatever the reason (noisier, since a loss is mostly the draw).

A region's success is divided by the typical size of successes seen so far, so ``boost`` means the same in both modes: a region
whose success is ``z`` typical sizes gets weight ``1 + boost * z`` (``z`` capped at ``limit``). Nothing is taken away unless ``penalty``
is set, which lowers the weight of a region that does worse than usual the same way, down to 0.25. Everything a weight reads comes
from the steps before this one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

MODES = ("trend", "current")
NEUTRAL_MASS = 3.0  # samples a bin needs behind its slow average before it speaks; fewer and its region stays at weight 1
RECENT_MASS = 1.5
SCALE_STEPS = 20   # successes seen before they are divided by their typical size


class TrendTracker:
    def __init__(self, bins: int = 12, window: int = 300, recent: int = 60, mode: str = "trend", boost: float = 0.5,
                 penalty: float = 0.0, limit: float = 3.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if bins < 2 or window < 2 or not 1 <= recent < window:
            raise ValueError("need bins >= 2 and 1 <= recent < window")
        if boost < 0 or penalty < 0 or limit <= 0:
            raise ValueError("boost and penalty must not be negative and limit must be positive")
        self.bins, self.mode, self.boost, self.penalty, self.limit = bins, mode, boost, penalty, limit
        self._decay_slow, self._decay_fast = math.exp(-1.0 / window), math.exp(-1.0 / recent)
        self._slow = [[0.0, 0.0] for _ in range(bins)]  # per bin: decayed sum of losses, decayed number of samples
        self._fast = [[0.0, 0.0] for _ in range(bins)]
        self._size, self._seen = 0.0, 0                  # decayed mean square of the successes, and how many were measured
        self._pending: list[tuple[list[tuple[int, float]], float]] = []
        self._weights: dict[int, list[float]] = {}
        self._z: dict[int, list[float]] = {}

    # -- one step ---------------------------------------------------------

    def begin(self) -> None:
        """A step (or its retry) starts: forget what the unfinished one collected."""
        self._pending, self._weights, self._z = [], {}, {}

    def weight(self, region: int, losses: Tensor, fractions: Tensor | Sequence[float]) -> float:
        """The weight of one region from its samples' losses and places in the range; the samples wait for ``commit``."""
        losses = [float(v) for v in losses.detach().flatten().float().cpu()]
        places = [float(v) for v in torch.as_tensor(fractions).flatten().float()]
        successes = []
        for loss, place in zip(losses, places):
            spread = self._spread(place)
            self._pending.append((spread, loss))
            s = self._success(spread, loss)
            if s is not None:
                successes.append(s)
        z = 0.0
        if successes:
            s = sum(successes) / len(successes)
            self._seen += 1
            self._size = s * s if self._seen == 1 else self._size + (s * s - self._size) / min(self._seen, SCALE_STEPS * 5)
            if self._seen >= SCALE_STEPS and self._size > 1e-12:
                z = s / math.sqrt(self._size)
        w = 1.0 + self.boost * min(z, self.limit) if z > 0 else max(0.25, 1.0 - self.penalty * min(-z, self.limit))
        self._weights.setdefault(region, []).append(w)
        self._z.setdefault(region, []).append(z)
        return w

    def commit(self) -> None:
        """The step is done: age every bin, then add the step's samples."""
        for table, decay in ((self._slow, self._decay_slow), (self._fast, self._decay_fast)):
            for cell in table:
                cell[0] *= decay
                cell[1] *= decay
        for spread, loss in self._pending:
            for k, share in spread:
                for table in (self._slow, self._fast):
                    table[k][0] += share * loss
                    table[k][1] += share
        self._pending = []

    def metrics(self) -> dict[str, float]:
        out = {}
        for region, values in self._weights.items():
            out[f"trend_w_region_{region}"] = sum(values) / len(values)
            out[f"trend_z_region_{region}"] = sum(self._z[region]) / len(self._z[region])
        return out

    # -- the bins ---------------------------------------------------------

    def _spread(self, place: float) -> list[tuple[int, float]]:
        """The two bins a place in the range feeds, with their shares (centres at the middle of equal parts)."""
        p = min(max(place * self.bins - 0.5, 0.0), self.bins - 1.0)
        low = min(int(p), self.bins - 2)
        share = p - low
        return [(low, 1.0 - share), (low + 1, share)]

    def _success(self, spread: list[tuple[int, float]], loss: float) -> float | None:
        total, mass = 0.0, 0.0
        for k, share in spread:
            slow_sum, slow_mass = self._slow[k]
            if slow_mass < NEUTRAL_MASS or share <= 0:
                continue
            slow = slow_sum / slow_mass
            if self.mode == "current":
                value = loss
            else:
                fast_sum, fast_mass = self._fast[k]
                if fast_mass < RECENT_MASS:
                    continue
                value = fast_sum / fast_mass
            total += share * (slow - value) / max(slow, 1e-12)
            mass += share
        return total / mass if mass > 0 else None
