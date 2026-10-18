"""Keeping the three parts of the timestep range moving together.

The optimizer minimises one number, the mean of the weighted per-sample losses, so a part of the range that stalls or
regresses changes nothing about the gradient while others improve. ``TimeBalance`` follows the loss of the low, middle
and high third of the range (the ``loss_t_*`` metrics) against each third's own history and offers two independent
levers: weights on the loss (``weights_for``) and a bias in which timesteps are drawn (``third_probabilities``, read by
the ``Noiser``). Thirds with no baseline yet stay neutral; nothing is made up for a part of the range not measured.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .loss import T_THIRDS

MODES = ("off", "normalize", "speed", "dro")


class TimeBalance:
    """``observe`` runs on the training thread; ``third_probabilities`` can run in the batch worker. Both touch only
    small dicts of floats, so no lock is taken."""

    def __init__(self, mode: str = "off", warmup_reports: int = 10, ema_alpha: float = 0.05, fast_alpha: float = 0.25,
                 eta: float = 0.5, clip_min: float = 0.25, clip_max: float = 4.0, dro_lambda: float = 1.0, sample_bias: float = 0.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if warmup_reports < 1 or not 0 < ema_alpha <= 1 or not 0 < fast_alpha <= 1 or eta <= 0:
            raise ValueError("warmup_reports must be at least 1, the EMA rates in (0, 1] and eta above 0")
        if clip_min <= 0 or clip_max < clip_min or dro_lambda < 0 or sample_bias < 0:
            raise ValueError("need 0 < clip_min <= clip_max, dro_lambda >= 0 and sample_bias >= 0")
        self.mode, self.warmup_reports, self.eta = mode, warmup_reports, eta
        self.ema_alpha, self.fast_alpha = ema_alpha, fast_alpha
        self.clip_min, self.clip_max, self.dro_lambda, self.sample_bias = clip_min, clip_max, dro_lambda, sample_bias
        self._warmup: dict[str, list[float]] = {}
        self._baseline: dict[str, float] = {}
        self._slow: dict[str, float] = {}
        self._fast: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def observe(self, thirds: dict[str, float]) -> None:
        """One optimizer step's mean loss per third. Thirds the step did not sample are left as they were."""
        for name in T_THIRDS:
            mean = thirds.get(name)
            if mean is None or not math.isfinite(mean) or mean <= 0.0:
                continue
            self._slow[name] = mean if name not in self._slow else self.ema_alpha * mean + (1 - self.ema_alpha) * self._slow[name]
            self._fast[name] = mean if name not in self._fast else self.fast_alpha * mean + (1 - self.fast_alpha) * self._fast[name]
            if name not in self._baseline:
                seen = self._warmup.setdefault(name, [])
                seen.append(mean)
                if len(seen) >= self.warmup_reports:
                    self._baseline[name] = sum(seen) / len(seen)
                    del self._warmup[name]
        if self.mode != "off" and self._baseline:
            self._reweigh()

    def _reweigh(self) -> None:
        ready = [name for name in T_THIRDS if name in self._baseline]
        if self.mode == "normalize":
            raw = {k: 1.0 / self._baseline[k] for k in ready}
        elif self.mode == "dro":
            relative = {k: self._slow[k] / self._baseline[k] for k in ready}
            worst = max(relative.values())
            raw = {k: math.exp(self.dro_lambda * (v - worst)) for k, v in relative.items()}
        else:  # speed: a third that descends slower than the others gains weight
            rates = {k: self._fast[k] / self._slow[k] for k in ready}
            pack = sum(rates.values()) / len(rates)
            raw = {k: self._weights.get(k, 1.0) * (rates[k] / pack) ** self.eta for k in ready}
        mean = sum(raw.values()) / len(raw)
        self._weights = {k: min(self.clip_max, max(self.clip_min, v / mean)) for k, v in raw.items()}

    @property
    def weights(self) -> dict[str, float]:
        """Current multiplier per third (only thirds past warmup, and only when ``mode`` is not ``off``)."""
        return dict(self._weights)

    def weights_for(self, t_frac: Tensor) -> Tensor | None:
        """One multiplier per sample from its place in the range; None when nothing is weighted yet."""
        if not self._weights:
            return None
        out = torch.ones(t_frac.numel(), device=t_frac.device)
        flat = t_frac.detach().flatten().float()
        for i, name in enumerate(T_THIRDS):
            if name in self._weights:
                out[(flat >= i / 3) & ((flat < (i + 1) / 3) | (i == 2))] = self._weights[name]
        return out

    def third_probabilities(self) -> list[float] | None:
        """Chance of drawing a timestep from each third, ``(current / baseline) ** sample_bias`` normalised; None while
        the draw should stay as the Noiser's own (bias 0, or nothing past warmup)."""
        if self.sample_bias == 0.0 or not self._baseline:
            return None
        raw = [max(self._slow[name] / self._baseline[name], 1e-6) ** self.sample_bias if name in self._baseline else 1.0 for name in T_THIRDS]
        total = sum(raw)
        return [r / total for r in raw]
