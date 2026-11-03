"""Power-function moving averages of trained weights (EDM2): the averaging window grows in proportion to the run's age.

A fixed-decay average either lags far behind early on or stays noisy late; here the decay at update ``t`` is
``(1 - 1/t) ** (gamma + 1)``, so the average always covers about the same *fraction* of the run so far. ``sigma_rel``
is the width of the window relative to the age (0.05 = roughly the last 5-10%). Several widths run side by side, and
the one that gives the best pictures is picked by eye afterwards. The weights are only read; training never sees them."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor


def gamma_for(sigma_rel: float) -> float:
    """The exponent whose window has relative width ``sigma_rel``: solves ``s^2 = (g+1) / ((g+2)^2 (g+3))`` (decreasing in g)."""
    if not 0.0 < sigma_rel < 0.29:
        raise ValueError("sigma_rel must lie between 0 and 0.28 (a window of the whole run is about 0.29)")
    low, high = 0.0, 1e6
    for _ in range(200):
        mid = (low + high) / 2
        width = ((mid + 1) / ((mid + 2) ** 2 * (mid + 3))) ** 0.5
        low, high = (mid, high) if width > sigma_rel else (low, mid)
    return (low + high) / 2


class PowerEma:
    """Averages of ``params``, one per ``sigma_rel``; ``update`` is called once per optimizer step."""

    def __init__(self, params: Sequence[Tensor], sigma_rels: Sequence[float]):
        if not sigma_rels:
            raise ValueError("an average needs at least one sigma_rel")
        self._params = list(params)
        self.sigma_rels = tuple(sigma_rels)
        self._exponents = [gamma_for(s) + 1.0 for s in self.sigma_rels]
        self._shadows = [[p.detach().clone() for p in self._params] for _ in self.sigma_rels]
        self.updates = 0
        self._synced = 0  # the step number of the last update

    def decay(self, index: int) -> float:
        """The weight the old average keeps at the next update."""
        return (1.0 - 1.0 / (self.updates + 1)) ** self._exponents[index]

    @torch.no_grad()
    def update(self, step: int) -> None:
        """Fold the current weights in; a second call for the same step does nothing, so several events may ask for it."""
        if step <= self._synced:
            return
        self._synced = step
        current = [p.detach() for p in self._params]
        for index, shadow in enumerate(self._shadows):
            torch._foreach_lerp_(shadow, current, 1.0 - self.decay(index))
        self.updates += 1

    def state(self) -> tuple[dict[str, Tensor], dict]:
        """Everything a resumed run needs: the averages as host tensors and the counters."""
        tensors = {f"{i}.{j}": t.detach().to("cpu", copy=True).contiguous() for i, shadow in enumerate(self._shadows) for j, t in enumerate(shadow)}
        return tensors, {"updates": self.updates, "synced": self._synced, "sigma_rels": list(self.sigma_rels)}

    @torch.no_grad()
    def restore(self, tensors: dict[str, Tensor], meta: dict) -> None:
        if list(meta["sigma_rels"]) != list(self.sigma_rels):
            raise ValueError(f"the saved averages have the widths {meta['sigma_rels']}, this run asks for {list(self.sigma_rels)}")
        for i, shadow in enumerate(self._shadows):
            for j, t in enumerate(shadow):
                t.copy_(tensors[f"{i}.{j}"].to(device=t.device, dtype=t.dtype))
        self.updates, self._synced = int(meta["updates"]), int(meta["synced"])

    def values(self, index: int) -> list[Tensor]:
        """Average ``index`` as one tensor per parameter (the live buffers, not copies)."""
        return self._shadows[index]

    @contextmanager
    def applied(self, index: int) -> Iterator[None]:
        """Inside the block the parameters hold average ``index`` (swapped, not copied), afterwards their own values."""
        shadow = self._shadows[index]
        for p, s in zip(self._params, shadow):
            p.data, s.data = s.data, p.data
        try:
            yield
        finally:
            for p, s in zip(self._params, shadow):
                p.data, s.data = s.data, p.data

    def label(self, index: int) -> str:
        return f"ema{self.sigma_rels[index]:g}"
