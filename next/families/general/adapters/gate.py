"""Timestep gate: scale an adapter's contribution toward 1 inside the timestep range being trained and toward 0
outside it, so timesteps the data never covers stay close to the frozen base model.

One ``TimestepGate`` is shared by every layer of an adapted model; the training step sets it once per batch.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from torch import Tensor

from ...api.adapters import GateWindow


def compute_gate(t: Tensor, train_low: float, train_high: float, width: float) -> Tensor:
    """See ``GateWindow.gate``."""
    return GateWindow(train_low, train_high, width).gate(t)


class TimestepGate:
    def __init__(self) -> None:
        self.value: Tensor | None = None

    def set(self, gate: Tensor | None) -> None:
        self.value = gate

    @contextlib.contextmanager
    def override(self, gate: Tensor | None) -> Iterator[None]:
        previous, self.value = self.value, gate
        try:
            yield
        finally:
            self.value = previous

    def apply(self, out: Tensor) -> Tensor:
        """Multiply per-sample (the batch axis) by the current gate, if any."""
        if self.value is None:
            return out
        g = self.value.to(device=out.device, dtype=out.dtype)
        return out * g.view(-1, *([1] * (out.dim() - 1)))
