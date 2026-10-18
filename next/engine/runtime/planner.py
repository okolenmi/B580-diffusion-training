"""Predicting what a step needs from the size of its batch, and splitting batches that would not fit."""

from __future__ import annotations

from collections import deque


def chunk_sizes(count: int, parts: int) -> list[int]:
    """``count`` split into ``parts`` sizes that differ by at most one."""
    base, extra = divmod(count, parts)
    return [base + (i < extra) for i in range(parts)]


class StepPlanner:
    """Activation memory of a step is close to linear in the latent pixels it processes (the blocks recompute in
    backward, so only one block's activations are live at a time). The probe and the peak of each step give a cost per
    pixel; a batch whose predicted peak is over the budget runs as the fewest equal chunks that fit. Gradients of the
    chunks add up to those of the whole batch (the loss is a mean), so splitting changes memory and nothing else.
    """

    MARGIN = 1.15  # fragmentation and the first sight of a shape's kernels

    def __init__(self, window: int = 8):
        self._costs: deque[float] = deque(maxlen=window)

    @property
    def mb_per_unit(self) -> float | None:
        """Megabytes per latent pixel of the most demanding recent step (a unit is one pixel of one sample)."""
        return max(self._costs) if self._costs else None

    def reset(self) -> None:
        self._costs.clear()

    def learn(self, floor_mb: float, peak_mb: float, units: int) -> None:
        if units > 0 and peak_mb > floor_mb:
            self._costs.append((peak_mb - floor_mb) / units)

    def predicted_peak_mb(self, floor_mb: float, units: int) -> float | None:
        cost = self.mb_per_unit
        return None if cost is None else floor_mb + units * cost * self.MARGIN

    def chunks(self, count: int, units_per_sample: int, floor_mb: float, budget_mb: float | None) -> list[int]:
        """Sizes of the chunks the batch runs as (``[count]`` when it fits or nothing is known yet)."""
        if budget_mb is None or self.mb_per_unit is None:
            return [count]
        for parts in range(1, count + 1):
            if self.predicted_peak_mb(floor_mb, max(chunk_sizes(count, parts)) * units_per_sample) <= budget_mb:
                return chunk_sizes(count, parts)
        return chunk_sizes(count, count)
