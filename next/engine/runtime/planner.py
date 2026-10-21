"""Predicting what a step needs from the size of its batch, and splitting batches that would not fit."""

from __future__ import annotations

from collections import deque


def chunk_sizes(count: int, parts: int) -> list[int]:
    """``count`` split into ``parts`` sizes that differ by at most one."""
    base, extra = divmod(count, parts)
    return [base + (i < extra) for i in range(parts)]


class StepPlanner:
    """Memory a step needs beyond what is already live: a fixed part (workspaces, scratch that does not grow with the
    batch) plus a part linear in the latent pixels it processes (the blocks recompute in backward, so only one block's
    activations are live at a time). Every probe and step peak is a point (pixels, megabytes); the line is fitted through
    them and lifted to lie above every point seen, so a prediction is never below what was observed. One point alone is
    taken as proportional, which overstates a big batch whenever the fixed part is large. A batch whose predicted peak
    is over the budget runs as the fewest equal chunks that fit; gradients of the chunks add up to those of the whole
    batch (the loss is a mean), so splitting changes memory and nothing else.
    """

    RESERVE_MB = 256.0  # what the optimizer update and gradient zeroing hold at the peak, which the probe does not run; the cap already holds the margin
    MARGIN = 1.15  # on the part that grows with the batch: fragmentation and the first sight of a shape's kernels

    def __init__(self, window: int = 64):
        self._points: deque[tuple[int, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._points.clear()

    def learn(self, floor_mb: float, peak_mb: float, units: int) -> None:
        if units > 0 and peak_mb > floor_mb:
            self._points.append((units, peak_mb - floor_mb))

    def line(self) -> tuple[float, float] | None:
        """``(fixed megabytes, megabytes per latent pixel)`` through what has been seen, or None before anything was."""
        if not self._points:
            return None
        peaks: dict[int, float] = {}
        for units, extra in self._points:
            peaks[units] = max(peaks.get(units, 0.0), extra)
        proportional = (0.0, max(extra / units for units, extra in peaks.items()))
        if len(peaks) < 2:
            return proportional
        mean_u, mean_e = sum(peaks) / len(peaks), sum(peaks.values()) / len(peaks)
        spread = sum((u - mean_u) ** 2 for u in peaks)
        slope = sum((u - mean_u) * (e - mean_e) for u, e in peaks.items()) / spread
        if slope <= 0.0:
            return proportional
        return max(max(e - slope * u for u, e in peaks.items()), 0.0), slope

    @property
    def mb_per_unit(self) -> float | None:
        """Megabytes per latent pixel of the growing part."""
        line = self.line()
        return None if line is None else line[1]

    def cost_at(self, units: int) -> float | None:
        """Average megabytes per pixel of a step of ``units`` pixels, fixed part included, before the margin."""
        line = self.line()
        return None if line is None or units <= 0 else (line[0] + line[1] * units) / units

    def predicted_peak_mb(self, floor_mb: float, units: int) -> float | None:
        line = self.line()
        return None if line is None else floor_mb + line[0] + units * line[1] * self.MARGIN

    def chunks(self, count: int, units_per_sample: int, floor_mb: float, budget_mb: float | None) -> list[int]:
        """Sizes of the chunks the batch runs as (``[count]`` when it fits or nothing is known yet)."""
        if budget_mb is None or self.line() is None:
            return [count]
        for parts in range(1, count + 1):
            if self.predicted_peak_mb(floor_mb, max(chunk_sizes(count, parts)) * units_per_sample) <= budget_mb:
                return chunk_sizes(count, parts)
        return chunk_sizes(count, count)
