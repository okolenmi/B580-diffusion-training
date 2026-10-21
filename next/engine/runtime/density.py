"""How many transformer blocks recompute their activations in backward, chosen from what the card allows."""

from __future__ import annotations

from typing import Callable

from .planner import StepPlanner


class CheckpointDensity:
    """Recomputing a block costs about a third of its forward time again, and storing its activations costs memory. The
    share of checkpointed transformer blocks (1 checkpoints all, 0 none) is lowered, in quarters, as far as the worst
    batch of the run still fits with its predicted peak: the cost per latent pixel is measured at the starting share and
    at half of it, extended linearly, and each share that is picked is measured again before it is kept.
    """

    LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0)

    def __init__(self, planner: StepPlanner):
        self.planner = planner

    def choose(self, measure: Callable[[float], float], start: float, floor_mb: float, units: int, budget_mb: float) -> float:
        """The share to run with. ``measure(share)`` applies it and returns megabytes per latent pixel; ``units`` is the
        pixel count of the largest batch. The model is left set to the share returned."""
        usable = budget_mb - self.planner.RESERVE_MB

        def fits(cost: float) -> bool:
            return floor_mb + units * cost * self.planner.MARGIN <= usable

        last = start
        cost_start = measure(start)
        lower = [level for level in self.LEVELS if level < start]
        if not lower or not fits(cost_start):
            return start
        probe = min(lower, key=lambda level: abs(level - start / 2))
        slope = (measure(probe) - cost_start) / (start - probe)  # megabytes per pixel for one unit of share given up
        last = probe
        chosen = start
        if slope > 0:
            for level in lower:
                if fits(cost_start + slope * (start - level)):
                    chosen = level
                else:
                    break
        while chosen < start:
            cost = measure(chosen)
            last = chosen
            if fits(cost):
                break
            chosen = min(level for level in self.LEVELS if level > chosen)
        if last != chosen:
            measure(chosen)
        return chosen
