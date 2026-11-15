"""Which blocks to checkpoint, from what each one costs.

Checkpointing every block is the safe choice and the slowest. With the measured memory and recompute time of each
block, the blocks that save the most time per megabyte of activations are kept (not recomputed) for as long as the
largest batch of the run still fits the budget. Blocks the profile has no memory figure for stay checkpointed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .planner import StepPlanner
from .profiler import BlockCost, BlockProfiler


class BlockPlacement:
    SHRINK = 0.25  # share of the kept blocks given back to checkpointing when a placement does not fit once measured

    def __init__(self, planner: StepPlanner, profiler: BlockProfiler):
        self.planner, self.profiler = planner, profiler

    def choose(self, measure: Callable[[Sequence[bool]], float], floor_mb: float, units: int, probe_units: int,
               budget_mb: float) -> list[bool]:
        """The flags (True = checkpoint) the model is left with. ``measure(flags)`` applies them, runs the probe and
        returns megabytes per latent pixel (infinite when it ran out of memory). ``probe_units`` is the pixel count of
        the probe, ``units`` that of the largest batch of the run."""
        names = self.profiler.order
        everything = [True] * len(names)
        usable = budget_mb - self.planner.RESERVE_MB
        self.profiler.reset()
        with self.profiler:
            base = measure(everything)  # the recomputes of this run are what the profiler times
        costs = self.profiler.costs()
        scale = units / max(probe_units, 1) * self.planner.MARGIN
        slack = usable - floor_mb - units * base * self.planner.MARGIN
        candidates = [n for n in names if n in costs and costs[n].activation_mb > 0]
        if slack <= 0 or not candidates:
            return everything
        keep = self._fill(candidates, costs, slack / scale)
        while keep:
            flags = [n not in keep for n in names]
            if floor_mb + units * measure(flags) * self.planner.MARGIN <= usable:
                return flags
            drop = sorted(keep, key=lambda n: costs[n].recompute_ms / costs[n].activation_mb)[:max(1, int(len(keep) * self.SHRINK))]
            keep -= set(drop)
        measure(everything)  # leaves the model set to it
        return everything

    def describe(self, flags: Sequence[bool]) -> list[dict]:
        """The blocks left without checkpointing, with what the profile said of them."""
        costs = self.profiler.costs()
        return [{"block": n, "mb": round(costs[n].activation_mb), "ms": round(costs[n].recompute_ms, 2)}
                for n, on in zip(self.profiler.order, flags) if not on and n in costs]

    @staticmethod
    def _fill(candidates: list[str], costs: dict[str, BlockCost], room_mb: float) -> set[str]:
        """Time saved per megabyte, best first, while the megabytes fit."""
        keep, used = set(), 0.0
        for name in sorted(candidates, key=lambda n: -costs[n].recompute_ms / costs[n].activation_mb):
            if used + costs[name].activation_mb <= room_mb:
                keep.add(name)
                used += costs[name].activation_mb
        return keep
