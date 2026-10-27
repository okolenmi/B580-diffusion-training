"""What a batch costs in memory, measured before the first step, and the checkpointing chosen from it."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

from ..runtime.memory import ResidencyManager, is_out_of_memory
from .phases import FetchBatch, Pipeline
from .state import StepState


class MemoryProbe:
    """A sample and a few samples of the largest shape go through forward and backward with nothing changed, so the
    planner knows what a pixel costs; with a checkpointing chooser (``density`` or the measured ``placement``, which wins)
    each candidate is measured the same way and the cheapest that fits stays."""

    SAMPLES = 3  # samples in the larger probe batch

    def __init__(self, residency: ResidencyManager | None, optimizer, pipeline: Pipeline, batches: Iterable[dict],
                 fetch: FetchBatch, device, task: Callable[[str], AbstractContextManager],
                 density: tuple | None = None, placement: tuple | None = None, splits: bool = True) -> None:
        self.residency, self.optimizer, self.pipeline = residency, optimizer, pipeline
        self.batches, self.fetch, self.device, self.task = batches, fetch, device, task
        self.density = density  # (CheckpointDensity, TrainableDenoiser) or None
        self.placement = placement  # (BlockPlacement, TrainableDenoiser) or None
        self.splits = splits  # whether a step too big for the budget can run as chunks

    def run(self) -> None:
        make = getattr(self.batches, "probe_batch", None)
        if self.residency is None or make is None:
            return
        largest = self.batches.largest_units()
        self._choose(make, largest)
        self.residency.expect(largest)
        predicted, budget = self.residency.largest_peak_mb(), self.residency.budget_mb()
        if not self.splits and predicted is not None and budget is not None and predicted > budget:
            self.residency.note("unsplit_over_budget", predicted_mb=round(predicted), budget_mb=round(budget))

    def _choose(self, make: Callable, largest: Any) -> None:
        like = self.fetch.peek()
        sizes = [1] + ([min(self.batches.max_batch_size(), self.SAMPLES)] if self.batches.max_batch_size() > 1 else [])
        probes = [make(like, count) for count in sizes]  # two sizes of the same shape: the slope is what a sample costs, the rest is fixed
        units = probes[-1]["x_t"].shape[0] * probes[-1]["x_t"].shape[-2] * probes[-1]["x_t"].shape[-1]
        budget = self.residency.budget_mb()
        if budget is None or (self.density is None and self.placement is None):
            self._measure(probes, largest)
            return
        floor = self.residency.accelerator.memory().get("allocated", 0.0)
        if self.placement is not None:
            chooser, model = self.placement
            flags = chooser.choose(lambda flags: self._measure(probes, largest, lambda: model.set_checkpoint_flags(flags), tolerate=not all(flags)),
                                   floor, largest, units, budget)
            self.residency.note("checkpoint_plan", checkpointed=sum(flags), of=len(flags), kept=chooser.describe(flags))
            return
        tuner, model = self.density
        starting = model.checkpoint_share
        chosen = tuner.choose(lambda share: self._measure(probes, largest, lambda: model.set_checkpoint_share(share), tolerate=share != starting),
                              starting, floor, largest, budget)
        self.residency.note("checkpoint_share", share=chosen, of_blocks="transformer")

    def _measure(self, probes: list[dict], largest: Any, apply: Callable[[], None] | None = None, tolerate: bool = False) -> float:
        """``apply`` changes the checkpointing first; a candidate that cannot fit says so by costing more than any budget."""
        residency = self.residency
        if apply is not None:
            apply()
            residency.planner.reset()
        try:
            with self.task("Measuring what a pixel costs"):
                for batch in probes:
                    size = batch["x_t"].shape[0] * batch["x_t"].shape[-2] * batch["x_t"].shape[-1]
                    with residency.measuring(size), self.optimizer.suspended():
                        self.pipeline.run(StepState(-1, 0, self.device, batch=batch))
        except Exception as error:  # noqa: BLE001 - narrowed just below
            if not tolerate or not is_out_of_memory(error):
                raise
            error.__traceback__ = None
            gc.collect()
            residency.give_back("a checkpoint choice that did not fit")
            return math.inf
        return residency.planner.cost_at(largest) or 0.0
