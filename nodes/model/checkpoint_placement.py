"""BlockCost/CheckpointPlacementPolicy/EveryBlockPlacement/GreedyRatioPlacement.

See docs/training_pipeline_design.md section 2.3 for the full rationale
(Chen et al. 2016 arXiv:1604.06174's sqrt(N)-uniform result, generalized
by Korthikanti et al. 2022's cost-ratio ranking, which this is a direct,
simplified reading of). The design doc's own illustrative code is
reproduced here almost verbatim -- this half of the item was never the
hard part; nodes/model/block_profiler.py's real per-block instrumentation
(producing the BlockCost values a real GreedyRatioPlacement call would
use) was.

One real correction from the design doc's own sketch: GreedyRatioPlacement
below subtracts ResourceBudget.vram_reserve_mb from vram_budget_mb before
comparing -- the doc's illustrative `remaining <= budget.vram_budget_mb`
predates vram_reserve_mb existing as a field (nodes/resource_policy.py),
and ResourceBudget's own docstring is explicit that vram_budget_mb is a
ceiling *before* the safety margin, not the usable amount. Fitting
against the full ceiling would silently eat into that margin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..resource_policy import ResourceBudget


@dataclass(frozen=True)
class BlockCost:
    """Per-block estimates a placement policy needs. Producing these
    accurately (profiling real activation sizes and real recompute time
    per block, on real hardware) is nodes/model/block_profiler.py's job,
    not this module's -- a BlockCost here is plain data, however it was
    obtained (a real BlockProfiler run, a hand-built test fixture, or a
    future calibration file)."""
    activation_bytes: int
    recompute_ms: float


class CheckpointPlacementPolicy(ABC):

    @abstractmethod
    def select(self, blocks: list[BlockCost], budget: ResourceBudget) -> list[bool]:
        """One bool per block, same order as `blocks`: True = checkpoint
        it (recompute during backward, the VRAM-cheap/compute-expensive
        choice); False = keep its activations resident."""


class EveryBlockPlacement(CheckpointPlacementPolicy):
    """Today's actual, only behavior wherever use_checkpoint=True --
    checkpoint everything, unconditionally, regardless of budget. The
    safe default until real BlockCost numbers exist for this project's
    own UNet (see block_profiler.py) -- a wrong GreedyRatioPlacement
    decision built on guessed costs would be worse than this."""

    def select(self, blocks: list[BlockCost], budget: ResourceBudget) -> list[bool]:
        return [True] * len(blocks)


class GreedyRatioPlacement(CheckpointPlacementPolicy):
    """Ranks blocks by activation_bytes/recompute_ms (memory saved per
    unit recompute cost, highest first) and checkpoints the best-ratio
    blocks first until the *remaining* (kept-resident) blocks' activation
    memory fits within budget -- i.e. checkpoints the blocks that buy the
    most VRAM per unit of recompute time paid, leaving the
    already-cheap-to-keep-resident blocks alone. A direct, simplified
    reading of Korthikanti et al.'s cost-ratio ranking idea -- not their
    full method, which also reasons about which *operations within* a
    block to recompute, not just whole-block on/off.

    Unvalidated against a real training run -- see
    docs/training_pipeline_design.md section 9.2. Not wired into
    ComfyUNetLoRANode's real construction path (still EveryBlockPlacement's
    unconditional behavior, via use_checkpoint's existing bool) -- that's
    real, separate follow-up work, once real BlockCost numbers from an
    actual profiled run exist to validate a first real placement against.
    """

    def select(self, blocks: list[BlockCost], budget: ResourceBudget) -> list[bool]:
        order = sorted(range(len(blocks)),
                        key=lambda i: blocks[i].activation_bytes / max(blocks[i].recompute_ms, 1e-6),
                        reverse=True)
        checkpoint = [False] * len(blocks)
        remaining = sum(b.activation_bytes for b in blocks)
        usable_budget_bytes = max(0.0, budget.vram_budget_mb - budget.vram_reserve_mb) * (2 ** 20)
        for i in order:
            if remaining <= usable_budget_bytes:
                break
            checkpoint[i] = True
            remaining -= blocks[i].activation_bytes
        return checkpoint
