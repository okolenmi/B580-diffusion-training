"""BlockProfileCollector/ProfilingCheckpointing: real per-block
activation/recompute measurement, on real hardware, during a real
training step -- the actual blocker CheckpointPlacementPolicy
(checkpoint_placement.py) was waiting on, per
docs/training_pipeline_design.md section 2.3's own words: "the actual
blocker is the per-block profiling instrumentation, not the policy class
itself."

**Where the measurement happens, and why it's correct.** A checkpointed
block's activations are deliberately *not* kept after its forward pass
(that's the entire point of checkpointing) -- they're recomputed during
backward instead, by re-running the block's forward under
torch.enable_grad() (see gradient_checkpointing.py's
enable_frozen_param_safe_checkpointing()). That recompute *is* a real,
uncheckpointed forward pass through exactly this one block -- so timing
it and measuring the device memory delta around it gives the same
numbers a non-checkpointed run of this block would have shown, without
needing a second, separate, profiling-only forward pass. This module
hooks that existing recompute via enable_frozen_param_safe_checkpointing's
optional recompute_wrapper parameter (added for exactly this) rather
than re-implementing or wrapping the checkpoint mechanism a second time.

**Block identity, confirmed against ComfyUI's real source, not
guessed.** Cloned comfyanonymous/ComfyUI directly to check what
`func`/`ctx.run_function` actually is: in
comfy/ldm/modules/diffusionmodules/openaimodel.py, ResBlock.forward()
calls `checkpoint(self._forward, (x, emb), self.parameters(),
self.use_checkpoint)` -- `self._forward` is a *bound method*, so
`ctx.run_function.__self__` is the real ResBlock module instance.
Separately confirmed comfy/ldm/modules/attention.py's
BasicTransformerBlock.forward() does **not** call checkpoint() at all in
this pinned ComfyUI version (its `checkpoint=True` constructor
parameter is unused dead wiring) -- so in practice, only ResBlock
instances ever reach this profiler. A CheckpointPlacementPolicy built on
this profiler's output can only ever place ResBlocks, not attention
blocks, in this ComfyUI version -- a real, grounded constraint on what
item 1 can actually decide over, not a gap in this implementation.

**Naming, honestly bounded.** `type(instance).__name__` (e.g. "ResBlock")
plus a per-process first-seen ordinal gives a stable, real per-block
label ("ResBlock#4") without needing the block's dotted path in the
UNet (e.g. "input_blocks.4.0"). A dotted path would need the top-level
model's named_modules() -- TrainableModel (nodes/model/handle.py)
deliberately never exposes the raw nn.Module it wraps, so resolving
real dotted names would mean widening that ABC's surface specifically
for this. Real, separate follow-up if the ordinal+class-name label ever
proves not enough to act on -- not bundled into this change.

**Not wired into ComfyUNetLoRANode's real construction path.** Same
status as AdapterStrategy's seam (section 3.1): real, tested,
composable via the existing ActivationCheckpointingStrategy interface,
reachable by any caller that constructs a ProfilingCheckpointing
directly, but ComfyUNetLoRANode's use_checkpoint/resource_policy ports
don't yet have a way to select it. Exposing it as a real Node input is
its own scoped follow-up, once this is proven correct.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..components.device import DeviceContext
from .checkpoint_placement import BlockCost
from .gradient_checkpointing import (
    ActivationCheckpointingStrategy,
    enable_frozen_param_safe_checkpointing,
)


@dataclass
class _RunningStats:
    """Incremental max(activation_bytes)/mean(recompute_ms), not a
    stored list of every sample -- a real training run profiled for many
    steps calls record() dozens of times per step; keeping raw samples
    forever is unnecessary and unbounded. max() for activation_bytes
    because a placement decision needs to fit the worst case it'll
    actually see, not the typical one; mean() for recompute_ms because
    typical cost, not a rare spike, is what a placement tradeoff should
    weigh."""
    max_activation_bytes: int = 0
    sum_recompute_ms: float = 0.0
    count: int = 0

    def add(self, activation_bytes: int, recompute_ms: float) -> None:
        self.max_activation_bytes = max(self.max_activation_bytes, activation_bytes)
        self.sum_recompute_ms += recompute_ms
        self.count += 1

    def to_block_cost(self) -> BlockCost:
        mean_ms = self.sum_recompute_ms / self.count if self.count else 0.0
        return BlockCost(activation_bytes=self.max_activation_bytes, recompute_ms=mean_ms)


class BlockProfileCollector:
    """Owns both the accumulated per-block stats and the label-resolution
    state (id(module instance) -> "ClassName#ordinal", assigned on first
    sight) -- the two are naturally coupled: a label, once assigned, has
    to keep meaning the same block for every later record() call in this
    collector's lifetime."""

    def __init__(self):
        self._stats: dict[str, _RunningStats] = {}
        self._labels_by_instance_id: dict[int, str] = {}

    def label_for(self, run_function) -> str:
        instance = getattr(run_function, "__self__", None)
        if instance is None:
            # Real fallback, not expected to fire for ResBlock (see module
            # docstring) -- id(run_function) itself as a last resort so an
            # unexpected non-bound-method callable still gets a stable,
            # if opaque, per-process label instead of crashing.
            return f"unidentified_block#{id(run_function)}"
        key = id(instance)
        label = self._labels_by_instance_id.get(key)
        if label is None:
            label = f"{type(instance).__name__}#{len(self._labels_by_instance_id)}"
            self._labels_by_instance_id[key] = label
        return label

    def record(self, label: str, activation_bytes: int, recompute_ms: float) -> None:
        self._stats.setdefault(label, _RunningStats()).add(activation_bytes, recompute_ms)

    def block_costs(self) -> dict[str, BlockCost]:
        """One BlockCost per distinct block seen so far, keyed by its
        label -- CheckpointPlacementPolicy.select() takes a plain
        list[BlockCost] (order-only, no names), so a caller feeding this
        into a real policy does `list(collector.block_costs().values())`;
        the dict form stays here since the labels are real, standing
        diagnostic value on their own (which specific block is
        expensive), the same reason ResourceProfile.per_resident_bytes
        (nodes/memory/profile.py) is a dict and not a bare total."""
        return {label: stats.to_block_cost() for label, stats in self._stats.items()}

    def reset(self) -> None:
        """Clears accumulated stats, keeps label assignments -- the same
        physical blocks are still the same blocks next time record() is
        called (nothing about the model changed), only the running
        statistics should restart. Separate from constructing a new
        BlockProfileCollector() specifically so a caller profiling
        several separate windows (e.g. "steps 10-20" then "steps 50-60")
        gets comparable labels across both without a full model
        re-walk."""
        self._stats.clear()


class ProfilingCheckpointing(ActivationCheckpointingStrategy):
    """Same correctness fix as FrozenParamSafeCheckpointing, plus
    per-block instrumentation recorded into `collector`. Constructor
    takes both device_ctx (real synchronize()/memory_stats(), same
    object the rest of a real run already threads through
    TrainingStepPipeline -- not a second one) and collector (the actual
    accumulator, injected rather than owned, so a caller can read
    collector.block_costs() at any point during or after the run without
    going through this strategy object at all)."""

    def __init__(self, device_ctx: DeviceContext, collector: BlockProfileCollector):
        self._device_ctx = device_ctx
        self._collector = collector

    def apply(self) -> None:
        enable_frozen_param_safe_checkpointing(recompute_wrapper=self._recompute_and_record)

    def _recompute_and_record(self, run_function, args):
        """The recompute_wrapper enable_frozen_param_safe_checkpointing()
        calls in place of `run_function(*args)` during backward. mem_before
        is captured after this call's own synchronize(), not before it --
        matching TimedPhase's own "synchronize on both sides" discipline
        (nodes/train/step_pipeline.py) for a boundary genuinely known to be
        clean, not a guess about what finished by the time this runs."""
        self._device_ctx.synchronize()
        mem_before = self._device_ctx.memory_stats()
        t0 = time.perf_counter()
        output = run_function(*args)
        self._device_ctx.synchronize()
        t1 = time.perf_counter()
        mem_after = self._device_ctx.memory_stats()

        recompute_ms = (t1 - t0) * 1000
        activation_bytes = 0
        if mem_before is not None and mem_after is not None:
            # allocated_mb, not reserved_mb -- reserved includes the
            # allocator's held-but-idle pool (DeviceContext.memory_stats()'s
            # own docstring), which can show zero delta even when real
            # tensor bytes grew, if the allocator already had reserved
            # headroom to hand out. allocated_mb tracks real referenced
            # tensor bytes directly.
            delta_mb = mem_after["allocated_mb"] - mem_before["allocated_mb"]
            activation_bytes = max(0, int(delta_mb * (2 ** 20)))
        # mem_before/mem_after both None (CPU, no allocator concept) means
        # activation_bytes stays 0 for this observation -- "not measurable
        # on this backend," not "confirmed zero." Real numbers need an
        # XPU/CUDA backend -- see module docstring; running this profiler
        # on CPU exercises its bookkeeping/labeling, not real measurement.

        label = self._collector.label_for(run_function)
        self._collector.record(label, activation_bytes, recompute_ms)
        return output
