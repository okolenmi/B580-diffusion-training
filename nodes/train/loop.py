"""run_supervised_lora_training_loop: the real step-loop logic
SupervisedLoRATrainerNode.build() ran inline before this extraction --
pulled out so a second Node class (BudgetedLoRATrainerNode,
nodes/train/budgeted.py) can share exactly this implementation instead
of a second copy of it that can silently drift the next time either one
is touched.

Same shape, same reasoning, as nodes/model/lora_injector.py's
build_lora_injected_unet() extraction (docs/design/resources-controller/
08-consolidation.md's own "Real redundancy risk found" section): a
Node.build() resolves its own Port defaults (which can differ between
concrete Node classes -- see BudgetedLoRATrainerNode's different
resource_control/empty_cache_every_n_steps defaults) into plain keyword
arguments, then calls this one real implementation. This function knows
nothing about Port/INPUTS/validate_inputs at all -- every caller has
already done that before it's reached.

This is a pure extraction, not a rewrite: every line of actual logic
below is unchanged from SupervisedLoRATrainerNode.build()'s previous
inline version (see git history), except that startup/step print lines
that used to hardcode "SupervisedLoRATrainerNode" now take log_prefix
from the caller -- for SupervisedLoRATrainerNode itself that's
type(self).__name__, which *is* the literal string "SupervisedLoRATrainerNode",
so its own printed output is byte-for-byte identical to before.
"""

from __future__ import annotations

import gc
from typing import Callable, Optional

from ..components.device import DeviceContext, allocator_conf_env
from ..components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                     EpsParameterization, KarrasInputScaler)
from ..core import ExecutionContext
from ..dataset.handle import TrainingBatchSource
from ..memory.control_handle import ResourceControlHandle
from ..memory.coordinator import ResourceCoordinator
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..model.text_encoder_cache import CachingTextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import FusedOptimizerHandle, OptimizerHandle, describe_optimizer
from .loss import LossWeighting, UniformLossWeighting
from .schedule import LRSchedule
from .step_pipeline import (BackwardPhase, EncodeConditioningPhase, FetchBatchPhase,
                             ForwardPhase, LossPhase, MonitoringPhase,
                             OptimizerBeginStepPhase, OptimizerStepPhase,
                             PrepareDiffusionInputsPhase, StepState, TimedPhase,
                             TrainingStepPipeline)


def run_supervised_lora_training_loop(
    context: ExecutionContext,
    *,
    model: TrainableModel,
    batches: TrainingBatchSource,
    optimizer: OptimizerHandle,
    text_encoder: TextEncoder,
    lr_schedule: LRSchedule,
    steps: int,
    diffusion_process: Optional[DiffusionProcess] = None,
    gate_enabled: bool = False,
    gate_train_low: float = 0.0,
    gate_train_high: float = 999.0,
    gate_width: float = 100.0,
    profile: bool = False,
    empty_cache_every_n_steps: int = 0,
    profile_memory_per_phase: bool = False,
    resource_control: Optional[ResourceControlHandle] = None,
    loss_weighting: Optional[LossWeighting] = None,
    monitor: Optional[MonitorHandle] = None,
    on_step: Optional[Callable] = None,
    log_prefix: str = "TrainerNode",
) -> dict[str, TrainableModel]:
    """Runs `steps` optimizer updates over `model`. See this module's
    own docstring for why this is a standalone function rather than a
    method on any one Node class, and for the log_prefix caveat (the
    only parameter here with no Port counterpart)."""
    model.train()
    device = next(iter(model.trainable_parameters())).device
    is_fused = isinstance(optimizer, FusedOptimizerHandle)
    device_ctx = DeviceContext.for_device(device)

    # Three one-time, unconditional (not gated behind `profile`)
    # prints, cheap and directly answering: which concrete optimizer
    # is this run actually using, is an allocator-config env var
    # actually being read, and does this run's real LoRA shape
    # distribution have enough same-shape parameter groups for
    # shape-based batching to be worth using.
    optimizer_id = describe_optimizer(optimizer)
    print(f"[{log_prefix}] optimizer: {optimizer_id}")
    print(f"[{log_prefix}] allocator config env: {allocator_conf_env()}")
    _log_shape_histogram(model.trainable_parameters())
    diffusion_process = diffusion_process or DiffusionProcess(
        DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler())
    loss_weighting = loss_weighting or UniformLossWeighting()

    # Registered for profile=True's tracked_footprint_mb cross-check
    # (nodes/train/step_pipeline.py's MonitoringPhase) -- not driving
    # any offload decisions itself here. See
    # nodes/memory/coordinator.py's OffloadOrchestrator for that;
    # nothing here publishes a TrainingLifecycleEvent for it to react
    # to yet.
    coordinator = ResourceCoordinator()
    coordinator.register("model", model)
    coordinator.register("optimizer", optimizer)
    coordinator.register("text_encoder", text_encoder)

    if resource_control is not None:
        # model/optimizer stay offloadable=False -- both are needed unconditionally
        # every step's forward/backward/optimizer-step, and nothing here calls
        # ensure_loaded() on either before that compute runs, so marking them
        # offloadable would let before_step() offload one and never bring it back --
        # a real crash, not just a missed optimization. Making that safe needs
        # ensure_loaded("model")/ensure_loaded("optimizer") wired into the step
        # pipeline's own compute phase -- real, separate, scoped follow-up work
        # (docs/status/progress.md's own "still open" list), not done here.
        #
        # text_encoder is different: offloadable exactly when it's a
        # CachingTextEncoder, which calls ensure_loaded() on its own resource_name
        # before any cache-miss encode -- self-healing by construction, checked
        # directly rather than assumed (isinstance, not duck-typing, matching this
        # file's own existing FusedOptimizerHandle check above).
        resource_control.register("model", model, offloadable=False)
        resource_control.register("optimizer", optimizer, offloadable=False)
        resource_control.register(
            "text_encoder", text_encoder, offloadable=isinstance(text_encoder, CachingTextEncoder))

    phases = [
        FetchBatchPhase(batches),
        PrepareDiffusionInputsPhase(
            diffusion_process, gate_enabled=gate_enabled, gate_train_low=gate_train_low,
            gate_train_high=gate_train_high, gate_width=gate_width),
        EncodeConditioningPhase(text_encoder),
        OptimizerBeginStepPhase(optimizer, lr_schedule, is_fused),
        ForwardPhase(),
        LossPhase(loss_weighting),
        BackwardPhase(),
        OptimizerStepPhase(optimizer, is_fused),
    ]
    if profile:
        phases = [TimedPhase(p, device_ctx, _phase_label(p), capture_memory=profile_memory_per_phase)
                  for p in phases]
    phases.append(MonitoringPhase(
        total_steps=steps, device_ctx=device_ctx, on_step=on_step,
        monitor=monitor, profile=profile, coordinator=coordinator,
        optimizer_id=optimizer_id))
    pipeline = TrainingStepPipeline(phases)

    step = 0
    while step < steps:
        if context.should_cancel():
            # Cooperative stop, between steps only -- never mid
            # backward/optimizer-step. Not a failure: the model
            # trained so far is a normal, valid output, same as a
            # run that finished all its steps, just fewer of them.
            return {"model": model}
        state = StepState(step=step, batch=None, model=model, device=device)
        if resource_control is not None:
            resource_control.before_step(step)
        pipeline.run_step(state)
        step += 1
        if empty_cache_every_n_steps > 0 and step % empty_cache_every_n_steps == 0:
            gc.collect()
            device_ctx.empty_cache()

    return {"model": model}


def _log_shape_histogram(params) -> None:
    """One-time, unconditional -- cheap (a handful of ops over at most a
    few hundred small tensors), and directly answers whether this run's
    actual LoRA configuration has enough same-shape parameter groups for
    exact-shape-grouped batching to pay off, or is closer to the
    pathological all-unique-shapes case where it wouldn't help at all.
    Reports against the real, built graph's real parameters -- not a
    hand-approximated config -- so this number is trustworthy without
    needing a separate standalone script that could drift from what a
    real run actually does."""
    from collections import Counter
    shapes = Counter(tuple(p.shape) for p in params)
    total_params = len(params)
    grouped = sum(count for count in shapes.values() if count > 1)
    pct = (100 * grouped / total_params) if total_params else 0.0
    print(f"[shape_histogram] {total_params} trainable parameter tensor(s), "
          f"{len(shapes)} distinct shape(s), {grouped}/{total_params} "
          f"({pct:.0f}%) covered by a group of 2+ identical-shape parameters:")
    for shape, count in shapes.most_common():
        print(f"    {tuple(shape)}: {count}")


def _phase_label(phase) -> str:
    """CamelCase class name -> snake_case label, minus a trailing
    "Phase" -- FetchBatchPhase -> "fetch_batch". Mechanical, not
    hand-maintained per phase, so a new phase class gets a sensible
    label for free."""
    name = type(phase).__name__
    if name.endswith("Phase"):
        name = name[: -len("Phase")]
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
