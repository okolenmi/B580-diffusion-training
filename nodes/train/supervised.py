"""SupervisedLoRATrainerNode: training loop for the dataset -> LoRA case.

Scope, explicit: single conditioning pass per batch (no CFG cond/uncond
dual pass), no gradient accumulation, no cyclic/teacher-rollout caching,
no DAgger, no adversarial pre-conditioning, no resume/checkpoint cadence
(use `on_step` for that). Assumes the dataset's stored `target` is
already in the student's own prediction parameterization -- no
teacher/student eps<->vpred conversion at train time.

The step itself is a TrainingStepPipeline (nodes/train/step_pipeline.py)
-- build() constructs the phase list once per run; a future change (CFG
dual-pass, gradient accumulation) is "construct one more phase, insert it
in the list", not "edit the method that does everything" (see
step_pipeline.py's own docstring).
"""

from __future__ import annotations

from typing import ClassVar

from ..components.device import DeviceContext, allocator_conf_env
from ..components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                     EpsParameterization, KarrasInputScaler)
from ..core import Port
from ..dataset.handle import TrainingBatchSource
from ..memory.coordinator import ResourceCoordinator
from ..model.handle import TrainableModel
from ..optimizer.handle import FusedOptimizerHandle, describe_optimizer
from .loss import UniformLossWeighting
from .node import TrainerNode
from .step_pipeline import (BackwardPhase, EncodeConditioningPhase, FetchBatchPhase,
                             ForwardPhase, LossPhase, MonitoringPhase,
                             OptimizerBeginStepPhase, OptimizerStepPhase,
                             PrepareDiffusionInputsPhase, StepState, TimedPhase,
                             TrainingStepPipeline)


class SupervisedLoRATrainerNode(TrainerNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **TrainerNode.COMMON_INPUTS,
        "diffusion_process": Port(
            name="diffusion_process", type=DiffusionProcess, required=False, default=None,
            doc="None = today's actual behavior: DiscreteLinearNoiseSchedule's default "
                "linear beta schedule, epsilon prediction, ComfyUI's calculate_input "
                "scaling. Wire a different DiffusionProcess (e.g. built around "
                "RescaledZeroTerminalSNRSchedule + VPredParameterization) to change any "
                "of the three -- see nodes/components/diffusion.py.",
        ),
        "gate_enabled": Port(
            name="gate_enabled", type=bool, required=False, default=False,
            doc="Off by default -- LoRA applies uniformly across all "
                "timesteps. When True, keeps the LoRA's contribution close to the frozen "
                "base at timesteps outside [gate_train_low, gate_train_high] instead of "
                "applying the learned delta at full strength everywhere, including "
                "timesteps the dataset never actually supervised. Set gate_train_low/"
                "gate_train_high to match whatever t_low/t_high the dataset source node "
                "was configured with -- not automatically synced from it yet (see "
                "PrepareDiffusionInputsPhase's own docstring, nodes/train/step_pipeline.py, "
                "for exactly why and what a real auto-sync would need), so a mismatch here "
                "gates against the wrong range silently rather than erroring.",
        ),
        "gate_train_low": Port(
            name="gate_train_low", type=float, required=False, default=0.0,
            doc="Only used when gate_enabled=True. Must match the dataset source node's "
                "own t_low (ManagedDatasetSourceNode) -- see gate_enabled's doc.",
        ),
        "gate_train_high": Port(
            name="gate_train_high", type=float, required=False, default=999.0,
            doc="Only used when gate_enabled=True. Must match the dataset source node's "
                "own t_high (ManagedDatasetSourceNode) -- see gate_enabled's doc.",
        ),
        "gate_width": Port(
            name="gate_width", type=float, required=False, default=100.0,
            doc="Only used when gate_enabled=True. Smaller = sharper cutoff right at "
                "[gate_train_low, gate_train_high]'s edges, larger = more gradual handoff. "
                "Same parameter, same default, as the legacy pipeline's gate_width "
                "(core/config_model.py) -- see core/lora.py's compute_lora_gate for the "
                "exact formula and a worked numeric example.",
        ),
        "profile": Port(
            name="profile", type=bool, required=False, default=False,
            doc="Per-phase step timing (fetch batch / prepare diffusion inputs / encode "
                "conditioning / optimizer begin step / forward / loss / backward / "
                "optimizer step) printed every step, and included in monitor.report() if a "
                "monitor is wired -- the breakdown this project didn't have a way to see. "
                "Off by default: correct phase timing needs a device synchronize() between "
                "phases (DeviceContext.synchronize()), which blocks the async pipeline and "
                "makes steps measurably slower than a normal run while this is on. Use it "
                "for a short diagnostic run, not for real training. Also reports "
                "vram_allocated_mb/vram_reserved_mb each step when profiling -- allocated "
                "growing over many steps is a real, live reference leak; reserved growing "
                "while allocated stays flat is just the caching allocator's own bookkeeping, "
                "not necessarily a leak (see nodes/components/device.py's "
                "DeviceContext.memory_stats -- also reports vram_peak_reserved_mb/"
                "vram_peak_allocated_mb, "
                "vram_active_mb/vram_requested_mb (the fragmentation/rounding overhead as "
                "direct numbers, not an inference from the allocated/reserved gap alone), "
                "vram_num_alloc_retries (the strongest single fragmentation signal -- a real "
                "cache flush+retry, not a guess), vram_num_segments, and both "
                "vram_reserved_delta_mb and vram_alloc_retries_delta against this run's own "
                "first profiled step, so a run doesn't need external before/after comparison "
                "to tell 'stable but high' from 'climbing'). Also prints the concrete "
                "optimizer identity (optimizer.handle.describe_optimizer) and a one-time "
                "shape histogram of every trainable parameter, unconditionally, regardless of "
                "this profile flag -- see SupervisedLoRATrainerNode.build()'s own startup "
                "prints. Also reports tracked_footprint_mb -- the sum of every registered "
                "DeviceResident's own footprint_bytes() (model/optimizer/text_encoder), "
                "independent of what the device driver reports. The two staying roughly in "
                "sync is a sanity check that DeviceResident accounting reflects reality; it "
                "won't match vram_allocated_mb exactly (this doesn't account for "
                "activations), so don't expect equality, just no growing gap over time. "
                "Per-phase keys are named after each phase (see "
                "nodes/train/step_pipeline.py) -- more granular than, and not the same "
                "key names as, an older version of this Port's output.",
        ),
        "empty_cache_every_n_steps": Port(
            name="empty_cache_every_n_steps", type=int, required=False, default=0,
            doc="0 disables this. When >0, calls gc.collect() + DeviceContext.empty_cache() "
                "every N steps. This only returns *unused* cached memory to the driver -- it "
                "cannot free memory still genuinely referenced by something, so it won't help "
                "a real reference leak (check profile=True's vram_allocated_mb for that), only "
                "caching-allocator fragmentation/bookkeeping. Costs a device sync each time "
                "it runs (same as any other explicit synchronize), so a very small N will "
                "cost real step time -- start high (e.g. 50) and go lower only if needed.",
        ),
        "profile_memory_per_phase": Port(
            name="profile_memory_per_phase", type=bool, required=False, default=False,
            doc="Only meaningful when profile=True. Every other VRAM number this node reports "
                "is a single end-of-step snapshot -- enough to see reserved differs between "
                "two steps, nothing about which phase within a step actually did it. This "
                "records vram_reserved_mb right after *each* phase's own synchronize() (fetch "
                "batch / prepare diffusion inputs / encode conditioning / optimizer begin step "
                "/ forward / loss / backward / optimizer step), printed as a second line per "
                "step: 'reserved by phase: label=XMB(+delta) ...', each delta relative to the "
                "previous phase's own snapshot in that same step -- so a jump shows up "
                "attributed to the specific phase that caused it, not inferred after the fact. "
                "Real added cost on top of profile=True's own overhead (an extra "
                "memory_stats() call per phase, 8x instead of 1x) -- for a short, targeted "
                "run chasing exactly this question, not for real training.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)

        model: TrainableModel = inputs["model"]
        batches: TrainingBatchSource = inputs["batches"]
        steps: int = inputs["steps"]
        empty_cache_every_n_steps: int = inputs.get(
            "empty_cache_every_n_steps", self.INPUTS["empty_cache_every_n_steps"].default)
        profile: bool = inputs.get("profile", self.INPUTS["profile"].default)
        profile_memory_per_phase: bool = inputs.get(
            "profile_memory_per_phase", self.INPUTS["profile_memory_per_phase"].default)

        model.train()
        device = next(iter(model.trainable_parameters())).device
        optimizer = inputs["optimizer"]
        is_fused = isinstance(optimizer, FusedOptimizerHandle)
        device_ctx = DeviceContext.for_device(device)

        # Three one-time, unconditional (not gated behind `profile`)
        # prints, cheap and directly answering: which concrete optimizer
        # is this run actually using, is an allocator-config env var
        # actually being read, and does this run's real LoRA shape
        # distribution have enough same-shape parameter groups for
        # shape-based batching to be worth using.
        optimizer_id = describe_optimizer(optimizer)
        print(f"[SupervisedLoRATrainerNode] optimizer: {optimizer_id}")
        print(f"[SupervisedLoRATrainerNode] allocator config env: {allocator_conf_env()}")
        _log_shape_histogram(model.trainable_parameters())
        diffusion_process = inputs.get("diffusion_process") or DiffusionProcess(
            DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler())
        loss_weighting = inputs.get("loss_weighting") or UniformLossWeighting()

        # Registered for profile=True's tracked_footprint_mb cross-check
        # (nodes/train/step_pipeline.py's MonitoringPhase) -- not driving
        # any offload decisions itself here. See
        # nodes/memory/coordinator.py's OffloadOrchestrator for that;
        # nothing here publishes a TrainingLifecycleEvent for it to react
        # to yet.
        coordinator = ResourceCoordinator()
        coordinator.register("model", model)
        coordinator.register("optimizer", optimizer)
        coordinator.register("text_encoder", inputs["text_encoder"])

        phases = [
            FetchBatchPhase(batches),
            PrepareDiffusionInputsPhase(
                diffusion_process,
                gate_enabled=inputs.get("gate_enabled", self.INPUTS["gate_enabled"].default),
                gate_train_low=inputs.get("gate_train_low", self.INPUTS["gate_train_low"].default),
                gate_train_high=inputs.get("gate_train_high", self.INPUTS["gate_train_high"].default),
                gate_width=inputs.get("gate_width", self.INPUTS["gate_width"].default)),
            EncodeConditioningPhase(inputs["text_encoder"]),
            OptimizerBeginStepPhase(optimizer, inputs["lr_schedule"], is_fused),
            ForwardPhase(),
            LossPhase(loss_weighting),
            BackwardPhase(),
            OptimizerStepPhase(optimizer, is_fused),
        ]
        if profile:
            phases = [TimedPhase(p, device_ctx, _phase_label(p), capture_memory=profile_memory_per_phase)
                      for p in phases]
        phases.append(MonitoringPhase(
            total_steps=steps, device_ctx=device_ctx, on_step=inputs.get("on_step"),
            monitor=inputs.get("monitor"), profile=profile, coordinator=coordinator,
            optimizer_id=optimizer_id))
        pipeline = TrainingStepPipeline(phases)

        step = 0
        while step < steps:
            if self.context.should_cancel():
                # Cooperative stop, between steps only -- never mid
                # backward/optimizer-step. Not a failure: the model
                # trained so far is a normal, valid output, same as a
                # run that finished all its steps, just fewer of them.
                result = {"model": model}
                self.validate_outputs(result)
                return result
            state = StepState(step=step, batch=None, model=model, device=device)
            pipeline.run_step(state)
            step += 1
            if empty_cache_every_n_steps > 0 and step % empty_cache_every_n_steps == 0:
                import gc
                gc.collect()
                device_ctx.empty_cache()

        result = {"model": model}
        self.validate_outputs(result)
        return result


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
