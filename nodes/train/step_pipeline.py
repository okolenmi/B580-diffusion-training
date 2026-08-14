"""TrainingStepPipeline: a training step as an ordered list of StepPhases,
not one method.

Concrete phases below map onto what a training step actually does --
separated so a future change (CFG dual-pass, gradient accumulation,
DAgger chain-mixing) is "construct one more phase, insert it in the
list", not "edit the method that does everything". update_lr()/
zero_grad()/begin_step() and the diffusion-input prep (x_t/target/t/
sigma/xc) each get their own phase (OptimizerBeginStepPhase,
PrepareDiffusionInputsPhase) rather than folding into ForwardPhase/
EncodeConditioningPhase, because each is genuinely its own reason to
change (a new noise schedule doesn't touch optimizer state; a new
optimizer's begin-step semantics don't touch diffusion math).

**Profiling.** TimedPhase wraps any phase and records wall time, with a
real device_ctx.synchronize() before/after (async dispatch means an
untimed op can finish after the Python call that launched it returns).
Turning profiling on/off is "wrap every phase in TimedPhase or don't",
decided once where the pipeline is assembled -- not a `profile: bool`
parameter threaded through every phase's own logic. TimedPhase labels
each phase by its own name (fetch_batch/prepare_diffusion_inputs/
encode_conditioning/optimizer_begin_step/forward/loss/backward/
optimizer_step) -- the profile=True report/print shape reflects this
per-phase breakdown; training behavior (gradients, loss values,
parameter updates, LR schedule) is unaffected either way.

**Cancellation is deliberately NOT a StepPhase** -- it's the outer loop's
concern (whether to run another step at all), not a transformation of
StepState mid-pipeline: only between steps, never mid backward/
optimizer-step. See SupervisedLoRATrainerNode.build().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..components.device import DeviceContext
from ..components.diffusion import DiffusionProcess
from ..dataset.handle import TrainingBatchSource
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import OptimizerHandle
from .loss import LossWeighting
from .schedule import LRSchedule


@dataclass
class StepState:
    """Mutable, passed through every phase in sequence -- generalizes the
    old _run_step's method-local variable soup (x_t, target, ctx_emb,
    pred, loss, timing dict) into one object phases read from and write
    to. Fresh per step (SupervisedLoRATrainerNode.build() constructs a
    new one each iteration) -- extras doesn't carry stale data from a
    previous step by construction, not by convention."""
    step: int
    batch: Optional[dict]
    model: TrainableModel
    device: Any
    extras: dict[str, Any] = field(default_factory=dict)


class StepPhase(ABC):
    @abstractmethod
    def run(self, state: StepState) -> StepState:
        ...


class TrainingStepPipeline:
    """An ordered list of StepPhases. Owns nothing itself beyond that
    list -- adding a phase later is insertion into this list, not
    editing a method that does everything."""

    def __init__(self, phases: list[StepPhase]):
        self.phases = phases

    def run_step(self, state: StepState) -> StepState:
        for phase in self.phases:
            state = phase.run(state)
        return state


class TimedPhase(StepPhase):
    """Wraps any StepPhase, records wall time with a real synchronize()
    on both sides (see this module's docstring for why).

    **capture_memory.** A single per-step VRAM snapshot at the end can
    show *that* reserved memory jumped between two steps, but nothing
    about *which phase within the step* did it. When True, records
    device_ctx.memory_stats() right after each phase's own synchronize()
    -- so each phase's number reflects memory state at a point genuinely
    known to be after that phase finished, not a guess. Off by default,
    and only meaningful when `profile` is already True (real per-step
    overhead on top of profile's own -- an 8x memory_stats() call
    instead of 1x -- so this is for a short, targeted diagnostic run,
    same caveat as profile itself, not for real training)."""

    def __init__(self, inner: StepPhase, device_ctx: DeviceContext, label: str,
                 capture_memory: bool = False):
        self.inner = inner
        self.device_ctx = device_ctx
        self.label = label
        self.capture_memory = capture_memory

    def run(self, state: StepState) -> StepState:
        self.device_ctx.synchronize()
        t0 = time.perf_counter()
        state = self.inner.run(state)
        self.device_ctx.synchronize()
        state.extras.setdefault("timing_ms", {})[self.label] = (time.perf_counter() - t0) * 1000
        if self.capture_memory:
            mem = self.device_ctx.memory_stats()
            if mem is not None:
                state.extras.setdefault("phase_mem", {})[self.label] = mem
        return state


class FetchBatchPhase(StepPhase):
    """Fetches the next batch, wrapping back to the start of the dataset
    when exhausted -- a dataset with fewer batches than `steps` needs is
    trained over multiple passes, not an error. Replaces the old
    while/for nesting in build() (`while step < steps: for batch in
    batches: ...`) with a flat loop plus this phase owning the
    epoch-wrap itself.

    No internal wait-time tracking (unlike an earlier version of this
    phase, before this comment) -- TimedPhase wrapping this phase already
    measures exactly next()'s own blocking duration, which is what
    "how long did we wait for a batch" actually means, and nothing runs
    between the previous step's last phase finishing and this phase's
    next() call except cheap, near-instant bookkeeping in build()'s outer
    loop (same as the old code's equivalent gap) -- so a separate
    "time since the previous step ended" measurement would give the same
    answer through more machinery, not a different one."""

    def __init__(self, batches: TrainingBatchSource):
        self._batches = batches
        self._iterator = iter(batches)

    def run(self, state: StepState) -> StepState:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._batches)
            batch = next(self._iterator)
        state.batch = batch
        return state


class PrepareDiffusionInputsPhase(StepPhase):
    """x_t/target/t onto the training device, then the noise schedule and
    input scaling from DiffusionProcess -- everything the forward pass
    needs about the diffusion process specifically, separated from both
    "fetch a batch" and "run the model" so a different NoiseSchedule/
    Parameterization/ModelInputTransform touches only this phase.

    **LoRA timestep gate.** core/lora.py's set_lora_gate()/
    compute_lora_gate() keeps a LoRA's contribution close to the frozen
    base at timesteps outside the dataset's own actually-trained range.
    Wired here at the same point the legacy pipeline calls it (right
    after the per-batch `t` tensor is available, before the forward pass
    that would apply the gated delta), using the same function, not a
    reimplementation.

    Off by default (gate_enabled=False) -- LoRA applies uniformly across
    all timesteps unless explicitly turned on. When enabled,
    gate_train_low/gate_train_high must be set to match whatever
    t_low/t_high the dataset source node was actually configured with
    (ManagedDatasetSourceNode's own t_low/t_high inputs) -- there is
    deliberately no automatic sync between the two. Mismatched values
    here don't error -- they just gate against the wrong range silently,
    so this needs to be set deliberately, not guessed.

    Known limitation, inherited from core/lora.py's own design:
    `_current_gate` is a module-level global, not scoped to any one
    model/build -- if a single process ever runs multiple concurrent
    trainer builds, one build's gate could leak into another's forward
    calls. set_lora_gate(None) is called unconditionally every step when
    disabled, specifically so a previous step or build's gate can never
    silently persist."""

    def __init__(self, diffusion_process: DiffusionProcess,
                 gate_enabled: bool = False, gate_train_low: float = 0.0,
                 gate_train_high: float = 999.0, gate_width: float = 100.0):
        self._process = diffusion_process
        self._gate_enabled = gate_enabled
        self._gate_train_low = gate_train_low
        self._gate_train_high = gate_train_high
        self._gate_width = gate_width

    def run(self, state: StepState) -> StepState:
        import torch
        from core.lora import compute_lora_gate, set_lora_gate

        batch = state.batch
        x_t = batch["x_t"].to(state.device)
        target = batch["target"].to(state.device)
        t = batch["t"].to(device=state.device, dtype=torch.long).view(-1)
        _, sigma = self._process.schedule.alpha_sigma(t)
        xc = self._process.input_transform.scale_input(x_t, sigma)
        state.extras["x_t"] = x_t
        state.extras["target"] = target
        state.extras["t"] = t
        state.extras["sigma"] = sigma
        state.extras["xc"] = xc

        if self._gate_enabled:
            set_lora_gate(compute_lora_gate(
                t, self._gate_train_low, self._gate_train_high, self._gate_width))
        else:
            set_lora_gate(None)

        return state


class EncodeConditioningPhase(StepPhase):
    """Wraps a TextEncoder. A CFG-dual-pass variant is a second class
    implementing this same contract (two encode() calls, cond and
    uncond), not a branch inside this one."""

    def __init__(self, text_encoder: TextEncoder):
        self._text_encoder = text_encoder

    def run(self, state: StepState) -> StepState:
        import torch

        x_t = state.extras["x_t"]
        batch = state.batch
        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        ctx_emb, y = self._text_encoder.encode(batch["prompt"], batch_size=x_t.shape[0],
                                                height=batch_h, width=batch_w)
        state.extras["ctx_emb"] = ctx_emb.to(device=state.device, dtype=torch.bfloat16)
        state.extras["y"] = y.to(device=state.device, dtype=torch.bfloat16)
        return state


class OptimizerBeginStepPhase(StepPhase):
    """LR update and gradient-buffer reset (zero_grad, or begin_step for
    a fused optimizer) -- runs before the forward pass, matching the old
    _run_step's exact order (zero_grad must precede the backward() that
    accumulates into it; a fused optimizer's begin_step() sets up the
    backward-hook state forward() implicitly depends on existing).
    A genuinely separate reason to change from ForwardPhase (a new
    optimizer's begin-step semantics, not a new model architecture)."""

    def __init__(self, optimizer: OptimizerHandle, lr_schedule: LRSchedule, is_fused: bool):
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._is_fused = is_fused

    def run(self, state: StepState) -> StepState:
        lr = self._lr_schedule.value(state.step)
        self._optimizer.update_lr(lr)
        if self._is_fused:
            self._optimizer.begin_step(sub_steps=1)
        else:
            self._optimizer.zero_grad()
        state.extras["lr"] = lr
        return state


class ForwardPhase(StepPhase):

    def run(self, state: StepState) -> StepState:
        state.extras["pred"] = state.model.forward(
            state.extras["xc"], state.extras["t"], state.extras["ctx_emb"], state.extras["y"])
        return state


class LossPhase(StepPhase):
    """Wraps a LossWeighting (nodes/train/loss.py, section 4)."""

    def __init__(self, loss_weighting: LossWeighting):
        self._loss_weighting = loss_weighting

    def run(self, state: StepState) -> StepState:
        pred = state.extras["pred"]
        target = state.extras["target"]
        sigma = state.extras["sigma"]
        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        # float(sigma...item()) here, not deferred -- matches the old
        # _run_step's exact sync point, not moved earlier or later.
        weight = self._loss_weighting.weight(float(sigma.float().mean().item()))
        state.extras["loss"] = per_sample.mean() * weight
        return state


class BackwardPhase(StepPhase):

    def run(self, state: StepState) -> StepState:
        state.extras["loss"].backward()
        return state


class OptimizerStepPhase(StepPhase):
    """Already has to branch on fused vs. non-fused (a fused optimizer's
    updates already happened via backward-pass hooks -- nothing left to
    do here) -- that branch stays, it's genuine, but it's now the entire
    content of one small class instead of interleaved with everything
    else _run_step used to do."""

    def __init__(self, optimizer: OptimizerHandle, is_fused: bool):
        self._optimizer = optimizer
        self._is_fused = is_fused

    def run(self, state: StepState) -> StepState:
        if not self._is_fused:
            self._optimizer.step(n_steps=1)
        return state


class MonitoringPhase(StepPhase):
    """Builds and sends/prints the same report the old _run_step's tail
    built inline. Always last, never TimedPhase-wrapped -- report-
    building/printing overhead was never counted in the old step_total_ms
    either, and wrapping it would be circular (it needs to read
    timing_ms to build the report it would also be timed into).

    Two additions beyond the base report, both gated behind the same
    `profile` flag as the existing vram_allocated_mb/vram_reserved_mb
    reporting, so this doesn't change anything about a real
    (non-profiled) run:

    1. `optimizer_id` (from optimizer.handle.describe_optimizer(),
       computed once by the caller and passed in as a plain string --
       this class doesn't need the actual optimizer object) is included
       in every report/print line, so which concrete optimizer
       implementation produced a given number is part of the number's
       own record.
    2. Baseline deltas for reserved_mb and num_alloc_retries, captured
       on this instance's first profiled step and diffed against every
       report after. A single snapshot line can't distinguish "stable
       but high" from "climbing" -- this makes that distinction
       possible from a handful of report lines pulled from anywhere in
       one run, not just by comparing separately-run reports by hand."""

    def __init__(self, total_steps: int, device_ctx: DeviceContext,
                 on_step: Optional[Callable] = None,
                 monitor: Optional[MonitorHandle] = None, profile: bool = False,
                 coordinator=None, optimizer_id: str = ""):
        self._total_steps = total_steps
        self._device_ctx = device_ctx
        self._on_step = on_step
        self._monitor = monitor
        self._profile = profile
        self._coordinator = coordinator
        self._optimizer_id = optimizer_id
        self._baseline_mem: Optional[dict[str, float]] = None

    def run(self, state: StepState) -> StepState:
        loss_value = float(state.extras["loss"].item())
        lr = state.extras["lr"]

        if self._on_step is not None:
            self._on_step(state.step, loss_value)

        timing = state.extras.get("timing_ms")
        mem = None
        tracked_mb = None
        if self._monitor is not None or self._profile:
            report = {
                "step": state.step, "total_steps": self._total_steps,
                "loss": loss_value, "lr": lr, "t": time.time(),
            }
            if self._optimizer_id:
                report["optimizer"] = self._optimizer_id
            if timing is not None:
                # timing_ms's own keys are bare phase labels (fetch_batch,
                # not fetch_batch_ms) -- the report dict is the one place
                # that needs the "_ms" suffix, matching the old
                # data_wait_ms/encode_ms/... naming convention.
                report.update({f"{label}_ms": ms for label, ms in timing.items()})
                report["step_total_ms"] = sum(timing.values())
                mem = self._device_ctx.memory_stats()
                if mem is not None:
                    report.update({f"vram_{k}": v for k, v in mem.items()})
                    if self._baseline_mem is None:
                        self._baseline_mem = mem
                    report["vram_reserved_delta_mb"] = (
                        mem["reserved_mb"] - self._baseline_mem["reserved_mb"])
                    report["vram_alloc_retries_delta"] = (
                        mem["num_alloc_retries"] - self._baseline_mem["num_alloc_retries"])
                if self._coordinator is not None:
                    # A cross-check, not a replacement for mem above: the
                    # sum of every registered DeviceResident's own
                    # footprint_bytes(), independent of what the device
                    # driver itself reports. The two agreeing (roughly --
                    # this doesn't account for activations, only the
                    # residents this run explicitly registered) is a
                    # useful sanity signal that DeviceResident accounting
                    # actually reflects reality; a growing gap between
                    # them is worth investigating on its own.
                    tracked_mb = self._coordinator.total_footprint_bytes() / (1024 ** 2)
                    report["tracked_footprint_mb"] = tracked_mb
            if self._monitor is not None:
                self._monitor.report(report)

        if self._profile:
            timing = timing or {}
            parts = " ".join(f"{label}={ms:.0f}ms" for label, ms in timing.items())
            total = sum(timing.values())
            if mem is None:
                mem = self._device_ctx.memory_stats()
                if mem is not None and self._baseline_mem is None:
                    self._baseline_mem = mem
            vram_part = ""
            if mem is not None:
                base = self._baseline_mem or mem
                reserved_delta = mem["reserved_mb"] - base["reserved_mb"]
                retries_delta = mem["num_alloc_retries"] - base["num_alloc_retries"]
                # allocated/reserved kept first, matching the original line's
                # shape exactly (nothing that was already grepping this line
                # for those two fields breaks); richer memory_stats() fields
                # appended after, not replacing it.
                vram_part = (
                    f" vram_allocated={mem['allocated_mb']:.0f}MB"
                    f" vram_reserved={mem['reserved_mb']:.0f}MB"
                    f" (peak={mem['peak_reserved_mb']:.0f}MB"
                    f" delta_from_baseline={reserved_delta:+.0f}MB)"
                    f" vram_active={mem['active_mb']:.0f}MB"
                    f" vram_requested={mem['requested_mb']:.0f}MB"
                    f" alloc_retries={mem['num_alloc_retries']:.0f}"
                    f" (delta={retries_delta:+.0f})"
                    f" segments={mem['num_segments']:.0f}"
                )
            tracked_part = ""
            if self._coordinator is not None:
                if tracked_mb is None:
                    tracked_mb = self._coordinator.total_footprint_bytes() / (1024 ** 2)
                tracked_part = f" tracked_footprint={tracked_mb:.0f}MB"
            optimizer_part = f" optimizer={self._optimizer_id}" if self._optimizer_id else ""
            print(f"  [step {state.step}]{optimizer_part} {parts} total={total:.0f}ms"
                  + vram_part + tracked_part)

            phase_mem = state.extras.get("phase_mem")
            if phase_mem:
                # Per-phase reserved_mb, each relative to the *previous
                # phase's own* snapshot -- directly shows which phase's
                # synchronize() boundary is where reserved actually
                # moves, instead of inferring it from end-of-step
                # snapshots two steps apart. First phase's delta is
                # against this step's own start (prev_reserved seeded
                # from the very first entry) so a jump in fetch_batch
                # itself isn't silently attributed to nothing.
                prev_reserved = None
                phase_parts = []
                for label, m in phase_mem.items():
                    r = m["reserved_mb"]
                    delta = "" if prev_reserved is None else f"{r - prev_reserved:+.0f}"
                    phase_parts.append(f"{label}={r:.0f}MB({delta})" if delta else f"{label}={r:.0f}MB")
                    prev_reserved = r
                print(f"    [step {state.step}] reserved by phase: " + " ".join(phase_parts))

        return state
