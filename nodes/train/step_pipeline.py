"""TrainingStepPipeline: a training step as an ordered list of StepPhases,
not one method. docs/training_pipeline_design.md section 2.1.

Concrete phases below map onto what SupervisedLoRATrainerNode._run_step
used to do as one method, before this refactor -- separated so a future
change (CFG dual-pass, gradient accumulation, DAgger chain-mixing) is
"construct one more phase, insert it in the list", not "edit the method
that does everything". More granular than the design doc's own
illustrative 7-phase list (FetchBatch/EncodeConditioning/Forward/Loss/
Backward/OptimizerStep/Monitoring): update_lr()/zero_grad()/begin_step()
and the diffusion-input prep (x_t/target/t/sigma/xc) each got their own
phase here (OptimizerBeginStepPhase, PrepareDiffusionInputsPhase) rather
than folding into ForwardPhase/EncodeConditioningPhase, because each is
genuinely its own reason to change (a new noise schedule doesn't touch
optimizer state; a new optimizer's begin-step semantics don't touch
diffusion math) -- the same "one reason to change" test this whole
section is built around.

**Profiling** (docs/training_pipeline_design.md section 2.1's own
example): TimedPhase wraps any phase and records wall time, with a real
device_ctx.synchronize() before/after (async dispatch means an untimed op
can finish after the Python call that launched it returns -- the same
correctness requirement the old profile=True implementation already
documented). Turning profiling on/off is "wrap every phase in TimedPhase
or don't", decided once where the pipeline is assembled -- not a
`profile: bool` parameter threaded through nine phases' own logic.

**The profiling *output* genuinely changes shape, on purpose, checked
directly rather than assumed to be safe:** the old timing dict's keys
(data_wait_ms/encode_ms/forward_ms/backward_ms/optim_ms/step_total_ms)
don't survive as-is -- TimedPhase labels each phase by its own name
(fetch_batch/prepare_diffusion_inputs/encode_conditioning/
optimizer_begin_step/forward/loss/backward/optimizer_step), which is more
accurate than the old grouping (the old "encode_ms" actually included
x_t/target/t loading and diffusion-schedule math ahead of the real
text-encoder call, not just encoding -- confirmed by re-reading the old
_run_step directly). Grepped server/*.py, manager/*.py, and every
server/static/*.js file for the old key names before making this change:
zero matches -- nothing in this codebase's server, manager, or frontend
code depends on the old names, so this is a safe, intentional
improvement to the diagnostic output, not a silent regression. The
*training* behavior (gradients, loss values, parameter updates, LR
schedule) is unchanged either way -- only the profile=True report/print
shape differs.

**Cancellation is deliberately NOT a StepPhase** -- it's the outer loop's
concern (whether to run another step at all), not a transformation of
StepState mid-pipeline, same "between steps only, never mid backward/
optimizer-step" guarantee as before. See SupervisedLoRATrainerNode.build().
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
    on both sides (see this module's docstring for why)."""

    def __init__(self, inner: StepPhase, device_ctx: DeviceContext, label: str):
        self.inner = inner
        self.device_ctx = device_ctx
        self.label = label

    def run(self, state: StepState) -> StepState:
        self.device_ctx.synchronize()
        t0 = time.perf_counter()
        state = self.inner.run(state)
        self.device_ctx.synchronize()
        state.extras.setdefault("timing_ms", {})[self.label] = (time.perf_counter() - t0) * 1000
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
    Parameterization/ModelInputTransform touches only this phase."""

    def __init__(self, diffusion_process: DiffusionProcess):
        self._process = diffusion_process

    def run(self, state: StepState) -> StepState:
        import torch

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
    timing_ms to build the report it would also be timed into)."""

    def __init__(self, total_steps: int, device_ctx: DeviceContext,
                 on_step: Optional[Callable] = None,
                 monitor: Optional[MonitorHandle] = None, profile: bool = False):
        self._total_steps = total_steps
        self._device_ctx = device_ctx
        self._on_step = on_step
        self._monitor = monitor
        self._profile = profile

    def run(self, state: StepState) -> StepState:
        loss_value = float(state.extras["loss"].item())
        lr = state.extras["lr"]

        if self._on_step is not None:
            self._on_step(state.step, loss_value)

        timing = state.extras.get("timing_ms")
        mem = None
        if self._monitor is not None or self._profile:
            report = {
                "step": state.step, "total_steps": self._total_steps,
                "loss": loss_value, "lr": lr, "t": time.time(),
            }
            if timing is not None:
                # timing_ms's own keys are bare phase labels (fetch_batch,
                # not fetch_batch_ms) -- the report dict is the one place
                # that needs the "_ms" suffix, matching the old
                # data_wait_ms/encode_ms/... naming convention.
                report.update({f"{label}_ms": ms for label, ms in timing.items()})
                report["step_total_ms"] = sum(timing.values())
                mem = self._device_ctx.memory_stats()
                if mem is not None:
                    report["vram_allocated_mb"] = mem["allocated_mb"]
                    report["vram_reserved_mb"] = mem["reserved_mb"]
            if self._monitor is not None:
                self._monitor.report(report)

        if self._profile:
            timing = timing or {}
            parts = " ".join(f"{label}={ms:.0f}ms" for label, ms in timing.items())
            total = sum(timing.values())
            if mem is None:
                mem = self._device_ctx.memory_stats()
            vram_part = ""
            if mem is not None:
                vram_part = (f" vram_allocated={mem['allocated_mb']:.0f}MB "
                             f"vram_reserved={mem['reserved_mb']:.0f}MB")
            print(f"  [step {state.step}] {parts} total={total:.0f}ms" + vram_part)

        return state
