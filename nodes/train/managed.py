"""ManagedLoRATrainerNode: the Resources Controller route's own training
step loop -- independent of nodes/train/step_pipeline.py and
nodes/train/supervised.py, not a variant of either. Written fresh,
using the main route's own step_pipeline.py only as a reference for the
underlying math (diffusion input prep, forward, loss, backward), which
is unrelated to what actually differs here and would be pointless to
re-derive by hand from nothing.

**What's actually different, and why it needs its own loop rather than
a flag on the existing one.** The main route keeps every resident
(model, optimizer, text_encoder) loaded for a step's entire duration by
default, offloading only reactively -- a wired VRAM budget (before_step()
below) offloads something registered offloadable *if and only if*
measured usage already exceeds it at that moment. That's a reasonable
default for a route that doesn't otherwise think about memory, but it
means a resource that's genuinely idle for most of a step (the text
encoder outside its own encode call, the optimizer's state outside its
own update) still just sits resident the common case, no pressure
required to justify moving it.

This route's own step loop can release each resident immediately after
the one phase that needs it, every step -- see EncodeConditioningPhase
and BackwardAndOptimizerStepPhase below for exactly which resident,
which window, and why -- but whether it actually *does* that is decided
once, adaptively, by AdaptiveResidencyController below, not
unconditionally. An earlier version of this file released
unconditionally, always, regardless of whether the stated budget needed
it -- real, measured cost (see AdaptiveResidencyController's own
docstring for the numbers and the actual report that motivated this),
paid on every run whether or not anything was ever close to the
ceiling. `resource_control.before_step()` still runs every step too, as
a safety net for whatever's registered non-offloadable (model, here --
see ManagedLoRATrainerNode's own docstring for why) and for whatever the
controller's own measured-peak estimate gets wrong.

Concretely, for a LoRA run: the frozen base dominates the model's own
footprint and is too expensive to move every step (a real multi-GB
transfer, likely making per-step offload/reload of the whole model
slower than the run it's meant to protect) -- it stays resident for the
run's duration, same conclusion the main route reaches, not a
carried-over assumption (see ManagedLoRATrainerNode's own docstring).
The optimizer's tracked state and SDXL's two text encoders (a full CLIP
ViT-L/14 plus OpenCLIP ViT-bigG/14) are both real, not-always-negligible
chunks of VRAM -- how large depends on LoRA rank (optimizer state) and
is simply fixed and large (~1.6GB combined) for the text encoders --
and are the two AdaptiveResidencyController actually chooses between
when the budget can't be honored with everything resident.

`ResourceControlHandle.release()` (nodes/memory/control_handle.py) is
new this session, specifically for this file: the existing handle only
ever offloaded reactively (before_step()/ensure_loaded()'s own
_make_room()), with no way for a caller who already knows precisely
when a resident's idle window starts to just say so. Everything else
this file uses from nodes/memory/ -- register()/ensure_loaded(),
ResourceCoordinator, DeviceResident -- is unchanged, already-established
machinery, used here as directly as the main route uses it.
"""

from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import torch

from ..components.device import DeviceContext
from ..components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                     EpsParameterization, KarrasInputScaler)
from ..core import Port
from ..dataset.handle import TrainingBatchSource
from ..components.layout import ProjectLayout
from ..memory.control_handle import ResourceControlHandle
from ..memory.coordinator import ResourceCoordinator
from ..memory.handle import DeviceResident
from ..model.handle import TrainableModel
from ..model.lora_training_resources import LoRATrainingSkeleton
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import FusedOptimizerHandle, OptimizerHandle, describe_optimizer
from .loss import LossWeighting, UniformLossWeighting, t_bucket_losses
from .node import TrainerNode
from .schedule import LRSchedule
from .step_pipeline import _phase_label
from .step_notify import notify_step
from .t_probe import TProbe, format_probe_line


@dataclass
class ManagedStepState:
    """One **micro-step**: a single batch fetched, forwarded, backwarded.

    `step` is the optimizer-step index (LR schedule, monitoring, save
    cadence all key on it); `micro` is the position within that step's
    grad_accum window (0 .. grad_accum-1), which is what phases gate on
    to run once per window (zero_grad/begin_step, optimizer.step,
    reporting) instead of once per batch. `micro` defaults to 0 == first
    position, but every phase treats a state built without it (grad_accum
    == 1 runs, every existing test) as a complete one-window step --
    `0 + 1 >= 1` holds, so boundaries fire on the first/only micro."""
    step: int
    batch: Optional[dict]
    model: TrainableModel
    device: Any
    micro: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class ManagedStepPhase(ABC):
    @abstractmethod
    def run(self, state: ManagedStepState) -> ManagedStepState:
        ...


class ManagedTrainingStepPipeline:
    """run_step() also carries optional per-phase timing, gated behind
    TRAIN_STEP_TIMING=1 (env var, same zero-overhead-unless-opted-in
    convention as core/comfy_setup.py's own TRAIN_VRAM_DEBUG -- checked
    once here, not re-read from os.environ every step). Added this
    session after two guesses at what a real, reported "fast steps, then
    a long stall, then a few more, correlated with new image
    resolutions" pattern's root cause was (SYCL/Level-Zero env vars;
    CachingTextEncoder's cache-key granularity) both turned out not to
    move the real number -- rather than guess a third time with no way
    to check it here (no XPU in this environment, still), this measures
    where a real run's own time actually goes, per phase, per step, so
    the next round of investigation starts from data instead of another
    guess. Off by default: an unconditional sync point after every phase
    (needed for the timing itself to mean anything against an async
    GPU queue) is real, deliberate overhead, not something to pay on
    every real training step.

    Usage: TRAIN_STEP_TIMING=1, run a short session (a few dozen steps
    covering at least one "long stall" is enough -- this doesn't need a
    full run), and the printed per-phase milliseconds will show which
    phase the stall actually lands in. That's the one piece of real
    information every guess so far has been missing."""

    def __init__(self, phases: list[ManagedStepPhase]):
        self.phases = phases
        self._timing = os.environ.get("TRAIN_STEP_TIMING", "0") == "1"

    def run_step(self, state: ManagedStepState) -> ManagedStepState:
        if not self._timing:
            for phase in self.phases:
                state = phase.run(state)
            return state

        device_module = None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_module = torch.xpu
        elif torch.cuda.is_available():
            device_module = torch.cuda

        timings = []
        for phase in self.phases:
            if device_module is not None:
                device_module.synchronize()
            start = time.perf_counter()
            state = phase.run(state)
            if device_module is not None:
                device_module.synchronize()
            ms = (time.perf_counter() - start) * 1000
            timings.append((type(phase).__name__, ms))
            # Land each measurement in extras *as phases complete*, not after
            # the loop: MonitoringPhase is one of these phases (the last), and
            # its report is what ships these as {label}_ms / step_total_ms to
            # the monitor -- same naming as the main route's TimedPhase, same
            # shared _phase_label, so the dashboard's timing chart reads both
            # routes identically. MonitoringPhase's own entry lands after its
            # report already went out (never read -- a phase can't time itself
            # into the report it's building); the printed line below still
            # covers every phase, class names unchanged for anyone grepping it.
            state.extras.setdefault("timing_ms", {})[_phase_label(phase)] = ms
        parts = ", ".join(f"{name}={ms:.1f}ms" for name, ms in timings)
        total = sum(ms for _, ms in timings)
        print(f"[step {state.step} timing] total={total:.1f}ms  {parts}")
        return state


class AdaptiveResidencyController:
    """Decides whether text_encoder/optimizer actually need to be
    released between uses at all -- an earlier version of this file
    always released both, every step, regardless of the stated budget.
    That version was real, measured, and wrong for a case that turned
    out to be common, not an edge case: a real run reported ~0.36
    steps/sec against this route vs. ~1.7 steps/sec on the main route
    (SupervisedLoRATrainerNode, same settings) -- a ~4.7x slowdown --
    with peak reserved VRAM around 9.0GB against a 12500MB budget the
    whole time. The offloading was never once necessary in that run;
    every release()/ensure_loaded() round trip was pure cost, bought
    nothing.

    This controller fixes that by measuring instead of assuming:
    `calibration_steps` steps run with *everything* resident (no
    release() calls at all -- see EncodeConditioningPhase/
    BackwardAndOptimizerStepPhase's own should_release() checks below),
    while DeviceContext.reset_peak_stats() + memory_stats()'s own
    peak_reserved_mb (nodes/components/device.py) track the real
    high-water mark. If that peak already fits the budget (with
    `safety_margin` headroom -- see below), nothing ever gets released
    -- full speed, same as the main route pays for the same reason. If
    it doesn't, candidates are released starting from the smallest
    footprint_bytes() first (a direct reading of "less performance
    costly options first" -- byte count is the one real, already-
    available number that's actually proportional to transfer cost, not
    a proxy for it), only as many as the shortfall actually needs,
    estimated by subtracting each candidate's own footprint_bytes()
    from the measured peak in that order.

    **Keeps watching after deciding, and escalates -- doesn't calibrate
    once and stop.** A second real report, from real use of the version
    of this class that only ever calibrated once: a variable-resolution
    ("non-square") dataset, where each step's own activation memory
    depends on that step's own image size, OOM'd partway through a run
    that had calibrated fine -- calibration_steps steps (default 3)
    happened to sample smaller images, so the true worst case never got
    measured before the decision to stay fully resident was locked in.
    ManagedLoRATrainerNode's own build() now calls
    DeviceContext.reset_peak_stats() every step, not just once before
    calibration, so every memory_stats() reading passed to
    record_step_peak() reflects *that one step's own* peak, not a
    cumulative one -- after the initial decision, a step whose own peak
    exceeds the (margined) usable budget triggers escalation:
    should_release() gains one more candidate (smallest-footprint-
    among-what's-not-already-released first, same ordering as the
    initial decision), permanently, for the rest of the run -- no
    de-escalating back once something's been added, to avoid thrashing
    between resident and released every time usage happens to dip.
    Escalation is real insurance, not a guarantee: it still can't react
    *within* the step that actually OOMs (a within-step activation
    spike isn't something any between-step check can catch in time --
    before_step()'s own reactive check has the same limit); what it
    does is make the *next* image of that size survive, once one
    instance has been seen and escalated for. `safety_margin` (default
    0.1 -- 10%) is the other, complementary piece: shaving the usable
    ceiling down before comparing against it, specifically so a
    somewhat-larger-than-calibrated step has a chance of still fitting
    without needing to escalate at all.

    Why measure real usage instead of estimating it analytically up
    front (batch size, resolution, rank, etc.): this project's own
    docs consistently favor a real, checked number over a predicted
    one (see e.g. nodes/model/checkpoint_placement.py's own
    BlockCost/GreedyRatioPlacement, explicitly not wired into real use
    yet because it has no real profiled numbers to validate a
    placement against). Analytically modeling this UNet's own
    activation memory across arbitrary batch/resolution/rank
    combinations is a much harder, more fragile problem than reading
    the number the allocator already tracks -- and here, unlike
    per-block checkpoint placement, there are only two candidates to
    choose between, so a few real, cheap calibration steps (plus
    ongoing escalation for what they miss) settle it directly rather
    than needing a model of the cost at all.

    What this still can't do anything about: for a real reported case,
    releasing *both* candidates (optimizer + text_encoder, together a
    small fraction of total usage next to activation memory for a large
    or variable-resolution image) wasn't enough headroom on its own --
    the dominant, unmanaged cost was activation memory, which neither
    this controller nor NF4/Int8 weight quantization touches at all
    (see docs/design/resources-controller/
    09-trainer-integration-and-vram-safety.md's addendum for
    the full reasoning, including why NF4/Int8's *storage* savings
    don't translate to comparable *peak-during-compute* savings -- both
    dequantize to a real, transient full-precision buffer on every use,
    by design, not a bug). Gradient checkpointing
    (nodes/model/gradient_checkpointing.py, real, working, not wired
    into this route) is the lever that actually addresses activation
    memory -- still deliberately not attempted here, same reasoning as
    before: a real, separate follow-up.
    """

    def __init__(self, usable_mb: Optional[float], candidates: dict[str, DeviceResident],
                 calibration_steps: int = 3, safety_margin: float = 0.1):
        self._raw_usable_mb = usable_mb
        self._usable_mb = usable_mb if usable_mb is None else usable_mb * (1.0 - safety_margin)
        self._candidates = candidates  # name -> resident, for footprint_bytes() at decision time
        self._calibration_steps = max(1, calibration_steps)
        self._steps_seen = 0
        self._peak_reserved_mb = 0.0
        self._decided: Optional[set] = None  # None while still calibrating

    @property
    def calibrating(self) -> bool:
        return self._decided is None

    @property
    def releases_anything(self) -> bool:
        """False while still calibrating (nothing decided yet) or once
        decided that nothing needs releasing. ManagedLoRATrainerNode's
        own empty_cache_every_n_steps loop checks this before paying for
        a gc.collect()/empty_cache() pass -- reclaiming unused cached
        memory back to the driver is pointless work when nothing was
        ever released in the first place, and a real, reported case: a
        run at 9964MB reserved against a 12500MB budget decided
        (correctly) to release nothing, and was still paying a full
        gc.collect()+empty_cache() pass every single step regardless,
        for nothing to actually reclaim."""
        return bool(self._decided)

    def record_step_peak(self, memory_stats: Optional[dict]) -> None:
        """Call every step -- caller (ManagedLoRATrainerNode.build())
        resets DeviceContext's own peak counter every step too, so each
        call here sees *that step's own* peak, not a cumulative one.
        Decides immediately, on the very first call, when there's no
        usable ceiling to plan against at all (usable_budget_mb()
        returned None -- ResourceControlHandle's own docstring: "no
        fixed ceiling concept") or no memory-stats concept on this
        device (memory_stats is None -- DeviceContext's own docstring:
        CPU, mainly) -- both cases mean there's nothing offloading
        could ever be measured against, so waiting calibration_steps
        for a number that will never arrive would just mean never
        deciding at all, staying fully resident is the only coherent
        answer either way (and there's nothing to escalate against
        later either, so this class has nothing further to do)."""
        if self._usable_mb is None or memory_stats is None:
            if self._decided is None:
                self._decided = set()
            return
        peak = memory_stats["peak_reserved_mb"]
        if self._decided is None:
            self._peak_reserved_mb = max(self._peak_reserved_mb, peak)
            self._steps_seen += 1
            if self._steps_seen >= self._calibration_steps:
                self._decide()
        else:
            self._maybe_escalate(peak)

    def _decide(self) -> None:
        order = sorted(self._candidates.items(), key=lambda kv: kv[1].footprint_bytes())
        release: set = set()
        remaining_mb = self._peak_reserved_mb
        for name, resident in order:
            if remaining_mb <= self._usable_mb:
                break
            release.add(name)
            remaining_mb -= resident.footprint_bytes() / (1024 ** 2)
        self._decided = release
        self._log(f"measured peak={self._peak_reserved_mb:.0f}MB over {self._steps_seen} "
                   f"calibration step(s)", release)

    def _maybe_escalate(self, this_step_peak_mb: float) -> None:
        if this_step_peak_mb <= self._usable_mb:
            return
        order = sorted(self._candidates.items(), key=lambda kv: kv[1].footprint_bytes())
        for name, _resident in order:
            if name not in self._decided:
                self._decided.add(name)
                self._log(f"a later step's own peak={this_step_peak_mb:.0f}MB exceeded "
                           f"budget even with the current release set -- escalating",
                           self._decided, escalated_name=name)
                return
        # Nothing left to escalate to. resource_control.before_step()'s own reactive
        # check (and strict=True, if set) is whatever's left from here -- and neither
        # of those can react *within* the step that's already over, only the next one.

    def _log(self, prefix: str, release: set, escalated_name: Optional[str] = None) -> None:
        if escalated_name is not None:
            verdict = f"now also releasing {escalated_name!r} (release set: {sorted(release)})"
        elif release:
            verdict = "releasing " + ", ".join(sorted(release))
        else:
            verdict = "nothing -- staying fully resident for the rest of this run"
        print(f"[AdaptiveResidencyController] {prefix}, usable budget={self._usable_mb:.0f}MB "
              f"(={self._raw_usable_mb:.0f}MB minus safety margin) -- {verdict}")

    def should_release(self, name: str) -> bool:
        """False during calibration (never release yet -- that's the
        whole point of measuring the unmodified peak first) and False
        after a decision that didn't select `name`."""
        if self._decided is None:
            return False
        return name in self._decided


class FetchBatchPhase(ManagedStepPhase):
    """No residency concern -- data, not a device resident."""

    def __init__(self, batches: TrainingBatchSource):
        self._batches = batches
        self._iterator = iter(batches)

    def run(self, state: ManagedStepState) -> ManagedStepState:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._batches)
            batch = next(self._iterator)
        state.batch = batch
        return state


class PrepareDiffusionInputsPhase(ManagedStepPhase):
    """x_t/target/t onto the device, noise schedule + input scaling, and
    the same LoRA timestep gate (nodes/model/lora.py) the main route's own
    equivalent phase wires -- no residency concern of its own (model
    isn't touched yet), and no reason for this project's LoRA-gate math
    itself to have two implementations."""

    def __init__(self, diffusion_process: DiffusionProcess, gate_enabled: bool = False,
                 gate_train_low: float = 0.0, gate_train_high: float = 999.0,
                 gate_width: float = 100.0):
        self._process = diffusion_process
        self._gate_enabled = gate_enabled
        self._gate_train_low = gate_train_low
        self._gate_train_high = gate_train_high
        self._gate_width = gate_width

    def run(self, state: ManagedStepState) -> ManagedStepState:
        from ..model.lora import compute_lora_gate, set_lora_gate

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
        # Shape-bucketing validity mask, when the dataset padded (LossPhase
        # divides by the valid element count rather than the total, so a
        # padded batch trains at the same scale as an unpadded one). Absent
        # for every graph that did not ask for bucketing, which is why the
        # loss keeps its original expression in that case.
        if batch.get("valid_mask") is not None:
            state.extras["valid_mask"] = batch["valid_mask"].to(state.device)

        if self._gate_enabled:
            set_lora_gate(compute_lora_gate(
                t, self._gate_train_low, self._gate_train_high, self._gate_width))
        else:
            set_lora_gate(None)

        return state


class EncodeConditioningPhase(ManagedStepPhase):
    """Loads the text encoder for exactly this phase's own encode()
    call, releases it immediately after -- this route's central
    difference from the main route's equivalent phase (which leaves it
    resident, relying only on reactive, pressure-triggered offloading).

    Correct to release unconditionally right after encoding, not just
    an optimization: the text encoder is always frozen in this design
    (never part of what TrainerParametersNode/nodes/model/
    trainer_parameters.py pulls trainable parameters from), so once
    ctx_emb/y are computed, nothing downstream needs the encoder's own
    weights resident -- there's no backward pass through it to support.

    device_ctx/profile: reports reserved_mb right after loading and
    right after releasing, when profile=True -- see this module's own
    ManagedLoRATrainerNode.build() for why this reporting lives here
    and in BackwardAndOptimizerStepPhase rather than in MonitoringPhase
    alone.

    controller: AdaptiveResidencyController -- release() only actually
    runs when controller.should_release("text_encoder") says so (False
    during calibration, and after calibration if the measured peak
    never needed it). ensure_loaded() runs unconditionally by default
    -- cheap and safe when nothing was ever offloaded, and correct
    if before_step()'s own reactive check offloaded this for some other
    reason between calls.

    ensure_loaded_before_encode=False (set by ManagedLoRATrainerNode's
    `prewarm_text_encoder` Port) is the one exception to that
    unconditional rule: when the cache was warmed over the exact keys
    training will ask for and the encoder unloaded, loading here would
    re-upload the encoder the prewarm just freed and re-reside it for
    the rest of the run (or re-offload it next phase, churning per
    step) -- the exact waste that flag exists to avoid. Correctness
    survives the skip because CachingTextEncoder only needs the inner
    encoder for a genuine miss, and the prewarm path binds its
    resource_control handle precisely so a miss still self-loads
    (misses outside the warmed set -- dataset changed after warm-up --
    degrade to slow-and-correct, never wrong). release() below is
    unchanged and stays correct either way: releasing an already-
    unloaded resident is that handle's own documented no-op."""

    def __init__(self, text_encoder: TextEncoder, resource_control: ResourceControlHandle,
                 controller: "AdaptiveResidencyController",
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False,
                 ensure_loaded_before_encode: bool = True):
        self._text_encoder = text_encoder
        self._resource_control = resource_control
        self._controller = controller
        self._device_ctx = device_ctx
        self._profile = profile
        self._ensure_loaded_before_encode = ensure_loaded_before_encode

    def run(self, state: ManagedStepState) -> ManagedStepState:
        if self._ensure_loaded_before_encode:
            self._resource_control.ensure_loaded("text_encoder")
            self._log("loaded")
        x_t = state.extras["x_t"]
        batch = state.batch
        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        ctx_emb, y = self._text_encoder.encode(
            batch["prompt"], batch_size=x_t.shape[0], height=batch_h, width=batch_w)
        state.extras["ctx_emb"] = ctx_emb.to(device=state.device, dtype=torch.bfloat16)
        state.extras["y"] = y.to(device=state.device, dtype=torch.bfloat16)
        if self._controller.should_release("text_encoder"):
            self._resource_control.release("text_encoder")
            self._log("released")
        return state

    def _log(self, moment: str) -> None:
        if not self._profile or self._device_ctx is None:
            return
        mem = self._device_ctx.memory_stats()
        reserved = f"{mem['reserved_mb']:.0f}MB" if mem is not None else "n/a"
        print(f"    [residency] text_encoder {moment}: vram_reserved={reserved}")


class ZeroGradPhase(ManagedStepPhase):
    """LR update + gradient-buffer reset, once per **optimizer step**
    (grad_accum window), not once per micro-step -- zeroing every
    micro-step would wipe gradients accumulated earlier in the same
    window before the boundary optimizer.step() ever saw them, silently
    defeating grad_accum (the same trap core/train_step.py's own
    `micro_step % effective_accum == 0` gating exists to avoid). The
    schedule value itself is read (and stashed into extras["lr"]) every
    micro-step so MonitoringPhase's boundary report always finds it in
    its own state's extras -- each micro-step gets a fresh ManagedStepState.

    No resource_control call here on purpose, unlike
    BackwardAndOptimizerStepPhase below -- checked directly against
    ComposedOptimizerHandle (nodes/optimizer/composed.py): zero_grad()
    only touches .grad on the model's own parameters
    (ExecutionStrategy.zero_grad(params)), update_lr() is pure Python
    float arithmetic -- neither one reads or writes the optimizer's own
    tracked state tensors (self.states), so this phase doesn't need
    optimizer state resident at all."""

    def __init__(self, optimizer: OptimizerHandle, lr_schedule: LRSchedule,
                 is_fused: bool, grad_accum: int = 1):
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._is_fused = is_fused
        self._grad_accum = grad_accum

    def run(self, state: ManagedStepState) -> ManagedStepState:
        lr = self._lr_schedule.value(state.step)
        state.extras["lr"] = lr
        if state.micro != 0:
            # Mid-window: this window's zero_grad/begin_step already ran
            # at micro 0 (and the optimizer can't be re-zeroed without
            # destroying what's accumulated so far).
            return state
        self._optimizer.update_lr(lr)
        if self._is_fused:
            # sub_steps=grad_accum tells the fused handle's backward
            # hooks to span this many passes before applying anything --
            # begin_step(sub_steps) is precisely this handle's multi-pass
            # entry point (nodes/optimizer/composed_fused.py's own
            # "multi-pass state machine" docstring), and
            # BackwardAndOptimizerStepPhase calls prepare_next_pass()
            # between passes to keep it counting.
            self._optimizer.begin_step(sub_steps=self._grad_accum)
        else:
            self._optimizer.zero_grad()
        return state


class ForwardPhase(ManagedStepPhase):
    """Runs the UNet forward, re-ensuring the model is resident first.

    That `ensure_loaded` is not ceremony and not free: it is a real device
    query (`memory_stats()`) every step, and it is here because the model is
    registered *sacrificable* (see below), so conditioning may have moved it
    to host RAM to make room for CLIP on a card that had none.

    It is the cost of having the capability at all: a step that never
    sacrifices the model pays one memory read for the privilege, and a step
    that does pays the reload -- measured at 1,145 ms on the B580. The
    alternative, registering the model as neither offloadable nor
    sacrificable, is today's behaviour and is correct on any card with room
    for CLIP alongside the model, which is most of them.
    """

    def __init__(self, resource_control=None):
        self._resource_control = resource_control

    def run(self, state: ManagedStepState) -> ManagedStepState:
        if self._resource_control is not None:
            self._resource_control.ensure_loaded("model")
        state.extras["pred"] = state.model.forward(
            state.extras["xc"], state.extras["t"], state.extras["ctx_emb"], state.extras["y"])
        return state


class LossPhase(ManagedStepPhase):
    """Per-sample loss weighting + per-t diagnostics stash -- same math
    as the main route's LossPhase (nodes/train/step_pipeline.py; both
    changed together in the same session, see that class's docstring for
    why weighting is per-sample rather than one scalar from the batch's
    mean sigma), plus this route's grad_accum loss scaling:

    extras["loss"] stays the *unscaled* weighted loss (what's reported,
    what on_step sees -- averaging window losses then equals the plain
    mean over the window's batches); extras["loss_for_backward"] is that
    same tensor divided by grad_accum, so backward() accumulates
    grad_accum micro-step gradients into the mean gradient over the whole
    window. Set only when backward_scale != 1.0 -- a plain grad_accum=1
    run's backward reads extras["loss"] exactly as before."""

    def __init__(self, loss_weighting: LossWeighting, backward_scale: float = 1.0,
                 bucket_balance=None):
        self._loss_weighting = loss_weighting
        self._backward_scale = backward_scale
        # Optional BucketBalance (nodes/train/bucket_balance.py): a second
        # per-sample factor, w(sigma) * w_bucket(t). weight_for_t() returns
        # None whenever the balance applies nothing (mode "off", warmup
        # unfinished, no t) -- then the branch below is the original code
        # expression-for-expression, so wiring a tracking-only balance is a
        # guaranteed no-op here (mirrors the main route's LossPhase).
        self._bucket_balance = bucket_balance

    def run(self, state: ManagedStepState) -> ManagedStepState:
        import torch

        pred = state.extras["pred"]
        target = state.extras["target"]
        sigma = state.extras["sigma"]
        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        sigmas = sigma.float().reshape(-1)
        w_bucket = None
        if self._bucket_balance is not None:
            w_bucket = self._bucket_balance.weight_for_t(
                state.extras.get("t"), dtype=per_sample.dtype,
                device=per_sample.device)
        if sigmas.numel() == per_sample.numel():
            weights = torch.tensor(
                [self._loss_weighting.weight(float(s)) for s in sigmas.tolist()],
                dtype=per_sample.dtype, device=per_sample.device)
            if w_bucket is not None:
                weights = weights * w_bucket
            loss = (per_sample * weights).mean()
        else:
            weight = self._loss_weighting.weight(float(sigmas.mean().item()))
            if w_bucket is not None:
                loss = (per_sample * w_bucket).mean() * weight
            else:
                loss = per_sample.mean() * weight
        state.extras["loss"] = loss
        state.extras["per_sample_loss"] = per_sample.detach()
        if self._backward_scale != 1.0:
            state.extras["loss_for_backward"] = loss * self._backward_scale
        return state


class BackwardAndOptimizerStepPhase(ManagedStepPhase):
    """Backward and the optimizer update as one phase, with one
    optimizer-residency window spanning both -- not two phases the way
    the main route splits them, and not a coincidence.

    Why they can't be split apart here: a fused optimizer's real update
    happens inside a backward-pass hook
    (ComposedFusedOptimizerHandle._on_grad_ready(), checked directly --
    fires per-parameter as each gradient becomes ready *during*
    backward() itself, via register_post_accumulate_grad_hook()), not in
    a separate call afterward (FusedOptimizerHandle.step() is a no-op --
    "real updates happen in the hook", that class's own comment). So for
    a fused optimizer, its state has to already be resident *before
    backward() starts*, not just before some later step()-shaped phase.
    A non-fused optimizer doesn't strictly need state resident during
    backward, only during its own step() call after -- but ensure_loaded()
    a little earlier than strictly required for that case is a small,
    deliberate over-inclusion, traded for one rule that's correct for
    both cases instead of a fused/non-fused branch in the residency
    logic itself.

    device_ctx/profile: same reporting as EncodeConditioningPhase's own
    -- see that class's docstring.

    controller: same gating as EncodeConditioningPhase's own -- see
    that class's docstring.

    grad_accum/grad_clip_max_norm: backward runs every micro-step (the
    graph must be freed per micro-step for accumulation to stay
    memory-neutral), but the *update* only happens on the boundary
    micro-step -- non-fused via optimizer.step() gated below, fused via
    the hook itself: begin_step(sub_steps=grad_accum) at window start
    made the fused handle count passes, so the boundary backward's hook
    fires the update on its own, and prepare_next_pass() between passes
    keeps that count advancing (without it the handle's _in_backward
    flag would stay True and it would never count a second pass). Grad
    clipping sits between backward and step, boundary only -- and is
    refused at build time for fused optimizers, since their update
    already happened inside backward() before any code here could clip
    (see ManagedLoRATrainerNode.build's own validation).
    """

    def __init__(self, optimizer: OptimizerHandle, is_fused: bool,
                 resource_control: ResourceControlHandle,
                 controller: "AdaptiveResidencyController",
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False,
                 grad_accum: int = 1, grad_clip_max_norm: float = 0.0):
        self._optimizer = optimizer
        self._is_fused = is_fused
        self._resource_control = resource_control
        self._controller = controller
        self._device_ctx = device_ctx
        self._profile = profile
        self._grad_accum = grad_accum
        self._grad_clip_max_norm = grad_clip_max_norm

    def run(self, state: ManagedStepState) -> ManagedStepState:
        self._resource_control.ensure_loaded("optimizer")
        self._log("loaded")
        # extras["loss_for_backward"] (grad_accum-scaled) when a window
        # is in flight, plain extras["loss"] otherwise -- see LossPhase.
        state.extras.get("loss_for_backward", state.extras["loss"]).backward()
        is_boundary = state.micro + 1 >= self._grad_accum
        if self._is_fused:
            if not is_boundary:
                self._optimizer.prepare_next_pass()
        else:
            if is_boundary:
                if self._grad_clip_max_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        state.model.trainable_parameters(), self._grad_clip_max_norm)
                    # clip_grad_norm_ already measured the pre-clip total norm --
                    # stashing it here makes `grad_norm` in the monitor report
                    # free. Deliberately absent when clipping is off
                    # (grad_clip_max_norm=0): measuring it separately would cost
                    # an extra pass over every grad plus a device sync every
                    # optimizer step, so the key present-or-absent means
                    # "measured here", never a stale placeholder.
                    state.extras["grad_norm"] = float(total_norm)
                self._optimizer.step(n_steps=1)
        if self._controller.should_release("optimizer"):
            self._resource_control.release("optimizer")
            self._log("released")
        return state

    def _log(self, moment: str) -> None:
        if not self._profile or self._device_ctx is None:
            return
        mem = self._device_ctx.memory_stats()
        reserved = f"{mem['reserved_mb']:.0f}MB" if mem is not None else "n/a"
        print(f"    [residency] optimizer {moment}: vram_reserved={reserved}")


class ProbePhase(ManagedStepPhase):
    """Optional fixed-probe diagnostics (nodes/train/t_probe.py -- read its
    module docstring for what the numbers mean).

    Runs after BackwardAndOptimizerStepPhase, every micro-step for
    *collection* (capturing element 0 of the first few batches -- already
    on device, conditioning already cached in extras -- as probe items,
    until the probe has n_items) and only at the optimizer-step boundary
    for *evaluation*: once as soon as the probe is ready (so the first
    numbers arrive early, when rel ~ 1.0 is the expected answer), then
    every `every_n_steps` optimizer steps. Position matters for the
    gradient-alignment diagnostic: after the update (its diagnostic
    backward would otherwise contaminate the step) and before the next
    window's ZeroGradPhase, which clears whatever .grad it leaves.

    The result lands in extras["probe_report"] for MonitoringPhase to merge
    into that step's report (one record per step -- the dashboard needs no
    new record type) and is also printed here, unconditionally: a probe is
    an explicit opt-in, and its numbers are useless if they only exist when
    a monitor is wired.
    """

    def __init__(self, probe: TProbe, process: DiffusionProcess, every_n_steps: int,
                 grad_accum: int = 1):
        self._probe = probe
        self._process = process
        self._every = every_n_steps
        self._grad_accum = grad_accum
        self._evaluated_once = False

    def run(self, state: ManagedStepState) -> ManagedStepState:
        ex = state.extras
        if self._probe.wants_items():
            self._probe.collect(self._process, ex["x_t"], ex["target"], ex["t"],
                                ex["sigma"], ex["ctx_emb"], ex["y"])
        if state.micro + 1 < self._grad_accum:
            return state
        if not self._probe.ready():
            return state
        due = (state.step + 1) % self._every == 0
        if not (due or not self._evaluated_once):
            return state
        self._evaluated_once = True
        report, detail = self._probe.evaluate(state.model, self._process)
        if self._probe.grad_alignment:
            report.update(self._probe.alignment(
                state.model, self._process, state.model.trainable_parameters()))
        if report:
            ex["probe_report"] = report
            print(format_probe_line(state.step, report, detail))
        return state


class MonitoringPhase(ManagedStepPhase):
    """Runs last in the step -- after EncodeConditioningPhase and
    BackwardAndOptimizerStepPhase have already released everything
    they each manage.

    **A correction to what this number used to be documented as.** It
    said here that ``per_resident_mb`` "will correctly show
    optimizer/text_encoder near 0 every step regardless of whether
    release() actually did anything", on the grounds that this phase runs
    too late to show their real peak. The first half is false and the
    second half is why nobody looked. These are *current* footprints
    (`ResourceCoordinator.per_resident_footprint_bytes`), not peaks, so
    they read near 0 only when something was actually released -- and at
    an operating point with headroom, nothing is. Measured on the B580 at
    batch 2 / 1024, where the controller measured peak 9,226 MB against a
    9,889 MB usable budget and logged "nothing -- staying fully resident":
    every step reported ``model=4897MB optimizer=714MB
    text_encoder=1561MB``. So CLIP's full 1,561 MB sits in the monitor's
    VRAM graph for the whole run, which is correct and not a leak --
    `AdaptiveResidencyController.should_release("text_encoder")` is False,
    so `EncodeConditioningPhase` never offloads it, so it is genuinely
    resident. Documenting the series as necessarily-uninformative is what
    let a real 1.5 GB go unexamined.

    The peak is still not this phase's job, and the two phases' own
    ``profile=True`` lines -- printed inline, when loading and releasing
    actually happen -- remain the informative place for it. Also
    meaningful here: the model's own footprint (steady, since it is never
    a release candidate) and one end-of-step summary line
    (loss/lr/vram_reserved_mb). Deliberately leaner than the main route's
    own MonitoringPhase (nodes/train/step_pipeline.py) beyond that -- not
    the main route's fuller set (per-phase timing, baseline deltas, a
    tracked-footprint cross-check against ResourceProfile). Someone wanting
    that level of detail can still build it the same way that file did;
    duplicating all of it here wasn't this file's job.

    Read the residents line as *what is on the card right now*, and expect
    a release candidate to be at its full size whenever the controller
    decided it could stay -- or at 0 when `prewarm_text_encoder` put it in
    host RAM instead. Those are the two real states, and both report
    themselves correctly as of the `footprint_bytes()` fix (see
    `nodes/model/text_encoder.py`'s `unload()`), which used to report
    CLIP's full 1,561 MB after a prewarm had freed exactly that much:

        prewarm off   peak 9,228 MB   residents: model=4897MB optimizer=714MB text_encoder=1561MB
        prewarm on    peak 7,666 MB   residents: model=4897MB optimizer=714MB text_encoder=0MB

    With the residents line correct, the gap between it and
    `vram_reserved_mb` is the same 2,05x MB either way -- activations,
    optimizer workspace and fragmentation -- which is the check that the
    line is now accounting for the difference rather than hiding it. The
    port worth reaching for is `prewarm_text_encoder`, which is off by
    default and is what buys that 1.5 GB.

    grad_accum: runs after every micro-step (it's last in the phase
    list), but *emits* only on the boundary micro-step -- one report /
    one on_step call / one profile line per optimizer step, with `loss`
    averaged over the window's micro-steps rather than any single
    batch's. It accumulates across micro-steps because each micro-step
    gets a fresh ManagedStepState whose extras die with it (see that
    class's docstring), so this instance -- constructed once per build,
    like FetchBatchPhase's iterator -- is the only thing that can carry
    a window's numbers forward.

    Report keys beyond loss/lr: `loss_t_low`/`loss_t_mid`/
    `loss_t_high`, the raw per-sample MSE per fixed third of the t range
    (loss.py's t_bucket_losses), accumulated over the whole window so a
    grad_accum=1 batch-2 step still usually covers two of the three, and
    buckets with no samples in the window emit no key at all (the
    monitor chart draws a gap, not a fabricated flat line) -- the series
    behind its per-t colored lines. Plus, when available:
    `vram_budget_mb` (the handle's usable_budget_mb(), constant --
    the ceiling the vram numbers are held under, for the dashboard's
    reference line), `vram_peak_reserved_mb`/`vram_peak_allocated_mb`
    (memory_stats()'s peaks -- since the build loop resets peak stats
    per micro-step, these are the *within-step* high-water mark, not a
    since-process-start one; same `vram_{k}` naming the main route's
    reports use), `grad_norm` (BackwardAndOptimizerStepPhase's stashed
    clip_grad_norm_ total -- only when clipping is on, see that stash),
    and `{label}_ms`/`step_total_ms` (TRAIN_STEP_TIMING=1's per-phase
    measurements, same keys as the main route's TimedPhase). All keys
    absent rather than fabricated when the source doesn't have them --
    the monitor chart's gap rule, applied to every series."""

    def __init__(self, total_steps: int, device_ctx: DeviceContext,
                 coordinator: ResourceCoordinator, on_step: Optional[Callable] = None,
                 monitor: Optional[MonitorHandle] = None, profile: bool = False,
                 optimizer_id: str = "", grad_accum: int = 1,
                 usable_budget_mb: Optional[float] = None,
                 bucket_balance=None):
        self._total_steps = total_steps
        self._device_ctx = device_ctx
        self._coordinator = coordinator
        self._on_step = on_step
        self._monitor = monitor
        self._profile = profile
        self._optimizer_id = optimizer_id
        self._grad_accum = grad_accum
        self._usable_budget_mb = usable_budget_mb
        # observe() runs at every boundary below, before the no-monitor
        # early return -- the balance drives training, so it keeps
        # tracking even in a run with no monitor and no profiling.
        self._bucket_balance = bucket_balance
        self._window_losses: list[float] = []
        self._window_ps: list[float] = []
        self._window_t: list[float] = []

    def run(self, state: ManagedStepState) -> ManagedStepState:
        # Accumulate this micro-step's contribution first, always -- the
        # boundary check below must not lose it.
        self._window_losses.append(float(state.extras["loss"].item()))
        ps = state.extras.get("per_sample_loss")
        t = state.extras.get("t")
        if ps is not None and t is not None:
            self._window_ps.extend(ps.detach().reshape(-1).tolist())
            self._window_t.extend(t.detach().reshape(-1).tolist())
        if state.micro + 1 < self._grad_accum:
            # Window still open -- emit once per optimizer step.
            return state

        loss_value = sum(self._window_losses) / len(self._window_losses)
        buckets = t_bucket_losses(self._window_ps, self._window_t)
        if self._bucket_balance is not None:
            # Window-accumulated means -- the right cadence for the
            # balance (batch-2 per-step numbers are too noisy to steer
            # by). Before the early return below, so observe() doesn't
            # depend on monitoring being on.
            self._bucket_balance.observe(buckets)
        self._window_losses.clear()
        self._window_ps.clear()
        self._window_t.clear()
        lr = state.extras["lr"]

        # The latent shape this step ran on, matching the main route's
        # MonitoringPhase (step_pipeline.py). With grad_accum > 1 a step spans
        # several micro-steps, so this is the LAST micro-step's shape -- the
        # same "which shape was this optimizer step" the main route reports,
        # not a summary of several. See that class's comment for why a shape
        # belongs in the step record at all.
        shape = None
        batch = state.batch
        if isinstance(batch, dict):
            latent = batch.get("x_t")
            latent_shape = getattr(latent, "shape", None)
            if latent_shape is not None and len(latent_shape) >= 2:
                shape = f"{int(latent_shape[-2])}x{int(latent_shape[-1])}"

        notify_step(self._on_step, state.step, loss_value, shape)

        if self._monitor is None and not self._profile:
            return state

        per_resident_mb = {
            name: nbytes / (1024 ** 2)
            for name, nbytes in self._coordinator.per_resident_footprint_bytes().items()
        }
        report = {
            "step": state.step, "total_steps": self._total_steps,
            "loss": loss_value, "lr": lr, "t": time.time(),
        }
        if self._optimizer_id:
            report["optimizer"] = self._optimizer_id
        report.update(buckets)
        probe_report = state.extras.get("probe_report")
        if probe_report:
            # probe_*/gc_* from ProbePhase -- present only on the steps a
            # probe actually ran (absent, not carried, in between).
            report.update(probe_report)
        if self._bucket_balance is not None:
            # weight_t_*/prob_t_* -- post-update weights (observe() ran
            # above), keys the balance doesn't have stay absent.
            report.update(self._bucket_balance.report())
        if self._usable_budget_mb is not None:
            # Constant per run -- the ceiling the vram_* numbers are held
            # under (ctor doc). None-handling is the key's whole contract:
            # absent means "this handle states no budget", never 0.
            report["vram_budget_mb"] = self._usable_budget_mb
        grad_norm = state.extras.get("grad_norm")
        if grad_norm is not None:
            report["grad_norm"] = grad_norm
        timing = state.extras.get("timing_ms")
        if timing:
            # Same {label}_ms / step_total_ms shape the main route's
            # TimedPhase-built reports carry -- one dashboard reads both.
            # Window nuance for grad_accum>1: extras dies with each micro
            # state, so this is the boundary micro-step's timing, not the
            # whole window's (honest label either way: it's timing for the
            # step that was measured).
            report.update({f"{label}_ms": ms for label, ms in timing.items()})
            report["step_total_ms"] = sum(timing.values())
        mem = self._device_ctx.memory_stats()
        if mem is not None:
            report["vram_reserved_mb"] = mem["reserved_mb"]
            report["vram_allocated_mb"] = mem["allocated_mb"]
            # Peaks since the build loop's per-micro reset -- the within-step
            # high-water mark (same `vram_{k}` key names the main route's
            # full memory_stats dump produces, so the dashboard's VRAM series
            # work identically on either route).
            report["vram_peak_reserved_mb"] = mem["peak_reserved_mb"]
            report["vram_peak_allocated_mb"] = mem["peak_allocated_mb"]
        for name, mb in per_resident_mb.items():
            report[f"resident_{name}_mb"] = mb

        if self._monitor is not None:
            self._monitor.report(report)
        if self._profile:
            resident_part = " ".join(f"{name}={mb:.0f}MB" for name, mb in per_resident_mb.items())
            mem_part = (f" vram_reserved={mem['reserved_mb']:.0f}MB"
                        if mem is not None else "")
            optimizer_part = f" optimizer={self._optimizer_id}" if self._optimizer_id else ""
            bucket_part = "".join(
                f" {key.replace('loss_t_', 't_')}={value:.4f}" for key, value in buckets.items())
            balance_part = "".join(
                f" {key.replace('weight_t_', 'w_')}={value:.2f}"
                for key, value in (self._bucket_balance.report().items()
                                   if self._bucket_balance is not None else ()))
            print(f"  [step {state.step}]{optimizer_part} loss={loss_value:.4f} lr={lr:.2e}"
                  f"{bucket_part}{balance_part}{mem_part} residents: {resident_part}")
        return state


class ManagedLoRATrainerNode(TrainerNode):
    """The Resources Controller route's own trainer -- see this module's
    own top docstring for the full reasoning. Takes `trainer` directly
    (a LoRATrainingConfigNode's own bundled output) rather than the main
    route's separate model/text_encoder ports: unlike
    TrainerResourcesUnpackNode's earlier, since-removed role of adapting
    that bundle into the main route's own TrainerNode ports, this node
    has no such ports to match -- it's not required to be interchangeable
    with the main route's trainer node, so there's nothing to adapt
    around.

    resource_control is required, not optional -- this design has
    nothing meaningful left to do without it (there's no other residency
    strategy here to fall back to; the main route's SupervisedLoRATrainerNode
    is that fallback, a genuinely separate route, not a mode of this one).

    empty_cache_every_n_steps defaults to 1 (every step), unlike the main
    route's 0 (off): a release() call only moves a resident's tensors off
    GPU inside this process's own caching allocator -- returning that
    freed reserved memory to the driver, which is the entire point for a
    route built around staying clear of a VRAM ceiling, needs an explicit
    empty_cache() on top. Still a plain int Port -- higher values trade
    some of that driver-visible headroom back for fewer synchronize()-
    forcing calls, for anyone who measures and decides that's the better
    trade for their own run.
    """

    #: Fields that change the peak VRAM this node's run will reach. Declarative
    #: data (not code) so the server can compute a graph fingerprint without
    #: importing torch or instantiating anything. See
    #: backend/application/memory_fingerprint.py.
    memory_fields: ClassVar[tuple[str, ...]] = (
        "model", "batch_size", "rank", "checkpointing", "optimizer",
    )

    INPUTS: ClassVar[dict[str, Port]] = {
        **{k: v for k, v in TrainerNode.COMMON_INPUTS.items() if k not in ("model", "text_encoder")},
        "trainer": Port(
            name="trainer", type=LoRATrainingSkeleton, required=True,
            doc="A LoRATrainingConfigNode's own `trainer` output. Unpacked into "
                "trainer.unet/trainer.clip internally -- see this module's own docstring.",
        ),
        "resource_control": Port(
            name="resource_control", type=ResourceControlHandle, required=True,
            doc="Wire a VRAM Budget Controller node's own output here -- required, not "
                "optional (see this class's own docstring). model is registered "
                "non-offloadable (the frozen base dominates its footprint and is too "
                "expensive to move every step); optimizer and text_encoder are each "
                "registered offloadable, but whether either is actually released between "
                "uses is decided once by AdaptiveResidencyController (this module's own "
                "docstring) after measuring real peak usage with everything resident -- "
                "not unconditionally, every step, regardless of whether the budget needed "
                "it. before_step() still runs every step too, as a safety net for model "
                "alone exceeding the budget, or for the controller's own estimate being "
                "wrong -- set strict=True on the connected VRAMBudgetControllerNode to "
                "raise instead of continuing if either happens.",
        ),
        "diffusion_process": Port(
            name="diffusion_process", type=DiffusionProcess, required=False, default=None,
            doc="None = DiscreteLinearNoiseSchedule's default linear beta schedule, "
                "epsilon prediction, ComfyUI's calculate_input scaling -- same default as "
                "the main route's SupervisedLoRATrainerNode.",
        ),
        "gate_enabled": Port(
            name="gate_enabled", type=bool, required=False, default=False,
            doc="Off by default. See SupervisedLoRATrainerNode's identically-named port "
                "for the full explanation -- same mechanism, same nodes/model/lora.py functions.",
        ),
        "gate_train_low": Port(name="gate_train_low", type=float, required=False, default=0.0,
                                visible_when=("gate_enabled", True)),
        "gate_train_high": Port(name="gate_train_high", type=float, required=False, default=999.0,
                                 visible_when=("gate_enabled", True)),
        "gate_width": Port(name="gate_width", type=float, required=False, default=100.0,
                            visible_when=("gate_enabled", True)),
        "profile": Port(
            name="profile", type=bool, required=False, default=False,
            doc="Two lines per phase that manages a resident (text_encoder, optimizer): "
                "'[residency] NAME loaded: vram_reserved=XMB' / '... released: "
                "vram_reserved=YMB' -- watch reserved drop right after each release, "
                "which is the concrete, checkable claim this route's whole design makes. "
                "An earlier version of this only reported per-resident footprint from "
                "MonitoringPhase, which runs last in the step, after both phases above "
                "have already released everything they manage -- structurally could "
                "never show anything but ~0MB for either, regardless of whether release() "
                "was actually working. Fixed to report from inside those two phases "
                "instead, right when loaded/released actually happen. MonitoringPhase "
                "still prints one summary line per step (loss/lr/vram_reserved/model's own "
                "footprint) -- meaningful there since model stays resident throughout.",
        ),
        "empty_cache_every_n_steps": Port(
            name="empty_cache_every_n_steps", type=int, required=False, default=1,
            doc="See this class's own docstring for why this defaults to 1 (every step) "
                "instead of the main route's 0 -- only actually fires once "
                "AdaptiveResidencyController has decided to release something; a no-op "
                "for the rest of the run once it's decided the budget was never tight "
                "enough to need offloading at all (nothing to reclaim in that case, so "
                "paying for gc.collect()/empty_cache() every step would be pure waste -- "
                "a real, reported case, not a hypothetical one).",
        ),
        "calibration_steps": Port(
            name="calibration_steps", type=int, required=False, default=3,
            doc="AdaptiveResidencyController (see this module's own docstring) runs this "
                "many steps with everything resident first, measuring real peak VRAM, "
                "before deciding whether optimizer/text_encoder need to be released at "
                "all. Not the only protection against an unrepresentative sample -- the "
                "same controller keeps watching every step after that and escalates "
                "(releases one more candidate) if a later step's own peak exceeds budget, "
                "real insurance for e.g. a variable-resolution dataset where the largest "
                "image doesn't show up in the first few steps. Higher calibration_steps "
                "still means more confidence in the *initial* decision, and less of the "
                "run spent finding out the hard way via escalation; lower means less of "
                "the run spent paying for a release() cycle nothing ever needed.",
        ),
        "residency_safety_margin": Port(
            name="residency_safety_margin", type=float, required=False, default=0.1,
            doc="Shaves this fraction off the connected VRAM Budget Controller's own "
                "usable ceiling before AdaptiveResidencyController compares anything "
                "against it -- 0.1 (default) means a 12500MB budget (minus its own "
                "vram_reserve_mb) is really treated as 90% of that. Headroom for a "
                "somewhat-larger-than-calibrated step to still fit without needing to "
                "escalate at all, on top of escalation itself. 0.0 disables it -- the "
                "original, un-margined comparison.",
        ),
        "prewarm_text_encoder": Port(
            name="prewarm_text_encoder", type=bool, required=False, default=True,
            doc="**On by default since 2026-10-04**, which is a deliberate "
                "reversal -- measured on the B580, this is 1,562 MB of peak "
                "device memory for ~3.5 s of startup and ~6 MB of host RAM. "
                "One pass over the *same* `batches` object training "
                "will consume to discover every (prompt, batch_size, height, width) "
                "key, encode them all into a CachingTextEncoder wrapped around "
                "trainer.clip, then unload the encoder entirely -- CLIP's ~1.5GB "
                "leaves VRAM before calibration and never returns for the rest of the "
                "run; every subsequent step's encode is a cache hit (which also skips "
                "the per-step CLIP forward). This is the Resources Controller route's "
                "entry point for what PrewarmedTextEncoderNode (main route) already "
                "does -- that route has no encoder graph port to wire it through, "
                "which is exactly why the Port lives here, where trainer.clip and "
                "batches coexist (see nodes/model/text_encoder_prewarm.py's own "
                "docstring). Coordinate with EncodeConditioningPhase: when prewarmed, "
                "that phase skips its unconditional ensure_loaded('text_encoder') -- "
                "otherwise the first step would re-upload the encoder just unloaded "
                "and defeat the whole point -- while a genuine cache miss (dataset "
                "changed after warm-up) still self-loads through the cache's bound "
                "resource_control handle, so a miss degrades to a slow correct "
                "answer, never a wrong one. Composes with LoRATrainingConfigNode's "
                "`cache_text_encoder` (an existing wrap is kept, its handle "
                "late-bound; its own max_entries cap then applies to the warm pass) "
                "-- and without `cache_text_encoder`, this Port *is* what installs "
                "the cache. Safe to combine with residency escalation: an unloaded "
                "encoder has 0 footprint, so AdaptiveResidencyController just stops "
                "considering it (nothing left to release). Requires `batches` to be "
                "finite per iteration (one pass = one epoch, same as "
                "ManagedDatasetSourceNode's own output) -- and that assumption is "
                "now bounded rather than trusted, because a default cannot be "
                "allowed to hang: the discovery pass stops at "
                "`MAX_DISCOVERY_BATCHES` (100,000, far above any real dataset) and "
                "warns. Truncation costs time, not correctness, since the keys past "
                "it are misses and a miss is correct.",
        ),
        "grad_accum": Port(
            name="grad_accum", type=int, required=False, default=1,
            doc="Gradient accumulation: one optimizer step per this many batches. The "
                "outer `steps` count stays optimizer steps -- with grad_accum=4, a "
                "steps=1000 run consumes 4000 batches, one "
                "LR-schedule tick, one monitor report, one on_step call, and one "
                "grad_accum-window average per step. Each micro-step does its own "
                "fetch/forward/backward and frees its graph immediately, so peak VRAM "
                "is identical to grad_accum=1 (accumulation lives in .grad, not in "
                "held activations); effective batch = batch_size * grad_accum, which is "
                "the lever for batch-2 gradient noise (the legacy loop trained with "
                "grad_accum=6). Loss is divided by grad_accum per micro-step so the "
                "accumulated gradient is the window's mean. Works with fused and "
                "non-fused optimizers (fused via begin_step(sub_steps) + "
                "prepare_next_pass between passes). Deliberate difference from legacy "
                "core/train_step.py: there `steps` counted *micro*-steps (updates = "
                "steps/grad_accum); here it counts optimizer updates, so `steps` keeps "
                "meaning 'how many training updates' regardless of this value.",
        ),
        "grad_clip_max_norm": Port(
            name="grad_clip_max_norm", type=float, required=False, default=0.0,
            doc="0.0 disables. >0 clips gradient global norm to this at the optimizer-"
                "step boundary, before the update (torch.nn.utils.clip_grad_norm_ over "
                "the model's trainable parameters) -- the dampener for occasional "
                "large-gradient batches that this loop otherwise passes straight "
                "through. Non-fused optimizers only: a fused optimizer's update fires "
                "inside backward() itself, before clipping could run, so a fused "
                "optimizer with clip >0 is rejected at build time rather than "
                "silently not clipping. Not applied during accumulation windows' "
                "interior micro-steps -- only the boundary's full-window gradient "
                "gets clipped.",
        ),
        "save_every_n_steps": Port(
            name="save_every_n_steps", type=int, required=False, default=0,
            doc="0 disables. >0 writes a full LoRA safetensors every N optimizer steps "
                "(at the window boundary, so the file always reflects completed "
                "updates) to `<save_prefix>_<step:06d>.safetensors` in the LoRA "
                "directory, sandboxed through the same resolve_safe_model_path as "
                "LoRACheckpointSaverNode (subfolders allowed, '..'/absolute rejected). "
                "The piece a kill-at-step-N run needs: intermediate checkpoints to "
                "evaluate instead of one file at the end. Safe mid-training: "
                "trained_state_dict() only reads/detaches weights, and optimizer hooks "
                "only fire during a backward() -- saves happen between steps. Pass "
                "`project_layout` to redirect where files land (tests do; default = "
                "ProjectLayout.from_paths_module()).",
        ),
        "save_prefix": Port(
            name="save_prefix", type=str, required=False, default="lora_step",
            doc="Filename prefix for save_every_n_steps output, e.g. 'lora_step' -> "
                "'lora_step_000100.safetensors'. Must be non-empty; the rest of the "
                "path is sandboxed by resolve_safe_model_path like every other "
                "graph-reachable path.",
        ),
        "probe_every_n_steps": Port(
            name="probe_every_n_steps", type=int, required=False, default=0,
            doc="0 disables (default -- zero cost, zero behavior change). >0 runs the "
                "fixed-probe diagnostic (nodes/train/t_probe.py) every N optimizer steps, "
                "and once as soon as the probe items are captured: a handful of probe "
                "images re-noised with FIXED seeded noise at a fixed t grid, forward-only, "
                "LoRA live vs. the frozen base (LoRA gated to exactly zero). Publishes "
                "probe_rel_t_low/mid/high (= LoRA loss / base loss on identical inputs; "
                ">1 means this run made that t region worse than no LoRA), "
                "probe_drift_t_* and probe_worst_rel, and prints a per-t line. Unlike "
                "loss_t_* (random samples every step), a change between two probe "
                "records is the model's change, not sampling noise.",
        ),
        "probe_items": Port(
            name="probe_items", type=int, required=False, default=2,
            doc="How many training images the probe captures (element 0 of the first "
                "batches seen). More = less noisy probe, proportionally more forwards.",
        ),
        "probe_points_per_bucket": Port(
            name="probe_points_per_bucket", type=int, required=False, default=2,
            doc="Fixed t's per low/mid/high third. Forwards per probe = probe_items * 3 * "
                "this (plus the same again once, for the cached frozen-base reference).",
        ),
        "probe_grad_alignment": Port(
            name="probe_grad_alignment", type=bool, required=False, default=False,
            doc="Also measure, per t bucket, the gradient norm and the cosine between "
                "buckets' gradients (gc_* keys): the direct test of whether one t "
                "region's improvement is being paid for by another. One forward+backward "
                "per probe point, so use a coarse probe_every_n_steps. Non-fused "
                "optimizers only (a fused optimizer would apply the diagnostic backward "
                "as a real update). Read gc_self_* first -- it is the noise floor for "
                "every cross-bucket cosine.",
        ),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/"
                "layout.py. Only used by save_every_n_steps.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)

        trainer: LoRATrainingSkeleton = inputs["trainer"]
        model: TrainableModel = trainer.unet
        text_encoder: TextEncoder = trainer.clip
        batches: TrainingBatchSource = inputs["batches"]
        optimizer: OptimizerHandle = inputs["optimizer"]
        lr_schedule: LRSchedule = inputs["lr_schedule"]
        steps: int = inputs["steps"]
        resource_control: ResourceControlHandle = inputs["resource_control"]
        diffusion_process = inputs.get("diffusion_process") or DiffusionProcess(
            DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler())
        loss_weighting = inputs.get("loss_weighting") or UniformLossWeighting()
        bucket_balance = inputs.get("bucket_balance")  # None = no rebalancing
        profile: bool = inputs.get("profile", self.INPUTS["profile"].default)
        empty_cache_every_n_steps: int = inputs.get(
            "empty_cache_every_n_steps", self.INPUTS["empty_cache_every_n_steps"].default)
        calibration_steps: int = inputs.get(
            "calibration_steps", self.INPUTS["calibration_steps"].default)
        residency_safety_margin: float = inputs.get(
            "residency_safety_margin", self.INPUTS["residency_safety_margin"].default)
        prewarm_text_encoder: bool = inputs.get(
            "prewarm_text_encoder", self.INPUTS["prewarm_text_encoder"].default)
        grad_accum: int = inputs.get("grad_accum", self.INPUTS["grad_accum"].default)
        grad_clip_max_norm: float = inputs.get(
            "grad_clip_max_norm", self.INPUTS["grad_clip_max_norm"].default)
        save_every_n_steps: int = inputs.get(
            "save_every_n_steps", self.INPUTS["save_every_n_steps"].default)
        save_prefix: str = inputs.get("save_prefix", self.INPUTS["save_prefix"].default)
        project_layout = inputs.get("project_layout")
        probe_every: int = inputs.get(
            "probe_every_n_steps", self.INPUTS["probe_every_n_steps"].default)
        probe_items: int = inputs.get("probe_items", self.INPUTS["probe_items"].default)
        probe_points: int = inputs.get(
            "probe_points_per_bucket", self.INPUTS["probe_points_per_bucket"].default)
        probe_grad_alignment: bool = inputs.get(
            "probe_grad_alignment", self.INPUTS["probe_grad_alignment"].default)

        # Prewarm step 1/2 -- discover the exact keys and wrap/bind trainer.clip
        # BEFORE registration below: warming itself (step 2/2, after registration)
        # takes cache misses on an empty cache, and a miss calls
        # resource_control.ensure_loaded("text_encoder"), which needs the name
        # registered first. Discovery needs no encoder at all -- just the batches.
        prewarm_keys = None
        if prewarm_text_encoder:
            from ..model.text_encoder_cache import CachingTextEncoder
            from ..model.text_encoder_prewarm import (
                MAX_DISCOVERY_BATCHES,
                discover_dataset_keys,
                prompt_capacity,
                warm_and_unload,
            )
            prewarm = discover_dataset_keys(
                batches, max_batches=MAX_DISCOVERY_BATCHES)
            prewarm_keys = prewarm.keys
            if isinstance(text_encoder, CachingTextEncoder):
                # LoRATrainingConfigNode's cache_text_encoder wrap already in place
                # -- keep it (its max_entries applies), late-bind the handle it
                # couldn't have been given at config time.
                text_encoder.bind_resource_control(resource_control)
            else:
                # Capacity from the host-RAM budget, not from the dataset's
                # key count. Sizing it to the dataset made host RAM a
                # function of dataset size -- a 1M-caption dataset wanted
                # 606 GB, which is the coupling that stops this scaling at
                # all -- and warm_and_unload() now warms only what fits, so
                # the two decisions cannot disagree.
                text_encoder = CachingTextEncoder(
                    text_encoder, max_entries=max(prompt_capacity(), 1),
                    resource_control=resource_control)
                trainer.clip = text_encoder

        model.train()
        device = next(iter(model.trainable_parameters())).device
        is_fused = isinstance(optimizer, FusedOptimizerHandle)
        device_ctx = DeviceContext.for_device(str(device))

        # Contract checks, up front rather than deep in a phase: each of
        # these would otherwise either silently do nothing (clip on a
        # fused optimizer) or produce confusing behavior mid-run.
        if grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
        if grad_clip_max_norm < 0.0:
            raise ValueError(f"grad_clip_max_norm must be >= 0.0, got {grad_clip_max_norm}")
        if grad_clip_max_norm > 0.0 and is_fused:
            raise ValueError(
                "grad_clip_max_norm > 0 is incompatible with a fused optimizer: its "
                "update fires inside backward() (per-parameter hooks), before any "
                "post-backward clip could run -- clipping here would silently do "
                "nothing. Use a non-fused optimizer node with clipping, or "
                "grad_clip_max_norm=0 with the fused one.")
        if probe_every < 0:
            raise ValueError(f"probe_every_n_steps must be >= 0, got {probe_every}")
        if probe_every > 0 and probe_grad_alignment and is_fused:
            raise ValueError(
                "probe_grad_alignment is incompatible with a fused optimizer: its update "
                "fires inside backward(), so the diagnostic backward passes would be "
                "applied as real training steps on probe data. Use a non-fused optimizer "
                "node, or probe_grad_alignment=False (the forward-only probe is fine).")
        if save_every_n_steps < 0:
            raise ValueError(f"save_every_n_steps must be >= 0, got {save_every_n_steps}")
        if save_every_n_steps > 0 and not str(save_prefix).strip():
            raise ValueError("save_prefix must be non-empty when save_every_n_steps > 0")

        optimizer_id = describe_optimizer(optimizer)
        print(f"[ManagedLoRATrainerNode] optimizer: {optimizer_id}")

        # Sacrificable, not offloadable. `offloadable` would mean "released
        # between uses", which is wrong for the model -- it is used every step,
        # so `before_step()`'s safety net would evict it eagerly and pay a
        # 2,594 ms round trip to no purpose. Sacrificable means "moves only if
        # something asks for the room", which is exactly the conditioning
        # miss path. ForwardPhase's own ensure_loaded("model") is what brings
        # it back, and that obligation is the price of the capability.
        resource_control.register("model", model, offloadable=False,
                                  sacrificable=True)
        resource_control.register("optimizer", optimizer, offloadable=True)
        resource_control.register("text_encoder", text_encoder, offloadable=True)

        controller = AdaptiveResidencyController(
            usable_mb=resource_control.usable_budget_mb(),
            candidates={"optimizer": optimizer, "text_encoder": text_encoder},
            calibration_steps=calibration_steps,
            safety_margin=residency_safety_margin,
        )

        # Separate from resource_control -- tracking/reporting only (MonitoringPhase's
        # own per-resident footprint numbers), same split the main route's own build()
        # makes between its ResourceCoordinator and its optional resource_control.
        coordinator = ResourceCoordinator()
        coordinator.register("model", model)
        coordinator.register("optimizer", optimizer)
        coordinator.register("text_encoder", text_encoder)

        if prewarm_keys is not None:
            # Prewarm step 2/2 -- now that registration is in place (a warm-up
            # miss's ensure_loaded("text_encoder") finds the encoder already
            # resident here: reload skipped, just _make_room()'s measure),
            # fill the cache and unload for good. Imported at the top of the
            # prewarm block above, where prompt_capacity() came from too.
            warm_and_unload(text_encoder, prewarm)

        if save_every_n_steps > 0:
            from ..model.lora_saver import save_trained_weights

        monitor = inputs.get("monitor")
        phases: list[ManagedStepPhase] = [
            FetchBatchPhase(batches),
            PrepareDiffusionInputsPhase(
                diffusion_process,
                gate_enabled=inputs.get("gate_enabled", self.INPUTS["gate_enabled"].default),
                gate_train_low=inputs.get("gate_train_low", self.INPUTS["gate_train_low"].default),
                gate_train_high=inputs.get(
                    "gate_train_high", self.INPUTS["gate_train_high"].default),
                gate_width=inputs.get("gate_width", self.INPUTS["gate_width"].default)),
            EncodeConditioningPhase(text_encoder, resource_control, controller,
                                     device_ctx=device_ctx, profile=profile,
                                     ensure_loaded_before_encode=not prewarm_text_encoder),
            ZeroGradPhase(optimizer, lr_schedule, is_fused, grad_accum=grad_accum),
            ForwardPhase(resource_control),
            LossPhase(loss_weighting,
                      backward_scale=1.0 / grad_accum if grad_accum > 1 else 1.0,
                      bucket_balance=bucket_balance),
            BackwardAndOptimizerStepPhase(optimizer, is_fused, resource_control, controller,
                                           device_ctx=device_ctx, profile=profile,
                                           grad_accum=grad_accum,
                                           grad_clip_max_norm=grad_clip_max_norm),
            *([ProbePhase(TProbe(n_items=probe_items, points_per_bucket=probe_points,
                                 grad_alignment=probe_grad_alignment),
                          diffusion_process, probe_every, grad_accum=grad_accum)]
              if probe_every > 0 else []),
            MonitoringPhase(
                total_steps=steps, device_ctx=device_ctx, coordinator=coordinator,
                on_step=inputs.get("on_step"), monitor=monitor,
                profile=profile, optimizer_id=optimizer_id, grad_accum=grad_accum,
                usable_budget_mb=resource_control.usable_budget_mb(),
                bucket_balance=bucket_balance),
        ]
        pipeline = ManagedTrainingStepPipeline(phases)

        step = 0  # optimizer-step index; each step runs a grad_accum
        # window of micro-steps (one batch each) inside the loop below.
        while step < steps:
            if self.context.should_cancel():
                # run_end still reports so a dashboard can tell "cancelled,
                # data is final" from "hung or still going" -- SSE stays open
                # either way, so silence alone can't distinguish them.
                if monitor is not None:
                    monitor.report({"type": "run_end", "step": step,
                                    "cancelled": True, "t": time.time()})
                return {"model": model}
            for micro in range(grad_accum):
                state = ManagedStepState(step=step, batch=None, model=model,
                                         device=device, micro=micro)
                device_ctx.reset_peak_stats()  # so this micro-step's own peak is what gets
                # read below -- not a cumulative one since process/run start
                # (AdaptiveResidencyController's own docstring: this is what makes ongoing
                # escalation, not just one-shot calibration, possible -- a stale cumulative
                # peak would falsely look "still over budget" on every subsequent read after
                # the first time it was, even once escalation had already reacted to it).
                # Reset per micro-step even for grad_accum>1: an over-budget micro-step is
                # exactly what escalation needs to see, not just an over-budget window.
                resource_control.before_step(step * grad_accum + micro)  # safety net -- see
                # this class's own docstring
                pipeline.run_step(state)
                controller.record_step_peak(device_ctx.memory_stats())  # every micro-step,
                # not just during calibration -- see AdaptiveResidencyController's own
                # "keeps watching after deciding" docstring for why this can't stop once a
                # decision is made.
                micro_index = step * grad_accum + micro
                if (controller.releases_anything and empty_cache_every_n_steps > 0
                        and (micro_index + 1) % empty_cache_every_n_steps == 0):
                    # Gated on releases_anything -- see that property's own docstring.
                    # Reclaiming unused cached memory back to the driver is pointless work
                    # when nothing was ever released in the first place (a real, reported
                    # case: this loop was paying a full gc.collect()+empty_cache() pass
                    # every step even when the controller had already decided to keep
                    # everything resident, with nothing to actually reclaim).
                    gc.collect()
                    device_ctx.empty_cache()
            step += 1
            if save_every_n_steps > 0 and step % save_every_n_steps == 0:
                # Window boundary == completed updates only -- the file never
                # reflects a half-accumulated window. See save_every_n_steps's
                # own Port doc for why this is safe mid-training.
                saved_path = save_trained_weights(
                    model, f"{save_prefix}_{step:06d}.safetensors", project_layout)
                print(f"[ManagedLoRATrainerNode] saved step {step}/{steps} -> {saved_path}")

        if monitor is not None:
            # Normal completion -- the dashboard's "finished" state, which
            # an open-but-quiet SSE connection can't signal on its own
            # (it stays open after the run ends either way).
            monitor.report({"type": "run_end", "step": step,
                            "cancelled": False, "t": time.time()})
        result = {"model": model}
        self.validate_outputs(result)
        return result
