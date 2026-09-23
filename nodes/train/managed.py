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
from ..memory.control_handle import ResourceControlHandle
from ..memory.coordinator import ResourceCoordinator
from ..memory.handle import DeviceResident
from ..model.handle import TrainableModel
from ..model.lora_training_resources import LoRATrainingSkeleton
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import FusedOptimizerHandle, OptimizerHandle, describe_optimizer
from .loss import LossWeighting, UniformLossWeighting
from .node import TrainerNode
from .schedule import LRSchedule


@dataclass
class ManagedStepState:
    step: int
    batch: Optional[dict]
    model: TrainableModel
    device: Any
    extras: dict[str, Any] = field(default_factory=dict)


class ManagedStepPhase(ABC):
    @abstractmethod
    def run(self, state: ManagedStepState) -> ManagedStepState:
        ...


class ManagedTrainingStepPipeline:
    def __init__(self, phases: list[ManagedStepPhase]):
        self.phases = phases

    def run_step(self, state: ManagedStepState) -> ManagedStepState:
        for phase in self.phases:
            state = phase.run(state)
        return state


class AdaptiveResidencyController:
    """Decides, once, whether text_encoder/optimizer actually need to be
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
    high-water mark. If that peak already fits the budget, nothing ever
    gets released for the rest of the run -- full speed, same as the
    main route pays for the same reason. If it doesn't, candidates are
    released starting from the smallest footprint_bytes() first (a
    direct reading of "less performance costly options first" -- byte
    count is the one real, already-available number that's actually
    proportional to transfer cost, not a proxy for it), only as many as
    the shortfall actually needs, estimated by subtracting each
    candidate's own footprint_bytes() from the measured peak in that
    order. That subtraction is an approximation, not a
    guarantee -- offloading X doesn't necessarily save exactly X's own
    byte count at the *real* peak moment (timing, fragmentation, and
    other residents' own footprint at that instant all affect it) --
    which is exactly why this controller doesn't replace
    resource_control.before_step()'s own reactive check, it front-runs
    it: before_step() (and strict=True, if set) is still there as the
    real backstop for whatever this estimate gets wrong.

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
    choose between, so a few real, cheap calibration steps settle it
    directly rather than needing a model of the cost at all.

    Not addressed here, and deliberately not attempted in the same
    change as this controller: an algorithmic alternative to offloading
    at all (gradient checkpointing trades recompute time for reduced
    *activation* memory -- a different axis than moving *weights*
    on/off device, and this project already has real, working
    infrastructure for it -- ActivationCheckpointingStrategy,
    nodes/model/gradient_checkpointing.py -- plus a real, matching
    budget-driven placement policy, GreedyRatioPlacement,
    nodes/model/checkpoint_placement.py, for deciding *which UNet
    blocks* to checkpoint under a budget, ranked the same
    memory-per-cost way this controller ranks text_encoder vs.
    optimizer). That policy's own docs already flag it as real but
    unvalidated against an actual training run and deliberately not
    wired into any node's construction path yet, for exactly the "measure,
    don't guess" reason above -- extending that same caution to a second,
    newer piece built in this same session, on hardware this was never
    run on, felt like the right call rather than rushing it in under the
    same patch. A natural, real next step once someone can confirm the
    existing per-block placement work on real hardware -- not done here.
    """

    def __init__(self, usable_mb: Optional[float], candidates: dict[str, DeviceResident],
                 calibration_steps: int = 3):
        self._usable_mb = usable_mb
        self._candidates = candidates  # name -> resident, for footprint_bytes() at decision time
        self._calibration_steps = max(1, calibration_steps)
        self._steps_seen = 0
        self._peak_reserved_mb = 0.0
        self._decided: Optional[set] = None  # None while still calibrating

    @property
    def calibrating(self) -> bool:
        return self._decided is None

    def record_step_peak(self, memory_stats: Optional[dict]) -> None:
        """Call once per step, only during calibration -- a no-op once
        a decision has been made (nothing left to calibrate). Decides
        immediately, on the very first call, when there's no usable
        ceiling to plan against at all (usable_budget_mb() returned
        None -- ResourceControlHandle's own docstring: "no fixed
        ceiling concept") or no memory-stats concept on this device
        (memory_stats is None -- DeviceContext's own docstring: CPU,
        mainly) -- both cases mean there's nothing offloading could
        ever be measured against, so waiting calibration_steps for a
        number that will never arrive would just mean never deciding
        at all, staying fully resident is the only coherent answer
        either way."""
        if self._decided is not None:
            return
        if self._usable_mb is None or memory_stats is None:
            self._decided = set()
            return
        self._peak_reserved_mb = max(self._peak_reserved_mb, memory_stats["peak_reserved_mb"])
        self._steps_seen += 1
        if self._steps_seen >= self._calibration_steps:
            self._decide()

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
        verdict = ("releasing " + ", ".join(sorted(release))) if release \
            else "nothing -- staying fully resident for the rest of this run"
        print(f"[AdaptiveResidencyController] measured peak={self._peak_reserved_mb:.0f}MB "
              f"over {self._steps_seen} calibration step(s), usable budget="
              f"{self._usable_mb:.0f}MB -- {verdict}")

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
    the same LoRA timestep gate (core/lora.py) the main route's own
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
    never needed it). ensure_loaded() still runs unconditionally either
    way -- cheap and safe when nothing was ever offloaded, and correct
    if before_step()'s own reactive check offloaded this for some other
    reason between calls."""

    def __init__(self, text_encoder: TextEncoder, resource_control: ResourceControlHandle,
                 controller: "AdaptiveResidencyController",
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False):
        self._text_encoder = text_encoder
        self._resource_control = resource_control
        self._controller = controller
        self._device_ctx = device_ctx
        self._profile = profile

    def run(self, state: ManagedStepState) -> ManagedStepState:
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
    """LR update + gradient-buffer reset. No resource_control call here
    on purpose, unlike BackwardAndOptimizerStepPhase below -- checked
    directly against ComposedOptimizerHandle (nodes/optimizer/composed.py):
    zero_grad() only touches .grad on the model's own parameters
    (ExecutionStrategy.zero_grad(params)), update_lr() is pure Python
    float arithmetic -- neither one reads or writes the optimizer's own
    tracked state tensors (self.states), so this phase doesn't need
    optimizer state resident at all."""

    def __init__(self, optimizer: OptimizerHandle, lr_schedule: LRSchedule, is_fused: bool):
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._is_fused = is_fused

    def run(self, state: ManagedStepState) -> ManagedStepState:
        lr = self._lr_schedule.value(state.step)
        self._optimizer.update_lr(lr)
        if self._is_fused:
            self._optimizer.begin_step(sub_steps=1)
        else:
            self._optimizer.zero_grad()
        state.extras["lr"] = lr
        return state


class ForwardPhase(ManagedStepPhase):
    def run(self, state: ManagedStepState) -> ManagedStepState:
        state.extras["pred"] = state.model.forward(
            state.extras["xc"], state.extras["t"], state.extras["ctx_emb"], state.extras["y"])
        return state


class LossPhase(ManagedStepPhase):
    def __init__(self, loss_weighting: LossWeighting):
        self._loss_weighting = loss_weighting

    def run(self, state: ManagedStepState) -> ManagedStepState:
        pred = state.extras["pred"]
        target = state.extras["target"]
        sigma = state.extras["sigma"]
        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        weight = self._loss_weighting.weight(float(sigma.float().mean().item()))
        state.extras["loss"] = per_sample.mean() * weight
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
    """

    def __init__(self, optimizer: OptimizerHandle, is_fused: bool,
                 resource_control: ResourceControlHandle,
                 controller: "AdaptiveResidencyController",
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False):
        self._optimizer = optimizer
        self._is_fused = is_fused
        self._resource_control = resource_control
        self._controller = controller
        self._device_ctx = device_ctx
        self._profile = profile

    def run(self, state: ManagedStepState) -> ManagedStepState:
        self._resource_control.ensure_loaded("optimizer")
        self._log("loaded")
        state.extras["loss"].backward()
        if not self._is_fused:
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


class MonitoringPhase(ManagedStepPhase):
    """Runs last in the step -- after EncodeConditioningPhase and
    BackwardAndOptimizerStepPhase have already released everything
    they each manage, so per_resident_mb below will correctly show
    optimizer/text_encoder near 0 every step regardless of whether
    release() actually did anything: this phase runs too late to ever
    show their real peak. That's not this phase's job -- see those two
    phases' own profile=True lines (printed inline, right when loaded/
    released actually happen) for the number that's actually
    informative. What's still meaningful here: model's own footprint
    (steady, since it stays resident throughout) and one end-of-step
    summary line (loss/lr/vram_reserved_mb). Deliberately leaner than
    the main route's own MonitoringPhase (nodes/train/step_pipeline.py)
    beyond that -- not the main route's fuller set (per-phase timing,
    baseline deltas, a tracked-footprint cross-check against
    ResourceProfile). Someone wanting that level of detail can still
    build it the same way that file did; duplicating all of it here
    wasn't this file's job."""

    def __init__(self, total_steps: int, device_ctx: DeviceContext,
                 coordinator: ResourceCoordinator, on_step: Optional[Callable] = None,
                 monitor: Optional[MonitorHandle] = None, profile: bool = False,
                 optimizer_id: str = ""):
        self._total_steps = total_steps
        self._device_ctx = device_ctx
        self._coordinator = coordinator
        self._on_step = on_step
        self._monitor = monitor
        self._profile = profile
        self._optimizer_id = optimizer_id

    def run(self, state: ManagedStepState) -> ManagedStepState:
        loss_value = float(state.extras["loss"].item())
        lr = state.extras["lr"]

        if self._on_step is not None:
            self._on_step(state.step, loss_value)

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
        mem = self._device_ctx.memory_stats()
        if mem is not None:
            report["vram_reserved_mb"] = mem["reserved_mb"]
            report["vram_allocated_mb"] = mem["allocated_mb"]
        for name, mb in per_resident_mb.items():
            report[f"resident_{name}_mb"] = mb

        if self._monitor is not None:
            self._monitor.report(report)
        if self._profile:
            resident_part = " ".join(f"{name}={mb:.0f}MB" for name, mb in per_resident_mb.items())
            mem_part = (f" vram_reserved={mem['reserved_mb']:.0f}MB"
                        if mem is not None else "")
            optimizer_part = f" optimizer={self._optimizer_id}" if self._optimizer_id else ""
            print(f"  [step {state.step}]{optimizer_part} loss={loss_value:.4f} lr={lr:.2e}"
                  f"{mem_part} residents: {resident_part}")
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
                "for the full explanation -- same mechanism, same core/lora.py functions.",
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
                "instead of the main route's 0.",
        ),
        "calibration_steps": Port(
            name="calibration_steps", type=int, required=False, default=3,
            doc="AdaptiveResidencyController (see this module's own docstring) runs this "
                "many steps with everything resident first, measuring real peak VRAM, "
                "before deciding whether optimizer/text_encoder need to be released at "
                "all for the rest of the run. Higher -- more confidence the measured peak "
                "is representative (a batch with unusual content, or one-time allocator "
                "warmup on the very first step, could make a single step's own peak "
                "unrepresentative); lower -- less of the run spent paying whatever this "
                "decides against (nothing, if it turns out headroom was never needed).",
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
        profile: bool = inputs.get("profile", self.INPUTS["profile"].default)
        empty_cache_every_n_steps: int = inputs.get(
            "empty_cache_every_n_steps", self.INPUTS["empty_cache_every_n_steps"].default)
        calibration_steps: int = inputs.get(
            "calibration_steps", self.INPUTS["calibration_steps"].default)

        model.train()
        device = next(iter(model.trainable_parameters())).device
        is_fused = isinstance(optimizer, FusedOptimizerHandle)
        device_ctx = DeviceContext.for_device(str(device))
        device_ctx.reset_peak_stats()  # calibration below measures THIS run, not process history

        optimizer_id = describe_optimizer(optimizer)
        print(f"[ManagedLoRATrainerNode] optimizer: {optimizer_id}")

        resource_control.register("model", model, offloadable=False)
        resource_control.register("optimizer", optimizer, offloadable=True)
        resource_control.register("text_encoder", text_encoder, offloadable=True)

        controller = AdaptiveResidencyController(
            usable_mb=resource_control.usable_budget_mb(),
            candidates={"optimizer": optimizer, "text_encoder": text_encoder},
            calibration_steps=calibration_steps,
        )

        # Separate from resource_control -- tracking/reporting only (MonitoringPhase's
        # own per-resident footprint numbers), same split the main route's own build()
        # makes between its ResourceCoordinator and its optional resource_control.
        coordinator = ResourceCoordinator()
        coordinator.register("model", model)
        coordinator.register("optimizer", optimizer)
        coordinator.register("text_encoder", text_encoder)

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
                                     device_ctx=device_ctx, profile=profile),
            ZeroGradPhase(optimizer, lr_schedule, is_fused),
            ForwardPhase(),
            LossPhase(loss_weighting),
            BackwardAndOptimizerStepPhase(optimizer, is_fused, resource_control, controller,
                                           device_ctx=device_ctx, profile=profile),
            MonitoringPhase(
                total_steps=steps, device_ctx=device_ctx, coordinator=coordinator,
                on_step=inputs.get("on_step"), monitor=inputs.get("monitor"),
                profile=profile, optimizer_id=optimizer_id),
        ]
        pipeline = ManagedTrainingStepPipeline(phases)

        step = 0
        while step < steps:
            if self.context.should_cancel():
                return {"model": model}
            state = ManagedStepState(step=step, batch=None, model=model, device=device)
            resource_control.before_step(step)  # safety net -- see this class's own docstring
            pipeline.run_step(state)
            controller.record_step_peak(device_ctx.memory_stats())
            step += 1
            if empty_cache_every_n_steps > 0 and step % empty_cache_every_n_steps == 0:
                gc.collect()
                device_ctx.empty_cache()

        result = {"model": model}
        self.validate_outputs(result)
        return result
