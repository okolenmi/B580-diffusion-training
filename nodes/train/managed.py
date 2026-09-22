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

This route's own step loop instead treats residency as deterministic,
not reactive: each resident is loaded immediately before the one phase
that needs it and released immediately after, every step, budget
exceeded or not -- see EncodeConditioningPhase and
BackwardAndOptimizerStepPhase below for exactly which resident, which
window, and why. `resource_control.before_step()` still runs every step
too, as a safety net for whatever's registered non-offloadable (model,
here -- see ManagedLoRATrainerNode's own docstring for why) rather than
the sole mechanism.

Concretely, for a LoRA run: the frozen base dominates the model's own
footprint and is too expensive to move every step (a real multi-GB
transfer, likely making per-step offload/reload of the whole model
slower than the run it's meant to protect) -- it stays resident for the
run's duration, same conclusion the main route reaches, not a
carried-over assumption (see ManagedLoRATrainerNode's own docstring).
The optimizer's tracked state, by contrast, is proportional only to the
trainable LoRA parameters -- a small fraction of the base model's size
-- so moving it every step is cheap, and SDXL's two text encoders
(a full CLIP ViT-L/14 plus OpenCLIP ViT-bigG/14) are large enough on
their own that keeping them off GPU outside their one phase is a real,
not token, VRAM reduction for the step's most memory-hungry stretch
(forward+backward, activations included). Whether this is actually
*worth* the transfer cost it adds -- a real, disclosed step-time
tradeoff on top of the main route -- is exactly the thing this route
exists to let a person measure against the main one, not something this
file can decide on anyone's behalf.

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
    alone."""

    def __init__(self, text_encoder: TextEncoder, resource_control: ResourceControlHandle,
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False):
        self._text_encoder = text_encoder
        self._resource_control = resource_control
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
    """

    def __init__(self, optimizer: OptimizerHandle, is_fused: bool,
                 resource_control: ResourceControlHandle,
                 device_ctx: Optional[DeviceContext] = None, profile: bool = False):
        self._optimizer = optimizer
        self._is_fused = is_fused
        self._resource_control = resource_control
        self._device_ctx = device_ctx
        self._profile = profile

    def run(self, state: ManagedStepState) -> ManagedStepState:
        self._resource_control.ensure_loaded("optimizer")
        self._log("loaded")
        state.extras["loss"].backward()
        if not self._is_fused:
            self._optimizer.step(n_steps=1)
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
                "loaded only for their own phase and released right after, every step, "
                "not just under pressure -- see EncodeConditioningPhase/"
                "BackwardAndOptimizerStepPhase in this module for exactly which window "
                "and why. before_step() still runs every step too, as a safety net for "
                "model alone exceeding the budget -- set strict=True on the connected "
                "VRAMBudgetControllerNode to raise instead of continuing if that happens.",
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

        model.train()
        device = next(iter(model.trainable_parameters())).device
        is_fused = isinstance(optimizer, FusedOptimizerHandle)
        device_ctx = DeviceContext.for_device(str(device))

        optimizer_id = describe_optimizer(optimizer)
        print(f"[ManagedLoRATrainerNode] optimizer: {optimizer_id}")

        resource_control.register("model", model, offloadable=False)
        resource_control.register("optimizer", optimizer, offloadable=True)
        resource_control.register("text_encoder", text_encoder, offloadable=True)

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
            EncodeConditioningPhase(text_encoder, resource_control,
                                     device_ctx=device_ctx, profile=profile),
            ZeroGradPhase(optimizer, lr_schedule, is_fused),
            ForwardPhase(),
            LossPhase(loss_weighting),
            BackwardAndOptimizerStepPhase(optimizer, is_fused, resource_control,
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
            step += 1
            if empty_cache_every_n_steps > 0 and step % empty_cache_every_n_steps == 0:
                gc.collect()
                device_ctx.empty_cache()

        result = {"model": model}
        self.validate_outputs(result)
        return result
