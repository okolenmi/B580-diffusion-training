"""The trainer and its trigger."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

from ...data.batcher import Batcher
from ...engine.conditioner import Conditioner
from ...engine.devices import Accelerator
from ...engine.handles import TrainableDenoiser
from ...engine.optim.handle import FusedOptimizer, Optimizer
from ...engine.runtime.density import CheckpointDensity
from ...engine.runtime.placement import BlockPlacement
from ...engine.runtime.profiler import BlockProfiler
from ...engine.step.alignment import GradientAlignment
from ...engine.step.balance import TimeBalance
from ...engine.step.graphed import GraphedCompute
from ...engine.step.loss import LossWeighting
from ...engine.step.probe import TimeProbe
from ...engine.step.schedule import LRSchedule
from ...engine.step.policy import make_policy
from ...engine.step.regional import RegionalUpdate
from ...engine.step.trainer import Trainer
from ...graph import Completion, EventIn, EventOut, Group, Node, Port
from ...graph.events import CheckpointDue, StepFinished


class DiffusionTrainerNode(Node):
    DISPLAY_NAME = "Diffusion Trainer"
    CATEGORY = "Training"
    READS_SETTINGS = ("device.default", "runtime.profile", "perf.dir", "perf.ceilings", "perf.warmup_steps", "perf.steps", "perf.modules", "perf.trace", "perf.provenance")
    GROUPS = (Group("loop", "Loop"), Group("speed", "Speed", collapsed=True), Group("diagnostics", "Diagnostics", collapsed=True, advanced=True))
    INPUTS = {
        "batches": Port("batches", Batcher),
        "model": Port("model", TrainableDenoiser),
        "conditioner": Port("conditioner", Conditioner),
        "optimizer": Port("optimizer", Optimizer),
        "schedule": Port("schedule", LRSchedule),
        "weighting": Port("weighting", LossWeighting),
        "balance": Port("balance", TimeBalance, required=False, doc="Weights the loss by how each third of the timestep range is doing."),
        "steps": Port("steps", int, required=False, default=1000, minimum=0, group="loop",
                      doc="Optimizer steps. 0 means no step limit (set epochs). With both set, the run ends at whichever comes first."),
        "epochs": Port("epochs", int, required=False, default=0, minimum=0, group="loop",
                       doc="Passes over the data (an epoch is the batcher's batches; with accumulation, steps are batches / accumulation). "
                           "0 means no epoch limit. With both set, the run ends at whichever comes first."),
        "accumulation": Port("accumulation", int, required=False, default=1, minimum=1, group="loop",
                             doc="Batches per optimizer step; their gradients average."),
        "start_step": Port("start_step", int, required=False, default=0, minimum=0, group="loop",
                           doc="Step to start from when resuming (the schedule and the logs continue from it)."),
        "tune_checkpointing": Port("tune_checkpointing", bool, required=False, default=False, group="speed",
                                   doc="Before step 1, measure what the card allows and recompute fewer transformer blocks in backward "
                                       "(faster steps) as long as the largest batch still fits the memory budget."),
        "checkpoint_by": Port("checkpoint_by", str, required=False, default="share", choices=("share", "measured"), group="speed",
                              visible_when=("tune_checkpointing", True),
                              doc="How blocks are picked. share: an even share of the transformer blocks (residual blocks always recompute). "
                                  "measured: the step times each block's recompute and counts its activation memory, then keeps the blocks "
                                  "that save the most time per megabyte, residual blocks included, as far as the largest batch fits."),
        "probe_every": Port("probe_every", int, required=False, default=0, minimum=0, group="diagnostics",
                            doc="Every N steps, measure whether the adapter makes any third of the timestep range worse than the "
                                "frozen model on fixed training samples (probe_rel_t_*: above 1 is worse; 0 = off). Costs a few "
                                "batch-1 forwards each time."),
        "probe_alignment": Port("probe_alignment", bool, required=False, default=False, group="diagnostics",
                                doc="With the probe: also measure whether the timestep thirds pull the adapter the same way "
                                    "(gc_*: gradient norm share, cosines between thirds, cosine with the sum, split-half noise floor). "
                                    "Costs a backward per probe point at every probe."),
        "probe_items": Port("probe_items", int, required=False, default=2, minimum=1, group="diagnostics",
                            doc="Training samples the probe keeps (the first ones seen)."),
        "probe_points": Port("probe_points", int, required=False, default=2, minimum=1, group="diagnostics",
                             doc="Fixed timesteps per third of the range."),
        "graph": Port("graph", bool, required=False, default=False, group="speed",
                      doc="Replay forward and backward from a captured device graph once a shape has been seen a few times: "
                          "about twice as fast per step where the step is bound by kernel launches. Costs memory for the graph "
                          "(shared by the shapes) and needs no dropout, no gate and no fused optimizer. Shapes that cannot be "
                          "captured keep running eagerly."),
        "graph_attention": Port("graph_attention", str, required=False, default="math", choices=("math", "math_bf16", "chunked_bf16"), group="speed",
                                visible_when=("graph", True),
                                doc="The attention a captured graph runs (the default fast kernel cannot be captured). math: the stock math backend (fp32 products, exact). "
                                    "math_bf16: the same with bf16 products. chunked_bf16: bf16 products over blocks of query rows, probabilities recomputed in backward "
                                    "(no score matrix kept). The bf16 modes round the scores before the softmax; compare the loss with math over a few hundred steps."),
        "gradient_policy": Port("gradient_policy", str, required=False, default="mean", choices=("mean", "project", "common"), group="loop",
                                doc="How the regions of a batch (see the noiser's `regions`) are reconciled. mean: one batch, one loss. "
                                    "project: a region's gradient loses the part that points against another region's. "
                                    "common: the shortest vector among the regions' gradients, which lowers every region's loss to first order. "
                                    "The last two run each region's backward alone (more memory-light but slower) and need the plain optimizer, no graph."),
        "policy_normalize": Port("policy_normalize", bool, required=False, default=True, group="loop",
                                 doc="Scale each region's gradient to unit length before combining, so heavily down-weighted regions count as much as the rest."),
        "policy_floor": Port("policy_floor", float, required=False, default=0.0, minimum=0.0, maximum=1.0, group="loop",
                             visible_when=("gradient_policy", "common"),
                             doc="Skip the update when the shared direction keeps less than this share of the average squared gradient norm "
                                 "(1 = regions identical, 1 / regions = unrelated, 0 = they cancel). 0 never skips."),
        "fused_geglu": Port("fused_geglu", bool, required=False, default=True, group="speed",
                            doc="Run the feed-forward activation (split, gelu, multiply, and their gradient) as one kernel pass each way, from the "
                                "xpu-kernels package. The kernel checks itself against the eager chain first and stays off, with a warning, if it differs."),
        "graph_shapes": Port("graph_shapes", int, required=False, default=8, minimum=1, group="speed", visible_when=("graph", True),
                             doc="Most shapes that get a graph; the rest run eagerly."),
        "timing": Port("timing", bool, required=False, default=False, group="diagnostics",
                       doc="Measure each phase of the step (synchronizes the device, so it slows training slightly)."),
    }
    OUTPUTS = {"model": Port("model", TrainableDenoiser, doc="The model after training (or after stopping).")}
    EVENTS_OUT = {"step_finished": EventOut(StepFinished, doc="After every optimizer step."),
                  "ended": EventOut(CheckpointDue, doc="Once when training ends, finished or stopped: wire it to a saver for the final file.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        model: TrainableDenoiser = inputs["model"]
        accelerator = Accelerator(model.device)
        self._warn_slow_int8(model, inputs)
        self._enable_speed_options(model, inputs)
        residency = getattr(self.context.runtime, "residency", None)
        if residency is not None:
            self._claim_memory(residency, model, inputs["optimizer"])
        density, placement = self._checkpoint_choice(residency, model, accelerator, inputs)
        graph = self._graph_factory(inputs, model, accelerator, residency) if inputs.get("graph") else None
        total = self._total_steps(inputs)
        inputs["schedule"].fit(total)
        trainer = Trainer(batches=inputs["batches"], denoiser=model.denoiser, conditioner=inputs["conditioner"],
                          optimizer=inputs["optimizer"], schedule=inputs["schedule"], weighting=inputs["weighting"],
                          steps=total, accumulation=inputs.get("accumulation", 1), device=model.device,
                          gate=model.gate, gate_window=model.gate_window,
                          emit=lambda event: self.emit("step_finished", event), should_stop=self.context.should_cancel,
                          synchronize=accelerator.synchronize,
                          timed=bool(inputs.get("timing") or self.context.settings.get("runtime.profile")),
                          residency=residency, density=density, placement=placement, balance=inputs.get("balance"),
                          probe=self._time_probe(model, inputs), task=self.context.task, graph=graph,
                          plan_chunks=getattr(getattr(self.context.runtime, "determinism", None), "plans_chunks", True),
                          perf=self._perf_session(model, accelerator), regional=self._regional(inputs),
                          warn=lambda message: self.context.report("warning", {"node": self.uid, "message": message}))
        start = inputs.get("start_step") or 0
        with self._attention_for_determinism(model, inputs):
            result = trainer.run(start)
        self.emit("ended", CheckpointDue(start + result.steps_done, "stopped" if result.stopped else "finished"))
        if result.stopped:
            self.report_completion(Completion.unfinished_at(start + result.steps_done, result.total))
        return {"model": model}

    @staticmethod
    def _regional(inputs: dict) -> RegionalUpdate | None:
        name = inputs.get("gradient_policy") or "mean"
        if name == "mean":
            return None
        if inputs["batches"].noiser.regions < 2:
            raise ValueError(f"gradient policy {name!r} reconciles regions: set the noiser's regions to 2 or more")
        policy = make_policy(name, inputs.get("policy_floor") or 0.0, int(getattr(inputs["batches"], "seed", 0)))
        return RegionalUpdate(policy, bool(inputs.get("policy_normalize", True)))

    def _attention_for_determinism(self, model: TrainableDenoiser, inputs: dict):
        """Deterministic kernels leave the stock attention for one that materialises the whole score matrix, which does not
        fit at large sizes; blockwise attention is deterministic and keeps the memory low. A graph run picks its own."""
        determinism = getattr(self.context.runtime, "determinism", None)
        scopes = getattr(model.denoiser, "attention_scopes", lambda: {})()
        if not (getattr(determinism, "enabled", False) and not inputs.get("graph") and "chunked_bf16" in scopes):
            return contextlib.nullcontext()
        self._warn("deterministic mode: attention runs blockwise with bf16 scores (the stock kernel's deterministic form needs far more memory)")
        return scopes["chunked_bf16"]()

    def _warn(self, message: str) -> None:
        self.context.report("warning", {"node": self.uid, "message": message})

    def _warn_slow_int8(self, model: TrainableDenoiser, inputs: dict) -> None:
        if (model.info or {}).get("base_precision") == "int8" and not inputs.get("graph"):
            self._warn("an int8 base launches about twice the kernels per layer: measured about 37% slower than native "
                       "per eager step, about 10% with graph replay (it saves about 2 GB)")

    def _enable_speed_options(self, model: TrainableDenoiser, inputs: dict) -> None:
        """Switch on what the model family offers for speed; an option the family lacks stays off, with a warning."""
        if inputs.get("fused_geglu"):
            enable = getattr(model.denoiser, "enable_fused_geglu", None)
            reason = enable(model.device) if enable else "this model family has no fused GEGLU"
            if reason:
                self._warn(f"fused_geglu stays off: {reason}")

    @staticmethod
    def _claim_memory(residency, model: TrainableDenoiser, optimizer: Optimizer) -> None:
        """Everything that stays on the device for the whole run is claimed first, then the cache is handed back once."""
        residency.claim("model", model, rank=10, pinned=True,
                        purpose="frozen weights, adapter factors and whatever else the denoiser keeps")
        residency.claim("optimizer", optimizer, rank=1, purpose="optimizer state (moments), one set per adapter factor")
        residency.claim("gradients", optimizer.hold_grads(), pinned=True,
                        purpose="adapter gradients, allocated once and zeroed in place so no step allocates between them")
        residency.settle()
        residency.start()

    @staticmethod
    def _checkpoint_choice(residency, model: TrainableDenoiser, accelerator: Accelerator, inputs: dict) -> tuple[tuple | None, tuple | None]:
        """``(density, placement)``: who picks which blocks recompute in backward, each paired with the model, or None."""
        if not (inputs.get("tune_checkpointing") and residency is not None and model.checkpointer is not None):
            return None, None
        if inputs.get("checkpoint_by") == "measured":
            profiler = BlockProfiler(model.checkpointer, model.checkpoint_blocks(),
                                     lambda: accelerator.memory().get("allocated", 0.0), accelerator.synchronize)
            return None, (BlockPlacement(residency.planner, profiler), model)
        return (CheckpointDensity(residency.planner), model), None

    @staticmethod
    def _time_probe(model: TrainableDenoiser, inputs: dict) -> TimeProbe | None:
        if not inputs.get("probe_every"):
            return None
        batches: Batcher = inputs["batches"]
        alignment = GradientAlignment(model.adapted.trainable_parameters, model.denoiser.module) if inputs.get("probe_alignment") else None
        return TimeProbe(denoiser=model.denoiser, conditioner=getattr(inputs["conditioner"], "inner", inputs["conditioner"]), gate=model.gate,
                         noiser=batches.noiser, item_of=batches.clean_item, device=model.device, every=inputs["probe_every"],
                         items=inputs.get("probe_items") or 2, points=inputs.get("probe_points") or 2, seed=batches.seed, alignment=alignment)

    def _perf_session(self, model: TrainableDenoiser, accelerator: Accelerator):
        settings = self.context.settings
        folder = settings.get("perf.dir")
        if not folder:
            return None
        from ...perf.session import PerfSession  # only a profiling run pays for the import
        return PerfSession(model.denoiser.module(), str(model.device), folder, warmup=int(settings.get("perf.warmup_steps")),
                           steps=int(settings.get("perf.steps")), modules=bool(settings.get("perf.modules")),
                           trace=bool(settings.get("perf.trace")), provenance=bool(settings.get("perf.provenance")), ceilings_path=settings.get("perf.ceilings") or None, say=lambda text: self.context.report("perf", {"node": self.uid, "message": text}),
                           meta={"base_precision": (model.info or {}).get("base_precision", "native")})

    @staticmethod
    def _total_steps(inputs: dict) -> int:
        """Optimizer steps of the run: the step limit, the epoch limit (batches per epoch / accumulation), whichever is smaller."""
        limits = []
        if inputs.get("steps", 1000):
            limits.append(int(inputs.get("steps", 1000)))
        if inputs.get("epochs"):
            limits.append(max(1, int(inputs["epochs"]) * len(inputs["batches"]) // (inputs.get("accumulation") or 1)))
        if not limits:
            raise ValueError("DiffusionTrainer: set steps or epochs")
        return min(limits)

    def _graph_factory(self, inputs: dict, model: TrainableDenoiser, accelerator: Accelerator, residency=None):
        """What the Trainer calls with the eager phases to get the phase that replays a captured graph; refuses the
        configurations a replay would silently get wrong."""
        optimizer = inputs["optimizer"]
        GraphedCompute.refuse_unless_capturable(model, optimizer, isinstance(optimizer, FusedOptimizer))
        xpu = getattr(torch, "xpu", None)
        capturable = accelerator.device.split(":")[0] == "xpu" and hasattr(xpu, "XPUGraph")
        if not capturable:
            self.context.report("graph", {"node": self.uid, "active": False, "why": "this device cannot capture graphs"})
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # The default kernel (flash) uses work-group scratch memory, which graph capture does not support; the memory-efficient
        # backend is not implemented on XPU and would only warn and fall back to this one.
        scopes = getattr(model.denoiser, "attention_scopes", lambda: {})()
        mode = inputs.get("graph_attention") or "math"
        if mode != "math" and mode not in scopes:
            raise ValueError(f"DiffusionTrainer: this model has no graph attention mode {mode!r}")
        attentions = [scopes.get(mode) or (lambda: sdpa_kernel(SDPBackend.MATH))]
        limit = inputs.get("graph_shapes") or 8
        counts = inputs["batches"].shape_counts()
        wanted = frozenset(shape for shape, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit])  # the shapes most steps use
        return lambda eager: GraphedCompute(wanted=wanted, eager=eager, denoiser=model.denoiser, weighting=inputs["weighting"], balance=inputs.get("balance"),
                                            optimizer=optimizer, accumulation=inputs.get("accumulation", 1),
                                            graph_api=xpu if capturable else None, synchronize=accelerator.synchronize,
                                            max_graphs=limit, attentions=attentions, clock=time.perf_counter,
                                            fits=residency.graph_fits if residency is not None else None,
                                            used=lambda: accelerator.memory().get("device_used", 0.0),
                                            say=lambda text: self.context.report("graph", {"node": self.uid, "message": text}))


class EveryEpochNode(Node):
    """Turns the end of an epoch into a checkpoint-due event (in addition to Every N Steps; wire either or both)."""

    DISPLAY_NAME = "Every Epoch"
    CATEGORY = "Training"
    INPUTS = {"every": Port("every", int, default=1, minimum=1, doc="Emit after every this many epochs.")}
    OUTPUTS = {"interval": Port("interval", int)}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.every = inputs["every"]
        return {"interval": self.every}

    def on_step(self, event: StepFinished, ctx) -> None:
        epoch = event.metrics.get("epoch", 0)
        if event.metrics.get("epoch_end") and (epoch + 1) % self.every == 0:
            ctx.emit("due", CheckpointDue(event.step, f"epoch {epoch + 1}"))


class EveryNStepsNode(Node):
    """Turns step events into checkpoint-due events."""

    DISPLAY_NAME = "Every N Steps"
    CATEGORY = "Training"
    INPUTS = {"every": Port("every", int, default=250, minimum=1, doc="Emit after every this many optimizer steps.")}
    OUTPUTS = {"interval": Port("interval", int)}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.every = inputs["every"]
        return {"interval": self.every}

    def on_step(self, event: StepFinished, ctx) -> None:
        if event.step % self.every == 0:
            ctx.emit("due", CheckpointDue(event.step, f"every {self.every} steps"))
