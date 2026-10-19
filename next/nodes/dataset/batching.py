"""Bucketer, Noiser and Batcher: how samples are grouped, noised and served."""

from __future__ import annotations

from typing import Any

from ...data.batcher import Batcher
from ...data.bucketing import BucketPolicy, padding_stats
from ...data.noise import Noiser, TimestepPolicy
from ...data.padding import FILLS
from ...engine.step.balance import TimeBalance
from ...families.api.parts import DenoiserPart
from ...families.registry import get_family
from ...graph import EventOut, Group, Node, Port
from ...graph.events import EpochEnding, EpochStarted, PromptsChanged
from .handles import LatentSet, PromptTable


class BucketerNode(Node):
    DISPLAY_NAME = "Bucketer"
    CATEGORY = "Dataset"
    INPUTS = {
        "latents": Port("latents", LatentSet),
        "multiple": Port("multiple", int, required=False, default=1, minimum=1, unit="latent px",
                         doc="Round every latent side up to this multiple and pad; 1 keeps shapes as they are."),
        "max_shapes": Port("max_shapes", int, required=False, default=0, minimum=0,
                           doc="At most this many padded shapes: the cheapest ones are merged into a common size until it holds. "
                               "Every shape the model meets for the first time costs seconds and driver memory; 0 keeps them all."),
        "batch_area": Port("batch_area", float, required=False, default=0.0, minimum=0.0, unit="x base area",
                           doc="Samples up to this many times the area of the base resolution's square run in full batches; a batch holds at "
                               "most batch size times that area in total, so bigger samples run in smaller batches, down to one. Bounds the memory a step can take. 0 = no limit."),
        "pack": Port("pack", float, required=False, default=0.3, minimum=0.0, maximum=0.9, unit="share of area",
                     doc="Fill short batches: samples left over after a shape's full batches are moved into a larger shape's short batch "
                         "(or combined into one batch of their bounding shape) when no sample ends up more than this share padding. "
                         "Padding is masked out of the loss and filled with noise. The batch-area limit still holds; without one, no batch gets a shape "
                         "larger than the largest existing one. 0 = off."),
        "fill": Port("fill", str, required=False, default="zero", choices=FILLS,
                     doc="What the padded part of the input holds. zero: a blank strip (noised like the rest). mirror: a reflection of the "
                         "sample's own noisy input, so normalisation and attention see image-like content; it adds no information. "
                         "The loss ignores the padding in both cases."),
        "keep_incomplete": Port("keep_incomplete", bool, required=False, default=False,
                                doc="Train on a short last batch of a group instead of dropping it (when shuffling)."),
    }
    OUTPUTS = {"buckets": Port("buckets", BucketPolicy)}

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        latents = inputs.get("latents")
        if latents is None:
            return {}
        stats = padding_stats(list(latents.shapes.values()), inputs.get("multiple") or 1, inputs.get("max_shapes") or 0)
        if stats["samples"] and stats["padded_samples"]:
            return {"multiple": [f"{stats['padded_samples']}/{stats['samples']} samples padded; mean padding "
                                 f"{stats['pad_mean']:.0%}, worst {stats['pad_max']:.0%}"]}
        return {}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"buckets": BucketPolicy(multiple=inputs.get("multiple") or 1, keep_incomplete=bool(inputs.get("keep_incomplete")),
                                         max_shapes=inputs.get("max_shapes") or 0, batch_area=float(inputs.get("batch_area") or 0.0),
                                         pack=float(inputs.get("pack", 0.3)), fill=inputs.get("fill") or "zero")}


class NoiserNode(Node):
    DISPLAY_NAME = "Noiser"
    CATEGORY = "Dataset"
    GROUPS = (Group("timesteps", "Timesteps"),)
    INPUTS = {
        "denoiser": Port("denoiser", DenoiserPart, doc="Tells which family's objective to use."),
        "parameterization": Port("parameterization", str, required=False, default="eps", choices=("eps", "v")),
        "zero_terminal_snr": Port("zero_terminal_snr", bool, required=False, default=False,
                                  doc="Rescale the schedule so the last step is pure noise (v-prediction)."),
        "mode": Port("mode", str, required=False, default="uniform", group="timesteps",
                     choices=("uniform", "low", "mid", "high", "logit")),
        "low": Port("low", float, required=False, default=None, group="timesteps", doc="Smallest timestep; empty = objective's."),
        "high": Port("high", float, required=False, default=None, group="timesteps", doc="Largest timestep; empty = objective's."),
        "balance": Port("balance", TimeBalance, required=False, doc="Steers the draw towards the thirds of the range that are behind."),
        "values": Port("values", str, required=False, default="", group="timesteps", advanced=True,
                       doc="Comma-separated timesteps to choose from instead of a distribution."),
    }
    OUTPUTS = {"noiser": Port("noiser", Noiser)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        family = get_family(inputs["denoiser"].family_id)
        objective = family.objective(inputs.get("parameterization") or "eps", bool(inputs.get("zero_terminal_snr")))
        values = tuple(float(v) for v in (inputs.get("values") or "").split(",") if v.strip())
        policy = TimestepPolicy(inputs.get("mode") or "uniform", inputs.get("low"), inputs.get("high"), values)
        return {"noiser": Noiser(objective, policy, inputs.get("balance"))}


class BatcherNode(Node):
    DISPLAY_NAME = "Batcher"
    CATEGORY = "Dataset"
    READS_SETTINGS = ("seed.base",)
    GROUPS = (Group("batching", "Batching"), Group("worker", "Worker", collapsed=True))
    INPUTS = {
        "latents": Port("latents", LatentSet),
        "prompts": Port("prompts", PromptTable),
        "noiser": Port("noiser", Noiser),
        "buckets": Port("buckets", BucketPolicy, required=False),
        "batch_size": Port("batch_size", int, required=False, default=1, minimum=1, group="batching"),
        "shuffle": Port("shuffle", bool, required=False, default=True, group="batching"),
        "seed": Port("seed", int, required=False, minimum=0, group="batching",
                     doc="One number for data order, padding offsets, timesteps and noise; empty derives it from the graph's base seed."),
        "start_epoch": Port("start_epoch", int, required=False, default=0, minimum=0, group="batching",
                            doc="Epoch to start from when resuming; prompts and data order follow from it."),
        "prefetch": Port("prefetch", int, required=False, default=2, minimum=1, group="worker",
                         doc="Batches prepared ahead of the training step."),
    }
    OUTPUTS = {"batches": Port("batches", Batcher, doc="Iterable of batches; one pass is one epoch.")}
    EVENTS_OUT = {
        "epoch_started": EventOut(EpochStarted, doc="Before the epoch's prompts are computed; handlers may edit captions."),
        "epoch_ending": EventOut(EpochEnding, doc="After the last batch has been handed out."),
        "prompts_changed": EventOut(PromptsChanged, doc="The active prompts differ from the previous epoch's."),
    }

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        buckets = inputs.get("buckets") or BucketPolicy()
        port = {EpochStarted: "epoch_started", EpochEnding: "epoch_ending", PromptsChanged: "prompts_changed"}
        emit = lambda event: self.emit(port[type(event)], event)  # noqa: E731
        prompts: PromptTable = inputs["prompts"]
        prompts.listener = emit
        batcher = Batcher(inputs["latents"], prompts, inputs["noiser"], batch_size=inputs.get("batch_size") or 1,
                          shuffle=inputs.get("shuffle", True), keep_incomplete=buckets.keep_incomplete,
                          multiple=buckets.multiple, max_shapes=buckets.max_shapes, batch_area=buckets.batch_area, pack=buckets.pack, fill=buckets.fill, seed=self.derived_seed(inputs.get("seed")), depth=inputs.get("prefetch") or 2,
                          start_epoch=inputs.get("start_epoch") or 0, emit=emit)
        self.context.report("buckets", {"node": self.uid, **batcher.shape_summary()})
        return {"batches": batcher}
