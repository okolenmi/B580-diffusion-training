"""Ready-made graphs. Each returns a ``GraphDoc`` that the editor, the headless runner and the tests all use."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Edge, GraphDoc, NodeSpec


@dataclass
class _Builder:
    nodes: list[tuple[str, str, dict]] = field(default_factory=list)
    wires: list[tuple[str, str, str, str]] = field(default_factory=list)
    events: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add(self, label: str, class_name: str, **params) -> None:
        self.nodes.append((label, class_name, params))

    def uid(self, name: str) -> str:
        return "n_" + "abcdefghijklmnopqrstuvwxyz"[[n for n, _, _ in self.nodes].index(name)] * 6

    def doc(self, settings: dict) -> GraphDoc:
        edge = lambda w: Edge(self.uid(w[0]), w[1], self.uid(w[2]), w[3])  # noqa: E731
        specs = tuple(NodeSpec(self.uid(n), cls, params, title=n) for n, cls, params in self.nodes)
        return GraphDoc(specs, tuple(map(edge, self.wires)), tuple(map(edge, self.events)), settings)


@dataclass(frozen=True)
class SdxlLoraOptions:
    """The choices of the SDXL LoRA starter, with the defaults the command line also uses. Optional features are off at their defaults."""

    name: str = "lora"
    resolution: int = 1024
    batch_size: int = 1
    accumulation: int = 1
    steps: int = 1000
    epochs: int = 0
    save_every: int = 250
    save_every_epoch: int = 0
    save_probe: bool = False
    lr: float = 1e-4
    schedule: str = "constant"
    warmup: int = 0
    optimizer: str = "AdamWNode"
    fused: bool = False
    rank: int = 16
    alpha: float | None = None
    lora_plus: float = 1.0
    checkpointing: bool = True
    graph: bool = False
    persist_conditioning: bool = False
    max_shapes: int = 16
    batch_area: float = 0.0
    pack: float = 0.3
    fill: str = "zero"
    auto_caption: bool = False
    tag_swap: bool = False
    time_balance: str = "off"
    balance_sampling: float = 0.0
    regions_from: int = 0
    regions_span: int = 120
    batch_first: int = 0
    batch_first_steps: int = 200
    ema: str = ""
    aligned_init: int = 0


class _SdxlLora:
    """Builds the SDXL LoRA graph in groups: data, model and optimizer, trainer and saver, then the optional extras."""

    SCHEDULES = {"constant": "ConstantLRNode", "cosine": "CosineLRNode"}

    def __init__(self, model: str, dataset: str, output: str, options: SdxlLoraOptions, source: tuple[str, dict] | None):
        self.o, self.dataset, self.output, self.source, self.model = options, dataset, output, source, model
        self.b = _Builder()
        self.prompts = "tag_swap" if options.tag_swap else "captions"

    def build(self, settings: dict | None, overrides: dict[str, dict] | None) -> GraphDoc:
        self._data()
        self._model_and_optimizer()
        self._training_and_saving()
        self._extras()
        self._override(overrides or {})
        return self.b.doc(settings or {})

    def _data(self) -> None:
        o, b, dataset = self.o, self.b, self.dataset
        src_class, src_params = self.source or ("LoadModelNode", {"file": self.model})
        b.add("model", src_class, **src_params)
        b.add("images", "DatasetLibraryNode", dataset=dataset)
        b.add("latents", "LatentCacheNode", dataset=dataset, base=o.resolution, mode="fit")
        b.add("captions", "CaptionsNode")
        if o.auto_caption:  # fills empty captions in the library first, then the prompts read them
            b.add("autocaption", "AutoCaptionNode", dataset=dataset)
        if o.tag_swap:  # reduces every caption to one random tag per epoch: an experiment, not a default
            b.add("tag_swap", "TagSwapNode")
        b.add("buckets", "BucketerNode", multiple=4, max_shapes=o.max_shapes, batch_area=o.batch_area, pack=o.pack, fill=o.fill, keep_incomplete=True)  # 32 px steps, padded and masked
        b.add("noiser", "NoiserNode")
        b.add("batcher", "BatcherNode", batch_size=o.batch_size)
        b.add("blend", "BlendConditioningNode")  # identity until its strength is raised
        b.add("conditioner", "ConditionerNode", persist=o.persist_conditioning, **({"dataset": dataset} if o.persist_conditioning else {}))
        captioned = [("images", "images", "autocaption", "images"), ("autocaption", "images", "captions", "images")] if o.auto_caption \
            else [("images", "images", "captions", "images")]
        if o.tag_swap:
            b.wires.append(("captions", "prompts", "tag_swap", "prompts"))
        b.wires += [*captioned, ("model", "vae", "latents", "vae"), ("images", "images", "latents", "images"),
                    ("latents", "latents", "buckets", "latents"),
                    ("model", "denoiser", "noiser", "denoiser"), ("latents", "latents", "batcher", "latents"),
                    (self.prompts, "prompts", "batcher", "prompts"), ("noiser", "noiser", "batcher", "noiser"),
                    ("buckets", "buckets", "batcher", "buckets"), ("model", "clip_l", "conditioner", "encoders"),
                    ("model", "clip_g", "conditioner", "encoders"), (self.prompts, "prompts", "conditioner", "prompts")]
        b.events.append(("batcher", "prompts_changed", "conditioner", "prompts_changed"))

    def _model_and_optimizer(self) -> None:
        o, b = self.o, self.b
        b.add("adapter", "InjectAdapterNode", rank=o.rank, alpha=float(o.alpha if o.alpha is not None else o.rank), checkpointing=o.checkpointing, lora_plus=o.lora_plus)
        b.add("optimizer", o.optimizer, lr=o.lr, fused=o.fused)
        b.add("schedule", self.SCHEDULES[o.schedule], lr=o.lr)
        if o.warmup:
            b.add("warmup", "WarmupLRNode", steps=o.warmup)
        b.add("weighting", "LossWeightingNode", kind="min_snr")
        b.wires += [("model", "denoiser", "adapter", "denoiser"), ("adapter", "model", "optimizer", "model")]

    def _training_and_saving(self) -> None:
        o, b = self.o, self.b
        b.add("trainer", "DiffusionTrainerNode", steps=o.steps, epochs=o.epochs, accumulation=o.accumulation, graph=o.graph, aligned_init_batches=o.aligned_init)
        b.add("trigger", "EveryNStepsNode", every=o.save_every)
        if o.save_every_epoch:
            b.add("epoch_trigger", "EveryEpochNode", every=o.save_every_epoch)
        b.add("saver", "SaveAdapterNode", directory=self.output, name=o.name, probe=o.save_probe)
        schedule = (("schedule", "schedule", "warmup", "schedule"), ("warmup", "schedule", "trainer", "schedule")) if o.warmup \
            else (("schedule", "schedule", "trainer", "schedule"),)
        b.wires += [("batcher", "batches", "trainer", "batches"), ("adapter", "model", "trainer", "model"),
                    ("conditioner", "conditioner", "blend", "conditioner"), ("blend", "conditioner", "trainer", "conditioner"), ("optimizer", "optimizer", "trainer", "optimizer"),
                    *schedule, ("weighting", "weighting", "trainer", "weighting"), ("adapter", "model", "saver", "model")]
        b.events += [("trainer", "step_finished", "trigger", "step"), ("trigger", "due", "saver", "checkpoint"), ("trainer", "ended", "saver", "checkpoint")]
        if o.save_probe:
            b.wires.append(("blend", "conditioner", "saver", "conditioner"))
        if o.save_every_epoch:
            b.events += [("trainer", "step_finished", "epoch_trigger", "step"), ("epoch_trigger", "due", "saver", "checkpoint")]

    def _extras(self) -> None:
        o, b = self.o, self.b
        if o.ema:  # averages of the adapter weights, written beside every saved file
            b.add("ema", "AdapterEmaNode", sigma_rel=o.ema)
            b.wires += [("adapter", "model", "ema", "model"), ("ema", "ema", "saver", "ema")]
            b.events.append(("trainer", "step_finished", "ema", "step"))
        if o.regions_from:
            b.add("ladder", "RegionLadderNode", start=o.regions_from, span=o.regions_span)
            b.wires.append(("ladder", "ladder", "noiser", "ladder"))
        if o.batch_first:
            b.add("ramp", "BatchRampNode", size=o.batch_first, batches=o.batch_first_steps)
            b.wires.append(("ramp", "ramp", "batcher", "ramp"))
        if o.time_balance != "off" or o.balance_sampling > 0:
            b.add("balance", "TimeBalanceNode", mode=o.time_balance, sample_bias=o.balance_sampling)
            b.wires += [("balance", "balance", "trainer", "balance"), ("balance", "balance", "noiser", "balance")]

    def _override(self, overrides: dict[str, dict]) -> None:
        names = [n for n, _, _ in self.b.nodes]
        for node, params in overrides.items():
            if node not in names:
                raise ValueError(f"no node called {node!r} in this graph (nodes: {', '.join(names)}); settings such as memory.vram_max_mb are not node parameters")
            next(p for n, _, p in self.b.nodes if n == node).update(params)


def sdxl_lora(*, model: str, dataset: str, output: str, settings: dict | None = None, source: tuple[str, dict] | None = None,
              overrides: dict[str, dict] | None = None, **options) -> GraphDoc:
    """SDXL LoRA training: dataset, latent cache, prompts, buckets, conditioner, adapter, optimizer, trainer, saver.

    ``options`` are the fields of ``SdxlLoraOptions``. ``source`` replaces Load Model (tests hand in a stand-in); ``overrides`` sets
    extra parameters by node name."""
    return _SdxlLora(model, dataset, output, SdxlLoraOptions(**options), source).build(settings, overrides)
