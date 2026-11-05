"""Ready-made graphs. Each returns a ``GraphDoc`` that the editor, the headless runner and the tests all use."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .graph import AddonSpec, Edge, GraphDoc, NodeSpec


@dataclass
class _Builder:
    nodes: list[tuple[str, str, dict]] = field(default_factory=list)
    wires: list[tuple[str, str, str, str]] = field(default_factory=list)
    events: list[tuple[str, str, str, str]] = field(default_factory=list)
    addons: list[tuple[str, str, str, dict]] = field(default_factory=list)  # (host label or "graph", id, kind, params)

    def add(self, label: str, class_name: str, **params) -> None:
        self.nodes.append((label, class_name, params))

    def addon(self, host: str, addon_id: str, kind: str, **params) -> None:
        self.addons.append((host, addon_id, kind, params))

    def uid(self, name: str) -> str:
        return "n_" + "abcdefghijklmnopqrstuvwxyz"[[n for n, _, _ in self.nodes].index(name)] * 6

    def doc(self, settings: dict) -> GraphDoc:
        edge = lambda w: Edge(self.uid(w[0]), w[1], self.uid(w[2]), w[3])  # noqa: E731
        on = lambda host: tuple(AddonSpec(i, kind, True, params) for h, i, kind, params in self.addons if h == host)  # noqa: E731
        specs = tuple(NodeSpec(self.uid(n), cls, params, title=n, addons=on(n)) for n, cls, params in self.nodes)
        return GraphDoc(specs, tuple(map(edge, self.wires)), tuple(map(edge, self.events)), settings, addons=on("graph"))


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
    init_adapter: str = ""
    clean: float = 0.0
    clean_at: int = 0
    clean_steps: int = 200
    clean_mode: str = "element"
    clean_lr: float = 0.1
    clean_sparsity: float = 0.0
    clean_freeze: bool = True
    clean_scope: str = "global"
    clean_drop: str = ""
    clean_rescale: bool = False
    stages: tuple[str, ...] = ()
    stage_warmup: int = 20
    anchor: float = 0.0
    anchor_every: int = 2
    anchor_prompts: str = ""
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
    resume_every: int = 0
    batch_first: int = 0
    batch_first_steps: int = 200
    ema: str = ""
    aligned_init: int = 0
    gradient_policy: str = "mean"
    policy_normalize: bool = True
    policy_floor: float = 0.0
    policy_calibrate_every: int = 0
    trend_boost: float = 0.0
    trend_mode: str = "trend"
    trend_penalty: float = 0.0


@dataclass(frozen=True)
class Stage:
    """One stretch of a staged run: ``steps`` at ``lr`` (None: the run's rate), ended by a cleaning to ``keep`` of the adapter (0: none)."""

    steps: int
    lr: float | None = None
    keep: float = 0.0
    mode: str = "magnitude"


def parse_stages(specs: tuple[str, ...] | list[str]) -> tuple[Stage, ...]:
    """Reads ``STEPS[@LR][:KEEP[/MODE]]`` (for example ``600:0.08``, ``400@5e-5:0.15/product``)."""
    stages = []
    for spec in specs:
        head, _, tail = spec.partition(":")
        count, _, rate = head.partition("@")
        keep, _, mode = tail.partition("/")
        try:
            stage = Stage(int(count), float(rate) if rate else None, float(keep) if keep else 0.0, mode or "magnitude")
        except ValueError as error:
            raise ValueError(f"a stage is STEPS[@LR][:KEEP[/MODE]], for example 600:0.08 or 400@5e-5:0.15; got {spec!r}") from error
        if stage.steps < 1 or not 0.0 <= stage.keep < 1.0 or (stage.lr is not None and stage.lr <= 0) or (tail and not keep):
            raise ValueError(f"stage {spec!r}: steps must be at least 1, the learning rate above 0 and the kept share between 0 and 1 (exclusive)")
        stages.append(stage)
    kept = [stage.keep for stage in stages if stage.keep]
    if kept != sorted(kept):
        raise ValueError("stages: the kept share is the share of the whole adapter that is frozen in total, so it can only grow from stage to stage")
    return tuple(stages)


class _SdxlLora:
    """Builds the SDXL LoRA graph in groups: data, model and optimizer, trainer and saver, then the optional extras."""

    SCHEDULES = {"constant": "ConstantLRNode", "cosine": "CosineLRNode"}

    def __init__(self, model: str, dataset: str, output: str, options: SdxlLoraOptions, source: tuple[str, dict] | None):
        self.o, self.dataset, self.output, self.source, self.model = options, dataset, output, source, model
        self.b = _Builder()
        self.prompts = "tag_swap" if options.tag_swap else "captions"
        self.stages = parse_stages(options.stages)
        if self.stages and options.clean:
            raise ValueError("--stage describes the cleanings of the run; it cannot be combined with --clean")

    def build(self, settings: dict | None, overrides: dict[str, dict] | None) -> GraphDoc:
        self._data()
        self._model_and_optimizer()
        self._training_and_saving()
        self._extras()
        self._resume(settings, overrides)
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
        b.add("adapter", "InjectAdapterNode", rank=o.rank, alpha=float(o.alpha if o.alpha is not None else o.rank), checkpointing=o.checkpointing, lora_plus=o.lora_plus,
              **({"init_adapter": o.init_adapter} if o.init_adapter else {}))
        b.add("optimizer", o.optimizer, lr=o.lr, fused=o.fused)
        b.add("schedule", self.SCHEDULES[o.schedule], lr=o.lr)
        if o.warmup:
            b.addon("schedule", "warmup", "WarmupAddon", steps=o.warmup)
        if self.stages:
            ends = [sum(stage.steps for stage in self.stages[:i + 1]) for i in range(len(self.stages))]
            b.addon("schedule", "stages", "StagedRateAddon", boundaries=",".join(map(str, ends[:-1])), warmup=o.stage_warmup,
                    ratios=",".join(f"{(stage.lr or o.lr) / o.lr:g}" for stage in self.stages))
        b.add("weighting", "LossWeightingNode", kind="min_snr")
        b.wires += [("model", "denoiser", "adapter", "denoiser"), ("adapter", "model", "optimizer", "model")]

    def _training_and_saving(self) -> None:
        o, b = self.o, self.b
        b.add("trainer", "DiffusionTrainerNode", steps=sum(stage.steps for stage in self.stages) if self.stages else o.steps, epochs=o.epochs, accumulation=o.accumulation, graph=o.graph)
        if o.gradient_policy != "mean":
            b.addon("trainer", "policy", "GradientPolicyAddon", policy=o.gradient_policy, normalize=o.policy_normalize, floor=o.policy_floor,
                    calibrate_every=o.policy_calibrate_every)
        if o.trend_boost:
            b.addon("trainer", "trend", "TrendWeightingAddon", boost=o.trend_boost, mode=o.trend_mode, penalty=o.trend_penalty)
        if o.clean:
            b.addon("trainer", "cleaner", "CleanerAddon", keep=o.clean, at=o.clean_at, steps=o.clean_steps, mode=o.clean_mode, lr=o.clean_lr,
                    sparsity=o.clean_sparsity, freeze=o.clean_freeze, directory=f"{self.output}/clean", scope=o.clean_scope, drop=o.clean_drop,
                    rescale=o.clean_rescale)
        if any(stage.keep for stage in self.stages):
            ends = [sum(stage.steps for stage in self.stages[:i + 1]) for i in range(len(self.stages))]
            events = ",".join(f"{end}:{stage.keep:g}:{stage.mode}" for end, stage in zip(ends, self.stages) if stage.keep)
            b.addon("trainer", "cleaner", "CleanerAddon", keep=next(s.keep for s in self.stages if s.keep), events=events, steps=o.clean_steps, lr=o.clean_lr,
                    sparsity=o.clean_sparsity, freeze=o.clean_freeze, directory=f"{self.output}/clean", scope=o.clean_scope, drop=o.clean_drop, rescale=o.clean_rescale)
        if o.anchor:
            b.addon("trainer", "anchor", "AnchorAddon", weight=o.anchor, every=o.anchor_every, prompts=o.anchor_prompts)
        if o.aligned_init and o.init_adapter:
            raise ValueError("--aligned-init sets the starting point of a fresh adapter; it cannot be combined with --init-adapter")
        if o.aligned_init:
            b.addon("trainer", "aligned_init", "AlignedInitAddon", batches=o.aligned_init)
        b.addon("graph", "save_steps", "EveryNSteps", every=o.save_every)
        if o.save_every_epoch:
            b.addon("graph", "save_epochs", "EveryEpoch", every=o.save_every_epoch)
        b.add("saver", "SaveAdapterNode", directory=self.output, name=o.name, probe=o.save_probe)
        b.wires += [("batcher", "batches", "trainer", "batches"), ("adapter", "model", "trainer", "model"),
                    ("conditioner", "conditioner", "blend", "conditioner"), ("blend", "conditioner", "trainer", "conditioner"), ("optimizer", "optimizer", "trainer", "optimizer"),
                    ("schedule", "schedule", "trainer", "schedule"), ("weighting", "weighting", "trainer", "weighting"), ("adapter", "model", "saver", "model")]
        b.events.append(("trainer", "ended", "saver", "checkpoint"))
        if o.save_probe:
            b.wires.append(("blend", "conditioner", "saver", "conditioner"))

    def _extras(self) -> None:
        o, b = self.o, self.b
        if o.ema:  # averages of the adapter weights, written beside every saved file
            b.add("ema", "AdapterEmaNode", sigma_rel=o.ema)
            b.wires += [("adapter", "model", "ema", "model"), ("ema", "ema", "saver", "ema")]
            b.events.append(("trainer", "step_finished", "ema", "step"))
        if o.regions_from:
            b.addon("noiser", "ladder", "RegionLadderAddon", start=o.regions_from, span=o.regions_span)
        if o.batch_first:
            b.addon("batcher", "ramp", "BatchRampAddon", size=o.batch_first, batches=o.batch_first_steps)
        if o.time_balance != "off" or o.balance_sampling > 0:
            b.add("balance", "TimeBalanceNode", mode=o.time_balance, sample_bias=o.balance_sampling)
            b.wires += [("balance", "balance", "trainer", "balance"), ("balance", "balance", "noiser", "balance")]

    def _resume(self, settings: dict | None, overrides: dict[str, dict] | None) -> None:
        """Periodic training states and continuing from the newest. The run's identity is a hash of everything that shapes it except how long it
        runs and when it saves, so a longer run may continue a shorter one and a different run never picks up this one's state."""
        o, b = self.o, self.b
        if not o.resume_every:
            return
        shaping = {k: v for k, v in asdict(o).items() if k not in ("steps", "epochs", "save_every", "save_every_epoch", "resume_every", "name")}
        identity = hashlib.sha256(json.dumps([shaping, self.model, self.dataset, settings, overrides, self.source], sort_keys=True, default=str).encode()).hexdigest()[:16]
        b.add("resume", "ResumeNode", directory=f"{self.output}/resume", every=o.resume_every, identity=identity)
        b.wires += [("adapter", "model", "resume", "model"), ("optimizer", "optimizer", "resume", "optimizer"),
                    ("resume", "start_step", "trainer", "start_step"), ("resume", "skip", "batcher", "skip")]
        if o.ema:
            b.wires.append(("ema", "ema", "resume", "ema"))
        b.events += [("trainer", "step_finished", "resume", "step"), ("trainer", "ended", "resume", "checkpoint")]

    def _override(self, overrides: dict[str, dict]) -> None:
        names = [n for n, _, _ in self.b.nodes]
        addons = [i for _, i, _, _ in self.b.addons]
        for node, params in overrides.items():
            if node in names:
                next(p for n, _, p in self.b.nodes if n == node).update(params)
            elif node in addons:
                next(p for _, i, _, p in self.b.addons if i == node).update(params)
            else:
                raise ValueError(f"no node or addon called {node!r} in this graph (nodes: {', '.join(names)}; addons: {', '.join(addons)}); "
                                 f"settings such as memory.vram_max_mb are not node parameters")


def sdxl_lora(*, model: str, dataset: str, output: str, settings: dict | None = None, source: tuple[str, dict] | None = None,
              overrides: dict[str, dict] | None = None, **options) -> GraphDoc:
    """SDXL LoRA training: dataset, latent cache, prompts, buckets, conditioner, adapter, optimizer, trainer, saver.

    ``options`` are the fields of ``SdxlLoraOptions``. ``source`` replaces Load Model (tests hand in a stand-in); ``overrides`` sets
    extra parameters by node name."""
    return _SdxlLora(model, dataset, output, SdxlLoraOptions(**options), source).build(settings, overrides)
