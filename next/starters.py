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


def sdxl_lora(*, model: str, dataset: str, output: str, name: str = "lora", resolution: int = 1024, batch_size: int = 1,
              accumulation: int = 1, steps: int = 1000, lr: float = 1e-4, rank: int = 16, alpha: float | None = None,
              optimizer: str = "AdamWNode", fused: bool = False, save_every: int = 250, checkpointing: bool = True,
              settings: dict | None = None, source: tuple[str, dict] | None = None, overrides: dict[str, dict] | None = None) -> GraphDoc:
    """SDXL LoRA training: dataset, latent cache, prompts, buckets, conditioner, adapter, optimizer, trainer, saver.

    ``source`` replaces Load Model (tests hand in a stand-in); ``overrides`` sets extra parameters by node name.
    """
    b = _Builder()
    src_class, src_params = source or ("LoadModelNode", {"file": model})
    b.add("model", src_class, **src_params)
    b.add("images", "DatasetLibraryNode", dataset=dataset)
    b.add("latents", "LatentCacheNode", dataset=dataset, base=resolution, mode="fit")
    b.add("captions", "CaptionsNode")
    b.add("tag_swap", "TagSwapNode")
    b.add("buckets", "BucketerNode", multiple=4, keep_incomplete=True)  # 32 px: few shapes, padded and masked
    b.add("noiser", "NoiserNode")
    b.add("batcher", "BatcherNode", batch_size=batch_size)
    b.add("conditioner", "ConditionerNode")
    b.add("adapter", "InjectAdapterNode", rank=rank, alpha=float(alpha if alpha is not None else rank), checkpointing=checkpointing)
    b.add("optimizer", optimizer, lr=lr, fused=fused)
    b.add("schedule", "ConstantLRNode", lr=lr)
    b.add("weighting", "LossWeightingNode", kind="min_snr")
    b.add("trainer", "DiffusionTrainerNode", steps=steps, accumulation=accumulation)
    b.add("trigger", "EveryNStepsNode", every=save_every)
    b.add("saver", "SaveAdapterNode", directory=output, name=name)
    b.wires += [("model", "vae", "latents", "vae"), ("images", "images", "latents", "images"), ("images", "images", "captions", "images"),
                ("captions", "prompts", "tag_swap", "prompts"), ("latents", "latents", "buckets", "latents"),
                ("model", "denoiser", "noiser", "denoiser"), ("latents", "latents", "batcher", "latents"),
                ("tag_swap", "prompts", "batcher", "prompts"), ("noiser", "noiser", "batcher", "noiser"),
                ("buckets", "buckets", "batcher", "buckets"), ("model", "clip_l", "conditioner", "encoders"),
                ("model", "clip_g", "conditioner", "encoders"), ("tag_swap", "prompts", "conditioner", "prompts"),
                ("model", "denoiser", "adapter", "denoiser"), ("adapter", "model", "optimizer", "model"),
                ("batcher", "batches", "trainer", "batches"), ("adapter", "model", "trainer", "model"),
                ("conditioner", "conditioner", "trainer", "conditioner"), ("optimizer", "optimizer", "trainer", "optimizer"),
                ("schedule", "schedule", "trainer", "schedule"), ("weighting", "weighting", "trainer", "weighting"),
                ("adapter", "model", "saver", "model")]
    b.events += [("batcher", "prompts_changed", "conditioner", "prompts_changed"), ("trainer", "step_finished", "trigger", "step"),
                 ("trigger", "due", "saver", "checkpoint"), ("trainer", "ended", "saver", "checkpoint")]
    for node, params in (overrides or {}).items():
        next(p for n, _, p in b.nodes if n == node).update(params)
    return b.doc(settings or {})
