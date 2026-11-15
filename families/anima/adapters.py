"""Where adapters go in Anima's transformer, and how they are saved (ComfyUI reads the kohya ``lora_unet_`` names)."""

from __future__ import annotations

import re

from ..api.adapters import AdapterPlan, ModuleGroup, TrainableSelection, descriptors_from_shapes, plan_adapter
from ..api.header import HeaderIndex
from ..general.adapters.inject import AdaptedPart, descriptors_from_model, inject
from ..general.adapters.schemes import KohyaScheme, NativeScheme
from .denoiser import AnimaDenoiser

GROUPS = (
    ModuleGroup("attn_self", "Self attention", True, "self_attn q/k/v/output of every block"),
    ModuleGroup("attn_cross", "Cross attention", True, "cross_attn q/k/v/output of every block (where the text enters)"),
    ModuleGroup("mlp", "Feed-forward", True, "mlp.layer1 and layer2 of every block"),
    ModuleGroup("modulation", "Timestep modulation", False, "the AdaLN layers that turn the timestep into shifts and scales"),
    ModuleGroup("embeddings", "Patch and timestep embedding, output layer", False, "x_embedder, t_embedder and final_layer.linear"),
    ModuleGroup("llm_adapter", "Text adapter", False, "the adapter that turns Qwen3 states and T5 ids into the context; "
                                                      "its model card says to leave it untrained"),
)

_RULES = (
    (re.compile(r"blocks\.\d+\.self_attn\.(?:q|k|v|output)_proj"), "attn_self"),
    (re.compile(r"blocks\.\d+\.cross_attn\.(?:q|k|v|output)_proj"), "attn_cross"),
    (re.compile(r"blocks\.\d+\.mlp\.layer[12]"), "mlp"),
    (re.compile(r"(?:blocks\.\d+\.adaln_modulation_\w+|final_layer\.adaln_modulation)\.[12]"), "modulation"),
    (re.compile(r"x_embedder\.proj\.1|t_embedder\.1\.linear_[12]|final_layer\.linear"), "embeddings"),
    (re.compile(r"llm_adapter\.blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)_proj|llm_adapter\.blocks\.\d+\.mlp\.[02]|llm_adapter\.out_proj"),
     "llm_adapter"),
)

SCHEMES = {"kohya": KohyaScheme("lora_unet_"), "native": NativeScheme()}


class AnimaAdapterLayout:
    def groups(self) -> tuple[ModuleGroup, ...]:
        return GROUPS

    def classify(self, path: str, kind: str) -> str | None:
        for pattern, group in _RULES:
            if pattern.fullmatch(path):
                return group
        return None

    def regions(self, paths) -> tuple[str, ...]:
        """Block prefixes present in ``paths``: ``blocks.N``."""
        found = {".".join(p.split(".")[:2]) for p in paths if p.startswith("blocks.") and p.count(".") >= 1}
        return tuple(sorted(found, key=lambda r: int(r.split(".")[1])))


LAYOUT = AnimaAdapterLayout()


def plan_from_header(index: HeaderIndex, prefix: str, selection: TrainableSelection, rank: int, alpha: float) -> AdapterPlan:
    """The plan for a DiT file, from key names and shapes alone."""
    shapes = {k: e.shape for k, e in index.entries.items()}
    return plan_adapter(LAYOUT, descriptors_from_shapes(shapes, prefix), selection, rank, alpha)


def inject_denoiser(denoiser: AnimaDenoiser, selection: TrainableSelection, *, rank: int, alpha: float, kind: str = "lora",
                    policy: str = "classic", dropout: float = 0.0, compute: str = "float32",
                    factor_device: str | None = None) -> AdaptedPart:
    plan = plan_adapter(LAYOUT, descriptors_from_model(denoiser.model), selection, rank, alpha)
    return inject(denoiser.model, plan, kind=kind, policy=policy, dropout=dropout, compute=compute, factor_device=factor_device)
