"""Where adapters go in SDXL's UNet, and how they are saved."""

from __future__ import annotations

import re

from ..api.adapters import AdapterPlan, ModuleGroup, TrainableSelection, descriptors_from_shapes, plan_adapter
from ..api.header import HeaderIndex
from ..general.adapters.inject import AdaptedPart, descriptors_from_model, inject
from ..general.adapters.schemes import KohyaScheme, NativeScheme
from .denoiser import SDXLDenoiser

GROUPS = (
    ModuleGroup("attn_self", "Self attention", True, "attn1 to_q/to_k/to_v/to_out"),
    ModuleGroup("attn_cross", "Cross attention", True, "attn2 to_q/to_k/to_v/to_out (the text-conditioned half)"),
    ModuleGroup("ff", "Feed-forward", True, "ff.net.0.proj and ff.net.2 of every transformer block"),
    ModuleGroup("transformer_proj", "Transformer projections", True, "proj_in and proj_out of every spatial transformer"),
    ModuleGroup("label_emb", "Size/pooled embedding", False, "the layers that embed the y vector"),
    ModuleGroup("time_embed", "Timestep embedding", False, "the layers that embed the timestep"),
)

_RULES = (
    (re.compile(r"(?:.*\.)?attn1\.(?:to_q|to_k|to_v|to_out\.0)"), "attn_self"),
    (re.compile(r"(?:.*\.)?attn2\.(?:to_q|to_k|to_v|to_out\.0)"), "attn_cross"),
    (re.compile(r".*\.ff\.net\.(?:0\.proj|2)"), "ff"),
    (re.compile(r".*\.(?:proj_in|proj_out)"), "transformer_proj"),
    (re.compile(r"label_emb\..*"), "label_emb"),
    (re.compile(r"time_embed\..*"), "time_embed"),
)

SCHEMES = {"kohya": KohyaScheme("lora_unet_"), "native": NativeScheme()}


class SDXLAdapterLayout:
    def groups(self) -> tuple[ModuleGroup, ...]:
        return GROUPS

    def classify(self, path: str, kind: str) -> str | None:
        for pattern, group in _RULES:
            if pattern.fullmatch(path):
                return group
        return None

    def regions(self, paths) -> tuple[str, ...]:
        """Block prefixes present in ``paths``: ``input_blocks.N``, ``middle_block``, ``output_blocks.N``."""
        found: set[str] = set()
        for p in paths:
            head = p.split(".")
            if head[0] in ("input_blocks", "output_blocks") and len(head) > 1:
                found.add(f"{head[0]}.{head[1]}")
            elif head[0] == "middle_block":
                found.add("middle_block")
        return tuple(sorted(found, key=lambda r: (r.split(".")[0], int(r.split(".")[1]) if "." in r else 0)))


LAYOUT = SDXLAdapterLayout()


def plan_from_header(index: HeaderIndex, prefix: str, selection: TrainableSelection, rank: int, alpha: float) -> AdapterPlan:
    """The plan for a UNet file, from key names and shapes alone."""
    shapes = {k: e.shape for k, e in index.entries.items()}
    return plan_adapter(LAYOUT, descriptors_from_shapes(shapes, prefix), selection, rank, alpha)


def inject_denoiser(denoiser: SDXLDenoiser, selection: TrainableSelection, *, rank: int, alpha: float, kind: str = "lora",
                    policy: str = "classic", dropout: float = 0.0, compute: str = "float32") -> AdaptedPart:
    plan = plan_adapter(LAYOUT, descriptors_from_model(denoiser.model), selection, rank, alpha)
    return inject(denoiser.model, plan, kind=kind, policy=policy, dropout=dropout, compute=compute)
