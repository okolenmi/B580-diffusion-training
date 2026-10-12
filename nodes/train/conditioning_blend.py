"""ConditioningBlend: ctx = p * ctx(caption) + (1 - p) * ctx("").

What it is. For each sample of a batch, the text conditioning the UNet sees
is a convex mix of the caption's conditioning and the *empty prompt's*:

    ctx = p * ctx(caption) + (1 - p) * ctx("")
    y   = p * y(caption)   + (1 - p) * y("")        (resolution part unchanged)

p is drawn per sample, per step, uniformly from [keep_min, keep_max]. p = 1
is the caption alone, p = 0 is the empty prompt alone, p = 0.5 is the
midpoint. This is the same mechanism ComfyUI's ``<w:...>`` blank-blend uses
at render time, applied to training so the run sees a *continuum* of
conditioning rather than one point per caption.

Why it exists. A run whose conditioning never varies (all-empty captions,
one shared caption) gives cross-attention's K/V projections no signal to be
shaped by -- see manager/loader.py's caption_dropout note and
archive/bad-LoRA-investigation/MEASURED-captionless-conditioning-break.md.
Caption dropout varies it in jumps (caption <-> another caption); this varies
it smoothly between a caption and nothing.

What it is NOT. It is not shown to fix the CLIP_L-only failure -- nothing
here has been run against real weights. It is the requested mechanism, built
so the experiment can be run.

Off by default, and "off" is exact: keep_min == keep_max == 1.0 (the node's
defaults) never touches the encoder -- the trainers see ``active`` False and
take the unchanged code path, so a graph that does not wire this node, or
wires it at the defaults, is bit-identical to one built before it existed.
A sample drawn p = 1.0 inside an active blend also keeps its caption
conditioning bit-exact (selected, not recomputed as ctx0 + 1.0*(ctx-ctx0)).

Where it lives. Not a TextEncoder decorator: the text cache is keyed on
(prompt, batch_size) and the trainers' prewarm/residency code finds the
cache by type, so a wrapper outside it would hide the cache from both, and a
wrapper inside it would cache a *random* blend. Applied after the encode
instead (``encode_with_true_sizes(..., blend=...)``), the blank prompt is one
more cached key -- ``("", 1)`` -- which ``with_blank_prompt`` adds to the
prewarm set so the encoder can still be unloaded.
"""

from __future__ import annotations

import random
from typing import ClassVar

from ..core import Node, Port

#: The empty prompt, the blend's other endpoint. A constant so the prewarm
#: set and the encode request cannot disagree about which key it is.
BLANK_PROMPT = ""


class ConditioningBlend:
    """Per-sample blend weights and the blend itself. No torch at import."""

    def __init__(self, keep_min: float = 1.0, keep_max: float = 1.0,
                 seed: int | None = None):
        keep_min, keep_max = float(keep_min), float(keep_max)
        if not (0.0 <= keep_min <= keep_max <= 1.0):
            raise ValueError(
                f"ConditioningBlend needs 0 <= keep_min <= keep_max <= 1, got "
                f"keep_min={keep_min}, keep_max={keep_max}")
        self.keep_min = keep_min
        self.keep_max = keep_max
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def active(self) -> bool:
        """False only for the exact identity (p == 1 for every sample)."""
        return not (self.keep_min == 1.0 and self.keep_max == 1.0)

    def draw(self, batch_size: int) -> list[float]:
        """One p per sample. Seeded, so a seeded run replays its blends."""
        if self.keep_min == self.keep_max:
            return [self.keep_min] * batch_size
        return [self._rng.uniform(self.keep_min, self.keep_max)
                for _ in range(batch_size)]

    def mix(self, ctx, y, ctx_blank, y_blank, p):
        """Blend a batch toward the blank conditioning.

        ``p`` is a list of per-sample weights on the *caption* side. Computed
        as blank + p*(caption - blank) so the resolution columns of ``y``
        (identical in both operands) come out exactly unchanged, and so a
        sample with p == 1 can be passed through untouched by ``where``.
        """
        import torch
        w = torch.tensor(p, dtype=torch.float32).to(ctx.device)
        w_ctx = w.view(-1, 1, 1)
        w_y = w.view(-1, 1)
        blank_ctx = ctx_blank.to(ctx.device, torch.float32)
        blank_y = y_blank.to(y.device, torch.float32)
        mixed_ctx = blank_ctx + w_ctx * (ctx.to(torch.float32) - blank_ctx)
        mixed_y = blank_y + w_y * (y.to(torch.float32) - blank_y)
        keep = w_ctx >= 1.0
        out_ctx = torch.where(keep, ctx.to(torch.float32), mixed_ctx).to(ctx.dtype)
        out_y = torch.where(w_y >= 1.0, y.to(torch.float32), mixed_y).to(y.dtype)
        return out_ctx, out_y


def with_blank_prompt(discovery):
    """The prewarm discovery plus the blank prompt's key, if it lacks it.

    The blend asks the encoder for ``BLANK_PROMPT`` at batch size 1 (the
    per-prompt path, so one key serves every batch size). Without it in the
    warm set, the first blended step is a cache miss that re-loads CLIP.
    An empty discovery (no batches seen) is returned as is.
    """
    import dataclasses
    keys = discovery.keys
    if not keys or any(k[0] == BLANK_PROMPT and k[1] == 1 for k in keys):
        return discovery
    _, _, h, w, n = next(iter(keys))
    return dataclasses.replace(
        discovery, keys=frozenset(keys | {(BLANK_PROMPT, 1, h, w, n)}))


class ConditioningBlendNode(Node):
    """Wire ``blend`` to a trainer's ``conditioning_blend`` port."""

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "blend": Port(
            name="blend", type=ConditioningBlend, required=True,
            doc="Wire to a trainer's conditioning_blend port. At the default "
                "keep range (1.0..1.0) it changes nothing, bit for bit.",
        ),
    }

    INPUTS: ClassVar[dict[str, Port]] = {
        "keep_min": Port(
            name="keep_min", type=float, required=False, default=1.0,
            doc="Lowest weight on the caption's conditioning, 0..1. Each "
                "sample's weight p is drawn uniformly from keep_min..keep_max "
                "every step; the rest (1 - p) goes to the empty prompt's "
                "conditioning. 1.0 = caption only, 0.0 = empty prompt only.",
        ),
        "keep_max": Port(
            name="keep_max", type=float, required=False, default=1.0,
            doc="Highest weight on the caption's conditioning, 0..1. Equal to "
                "keep_min = a fixed p (0.5 is the midpoint, ComfyUI's "
                "<w:0.5>); keep_min=keep_max=1.0 (default) disables the "
                "blend entirely.",
        ),
        "seed": Port(
            name="seed", type=int, required=False, default=-1,
            doc="Seed for the per-step draws of p. -1 = unseeded.",
        ),
    }

    def build(self, **inputs) -> dict[str, ConditioningBlend]:
        self.validate_inputs(inputs)
        seed = inputs.get("seed", self.INPUTS["seed"].default)
        result = {"blend": ConditioningBlend(
            keep_min=inputs.get("keep_min", self.INPUTS["keep_min"].default),
            keep_max=inputs.get("keep_max", self.INPUTS["keep_max"].default),
            seed=None if seed is None or seed < 0 else int(seed))}
        self.validate_outputs(result)
        return result
