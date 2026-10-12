"""CaptionPoolNode: settings for building each image's prompt out of several
of the dataset's real captions (manager/caption_pool.py has the semantics).

Wire ``pool`` to a dataset source node's ``caption_pool`` port. The node only
describes the pool; the loader, which has the captions, builds it. The handle
is a plain settings object for that reason: nodes/ does not import manager/ at
module level, and the loader duck-types what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core import Node, Port


@dataclass(frozen=True)
class CaptionPoolSettings:
    n_min: int = 1
    n_max: int = 3
    mode: str = "captions"
    include_own: bool = True
    variants: int = 4
    seed: int | None = None
    dividers: str = ".,"

    def build(self):
        """The manager-side pool, validated (ValueError on bad settings)."""
        from manager.caption_pool import CaptionPool
        return CaptionPool(
            n_min=self.n_min, n_max=self.n_max, mode=self.mode,
            include_own=self.include_own, variants=self.variants,
            seed=self.seed, dividers=self.dividers)


class CaptionPoolNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "pool": Port(
            name="pool", type=CaptionPoolSettings, required=True,
            doc="Wire to a dataset source node's caption_pool port.",
        ),
    }

    INPUTS: ClassVar[dict[str, Port]] = {
        "n_min": Port(
            name="n_min", type=int, required=False, default=1,
            doc="Fewest captions combined into one prompt.",
        ),
        "n_max": Port(
            name="n_max", type=int, required=False, default=3,
            doc="Most captions combined into one prompt. Each prompt draws n "
                "uniformly from n_min..n_max. CLIP reads 77 tokens; longer "
                "prompts are cut, so keep n modest.",
        ),
        "mode": Port(
            name="mode", type=str, required=False, default="captions",
            choices=("captions", "parts"),
            doc="'captions' = whole captions joined with '. '. 'parts' = every "
                "chosen caption split on the dividers into tags, de-duplicated "
                "and shuffled, joined with ', ' -- a random bag of the "
                "dataset's own tags.",
        ),
        "include_own": Port(
            name="include_own", type=bool, required=False, default=True,
            doc="Start every prompt with the image's own caption (when it has "
                "one) and add n-1 others. Off = n captions drawn from the whole "
                "set, so no prompt need describe its image at all.",
        ),
        "variants": Port(
            name="variants", type=int, required=False, default=4,
            doc="Prompts drawn per image, once, up front; each epoch serves one "
                "of them. The prompt set is then bounded, so the text-encoder "
                "prewarm can encode all of it and unload CLIP. 0 = a fresh "
                "prompt every time an image is served: more variety, but "
                "unbounded -- no prewarm, every new prompt is a cache miss.",
        ),
        "seed": Port(
            name="seed", type=int, required=False, default=-1,
            doc="Seed for the draws. -1 = unseeded.",
        ),
        "dividers": Port(
            name="dividers", type=str, required=False, default=".,",
            doc="Characters that divide a caption into parts / captions from "
                "each other. Default '.,'.",
        ),
    }

    def build(self, **inputs) -> dict[str, CaptionPoolSettings]:
        self.validate_inputs(inputs)
        seed = inputs.get("seed", self.INPUTS["seed"].default)
        settings = CaptionPoolSettings(
            n_min=inputs.get("n_min", self.INPUTS["n_min"].default),
            n_max=inputs.get("n_max", self.INPUTS["n_max"].default),
            mode=inputs.get("mode", self.INPUTS["mode"].default),
            include_own=inputs.get("include_own", self.INPUTS["include_own"].default),
            variants=inputs.get("variants", self.INPUTS["variants"].default),
            seed=None if seed is None or seed < 0 else int(seed),
            dividers=inputs.get("dividers", self.INPUTS["dividers"].default))
        settings.build()  # bad settings fail here, as a config error
        result = {"pool": settings}
        self.validate_outputs(result)
        return result
