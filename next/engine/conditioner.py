"""The conditioner: what the trainer asks for text conditioning, on the device, from a cache."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..data.conditioning_cache import ConditioningCache
from ..families.api.conditioning import Conditioning
from ..graph.events import PromptsChanged


class Conditioner:
    def __init__(self, cache: ConditioningCache, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.cache, self.device, self.dtype = cache, device, dtype

    def encode(self, prompts: Sequence[str], size: tuple[int, int] | None = None) -> Conditioning:
        return self.cache.get(prompts, size).to(self.device, self.dtype)

    def warm(self, prompts: Sequence[str], size: tuple[int, int] | None = None) -> int:
        return self.cache.warm(prompts, size)

    def on_prompts_changed(self, event: PromptsChanged, universe: set[str], size: tuple[int, int] | None = None) -> None:
        """Drop cached prompts outside ``universe``, then encode the ones that are missing."""
        self.cache.retain(universe)
        self.cache.warm(sorted(universe), size)


class BlendedConditioner(Conditioner):
    """Blends each batch's conditioning with that of other prompts after the cached encode, so a random blend is
    never cached. ``segments`` limits the blend to some parts of the conditioning (family-defined names)."""

    def __init__(self, inner: Conditioner, partners: Sequence[str], probability: float, segments: Sequence[str] | None = None,
                 generator: torch.Generator | None = None):
        super().__init__(inner.cache, inner.device, inner.dtype)
        self.partners, self.probability, self.segments = list(partners), probability, segments
        self.generator = generator or torch.Generator().manual_seed(0)

    def encode(self, prompts: Sequence[str], size: tuple[int, int] | None = None) -> Conditioning:
        base = super().encode(prompts, size)
        picks = torch.randint(0, len(self.partners), (len(prompts),), generator=self.generator).tolist()
        other = super().encode([self.partners[i] for i in picks], size)
        p = (torch.rand(len(prompts), generator=self.generator) < self.probability).float()
        return base.blend(other, p.to(self.device), self.segments)


def describe(conditioner: Any) -> str:
    return f"{type(conditioner).__name__}({len(conditioner.cache)} cached, {conditioner.cache.nbytes >> 20} MiB)"
