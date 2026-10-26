"""The conditioner: what the trainer asks for text conditioning, on the device, from a cache."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import torch

from ..data.conditioning_cache import ConditioningCache, Size
from ..families.api.conditioning import Conditioning
from ..graph.events import PromptsChanged


class Conditioner:
    def __init__(self, cache: ConditioningCache, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.cache, self.device, self.dtype = cache, device, dtype
        self.extra: set[str] = set()  # prompts beyond the dataset's that must stay encoded (blend partners)

    def encode(self, prompts: Sequence[str], sizes: Sequence[Size] | Size | None = None) -> Conditioning:
        return self.cache.get(prompts, sizes).to(self.device, self.dtype)

    def warm(self, prompts: Sequence[str]) -> int:
        return self.cache.warm(prompts)

    def on_prompts_changed(self, event: PromptsChanged, universe: set[str]) -> None:
        """Drop cached prompts outside ``universe``, then encode the ones that are missing."""
        universe = universe | self.extra
        self.cache.retain(universe)
        self.cache.warm(sorted(universe))


class BlendedConditioner(Conditioner):
    """Blends each sample's conditioning toward that of a partner prompt after the cached encode (so a random blend is
    never cached): with chance ``probability`` a sample gets a strength drawn uniformly from ``strength`` and becomes
    ``caption * (1 - s) + partner * s``. The partner is a random pick of ``partners`` (the empty prompt is ``""``).
    ``segments`` limits the blend to some parts of the conditioning (family-defined names)."""

    def __init__(self, inner: Conditioner, partners: Sequence[str], strength: tuple[float, float] = (1.0, 1.0),
                 probability: float = 1.0, segments: Sequence[str] | None = None, generator: torch.Generator | None = None):
        low, high = strength
        if not partners or not 0.0 <= low <= high <= 1.0 or not 0.0 <= probability <= 1.0:
            raise ValueError("a blend needs partners, 0 <= strength min <= max <= 1 and a probability within 0..1")
        super().__init__(inner.cache, inner.device, inner.dtype)
        self.inner, self.partners, self.segments = inner, list(partners), segments
        self.strength, self.probability = (low, high), probability
        self.generator = generator or torch.Generator().manual_seed(0)
        inner.extra.update(self.partners)  # the encoders must be able to answer for them without loading again

    def draw(self, count: int) -> tuple[list[int], torch.Tensor]:
        """A partner index and a strength per sample; strength 0 for samples the chance skipped."""
        picks = torch.randint(0, len(self.partners), (count,), generator=self.generator).tolist()
        low, high = self.strength
        strength = low + (high - low) * torch.rand(count, generator=self.generator)
        applies = (torch.rand(count, generator=self.generator) < self.probability).float()
        return picks, strength * applies

    def encode(self, prompts: Sequence[str], sizes: Sequence[Size] | Size | None = None) -> Conditioning:
        base = super().encode(prompts, sizes)
        picks, strength = self.draw(len(prompts))
        if not strength.any():
            return base
        other = super().encode([self.partners[i] for i in picks], sizes)
        return base.blend(other, strength.to(self.device), self.segments)


def describe(conditioner: Any) -> str:
    return f"{type(conditioner).__name__}({len(conditioner.cache)} cached, {conditioner.cache.nbytes >> 20} MiB)"
