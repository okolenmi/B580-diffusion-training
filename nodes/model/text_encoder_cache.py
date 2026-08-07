"""CachingTextEncoderNode: decorates any TextEncoder with an LRU cache
keyed on (prompt, batch_size, height, width).

VRAM/compute rationale: SupervisedLoRATrainerNode encodes every batch's
prompt from scratch, every step (see nodes/train/step_pipeline.py's
EncodeConditioningPhase) -- for a managed dataset whose captions repeat (the common
case: a handful of style/character tags reused across many images), that
means CLIP re-runs a full forward pass, with its own real activation
memory on top of the UNet's, for conditioning this codebase already
computed. Caching skips that entirely on a hit. Bounded (default 512
entries) rather than unbounded, so an open-ended/randomized-prompt
dataset can't grow this without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

from ..core import Port
from .text_encoder import TextEncoder, TextEncoderNode


class CachingTextEncoder(TextEncoder):
    """Every caller in this codebase already does .to(device=...) on
    encode()'s return value (see nodes/train/step_pipeline.py's
    EncodeConditioningPhase), so
    caching on CPU and handing the same tensors back on a hit needs no
    device-placement special-casing here -- correct whether the caller
    gets a fresh (already-on-device) or cached (CPU) pair."""

    def __init__(self, inner: TextEncoder, max_entries: int = 512):
        self._inner = inner
        self._max_entries = max_entries
        self._cache: OrderedDict = OrderedDict()

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        key = (prompt, batch_size, height, width)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        ctx, y = self._inner.encode(prompt, batch_size, height, width)
        entry = (ctx.detach().cpu(), y.detach().cpu())
        self._cache[key] = entry
        if len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return entry

    def unload(self) -> None:
        self._inner.unload()

    def clear_cache(self) -> None:
        self._cache.clear()


class CachingTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True,
                         doc="The real encoder to wrap, e.g. an SDXLTextEncoderNode's output."),
        "max_entries": Port(
            name="max_entries", type=int, required=False, default=512,
            doc="Oldest entry is evicted once the cache holds more than this many "
                "distinct (prompt, batch_size, height, width) combinations.",
        ),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        encoder: TextEncoder = inputs["encoder"]
        max_entries = inputs.get("max_entries", self.INPUTS["max_entries"].default)
        result = {"encoder": CachingTextEncoder(encoder, max_entries=max_entries)}
        self.validate_outputs(result)
        return result
