"""CachingTextEncoderNode: decorates any TextEncoder with two independent
LRU caches -- one for the prompt-only half of encode() (keyed on
(prompt, batch_size)), one for the resolution-only half (keyed on
(height, width, batch_size)) -- combined into the same (ctx, y) pair
encode() always returned.

VRAM/compute rationale: SupervisedLoRATrainerNode encodes every batch's
prompt from scratch, every step (see nodes/train/step_pipeline.py's
EncodeConditioningPhase) -- for a managed dataset whose captions repeat (the common
case: a handful of style/character tags reused across many images), that
means CLIP re-runs a full forward pass, with its own real activation
memory on top of the UNet's, for conditioning this codebase already
computed. Caching skips that entirely on a hit. Bounded (default 512
entries, each cache independently) rather than unbounded, so an
open-ended/randomized-prompt (or, symmetrically, open-ended-resolution)
dataset can't grow either without limit.

Two caches, not one keyed on all four fields together (this class's own
original design, until this session): CLIP's own forward pass --
encode_prompt_only(), the genuinely expensive half -- depends only on
the prompt string, never on height/width; TextEncoder.resolution_embedding()
(nodes/model/text_encoder.py), the cheap half, is the only part that
depends on height/width at all. A single combined key meant the SAME
prompt at two different resolutions was two unrelated cache misses, each
paying the full CLIP forward pass again -- for a mixed-aspect-ratio
dataset with repeated (or, worse, this project's own real "100 images
without captions" case -- a single shared empty-string prompt) captions,
that's close to zero caching benefit at all for exactly the dataset
shape this project's own real use has hit, despite this class existing
specifically to avoid that cost. Splitting the key means the SAME prompt
at a new resolution is a resolution-cache miss (cheap: one small
sinusoidal-embedding computation) composed with a prompt-cache *hit*
(skips CLIP entirely) -- the two are recombined into the same (ctx, y)
shape every caller already expects, so this is invisible from outside
this class.

Optional resource_control (a ResourceControlHandle,
nodes/memory/control_handle.py) is what actually makes a warm cache pay
off in VRAM, not just compute: on a hit (of either cache), the inner
encoder genuinely isn't touched for that half, so it's safe for it to be
offloaded between hits; on a miss of either, ensure_loaded() below
brings it back first -- including, if something else is currently using
the room, offloading that to make space (ResourceControlHandle.ensure_loaded()'s
own "offload everything else until this one's done" behavior, direct
feedback on the case where model+optimizer are already near budget when
a miss happens). This class doesn't decide *when* the inner encoder gets
offloaded in the first place -- that's whatever holds this handle
calling before_step() between steps (nodes/train/supervised.py), same as
any other registered resident.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

from ..core import Port
from ..memory.control_handle import ResourceControlHandle
from .text_encoder import TextEncoder, TextEncoderNode


class CachingTextEncoder(TextEncoder):
    """Every caller in this codebase already does .to(device=...) on
    encode()'s return value (see nodes/train/step_pipeline.py's
    EncodeConditioningPhase), so
    caching on CPU and handing the same tensors back on a hit needs no
    device-placement special-casing here -- correct whether the caller
    gets a fresh (already-on-device) or cached (CPU) pair."""

    def __init__(self, inner: TextEncoder, max_entries: int = 512,
                 resource_control: ResourceControlHandle | None = None,
                 resource_name: str = "text_encoder"):
        self._inner = inner
        self._max_entries = max_entries
        self._prompt_cache: OrderedDict = OrderedDict()
        self._resolution_cache: OrderedDict = OrderedDict()
        self._resource_control = resource_control
        # Matched against whatever name the trainer registers this same object
        # under (nodes/train/supervised.py registers text_encoder as
        # "text_encoder") -- both sides need to agree on the string, so it's a
        # real, if small, shared convention rather than something either side
        # derives independently. Overridable in case that convention doesn't
        # hold for some future caller.
        self._resource_name = resource_name

    def encode_prompt_only(self, prompt: str, batch_size: int):
        """The base class's own encode() (nodes/model/text_encoder.py)
        calls this + resolution_embedding() below and combines them --
        not overridden here, so encode() on this class already gets the
        two-cache behavior below for free, with nothing here needing to
        duplicate that combining step itself."""
        key = (prompt, batch_size)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            self._prompt_cache.move_to_end(key)
            return cached
        if self._resource_control is not None:
            self._resource_control.ensure_loaded(self._resource_name)
        ctx, pooled = self._inner.encode_prompt_only(prompt, batch_size)
        entry = (ctx.detach().cpu(), pooled.detach().cpu())
        self._prompt_cache[key] = entry
        if len(self._prompt_cache) > self._max_entries:
            self._prompt_cache.popitem(last=False)
        return entry

    def resolution_embedding(self, height: int, width: int, batch_size: int):
        key = (height, width, batch_size)
        cached = self._resolution_cache.get(key)
        if cached is not None:
            self._resolution_cache.move_to_end(key)
            return cached
        if self._resource_control is not None:
            self._resource_control.ensure_loaded(self._resource_name)
        res_emb = self._inner.resolution_embedding(height, width, batch_size).detach().cpu()
        self._resolution_cache[key] = res_emb
        if len(self._resolution_cache) > self._max_entries:
            self._resolution_cache.popitem(last=False)
        return res_emb

    def unload(self) -> None:
        self._inner.unload()

    def footprint_bytes(self) -> int:
        # Both caches hold only CPU tensors -- doesn't count toward device
        # footprint at all, so this is exactly self._inner's own.
        return self._inner.footprint_bytes()

    def offload(self) -> None:
        self._inner.offload()

    def reload(self, device: str | None = None) -> None:
        self._inner.reload(device)

    def release(self) -> None:
        self._inner.release()

    def clear_cache(self) -> None:
        self._prompt_cache.clear()
        self._resolution_cache.clear()


class CachingTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True,
                         doc="The real encoder to wrap, e.g. an SDXLTextEncoderNode's output."),
        "max_entries": Port(
            name="max_entries", type=int, required=False, default=512,
            doc="Oldest entry is evicted once either the prompt cache holds more than "
                "this many distinct (prompt, batch_size) pairs, or the resolution cache "
                "holds more than this many distinct (height, width, batch_size) triples "
                "-- independent limits, not a combined one.",
        ),
        "resource_control": Port(
            name="resource_control", type=ResourceControlHandle, required=False, default=None,
            doc="Wire a VRAM Budget Controller node's own output here (the same one "
                "given to the trainer) so a warm cache can actually free VRAM between "
                "hits, not just skip the compute. Optional -- caching still works for "
                "compute without it, just doesn't offload anything on its own.",
        ),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        encoder: TextEncoder = inputs["encoder"]
        max_entries = inputs.get("max_entries", self.INPUTS["max_entries"].default)
        result = {"encoder": CachingTextEncoder(
            encoder, max_entries=max_entries, resource_control=inputs.get("resource_control"))}
        self.validate_outputs(result)
        return result


