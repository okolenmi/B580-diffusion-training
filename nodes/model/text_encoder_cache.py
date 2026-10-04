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
offloaded between hits; on a miss of either half, encode() below calls
ensure_loaded() once for the whole encode -- both keys checked before
either half loads, never once per cold half -- bringing the encoder
back first, including, if something else is currently using the room,
offloading that to make space (ResourceControlHandle.ensure_loaded()'s
own "offload everything else until this one's done" behavior, direct
feedback on the case where model+optimizer are already near budget when
a miss happens). Once per encode, not once per half, because
ensure_loaded() with the encoder already resident is correct but not
free: BudgetedResourceControlHandle.ensure_loaded() unconditionally
runs _make_room(), which reads memory_stats() -- a real device query,
not a cache lookup -- every single call (nodes/memory/control_handle.py).
This class doesn't decide *when* the inner encoder gets
offloaded in the first place -- that's whatever holds this handle
calling before_step() between steps (nodes/train/supervised.py), same as
any other registered resident.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

import torch

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
        # True only while encode() is in flight (set/reset in its
        # try/finally): encode() has already done the both-keys check and
        # called ensure_loaded() at most once, so the halves' own
        # per-miss calls must not fire a second time for the same encode.
        # Direct half calls (outside encode()) see False and keep their
        # own load -- they're public interface with no one else to check.
        self._ensured_for_encode = False

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
        if self._resource_control is not None and not self._ensured_for_encode:
            # Suppressed while encode() is in flight: it already ensured
            # once for this encode (see encode()'s both-keys check).
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
        if self._resource_control is not None and not self._ensured_for_encode:
            # Same suppression as encode_prompt_only()'s -- one
            # ensure_loaded() per encode(), not one per cold half.
            self._resource_control.ensure_loaded(self._resource_name)
        res_emb = self._inner.resolution_embedding(height, width, batch_size).detach().cpu()
        self._resolution_cache[key] = res_emb
        if len(self._resolution_cache) > self._max_entries:
            self._resolution_cache.popitem(last=False)
        return res_emb

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        """Both cache keys checked *before* either half loads: a full
        miss (both halves cold) calls ensure_loaded() exactly once, not
        once per half. ensure_loaded() with the encoder already resident
        is correct but not free -- BudgetedResourceControlHandle runs
        _make_room()'s memory_stats() device query on every call,
        resident or not (nodes/memory/control_handle.py) -- so a
        both-cold encode firing it twice paid that query twice for one
        load, on top of the reload path's own work. This was once
        asserted as one call, relaxed to two when the split-key
        redesign moved ensure into each half (c703aa6 edited the test,
        not this class), and is now one again for real: the halves keep
        their own per-miss ensure for direct callers (they're public
        interface; base TextEncoder.encode() is the only in-repo path,
        which is why this override can suppress the duplicates for the
        duration of the call -- see _ensured_for_encode).
        """
        if self._resource_control is not None:
            prompt_miss = (prompt, batch_size) not in self._prompt_cache
            resolution_miss = (height, width, batch_size) not in self._resolution_cache
            if prompt_miss or resolution_miss:
                self._resource_control.ensure_loaded(self._resource_name)
            self._ensured_for_encode = True
        try:
            return super().encode(prompt, batch_size, height, width)
        finally:
            # Restored even if the inner encoder raises: a flag stuck True
            # would make every later *direct* half call skip its own
            # ensure_loaded() and touch an offloaded encoder.
            self._ensured_for_encode = False

    def unload(self) -> None:
        self._inner.unload()

    def footprint_bytes(self) -> int:
        # Both caches hold only CPU tensors -- doesn't count toward device
        # footprint at all, so this is exactly self._inner's own.
        return self._inner.footprint_bytes()

    def cache_bytes(self) -> int:
        """Host RAM the two caches currently occupy. 0 device bytes.

        `footprint_bytes()` deliberately answers a *device* question and so
        reports 0 for both caches, which is right and useless for the one
        question that decides whether prewarm is affordable: how big does
        warming this dataset get?

        Summed from the tensors actually stored rather than estimated,
        because the estimate is what has to be checked. Measured per entry
        on the B580: **621 KB per prompt key** (a (1, 77, 2048) float32
        context plus ~0.3 KB pooled) and ~0.3 KB per resolution key. So a
        dataset with 100,000 distinct captions costs 60.6 GB of host RAM
        to prewarm, and 1,000,000 costs 606 GB -- which is why the
        prewarm path reports this number rather than leaving it to be
        discovered by the machine running out.
        """
        total = 0
        for cache in (self._prompt_cache, self._resolution_cache):
            for entry in cache.values():
                for tensor in entry:
                    if torch.is_tensor(tensor):
                        total += tensor.numel() * tensor.element_size()
        return total

    def offload(self) -> None:
        self._inner.offload()

    def reload(self, device: str | None = None) -> None:
        self._inner.reload(device)

    def release(self) -> None:
        self._inner.release()

    def bind_resource_control(self, resource_control: ResourceControlHandle,
                              resource_name: str | None = None) -> None:
        """Late-bind the handle this class loads its inner encoder
        through on a cache miss (see this class's own docstring). For a
        cache built before its handle existed -- e.g.
        LoRATrainingConfigNode's `cache_text_encoder` wrap, which has
        no handle to give (it runs before the trainer node's
        `resource_control` input is anywhere in scope), later bound by
        ManagedLoRATrainerNode's `prewarm_text_encoder` Port. A no-op
        when a handle is already bound -- the constructor's own value
        wins, first one in keeps the slot, same first-wins shape this
        codebase's other idempotent wiring uses.
        """
        if self._resource_control is None:
            self._resource_control = resource_control
            if resource_name is not None:
                self._resource_name = resource_name

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


