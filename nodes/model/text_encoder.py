"""TextEncoderNode: builds a text encoder from checkpoint weights (SDXL dual CLIP).

TextEncoder extends DeviceResident (nodes/memory/handle.py)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..core import Node, Port
from ..memory.handle import DeviceResident
from .handle import ModelWeights


class TextEncoder(DeviceResident, ABC):

    @abstractmethod
    def encode_prompt_only(self, prompt: str, batch_size: int):
        """The expensive half of encode(): (context, pooled_y) for this
        prompt alone -- depends only on the prompt string (and
        batch_size, for the repeat), never on height/width. Split out
        from encode() this session specifically so a caller can cache
        this half separately from resolution_embedding()'s (cheap,
        resolution-dependent) one -- see
        nodes/model/text_encoder_cache.py's CachingTextEncoder, the
        reason this split exists at all."""

    @abstractmethod
    def resolution_embedding(self, height: int, width: int, batch_size: int):
        """The cheap half of encode(): the resolution-dependent
        SDXL time-embedding tensor that gets concatenated onto pooled_y.
        Never involves the prompt at all."""

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        """Return (context, pooled_y) tensors for the UNet's conditioning
        inputs -- concrete here, combining the two pieces above, so a
        subclass only has to implement the two granular methods once
        rather than every subclass re-deriving the same "concatenate
        pooled text conditioning with the resolution embedding" step.
        Not abstract anymore as of this session (previously subclasses
        implemented this directly, single-piece); CachingTextEncoder is
        the only reason it needed to become two pieces, but this default
        keeps calling it exactly this way meaning exactly what it always
        meant for every other caller."""
        return self.encode_per_sample(
            prompt, batch_size, [(height, width)] * batch_size)

    @staticmethod
    def _group_by_size(sizes) -> dict:
        """(height, width) -> the sample rows it fills, in order.

        Coercing each component with `int()` rather than indexing it is
        deliberate on both counts: a caller deriving sizes from a tensor
        hands over 0-dim tensors, which are not subscriptable at all, and
        `hash(torch.tensor(512)) != hash(512)`, so without the coercion the
        same size reached through two routes would be two cache keys. It is
        also what CachingTextEncoder reuses to check the keys it is about
        to request, so that check and this request cannot drift apart.
        """
        by_size: dict[tuple[int, int], list[int]] = {}
        for i, (height, width) in enumerate(sizes):
            # int() on each component rather than a tuple(...) or a dict
            # lookup of the pair: a caller that derived sizes from a mask
            # hands over 0-dim tensors, which are not subscriptable, and
            # hash(torch.tensor(512)) != hash(512) anyway.
            by_size.setdefault((int(height), int(width)), []).append(i)
        return by_size

    def encode_per_sample(self, prompt: str, batch_size: int, sizes):
        """`encode()` for a batch whose samples do NOT all share one size.

        `sizes` is one (height, width) pair per sample, in order. The
        returned `y` carries each sample's own resolution embedding on its
        own row, where `encode()` puts the same one on every row. This is
        the one place either of them assembles `y`, so the two cannot
        disagree about what a resolution embedding contributes to it.

        `resolution_embedding()` is asked for one (height, width, count) at
        a time rather than per sample, so a batch of N samples at one size
        costs the same single call `encode()` always made, and only a
        genuinely mixed batch asks more than once. Grouping by *count* as
        well as by size is what keeps CachingTextEncoder's key meaningful:
        its resolution key includes batch_size, so asking count=1 per
        sample would be a different -- and far more numerous -- set of keys
        than anything else in the codebase asks for.
        """
        ctx, pooled = self.encode_prompt_only(prompt, batch_size)
        import torch
        if len(sizes) != batch_size:
            raise ValueError(
                f"encode_per_sample: got {len(sizes)} size(s) for a batch of "
                f"{batch_size}; every sample needs exactly one (height, "
                f"width) or the rows of y cannot be lined up with it")
        by_size = self._group_by_size(sizes)
        rows: list = [None] * batch_size
        for (height, width), idx in by_size.items():
            # One call per distinct size, at the number of rows it fills.
            # `resolution_embedding` repeats its own row, so this block is
            # exactly the tensor `encode()` would have produced for a
            # same-sized batch of this many -- spread back onto the samples
            # that asked for it.
            block = self.resolution_embedding(height, width, len(idx))
            for slot, row in zip(idx, block.unbind(0)):
                rows[slot] = row
        res_emb = torch.stack(rows, dim=0)
        return ctx, torch.cat([pooled, res_emb], dim=-1)

    @abstractmethod
    def unload(self) -> None:
        ...

    def encode_prompts(self, prompts, batch_size: int = 1):
        """Bulk sibling of `encode_prompt_only`; returns a list of pairs.

        Concrete here as *a loop*, deliberately, and that is the honest
        default rather than a placeholder: it is correct for every
        `TextEncoder`, and a subclass that has a batched path overrides it
        to go faster. The contract that matters is that both paths return
        the same values for the same prompt -- a warm pass filling the
        cache through one and a cache miss serving through the other must
        not disagree, or a run's conditioning would depend on its own cache
        state. `SDXLClipEncoder.encode_prompts` is the batched
        implementation, and its docstring has the measured difference
        (19% in float16, 0.17% in float32) that is why it is not simply
        faster.
        """
        return [self.encode_prompt_only(prompt, batch_size)
                for prompt in prompts]


class TextEncoderNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "weights": Port(name="weights", type=ModelWeights, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, TextEncoder]:
        ...


class SDXLTextEncoder(TextEncoder):

    def __init__(self, encoder):
        self._encoder = encoder
        self._device_before_offload = None

    def encode_prompt_only(self, prompt: str, batch_size: int):
        return self._encoder.encode_prompt_and_pool(prompt, batch_size)

    def encode_prompts(self, prompts, batch_size: int = 1):
        """Delegates to the wrapped encoder's bulk path.

        Not the ABC's loop: `SDXLClipEncoder` has a real batched
        implementation, and using the loop here would quietly throw away a
        3x speedup whenever a caller warms through the wrapper -- which is
        the only way the trainer ever reaches it.
        """
        return self._encoder.encode_prompts(prompts, batch_size)

    def resolution_embedding(self, height: int, width: int, batch_size: int):
        return self._encoder.resolution_embedding(height, width, batch_size)

    def unload(self) -> None:
        """Record the device before delegating, or the footprint lies.

        `unload()` and `offload()` are two routes to the same state --
        CLIP in host RAM, nothing on the device -- and they reach it
        differently: `offload()` records the device itself and moves the
        tensors, `unload()` delegates to `SDXLClipEncoder.unload()`, which
        moves them and sets its own `device = "cpu"`. `footprint_bytes()`
        consulted only `offload()`'s record, so a `unload()` left it
        reporting CLIP's full size for a CLIP that was entirely on the
        host.

        That is not cosmetic. `ManagedLoRATrainerNode`'s
        `prewarm_text_encoder` Port warms the cache and then calls this,
        and that Port is the one that frees CLIP's ~1.5 GB for the rest of
        a run -- so under exactly the setting where CLIP is *not* on the
        card, the monitor's VRAM graph reported 1,561 MB of it. Measured
        on the B580: peak reserved 7,666 MB with prewarm on against
        9,228 MB with it off, while the residents line read
        `text_encoder=1561MB` on both.

        Guarded rather than assigned so an `offload()` first is not
        overwritten with the CPU device `unload()` has already moved to.
        """
        if self._device_before_offload is None:
            self._device_before_offload = self._encoder.device
        self._encoder.unload()

    def footprint_bytes(self) -> int:
        """clip_encoder.SDXLClipEncoder has no footprint accessor of
        its own -- summed here directly from clip_model's (always real)
        and _embedder's (None until encode_for_unet()'s first real call,
        via _get_embedder()'s lazy construction) parameters/buffers.
        0 while offloaded (self._device_before_offload set) -- the
        tensors still exist, just not on any device this counts:
        offload()'s own device-memory usage is 0 by definition, and
        numel()*element_size() alone can't tell CPU-resident from
        device-resident, so this has to be checked explicitly rather
        than left to the summing loop below to get right by accident.

        The flag is set by **both** routes that free the device --
        `offload()` and `unload()` -- and that is load-bearing rather than
        tidiness: `unload()` frees the memory inside
        `SDXLClipEncoder` without touching this flag, so a check that
        only knew about `offload()` reported CLIP's full 1,561 MB for a
        CLIP sitting in host RAM. See `unload()`'s own docstring and
        nodes/smoke_tests/smoke_test_sdxl_text_encoder_offload.py, which
        checks both routes."""
        if self._encoder is None:
            return 0
        if self._device_before_offload is not None:
            return 0
        total = sum(p.numel() * p.element_size() for p in self._encoder.clip_model.parameters())
        total += sum(b.numel() * b.element_size() for b in self._encoder.clip_model.buffers())
        embedder = self._encoder._embedder
        if embedder is not None:
            total += sum(p.numel() * p.element_size() for p in embedder.parameters())
            total += sum(b.numel() * b.element_size() for b in embedder.buffers())
        return total

    def offload(self) -> None:
        """A direct move, not unload() -- unload() (clip_encoder.
        SDXLClipEncoder.unload()) also calls gc.collect() +
        empty_cache() internally, appropriate for a one-time "done with
        this encoder for the rest of the run" call, real, avoidable cost
        every time for a per-step offload/reload cycle (this class's own
        DeviceResident.offload(), called every step by
        ManagedLoRATrainerNode's EncodeConditioningPhase,
        nodes/train/managed.py). The caller managing this resident's own
        step loop already has its own empty_cache_every_n_steps
        mechanism for when a real cache-reclaim is actually worth
        its cost; this method shouldn't force one on every single call
        on its own account. Confirmed via profiling this was a real,
        measurable, previously-invisible cost, not a theoretical one --
        see docs/known-issues/pending-testing.md's entry on this."""
        self._device_before_offload = self._encoder.device
        self._encoder.clip_model = self._encoder.clip_model.cpu()
        if self._encoder._embedder is not None:
            self._encoder._embedder = self._encoder._embedder.cpu()
        self._encoder.device = "cpu"

    def reload(self, device: str | None = None) -> None:
        """No reload() on the wrapped encoder to delegate to -- unload() is
        one-directional there. Moves clip_model back explicitly; resets
        _embedder to None rather than moving it, so _get_embedder()'s own
        existing lazy-construction path rebuilds it on the right device
        next time it's actually needed, instead of duplicating that
        device-placement logic here."""
        target = device or self._device_before_offload
        if target is None:
            raise RuntimeError(
                "reload() needs an explicit device, or a prior offload() to "
                "remember one -- neither was given."
            )
        self._encoder.clip_model = self._encoder.clip_model.to(
            device=target, dtype=self._encoder.dtype)
        self._encoder._embedder = None
        self._encoder.device = target
        self._device_before_offload = None

    def release(self) -> None:
        """Genuinely drops the encoder -- unload() alone doesn't (the
        encoder and its weights survive, just moved to CPU, so
        reload() would still work after it). Moves to CPU first for a
        clean drop, then drops the reference itself, matching
        ComfyUNetTrainableModel.release()'s exact pattern."""
        if self._encoder is not None:
            self._encoder.unload()
            self._encoder = None


class SDXLTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **TextEncoderNode.COMMON_INPUTS,
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        from .clip_encoder import SDXLClipEncoder

        weights: ModelWeights = inputs["weights"]
        encoder = SDXLClipEncoder(weights.non_unet_sd,
                                  device=inputs.get("device", self.INPUTS["device"].default))
        result = {"encoder": SDXLTextEncoder(encoder)}
        self.validate_outputs(result)
        return result
