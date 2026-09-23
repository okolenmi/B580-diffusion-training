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
    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        """Return (context, pooled_y) tensors for the UNet's conditioning inputs."""

    @abstractmethod
    def unload(self) -> None:
        ...


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

    def __init__(self, legacy_encoder):
        self._legacy = legacy_encoder
        self._device_before_offload = None

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        return self._legacy.encode_for_unet(prompt, batch_size=batch_size, height=height, width=width)

    def unload(self) -> None:
        self._legacy.unload()

    def footprint_bytes(self) -> int:
        """core.clip_encode.SDXLClipEncoder has no footprint accessor of
        its own -- summed here directly from clip_model's (always real)
        and _embedder's (None until encode_for_unet()'s first real call,
        via _get_embedder()'s lazy construction) parameters/buffers.
        0 while offloaded (self._device_before_offload set) -- the
        tensors still exist, just not on any device this counts:
        offload()'s own device-memory usage is 0 by definition, and
        numel()*element_size() alone can't tell CPU-resident from
        device-resident, so this has to be checked explicitly rather
        than left to the summing loop below to get right by accident."""
        if self._legacy is None:
            return 0
        if self._device_before_offload is not None:
            return 0
        total = sum(p.numel() * p.element_size() for p in self._legacy.clip_model.parameters())
        total += sum(b.numel() * b.element_size() for b in self._legacy.clip_model.buffers())
        embedder = self._legacy._embedder
        if embedder is not None:
            total += sum(p.numel() * p.element_size() for p in embedder.parameters())
            total += sum(b.numel() * b.element_size() for b in embedder.buffers())
        return total

    def offload(self) -> None:
        """A direct move, not unload() -- unload() (core.clip_encode.
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
        self._device_before_offload = self._legacy.device
        self._legacy.clip_model = self._legacy.clip_model.cpu()
        if self._legacy._embedder is not None:
            self._legacy._embedder = self._legacy._embedder.cpu()
        self._legacy.device = "cpu"

    def reload(self, device: str | None = None) -> None:
        """No reload() on the legacy class to delegate to -- unload() is
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
        self._legacy.clip_model = self._legacy.clip_model.to(
            device=target, dtype=self._legacy.dtype)
        self._legacy._embedder = None
        self._legacy.device = target
        self._device_before_offload = None

    def release(self) -> None:
        """Genuinely drops the encoder -- unload() alone doesn't (the
        legacy object and its weights survive, just moved to CPU, so
        reload() would still work after it). Moves to CPU first for a
        clean drop, then drops the reference itself, matching
        ComfyUNetTrainableModel.release()'s exact pattern."""
        if self._legacy is not None:
            self._legacy.unload()
            self._legacy = None


class SDXLTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **TextEncoderNode.COMMON_INPUTS,
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        from core.clip_encode import SDXLClipEncoder

        weights: ModelWeights = inputs["weights"]
        legacy = SDXLClipEncoder(weights.non_unet_sd,
                                  device=inputs.get("device", self.INPUTS["device"].default))
        result = {"encoder": SDXLTextEncoder(legacy)}
        self.validate_outputs(result)
        return result
