"""TextEncoderNode: builds a text encoder from checkpoint weights (SDXL dual CLIP).

TextEncoder extends DeviceResident (nodes/memory/handle.py,
docs/training_pipeline_design.md section 1.2) as of backlog item 12 --
the last conformance gap item 3/9 left open (TrainableModel's landed in
item 9; TextEncoder wasn't blocked on anything, just not itemized until
ResourceCoordinator/OffloadOrchestrator (5.1/5.2) actually needed a
second real DeviceResident besides the model to coordinate anything
meaningful)."""

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
        via _get_embedder()'s lazy construction) parameters/buffers."""
        if self._legacy is None:
            return 0
        total = sum(p.numel() * p.element_size() for p in self._legacy.clip_model.parameters())
        total += sum(b.numel() * b.element_size() for b in self._legacy.clip_model.buffers())
        embedder = self._legacy._embedder
        if embedder is not None:
            total += sum(p.numel() * p.element_size() for p in embedder.parameters())
            total += sum(b.numel() * b.element_size() for b in embedder.buffers())
        return total

    def offload(self) -> None:
        """unload() already does exactly this -- move clip_model/_embedder
        to CPU, clear the device cache -- just not under this name.
        Remembering the pre-offload device here since unload() overwrites
        self._legacy.device to "cpu", same as ComfyUNetTrainableModel's
        offload()/reload() pair in nodes/model/lora_injector.py."""
        self._device_before_offload = self._legacy.device
        self._legacy.unload()

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
