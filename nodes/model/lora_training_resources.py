"""LoRATrainingSkeleton + SDXL_LoraTrainer: builds a real unet + clip +
vae + lora resources object from a base checkpoint's state dict.
LoRATrainingSkeleton owns everything about LoRA training that doesn't
depend on architecture; SDXLArchitecture (sdxl_architecture.py) owns
the SDXL-specific mechanics. The two combine via inheritance --
SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton), base order
required, see that class's own docstring.

Construction does the real work: __init__ runs the actual pipeline
(merge an optional frozen LoRA into the base weights, split checkpoint,
inject LoRA, build the text encoder, load an optional continue-from
LoRA into the freshly-injected adapter) and the resulting object
already has real .unet/.clip/.vae_sd/.lora attributes -- no separate
build step.

Three methods below are abstract because they're architecture-specific:
an architecture class (SDXLArchitecture) has to provide them.

LoRATrainingSkeleton is also a DeviceResident (nodes/memory/handle.py),
built on nodes/memory/coordinator.py's ResourceCoordinator -- the same
mechanism nodes/train/supervised.py's SupervisedLoRATrainerNode already
uses for its own model/optimizer/text_encoder. .unet and .clip register
with an internal coordinator; footprint_bytes()/offload()/reload()/
release() delegate to it and to .unet/.clip's own DeviceResident
implementations. .vae_sd isn't a DeviceResident (see __init__), so it's
moved/dropped directly alongside the coordinator's work rather than
through it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..memory.coordinator import ResourceCoordinator
from ..memory.handle import DeviceResident, sum_tensor_bytes
from .sdxl_architecture import SDXLArchitecture


class LoRATrainingSkeleton(DeviceResident, ABC):

    @abstractmethod
    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """{"unet": ..., "clip": ..., "vae": ...} -- see
        SDXLArchitecture.split_checkpoint for the implementation."""

    @abstractmethod
    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single object masking however many real text encoders the
        architecture has -- see SDXLArchitecture.build_text_encoder."""

    @abstractmethod
    def inject_lora(self, unet_sd: dict, **kwargs):
        """The adapter-injection entry point -- see
        SDXLArchitecture.inject_lora."""

    def __init__(self, base_model_sd: dict, *, device: str = "xpu", dtype=None,
                 rank: int = 64, alpha: float = 1.0,
                 frozen_lora_sd: dict | None = None, frozen_lora_strength: float = 1.0,
                 continue_lora_sd: dict | None = None,
                 **inject_kwargs):
        """dtype/rank/alpha/**inject_kwargs are resolved values this is
        handed, not detected here -- whatever calls this decides them.

        frozen_lora_sd: a saved LoRA (lora_unet_*.lora_down.weight/
        lora_up.weight/alpha format -- lora_merge.py) merged directly
        into the UNet's base weights before injection. Permanent,
        untrainable: the merged LoRA has no separate identity
        afterward, no object holds it, only its effect on the weights
        remains. frozen_lora_strength scales the merge -- see
        lora_merge.merge_lora_into_state_dict() for exactly what it
        does to the math.

        continue_lora_sd: an existing saved LoRA (same file format as
        frozen_lora_sd) loaded into self.unet's own trainable adapter
        instead of merged into the base -- training continues from
        these weights rather than starting fresh. A different feature
        from frozen_lora_sd: this one stays trainable afterward,
        frozen_lora_sd doesn't and has no separate identity at all.
        Uses load_lora_into_registry() (lora_checkpoint_loader.py) --
        the same validation and loading LoRACheckpointLoaderNode uses
        for the node-graph case, so a missing key or a rank mismatch
        between continue_lora_sd and rank/target_modules above raises
        rather than silently loading a partial LoRA. self.lora is
        continue_lora_sd itself when given, None otherwise -- a plain
        reference (matching self.vae_sd's own raw-dict pattern), not
        the weights themselves, which live inside self.unet's own
        registry once loaded.

        self.vae_sd stays the raw split-out state dict, not a VAE
        object -- nothing in nodes/ builds one yet (only legacy
        core.vae_decode.VAEDecoder, unused elsewhere in nodes/).
        """
        components = self.split_checkpoint(base_model_sd)
        self.frozen_lora_merged_count = 0
        if frozen_lora_sd is not None:
            from .lora_merge import merge_lora_into_state_dict
            components["unet"], self.frozen_lora_merged_count = merge_lora_into_state_dict(
                components["unet"], frozen_lora_sd, strength=frozen_lora_strength)
        self.unet = self.inject_lora(components["unet"], device=device, dtype=dtype,
                                      rank=rank, alpha=alpha, **inject_kwargs)
        self.clip = self.build_text_encoder(components["clip"], device=device)
        self.vae_sd = components["vae"]

        self.lora = continue_lora_sd
        if continue_lora_sd is not None:
            from .lora_checkpoint_loader import load_lora_into_registry
            load_lora_into_registry(self.unet.raw.lora_registry, continue_lora_sd,
                                     source_description="continue_lora_sd")

        self._device = device

        self._coordinator = ResourceCoordinator()
        self._coordinator.register("unet", self.unet)
        self._coordinator.register("clip", self.clip)

    def footprint_bytes(self) -> int:
        """Coordinator's total (unet + clip) plus vae_sd's raw tensors,
        via the same sum_tensor_bytes() helper other DeviceResident
        implementations in this project use for a list of tensors."""
        return self._coordinator.total_footprint_bytes() + sum_tensor_bytes(self.vae_sd.values())

    def offload(self) -> None:
        """unet/clip via the coordinator's bulk offload; vae_sd's raw
        tensors moved to CPU directly since they aren't registered."""
        self._coordinator.offload_all_except(set())
        self.vae_sd = {k: v.cpu() for k, v in self.vae_sd.items()}

    def reload(self, device: str | None = None) -> None:
        """None reloads to the device this was constructed for."""
        target = device or self._device
        self._coordinator.reload("unet", target)
        self._coordinator.reload("clip", target)
        self.vae_sd = {k: v.to(target) for k, v in self.vae_sd.items()}

    def release(self) -> None:
        """Not reversible. unet/clip via their own release() (each
        moves to CPU first, then drops); vae_sd cleared directly."""
        self.unet.release()
        self.clip.release()
        self.vae_sd = {}


class SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton):
    """LoRA training resources for SDXL. An instance has real
    .unet/.clip/.vae_sd/.lora attributes the moment it's constructed.

    Base order (SDXLArchitecture before LoRATrainingSkeleton) is
    required, not stylistic. Python resolves a method by the first
    match walking the MRO left to right. Listing LoRATrainingSkeleton
    first would mean split_checkpoint/build_text_encoder/inject_lora
    all resolve to its @abstractmethod stubs (found first), which never
    call super() -- so SDXLArchitecture's real implementations would
    never be reached and the class would fail to instantiate at all.
    See this module's smoke test for a negative-case check that the
    wrong order genuinely does fail this way."""
    pass
