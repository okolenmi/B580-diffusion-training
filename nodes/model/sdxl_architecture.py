"""SDXLArchitecture: SDXL-specific mechanics for building training
resources from a checkpoint -- splitting it into unet/clip/vae,
masking SDXL's two real text encoders behind one simple object, and
adapter injection. No generic multi-architecture abstraction exists;
this is deliberately specific to SDXL, the only architecture this
project supports.

Has no relationship to dtype -- every method here moves or wraps data
at whatever dtype it already has. Dtype decisions belong to the task
layer (nodes/model/lora_training_resources.py), not here.

Each method delegates to an existing implementation rather than
duplicating it:
- split_checkpoint(): resource_inspection.py's classify_key(), the
  SDXL key-prefix classifier.
- build_text_encoder(): core.clip_encode.SDXLClipEncoder wrapped in
  text_encoder.py's SDXLTextEncoder, exactly as SDXLTextEncoderNode
  already builds it.
- inject_lora(): lora_injector.py's build_lora_injected_unet().
"""

from __future__ import annotations


class SDXLArchitecture:
    """Combined with a task class via inheritance. Must be listed
    before the task class in the bases -- Python resolves a method by
    the first match in the MRO, and the task class's abstract stubs
    would otherwise shadow these real implementations. Methods take
    self for consistency with the abstract methods they satisfy, even
    though none currently use it."""

    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """Splits state_dict into {"unet": ..., "clip": ..., "vae": ...}
        by key prefix -- every key ends up in exactly one bucket."""
        from .resource_inspection import classify_key

        result: dict[str, dict] = {"unet": {}, "clip": {}, "vae": {}}
        for key, tensor in state_dict.items():
            result[classify_key(key)][key] = tensor
        return result

    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single TextEncoder masking SDXL's two real text encoders
        (CLIP-L, OpenCLIP-G) behind one object.

        core.clip_encode.SDXLClipEncoder hardcodes its own dtype
        (float16 compute, bfloat16 output) -- there's no parameter to
        override it yet, so a dtype choice for CLIP isn't honored here,
        only whatever SDXLClipEncoder itself uses."""
        from core.clip_encode import SDXLClipEncoder

        from .text_encoder import SDXLTextEncoder

        return SDXLTextEncoder(SDXLClipEncoder(clip_sd, device=device))

    def inject_lora(self, unet_sd: dict, **kwargs):
        """LoRA injection -- delegates to build_lora_injected_unet();
        kwargs (device/dtype/rank/alpha/scaling_policy/dropout/
        target_modules/use_checkpoint/resource_policy/adapter_strategy/
        frozen_weight_store_factory) pass straight through to it."""
        from .handle import ModelWeights
        from .lora_injector import build_lora_injected_unet

        weights = ModelWeights.from_state_dicts(unet_sd, {})
        return build_lora_injected_unet(weights, **kwargs)
