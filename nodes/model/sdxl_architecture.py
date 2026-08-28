"""SDXLArchitecture: the concrete, architecture-specific "workplace"
half of Phase 4's resources-controller design
(docs/resources_controller_redesign_plan.md) -- checkpoint splitting,
masking SDXL's real two text encoders behind one simple surface,
adapter injection. Fixed and concrete on purpose, not a pluggable
N-architecture abstraction: SDXL is the only architecture this project
supports today, and generalizing past it happens later, when a second
real architecture actually needs it -- matches this project's own
consistent preference throughout docs/training_pipeline_design.md for
building the one real thing before extracting an abstraction from it,
and the resources-controller design conversation's own explicit
instruction to the same effect.

Deliberately has no relationship to dtype at all -- every method here
just moves/wraps real data at whatever dtype it's already in. Dtype
decisions live entirely in the ResourcePreset layer (Phase 4's other
half, see nodes/model/lora_training_resources.py), not here.

Every method below delegates to something already real and tested in
this project rather than reimplementing it -- this class is assembly,
not new logic:
- split_checkpoint(): nodes/model/resource_inspection.py's classify_key(),
  the single real implementation of this project's SDXL key-prefix
  convention (Phase 1).
- build_text_encoder(): exactly what nodes/model/text_encoder.py's
  SDXLTextEncoderNode already builds -- core.clip_encode.SDXLClipEncoder
  wrapped in SDXLTextEncoder, masking SDXL's real two encoders (CLIP-L,
  OpenCLIP-G) behind one simple TextEncoder object. That masking already
  existed; this doesn't reinvent it, just gives it a second, reusable
  entry point outside the node.
- inject_lora(): nodes/model/lora_injector.py's build_lora_injected_unet(),
  extracted for exactly this kind of reuse.
"""

from __future__ import annotations


class SDXLArchitecture:
    """Combined with a task skeleton via inheritance (concrete mixin
    listed first in the bases, per this project's own choice after
    working through the real MRO mechanics -- see
    docs/resources_controller_redesign_plan.md's Phase 4 section for
    why composition was considered and inheritance chosen instead).
    Every method here is a plain instance method (not @staticmethod)
    for consistency with how a task skeleton declares its abstract
    counterparts, even though none of them currently need self --
    a future architecture mixin might."""

    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """{"unet": ..., "clip": ..., "vae": ...} -- three real,
        disjoint sub-dicts of state_dict, split by
        resource_inspection.classify_key()'s real SDXL prefixes. Every
        key in state_dict ends up in exactly one bucket."""
        from .resource_inspection import classify_key

        result: dict[str, dict] = {"unet": {}, "clip": {}, "vae": {}}
        for key, tensor in state_dict.items():
            result[classify_key(key)][key] = tensor
        return result

    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single TextEncoder (nodes/model/text_encoder.py) masking
        SDXL's real two encoders behind one simple object -- see this
        module's own docstring for why this is a second entry point to
        existing masking, not new masking logic.

        core.clip_encode.SDXLClipEncoder (frozen legacy code) hardcodes
        its own dtype (float16 compute, bfloat16 output) -- no dtype
        parameter exists to pass through yet. Real, current limitation,
        not silently pretended to work: a dtype choice for CLIP in a
        future Resources Controller UI can exist as a control, but
        honoring anything other than this hardcoded default needs
        further work in core.clip_encode (frozen) or a casting wrapper
        around it, neither of which is attempted here."""
        from core.clip_encode import SDXLClipEncoder

        from .text_encoder import SDXLTextEncoder

        return SDXLTextEncoder(SDXLClipEncoder(clip_sd, device=device))

    def inject_lora(self, unet_sd: dict, **kwargs):
        """The real adapter-injection entry point -- delegates directly
        to build_lora_injected_unet(), not reimplemented. kwargs passed
        straight through (device/dtype/rank/alpha/scaling_policy/
        dropout/target_modules/use_checkpoint/resource_policy/
        adapter_strategy/frozen_weight_store_factory -- see that
        function's own signature, this doesn't duplicate its defaults
        by repeating them here)."""
        from .handle import ModelWeights
        from .lora_injector import build_lora_injected_unet

        weights = ModelWeights.from_state_dicts(unet_sd, {})
        return build_lora_injected_unet(weights, **kwargs)
