"""Cheap, file-header-only inspection of a safetensors checkpoint --
dtype per component (unet/clip/vae), without loading any tensor data.

Real capability grounded in safetensors' own file format, confirmed
directly against a real file during development rather than assumed
from documentation: a JSON header (tensor name -> {dtype, shape,
offsets}) sits before the actual weight bytes, and safe_open()'s
get_slice(name).get_dtype() reads only that header entry for the named
key -- verified this returns a string ("BF16", "F32", ...), not a
torch.dtype, and that reading it doesn't materialize any tensor data.
This is what makes resource-dtype detection cheap enough to run the
instant a resource is attached, the way
docs/resources_controller_redesign_plan.md's Phase 2 (server query
endpoint) needs it to be -- and what ModelWeights.inspect_dtypes()
(nodes/model/handle.py) is built on for the same reason.

Key prefixes below match this project's own real checkpoint format --
SDXL, the only architecture this project currently supports.
Generalizing past it is explicitly out of scope until a second real
architecture actually needs it (see the redesign plan's own reasoning
for why, and docs/training_pipeline_design.md's consistent preference
throughout for building the one real thing before extracting an
abstraction from it). UNet lives under "model.diffusion_model.", VAE
under "first_stage_model.", and everything else is treated as CLIP --
matching how this project's own non_unet_sd split has always treated
"not UNet" as one bucket, and how core.clip_encode.SDXLClipEncoder
already does its own internal extraction from exactly that bucket
rather than the nodes/ layer needing to carve out SDXL's two real
text-encoder prefixes (conditioner.embedders.0./.1., CLIP-L and
OpenCLIP-G) itself. That's real, deliberate masking, not a shortcut --
per the resources-controller design, the architecture layer should
present a single, simple "clip" surface regardless of SDXL actually
having two encoders underneath.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# safetensors' own header dtype strings, confirmed directly against a
# real file (not every dtype the format supports -- F8_E4M3/F8_E5M2
# omitted, not real for this project's checkpoints today; an unknown
# string maps to None below rather than raising, since inspection
# should degrade to "couldn't determine" rather than crash the picker
# over a component this project doesn't actually use).
_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _is_unet_key(key: str) -> bool:
    return key.startswith("model.diffusion_model.")


def _is_vae_key(key: str) -> bool:
    return key.startswith("first_stage_model.")


def classify_key(key: str) -> str:
    """'unet' / 'vae' / 'clip' -- see this module's docstring for the
    real prefixes this is grounded in and why "everything else" is
    CLIP rather than SDXL's own two real encoder prefixes being carved
    out here."""
    if _is_unet_key(key):
        return "unet"
    if _is_vae_key(key):
        return "vae"
    return "clip"


@dataclass
class ComponentDtype:
    dtype: torch.dtype | None  # None when key_count == 0 (absent) OR
                                # key_count > 0 with disagreeing dtypes
                                # across that component's own tensors
                                # (a real, honestly-reported condition,
                                # not silently resolved to a majority)
    key_count: int             # 0 means this component has no keys in
                                # the file at all -- distinguishes
                                # "absent" from "present but mixed",
                                # which dtype alone can't, since both
                                # would otherwise collapse to None


def inspect_checkpoint_dtypes(path) -> dict[str, ComponentDtype]:
    """{'unet': ComponentDtype, 'clip': ..., 'vae': ...} for the
    checkpoint at `path`, read from the file's header only -- no tensor
    data is loaded, so this is cheap even for a multi-GB checkpoint."""
    from safetensors import safe_open

    per_component_dtypes: dict[str, set] = {"unet": set(), "clip": set(), "vae": set()}
    per_component_counts: dict[str, int] = {"unet": 0, "clip": 0, "vae": 0}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            component = classify_key(key)
            dtype_str = f.get_slice(key).get_dtype()
            per_component_dtypes[component].add(_SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str))
            per_component_counts[component] += 1

    result = {}
    for component, dtypes in per_component_dtypes.items():
        count = per_component_counts[component]
        resolved_dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
        result[component] = ComponentDtype(dtype=resolved_dtype, key_count=count)
    return result
