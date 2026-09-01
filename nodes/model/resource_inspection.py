"""Cheap, header-only inspection of safetensors files -- checkpoint
dtype per component (unet/clip/vae), and saved-LoRA dtype/rank --
without loading any tensor data.

safetensors stores a JSON header (tensor name -> {dtype, shape,
offsets}) before the actual weight bytes. safe_open()'s get_slice(name)
reads only that header entry -- get_dtype() returns a string ("BF16",
"F32", ...), not a torch.dtype, get_shape() returns the tensor's shape
-- without materializing any tensor data. This is why inspection here
is cheap even for a multi-GB checkpoint.

Checkpoint key prefixes are SDXL-specific, the only architecture this
project currently supports: UNet lives under "model.diffusion_model.",
VAE under "first_stage_model.", and everything else is treated as
CLIP. SDXL actually has two text encoders
(conditioner.embedders.0./.1., CLIP-L and OpenCLIP-G); they're
deliberately not split apart here -- core.clip_encode.SDXLClipEncoder
already extracts both from this same "everything else" bucket
internally, so downstream code only ever needs to deal with one "clip"
component.

Saved-LoRA files use lora_unet_<module>.lora_down.weight/lora_up.weight/
alpha keys (lora_phases.py's lora_key()) -- architecture-independent,
not SDXL-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# safetensors' own header dtype strings. Not exhaustive (F8_E4M3/F8_E5M2
# omitted -- not used by any checkpoint this project handles); an
# unknown string maps to None rather than raising.
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
    """'unet' / 'vae' / 'clip', by SDXL key prefix."""
    if _is_unet_key(key):
        return "unet"
    if _is_vae_key(key):
        return "vae"
    return "clip"


def dtype_to_str(dtype: torch.dtype | None) -> str | None:
    """torch.bfloat16 -> "bfloat16", None -> None. str(torch.bfloat16)
    alone returns "torch.bfloat16"; this is the plain form for an HTTP
    response or a UI label."""
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


@dataclass
class ComponentDtype:
    dtype: torch.dtype | None  # None when key_count == 0 (component absent) or
                                # key_count > 0 with disagreeing dtypes across
                                # that component's own tensors (mixed) -- these
                                # two cases are distinguished by key_count, not
                                # collapsed into one meaning of None
    key_count: int             # 0 = this component has no keys in the file


def inspect_checkpoint_dtypes(path) -> dict[str, ComponentDtype]:
    """{'unet': ComponentDtype, 'clip': ..., 'vae': ...} for the
    checkpoint at `path`, read from the file's header only."""
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


@dataclass
class LoRAInspection:
    dtype: torch.dtype | None  # None when key_count == 0 (not a LoRA file, or
                                # empty) or when key_count > 0 with disagreeing
                                # dtypes across the file's own lora_down.weight
                                # tensors -- same absent-vs-mixed distinction as
                                # ComponentDtype, via key_count
    rank: int | None           # None under the same two conditions as dtype
    key_count: int              # number of lora_down.weight keys found


def inspect_lora(path) -> LoRAInspection:
    """dtype and rank for a saved LoRA file, read from the header only
    -- rank is each lora_down.weight tensor's own shape[0], which
    safe_open()'s get_shape() reads from the header the same way
    get_dtype() does."""
    from safetensors import safe_open

    dtypes: set = set()
    ranks: set = set()
    count = 0
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if not key.endswith(".lora_down.weight"):
                continue
            slice_ = f.get_slice(key)
            dtypes.add(_SAFETENSORS_DTYPE_TO_TORCH.get(slice_.get_dtype()))
            ranks.add(slice_.get_shape()[0])
            count += 1

    return LoRAInspection(
        dtype=next(iter(dtypes)) if len(dtypes) == 1 else None,
        rank=next(iter(ranks)) if len(ranks) == 1 else None,
        key_count=count,
    )
