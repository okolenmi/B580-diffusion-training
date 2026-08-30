"""Merges a saved LoRA's weights directly into a checkpoint's own raw
state dict -- the LoRA becomes a permanent part of the base weights
before anything else touches them, not a separate trainable adapter
sitting alongside the base. There's no "frozen LoRA" object anywhere
after this runs; the effect is baked into the tensors themselves.

Operates on plain state dicts (checkpoint-style {module.path.weight:
tensor} on one side, saved-LoRA-style {lora_unet_module_path.lora_down.weight:
tensor, ...} on the other), not live nn.Module trees -- a different
operation from core.lora.LoRALinear.merge()/DoRALinear (which merge an
already-injected module's own live parameters in place). Same
underlying math (delta = B @ A, scaled by alpha/rank), applied to raw
tensors before injection instead of to an injected layer after it.
"""

from __future__ import annotations

import torch

from .lora_phases import lora_key


def merge_lora_into_state_dict(base_sd: dict, lora_sd: dict, strength: float = 1.0) -> tuple[dict, int]:
    """For every weight in base_sd that has a matching LoRA entry in
    lora_sd, replaces it with base_weight + strength * (alpha/rank) *
    (B @ A) -- the standard LoRA merge formula, matching
    core.lora.LoRALinear.merge()/LoRAConv2d.merge() exactly, just
    computed on raw tensors instead of live parameters.

    strength scales the merge on top of the LoRA's own saved alpha/rank
    scaling: 1.0 applies it as saved, 0.0 leaves base_sd untouched,
    other values scale the effect up or down. Corresponds to
    core.lora.LoRALinear's own `weight` constructor argument (also a
    multiplier on scaling), applied here at merge time instead of at
    injection time.

    Mutates base_sd's matched entries in place and returns
    (base_sd, merged_count) -- merged_count is how many weights
    actually got a LoRA applied, 0 meaning nothing in lora_sd matched
    anything in base_sd (wrong checkpoint, wrong LoRA, or strength=0).
    Caller decides whether to pass a copy of base_sd if the original
    needs to stay untouched.
    """
    merged_count = 0
    for full_key in list(base_sd.keys()):
        if not full_key.endswith(".weight"):
            continue
        module_name = full_key[: -len(".weight")]
        prefix = lora_key(module_name)
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"
        if down_key not in lora_sd or up_key not in lora_sd:
            continue

        W = base_sd[full_key]
        A = lora_sd[down_key]
        B = lora_sd[up_key]
        rank = A.shape[0]
        alpha_key = f"{prefix}.alpha"
        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else float(rank)
        scaling = strength * (alpha / rank)

        if A.dim() == 4:
            out_channels = W.shape[0]
            re_A = A.view(rank, -1)
            re_B = B.view(out_channels, rank)
            delta = (re_B @ re_A).view_as(W)
        else:
            delta = B @ A

        base_sd[full_key] = W + (delta * scaling).to(W.dtype)
        merged_count += 1

    return base_sd, merged_count
