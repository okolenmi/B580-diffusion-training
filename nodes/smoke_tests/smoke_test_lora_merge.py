"""Checks nodes/model/lora_merge.py's merge_lora_into_state_dict()
against core.lora.LoRALinear.merge()/LoRAConv2d.merge() -- the
already-real, already-tested reference implementation of this same
merge math, just operating on a live injected module instead of a raw
state dict. Building a real LoRALinear/LoRAConv2d, running its own
.merge(), and comparing the resulting weight against what this
function produces for the same base weight + LoRA A/B/alpha is a
direct proof of correctness, not just an independent reimplementation
checked against itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from core.lora import LoRAConv2d, LoRALinear
from nodes.model.lora_merge import merge_lora_into_state_dict


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_linear_merge_matches_core_lora_reference():
    print("[merge_lora_into_state_dict() matches LoRALinear.merge() exactly, "
          "for a real, non-trivial (trained) A/B]")
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    original_weight = base.weight.detach().clone()
    layer = LoRALinear(base, rank=4, alpha=6.0)
    # Move A/B off their zero-init so the merge isn't a no-op.
    opt = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.5)
    for _ in range(3):
        x = torch.randn(3, 8)
        layer(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

    lora_A, lora_B = layer.lora_A.detach().clone(), layer.lora_B.detach().clone()
    layer.merge()
    reference_weight = layer.base_weight.detach().clone()

    base_sd = {"model.diffusion_model.some.module.weight": original_weight.clone()}
    lora_sd = {
        "lora_unet_some_module.lora_down.weight": lora_A,
        "lora_unet_some_module.lora_up.weight": lora_B,
        "lora_unet_some_module.alpha": torch.tensor([6.0]),
    }
    merged_sd, count = merge_lora_into_state_dict(base_sd, lora_sd, strength=1.0)
    check(count == 1, count)
    torch.testing.assert_close(merged_sd["model.diffusion_model.some.module.weight"],
                                reference_weight)
    print("    PASS")


def check_conv2d_merge_matches_core_lora_reference():
    print("[merge_lora_into_state_dict() matches LoRAConv2d.merge() exactly]")
    torch.manual_seed(1)
    base = nn.Conv2d(4, 6, kernel_size=3, padding=1)
    original_weight = base.weight.detach().clone()
    layer = LoRAConv2d(base, rank=2, alpha=4.0)
    opt = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.5)
    for _ in range(3):
        x = torch.randn(2, 4, 8, 8)
        layer(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

    lora_A, lora_B = layer.lora_A.detach().clone(), layer.lora_B.detach().clone()
    layer.merge()
    reference_weight = layer.base_weight.detach().clone()

    base_sd = {"model.diffusion_model.some.conv.weight": original_weight.clone()}
    lora_sd = {
        "lora_unet_some_conv.lora_down.weight": lora_A,
        "lora_unet_some_conv.lora_up.weight": lora_B,
        "lora_unet_some_conv.alpha": torch.tensor([4.0]),
    }
    merged_sd, count = merge_lora_into_state_dict(base_sd, lora_sd, strength=1.0)
    check(count == 1, count)
    torch.testing.assert_close(merged_sd["model.diffusion_model.some.conv.weight"],
                                reference_weight)
    print("    PASS")


def check_strength_scales_the_merge():
    print("[strength scales the merge -- 0.0 leaves base_sd untouched, 0.5 is "
          "exactly half the 1.0 effect]")
    base = nn.Linear(4, 4)
    original_weight = base.weight.detach().clone()
    lora_A = torch.randn(2, 4)
    lora_B = torch.randn(4, 2)
    lora_sd = {
        "lora_unet_x.lora_down.weight": lora_A,
        "lora_unet_x.lora_up.weight": lora_B,
        "lora_unet_x.alpha": torch.tensor([2.0]),
    }

    zero_sd, count0 = merge_lora_into_state_dict(
        {"model.diffusion_model.x.weight": original_weight.clone()}, lora_sd, strength=0.0)
    check(count0 == 1, "strength=0.0 still counts as a match, just a no-op delta")
    torch.testing.assert_close(zero_sd["model.diffusion_model.x.weight"], original_weight)

    full_sd, _ = merge_lora_into_state_dict(
        {"model.diffusion_model.x.weight": original_weight.clone()}, lora_sd, strength=1.0)
    half_sd, _ = merge_lora_into_state_dict(
        {"model.diffusion_model.x.weight": original_weight.clone()}, lora_sd, strength=0.5)

    full_delta = full_sd["model.diffusion_model.x.weight"] - original_weight
    half_delta = half_sd["model.diffusion_model.x.weight"] - original_weight
    torch.testing.assert_close(half_delta, full_delta * 0.5)
    print("    PASS")


def check_no_match_leaves_base_sd_untouched():
    print("[a lora_sd with nothing matching base_sd -- merged_count == 0, "
          "base_sd's values genuinely untouched]")
    original_weight = torch.randn(4, 4)
    base_sd = {"model.diffusion_model.x.weight": original_weight.clone()}
    lora_sd = {"lora_unet_completely_different.lora_down.weight": torch.randn(2, 4),
               "lora_unet_completely_different.lora_up.weight": torch.randn(4, 2)}
    merged_sd, count = merge_lora_into_state_dict(base_sd, lora_sd)
    check(count == 0, count)
    torch.testing.assert_close(merged_sd["model.diffusion_model.x.weight"], original_weight)
    print("    PASS")


def check_mutates_and_returns_the_same_dict():
    print("[mutates base_sd in place and returns the same object -- not a copy]")
    base_sd = {"model.diffusion_model.x.weight": torch.randn(4, 4)}
    lora_sd = {}
    merged_sd, _ = merge_lora_into_state_dict(base_sd, lora_sd)
    check(merged_sd is base_sd, "should return the exact same dict object")
    print("    PASS")


def main():
    check_linear_merge_matches_core_lora_reference()
    check_conv2d_merge_matches_core_lora_reference()
    check_strength_scales_the_merge()
    check_no_match_leaves_base_sd_untouched()
    check_mutates_and_returns_the_same_dict()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
