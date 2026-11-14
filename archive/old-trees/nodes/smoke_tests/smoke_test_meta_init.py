"""Meta-device construction: eager vs meta builds compute the same function.

Evidence class throughout: tiny random weights, CPU. (Real-weight
confirmation -- full UNet + both CLIP towers -- is a maintainer-queue item;
the mechanism is identical, but only the tiny case runs here.)

Covers task A2: after every load_state_dict(strict=False) on a meta-built
model, uncovered tensors are filled (allowlisted) or raise; forward outputs
are compared bit-for-bit, not state dicts; a pruned state dict fails loudly.
"""

import sys
import os
sys.path.insert(0, os.environ.get("REPO", "."))

import torch

from nodes.model.lora import LoRAConfig, inject_lora_into_unet
from nodes.model.meta_init import verify_loaded
from nodes.model.unet import UNetModel

# Same shape family as archive/bad-LoRA-investigation/unet_vs_diffusers.py
# (which checks this UNet against diffusers itself).
TINY_UNET = dict(image_size=16, in_channels=4, out_channels=4, model_channels=32,
                 num_res_blocks=[2, 2, 2], channel_mult=[1, 2, 4],
                 num_head_channels=16, use_spatial_transformer=True,
                 transformer_depth=[0, 0, 1, 1, 2, 2], transformer_depth_middle=2,
                 transformer_depth_output=[0, 0, 0, 1, 1, 1, 2, 2, 2],
                 context_dim=64, use_linear_in_transformer=True,
                 num_classes="sequential", adm_in_channels=64,
                 use_checkpoint=False)

# Same shape family as archive/bad-LoRA-investigation/clip_vs_hf.py.
TINY_CLIP = {"hidden_size": 64, "intermediate_size": 128,
             "num_attention_heads": 4, "num_hidden_layers": 4,
             "max_position_embeddings": 77, "hidden_act": "quick_gelu",
             "eos_token_id": 49407, "vocab_size": 4096}


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _rand_sd(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def check_unet_meta_matches_eager():
    print("[tiny UNet: meta+to_empty+load computes the same function as "
          "eager+load, before and after LoRA injection]")
    torch.manual_seed(11)
    eager = UNetModel(**TINY_UNET).eval()
    sd = _rand_sd(eager)
    with torch.device("meta"):
        meta = UNetModel(**TINY_UNET).eval()
    meta.to_empty(device="cpu")
    miss, unexp = meta.load_state_dict(sd, strict=False)
    verify_loaded(meta, miss, unexp, owner="tiny-UNet")
    torch.manual_seed(13)
    x = torch.randn(2, 4, 16, 16)
    ctx = torch.randn(2, 77, 64)
    y = torch.randn(2, 64)
    t = torch.tensor([30.0, 700.0])
    with torch.no_grad():
        a = eager(x, t, context=ctx, y=y)
        b = meta(x, t, context=ctx, y=y)
    check(torch.equal(a, b),
          f"meta/eager forward differs (max {(a - b).abs().max().item():.2e})")
    check(a.abs().max() > 0, "vacuous forward (all zeros)")
    # Same weights, same LoRA init seed -> injected forwards agree too.
    torch.manual_seed(17)
    reg_e = inject_lora_into_unet(eager, LoRAConfig(rank=2, alpha=1.0))
    torch.manual_seed(17)
    reg_m = inject_lora_into_unet(meta, LoRAConfig(rank=2, alpha=1.0))
    check(len(reg_e) == len(reg_m) and len(reg_e) > 0,
          "LoRA injection hit different module sets")
    with torch.no_grad():
        a2 = eager(x, t, context=ctx, y=y)
        b2 = meta(x, t, context=ctx, y=y)
    check(torch.equal(a2, b2),
          "meta/eager forward differs after identical LoRA injection")
    print("    PASS")


def check_pruned_state_dict_fails_loudly():
    print("[deleted key: verify_loaded raises naming it, no silent garbage]")
    torch.manual_seed(11)
    eager = UNetModel(**TINY_UNET).eval()
    sd = _rand_sd(eager)
    victim = sorted(sd.keys())[len(sd) // 2]
    del sd[victim]
    with torch.device("meta"):
        meta = UNetModel(**TINY_UNET).eval()
    meta.to_empty(device="cpu")
    miss, unexp = meta.load_state_dict(sd, strict=False)
    try:
        verify_loaded(meta, miss, unexp, owner="tiny-UNet")
    except ValueError as e:
        check(victim in str(e),
              f"error must name the missing key {victim!r}; got: {e}")
        print("    PASS")
        return
    check(False, f"pruned key {victim!r} loaded silently")


def check_allow_fill_zeroes_and_unknown_raises():
    print("[allowlist fills zeros; anything else missing or unexpected raises]")
    mod = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    sd = _rand_sd(mod)
    del sd["0.weight"]
    miss, unexp = mod.load_state_dict(sd, strict=False)
    verify_loaded(mod, miss, unexp, allow_fill=frozenset({"0.weight"}),
                  owner="seq")
    check(bool((mod[0].weight == 0).all()),
          "allowlisted tensor was not deterministically zeroed")
    sd2 = _rand_sd(mod)
    del sd2["1.weight"]
    miss2, unexp2 = mod.load_state_dict(sd2, strict=False)
    try:
        verify_loaded(mod, miss2, unexp2, allow_fill=frozenset({"0.weight"}),
                      owner="seq")
    except ValueError:
        pass
    else:
        check(False, "non-allowlisted missing key passed silently")
    try:
        verify_loaded(mod, [], ["mystery.key"], owner="seq")
    except ValueError as e:
        check("mystery.key" in str(e), "unexpected key not named")
    else:
        check(False, "unexpected key passed silently")
    print("    PASS")


def check_clip_meta_matches_eager():
    print("[tiny CLIP tower: meta+to_empty+load matches eager forward]")
    from nodes.model.clip import SDClipModel
    cfg = dict(TINY_CLIP)
    torch.manual_seed(21)
    eager = SDClipModel(cfg, layer="hidden", layer_idx=-2,
                        layer_norm_hidden_state=False,
                        special_tokens={"start": 1, "end": 2, "pad": 2},
                        return_projected_pooled=True).eval()
    sd = _rand_sd(eager)
    with torch.device("meta"):
        meta = SDClipModel(cfg, layer="hidden", layer_idx=-2,
                           layer_norm_hidden_state=False,
                           special_tokens={"start": 1, "end": 2, "pad": 2},
                           return_projected_pooled=True).eval()
    meta.to_empty(device="cpu")
    miss, unexp = meta.load_state_dict(sd, strict=False)
    verify_loaded(meta, miss, unexp, owner="tiny-CLIP")
    torch.manual_seed(23)
    tokens = torch.randint(3, cfg["vocab_size"], (2, 77))
    with torch.no_grad():
        za, pa = eager.encode(tokens)
        zb, pb = meta.encode(tokens)
    check(torch.equal(za, zb) and torch.equal(pa, pb),
          "meta/eager CLIP forward differs")
    check(za.abs().max() > 0, "vacuous CLIP forward")
    print("    PASS")


def main():
    check_unet_meta_matches_eager()
    check_pruned_state_dict_fails_loudly()
    check_allow_fill_zeroes_and_unknown_raises()
    check_clip_meta_matches_eager()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
