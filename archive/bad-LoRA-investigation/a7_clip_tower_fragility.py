"""Why does one LoRA break under CLIP-L-only conditioning but not CLIP-G?

SDXL's cross-attention context is 2048-wide: CLIP-L's 768 dims
concatenated with CLIP-G's 1280. At inference ComfyUI can build a
context with either half zeroed ("only CLIP-L" / "only CLIP-G"). A LoRA
trained on the joint context then sees an input it never saw, and the
question is *how much* the LoRA's delta moves the prediction under each.

This measures it on real weights: same latent, same seed, same timestep,
three contexts, run with and without the LoRA. The reported number is the
relative size of the LoRA-induced change in the predicted noise, per
conditioning. A LoRA that generalises moves all three about equally; one
that has latched onto a half it should not have moves one of them far
more -- which is the signature of a broken render.

Usage:
    python -u archive/bad-LoRA-investigation/a7_clip_tower_fragility.py \
        --lora TestAes --compare B_arm5_const_722
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths


def build_contexts(clip_model, tokenizer, prompt: str):
    """(full, clip_l_only, clip_g_only, pooled) -- all 2048-wide."""
    pair = tokenizer.tokenize(prompt)      # {"l": [[ids]], "g": [[ids]]}
    with torch.no_grad():
        l_out, _ = clip_model.clip_l.encode_token_ids(pair["l"])
        g_out, g_pooled = clip_model.clip_g.encode_token_ids(pair["g"])
        length = min(l_out.shape[1], g_out.shape[1])
        l_out, g_out = l_out[:, :length], g_out[:, :length]
        full = torch.cat([l_out, g_out], dim=-1)
        l_only = torch.cat([l_out, torch.zeros_like(g_out)], dim=-1)
        g_only = torch.cat([torch.zeros_like(l_out), g_out], dim=-1)
    return full, l_only, g_only, g_pooled


def build_unet(weights):
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W
    cfg = dict(W.SDXL_CONFIG)
    cfg["use_checkpoint"] = False
    with torch.device("meta"):
        unet = UNetModel(**cfg)
    unet.to_empty(device="cpu")
    sd = {k.replace("model.diffusion_model.", ""): v
          for k, v in weights.unet_sd.items()
          if k.startswith("model.diffusion_model.")}
    unet.load_state_dict(sd, strict=False)
    return unet.to(device="xpu", dtype=torch.bfloat16)


def apply_lora(unet, state, rank=None, alpha=None, zero_group=None):
    """Inject at the shape the FILE actually is, not what we guessed.

    Rank is read off a `lora_down` tensor rather than passed in: TestAes is
    rank 64 while the study arms are rank 16, and a mismatch here is a
    hard assert that says nothing about the question being asked.
    """
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet, load_lora_into_model
    down = next(v for k, v in state.items() if k.endswith(".lora_down.weight"))
    real_rank = int(down.shape[0])
    akey = next(k for k in state if k.endswith(".alpha"))
    real_alpha = float(state[akey].reshape(-1)[0])
    cfg = LoRAConfig(rank=real_rank, alpha=real_alpha,
                     target_modules="kohya_label", target_conditioning_path=False)
    registry = inject_lora_into_unet(unet, cfg)
    load_lora_into_model(registry, state)
    if zero_group:
        # Neutralise exactly one adapter group, leaving everything else, so
        # its share of the LoRA's effect is attributable rather than argued.
        n = 0
        with torch.no_grad():
            for full_name, _p, _a, layer in registry:
                if full_name.split(".")[0] == zero_group:
                    layer.lora_A.zero_(); layer.lora_B.zero_(); n += 1
        print(f"    zeroed {n} adapters in group {zero_group!r}", flush=True)
    return unet, real_rank


def forward_all(unet, x, timesteps, time_emb, contexts):
    out = {}
    with torch.no_grad():
        for name, cond in contexts.items():
            c = cond.to("xpu", torch.bfloat16)
            out[name] = unet(x, timesteps, context=c, y=time_emb).float().cpu()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True, help="LoRA stem in ComfyUI models/loras")
    ap.add_argument("--compare", default=None, help="a second LoRA stem to contrast")
    ap.add_argument("--zero-group", default=None,
                    help="adapter group to zero after loading (e.g. label_emb), "
                         "to attribute the LoRA's effect to one group")
    ap.add_argument("--prompt", default="a photo",
                    help="prompt to condition on; pass '' to reproduce the "
                         "training regime of a captionless dataset")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()

    from nodes.core import ExecutionContext
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.text_encoder import SDXLTextEncoderNode

    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path="div_4.safetensors")["weights"]
    encoder = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    clip_model = encoder._encoder.clip_model
    tokenizer = encoder._encoder.tokenizer
    full, l_only, g_only, pooled = build_contexts(clip_model, tokenizer, args.prompt)
    print(f"prompt={args.prompt!r}")
    print(f"contexts: {tuple(full.shape)}  (CLIP-L 768 + CLIP-G 1280)")
    print(f"  mean |activation|: CLIP-L half {full[..., :768].abs().mean():.5f} | "
          f"CLIP-G half {full[..., 768:].abs().mean():.5f}")
    encoder.unload()
    del encoder
    gc.collect()

    contexts = {"full (L+G)": full, "CLIP-L only": l_only, "CLIP-G only": g_only}
    torch.manual_seed(1234)
    x = torch.randn(1, 4, 64, 64, device="xpu", dtype=torch.bfloat16)
    timesteps = torch.tensor([500.0], device="xpu")
    time_emb = torch.zeros(1, 2816, device="xpu", dtype=torch.bfloat16)
    time_emb[:, :1280] = pooled.to("xpu", torch.bfloat16)

    print("\nbuilding base UNet (no LoRA)...", flush=True)
    unet = build_unet(weights)
    base = forward_all(unet, x, timesteps, time_emb, contexts)
    print("  base forwards done", flush=True)
    del unet
    gc.collect()
    torch.xpu.empty_cache()

    rows = []
    for stem in [args.lora] + ([args.compare] if args.compare else []):
        from nodes.model.lora_file import load_lora_state_dict
        p = paths.get_comfy_dir() / "models" / "loras" / f"{stem}.safetensors"
        state = load_lora_state_dict(p)
        print(f"\nloading {stem} (rank from file) ...", flush=True)
        unet = build_unet(weights)
        unet, real_rank = apply_lora(unet, state, zero_group=args.zero_group)
        got = forward_all(unet, x, timesteps, time_emb, contexts)
        del unet
        gc.collect()
        torch.xpu.empty_cache()
        rel = {}
        for name in contexts:
            d = (got[name] - base[name]).abs().max()
            rel[name] = float(d / base[name].abs().max().clamp_min(1e-9))
        rows.append((stem, rel))
        print("   ", flush=True)

    print("\n=== LoRA-induced change in the predicted noise (relative) ===")
    print(f"{'LoRA':24s} {'full (L+G)':>12s} {'CLIP-L only':>13s} {'CLIP-G only':>13s} {'L-only / full':>14s}")
    for stem, rel in rows:
        ratio = rel["CLIP-L only"] / max(rel["full (L+G)"], 1e-12)
        print(f"{stem:24s} {rel['full (L+G)']:12.4f} {rel['CLIP-L only']:13.4f} "
              f"{rel['CLIP-G only']:13.4f} {ratio:14.2f}x")
    print("\nA ratio near 1.0 means the LoRA behaves the same whether or not the\n"
          "CLIP-G half is present; >>1 under CLIP-L-only means it has learned to\n"
          "lean on the half that disappears, which is what breaks the render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())