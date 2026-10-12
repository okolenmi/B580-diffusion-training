"""Why does CLIP-L-only conditioning break a LoRA on one checkpoint but not another?

`CLIPTextEncodeSDXL` takes `text_g` and `text_l` separately. "Only
CLIP-L" means text_l carries the prompt and text_g is empty, so the
CLIP-G half of the 2048-wide context is the encoding of the empty
string -- and ComfyUI pads the shorter tower's sections with `tokenize("")`.

A LoRA trained on a captionless dataset was fitted against
`[encode_L("") | encode_G("")]`. The question this measures, per
checkpoint, is how far each rendering mode moves the conditioning away
from that training-time value:

  full(L+G)  a real prompt in both towers      -- "best quality"
  CLIP-L only  text_l=prompt, text_g=""        -- the reported failure
  CLIP-G only  text_g=prompt, text_l=""        -- reported to work

If the CLIP-L-only deviation is much larger than the CLIP-G-only one on
one checkpoint and not the other, that is the asymmetry.

Usage:
    python -u archive/bad-LoRA-investigation/a8_tower_asymmetry.py \
        --checkpoints div_4.safetensors divA.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths


def tokenize_pair(tokenizer, text_g: str, text_l: str):
    """Reproduce CLIPTextEncodeSDXL's token handling exactly.

    ComfyUI tokenises each tower from its OWN text and pads the shorter
    tower's section list with `tokenize("")`, so the sequence length is
    the max of the two. Mirrored here rather than approximated, because
    the padding is part of what differs between the modes.
    """
    tokens = tokenizer.tokenize(text_g)
    tokens["l"] = tokenizer.tokenize(text_l)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = tokenizer.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] = tokens["l"] + empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] = tokens["g"] + empty["g"]
    return tokens


def build(clip_model, tokenizer, text_g, text_l):
    tokens = tokenize_pair(tokenizer, text_g, text_l)
    with torch.no_grad():
        l_out, _ = clip_model.clip_l.encode_token_ids(tokens["l"])
        g_out, pooled = clip_model.clip_g.encode_token_ids(tokens["g"])
        length = min(l_out.shape[1], g_out.shape[1])
        l_out, g_out = l_out[:, :length], g_out[:, :length]
        ctx = torch.cat([l_out, g_out], dim=-1)
    return ctx, l_out, g_out, pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=["div_4.safetensors", "divA.safetensors"])
    ap.add_argument("--prompt", default="a photo")
    args = ap.parse_args()

    from safetensors.torch import load_file
    from nodes.model.clip_encoder import SDXLClipEncoder

    ck_dir = paths.get_comfy_dir() / "models" / "checkpoints"
    prompt = args.prompt

    for name in args.checkpoints:
        p = ck_dir / name
        if not p.exists():
            print(f"{name}: not found, skipping")
            continue
        sd = load_file(str(p))
        # SDXLClipEncoder wants the checkpoint's own CLIP tensors under the
        # keys this project's converter reads.
        from nodes.model.clip_encoder import _extract_and_convert_clip_state_dict
        enc = SDXLClipEncoder(_wrap(sd), device="cpu")
        cm, tok = enc.clip_model, enc.tokenizer

        base, l0, g0, pooled0 = build(cm, tok, "", "")            # training-time
        full, l1, g1, _ = build(cm, tok, prompt, prompt)
        l_only, l2, g2, _ = build(cm, tok, "", prompt)           # text_g empty
        g_only, l3, g3, _ = build(cm, tok, prompt, "")           # text_l empty

        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp_min(1e-9))

        print(f"\n=== {name} ===")
        print(f"  empty-prompt context: |CLIP-L half| mean {l0.abs().mean():.5f} | "
              f"|CLIP-G half| mean {g0.abs().mean():.5f} | seq {base.shape[1]}")
        print(f"  relative deviation from the TRAINING (empty) conditioning:")
        print(f"    full (L+G)    {rel(full, base):.4f}")
        print(f"    CLIP-L only   {rel(l_only, base):.4f}   <-- reported to break")
        print(f"    CLIP-G only   {rel(g_only, base):.4f}   <-- reported to work")
        asym = rel(l_only, base) / max(rel(g_only, base), 1e-9)
        print(f"    L-only / G-only asymmetry: {asym:.2f}x")
        print(f"  which half moved:")
        print(f"    CLIP-L only: CLIP-L half {rel(l2, l0):.4f}, CLIP-G half {rel(g2, g0):.4f}")
        print(f"    CLIP-G only: CLIP-L half {rel(l3, l0):.4f}, CLIP-G half {rel(g3, g0):.4f}")
        del enc, cm
        import gc
        gc.collect()

    return 0


def _wrap(sd):
    """SDXLClipEncoder reads non-UNet keys; hand it the whole checkpoint."""
    return {k: v for k, v in sd.items() if not k.startswith("model.diffusion_model.")}


if __name__ == "__main__":
    raise SystemExit(main())