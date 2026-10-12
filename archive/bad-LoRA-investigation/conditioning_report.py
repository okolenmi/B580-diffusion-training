"""Conditioning-robustness report: how much does this LoRA depend on WHICH
conditioning it is given?

Why this exists. A LoRA trained on a captionless dataset (test2: 204
images, 204 empty captions) presents a constant (ctx, y) every step. A
cross-attention projection `W_k . c` whose input never changes gets zero
gradient from any loss-dependent path, so its key/query/value projections
are never shaped to interpret a context -- unconstrained, not underfit.
The result falls apart when conditioning changes at render time, and
measured on TestAes the LEAST disturbing change was the damaging one:
CLIP-L-only moved the context only 0.081 from the training value, yet it
was the worst case to render with.

So the useful question is not "how big is this LoRA" but "does its effect
depend on the conditioning". This answers it directly: run the UNet on
one fixed latent under several conditionings with the LoRA active and
with it neutralised, and report the LoRA-induced change in each. A LoRA
whose effect barely moves across conditionings is conditioning-robust; one
whose effect swings by an order of magnitude has been fitted to a single
input value and will break when that value changes.

Usage:
    python -u archive/bad-LoRA-investigation/conditioning_report.py \
        --lora TestAes --lora B_arm5_const_722 [--prompt "a photo"]

Needs the real checkpoint and the GPU/CPU forward path; ~2 min per LoRA.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths

#: Verdict threshold. The spread is max/min of the LoRA-induced relative
#: change across conditionings. 1.0 = perfectly conditioning-invariant.
#: A healthy LoRA lands well under 2; a captionless-fitted one is far above.
SPREAD_WARN = 2.0


def build_contexts(clip_model, tokenizer, prompt: str):
    """The four conditioning modes, each 2048-wide like the real thing.

    The single-tower modes zero the other half rather than slicing: the
    UNet's cross-attention is Linear(2048 -> inner), so a zeroed half is
    what ComfyUI's one-tower conditioning actually feeds it.
    """
    def pair(text_g, text_l):
        tokens = tokenizer.tokenize(text_g)
        tokens["l"] = tokenizer.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = tokenizer.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] = tokens["l"] + empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] = tokens["g"] + empty["g"]
        return tokens

    def encode(text_g, text_l):
        with torch.no_grad():
            t = pair(text_g, text_l)
            l_out, _ = clip_model.clip_l.encode_token_ids(t["l"])
            g_out, pooled = clip_model.clip_g.encode_token_ids(t["g"])
            n = min(l_out.shape[1], g_out.shape[1])
            l_out, g_out = l_out[:, :n], g_out[:, :n]
            return l_out, g_out, pooled

    l0, g0, pooled_empty = encode("", "")
    l1, g1, pooled_real = encode(prompt, prompt)
    l_only, g_only, _ = encode("", prompt)          # text_l only
    g_only2, _, _ = encode(prompt, "")              # text_g only
    modes = {
        "empty (training)": torch.cat([l0, g0], dim=-1),
        f"full  {prompt[:12]!r}": torch.cat([l1, g1], dim=-1),
        "CLIP-L only": torch.cat([l_only, torch.zeros_like(g_only)], dim=-1),
        "CLIP-G only": torch.cat([torch.zeros_like(l_only), g_only2], dim=-1),
    }
    return modes, pooled_empty, pooled_real


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


def forward_all(unet, x, t, y, modes):
    out = {}
    with torch.no_grad():
        for name, cond in modes.items():
            out[name] = unet(x, t, context=cond.to("xpu", torch.bfloat16), y=y).float().cpu()
    return out


def neutralise(registry):
    """Zero every adapter, returning the saved weights so it can be undone."""
    saved = []
    with torch.no_grad():
        for _n, _p, _a, layer in registry:
            for attr in ("lora_A", "lora_B"):
                t = getattr(layer, attr, None)
                if t is not None:
                    saved.append((t, t.detach().clone()))
                    t.zero_()
    return saved


def restore(saved):
    with torch.no_grad():
        for t, v in saved:
            t.copy_(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", action="append", required=True,
                    help="LoRA stem in ComfyUI models/loras (repeatable)")
    ap.add_argument("--prompt", default="a photo")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the summary table")
    args = ap.parse_args()

    from nodes.core import ExecutionContext
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.text_encoder import SDXLTextEncoderNode

    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path="div_4.safetensors")["weights"]
    enc = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    modes, pooled_empty, pooled_real = build_contexts(
        enc._encoder.clip_model, enc._encoder.tokenizer, args.prompt)
    enc.unload()
    del enc
    gc.collect()

    torch.manual_seed(1234)
    x = torch.randn(1, 4, 64, 64, device="xpu", dtype=torch.bfloat16)
    t = torch.tensor([500.0], device="xpu")

    def y_for(pooled):
        y = torch.zeros(1, 2816, device="xpu", dtype=torch.bfloat16)
        y[:, :1280] = pooled.to("xpu", torch.bfloat16)
        return y

    print("\nBuilding the UNet once for each LoRA...", flush=True)
    unet = build_unet(weights)
    del unet
    gc.collect()
    torch.xpu.empty_cache()

    rows = []
    for stem in args.lora:
        from nodes.model.lora import (LoRAConfig, inject_lora_into_unet,
                                      load_lora_into_model)
        from nodes.model.lora_file import load_lora_state_dict
        p = paths.get_comfy_dir() / "models" / "loras" / f"{stem}.safetensors"
        state = load_lora_state_dict(p)
        rank = int(next(v for k, v in state.items()
                        if k.endswith(".lora_down.weight")).shape[0])
        alpha = float(next(v for k, v in state.items()
                           if k.endswith(".alpha")).reshape(-1)[0])
        print(f"\n{stem} (rank {rank}, alpha {alpha}) ...", flush=True)
        unet = build_unet(weights)
        registry = inject_lora_into_unet(
            unet, LoRAConfig(rank=rank, alpha=alpha,
                             target_modules="kohya_label",
                             target_conditioning_path=False))
        load_lora_into_model(registry, state)
        active = forward_all(unet, x, t, y_for(pooled_real), modes)
        saved = neutralise(registry)
        off = forward_all(unet, x, t, y_for(pooled_real), modes)
        restore(saved)
        del unet
        gc.collect()
        torch.xpu.empty_cache()
        rows.append((stem, {
            n: float((active[n] - off[n]).abs().max()
                     / off[n].abs().max().clamp_min(1e-9))
            for n in modes}))

    names = list(modes)
    print("\n" + "=" * 78)
    print("LoRA-induced change in the predicted noise, per conditioning")
    print("(lower = weaker LoRA; the SHAPE of the row is what matters)")
    print(f"{'LoRA':26s}" + "".join(f"{n[:14]:>16s}" for n in names) + f"{'spread':>9s}")
    for stem, rel in rows:
        vals = [rel[n] for n in names]
        spread = max(vals) / max(min(vals), 1e-9)
        print(f"{stem[:25]:26s}" + "".join(f"{v:16.4f}" for v in vals)
              + f"{spread:8.1f}x")
    print("=" * 78)
    print("spread = max/min of the row. Near 1.0x = the LoRA behaves the")
    print(f"same whatever conditioning it is given. Above ~{SPREAD_WARN}x = it")
    print("has been fitted to one conditioning value and will break when that")
    print("value changes -- the captionless-failure signature.")
    print("\nReference: measured deviations of the CONDITIONING itself from the")
    print("empty-caption training value -- CLIP-L only 0.081, CLIP-G only 0.307,")
    print("full 0.318 -- i.e. the damaging mode is the one that moved least.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())