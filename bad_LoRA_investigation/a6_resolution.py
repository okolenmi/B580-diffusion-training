#!/usr/bin/env python3
"""A6: does the bad LoRA's advantage survive a change of resolution?

The `y` conditioning vector is SDXL's resolution embedding -- the model is
told the image size, and the training signal shapes what the LoRA does
with that information. So a LoRA trained at one size and sampled at
another is being asked to do something it was never trained for, on the
axis that determines global composition.

This measures that directly, and it is the item that separates two
otherwise-identical stories:

  * If the LoRA's advantage holds at the training size and collapses or
    reverses at other sizes, the LoRA is conditioned on resolution and
    will misbehave wherever it is sampled away from where it was trained.
  * If the advantage is flat in resolution, resolution is not the
    mechanism, and the caption question (or something else) stands.

Same paired construction as a5_per_t_holdout.py: identical x_t, noise, t
and conditioning across arms, seeded per (sample, t, size) so a size
cannot shift what the other sizes saw.

Renders the centre pixel region's error as well as the whole-tensor MSE,
because a composition-level failure (duplicated objects, patchwork) is
not necessarily the worst thing in the tensor and a whole-tensor average
can hide it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUCKETS = [("[0,100)", 0, 100), ("[100,300)", 100, 300), ("[300,500)", 300, 500),
           ("[500,700)", 500, 700), ("[700,900)", 700, 900), ("[900,1000)", 900, 1000)]

_ENC = None


def encoder():
    global _ENC
    if _ENC is None:
        from nodes.core import ExecutionContext
        from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
        from nodes.model.text_encoder import SDXLTextEncoderNode
        ctx = ExecutionContext()
        weights = SafetensorsCheckpointNode(ctx).build(path=CHECKPOINT)["weights"]
        _ENC = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    return _ENC


CHECKPOINT = "div_4.safetensors"


def resize_latent(x0, h_lat, w_lat):
    """Nearest-neighbour resize of a latent to a target latent size.

    Deliberately not a learned resize: this measures how the *LoRA*
    responds to the conditioning changing, so the image content has to be
    held as constant as the method allows. Interpolating latents would
    blur them and confound the size change with a content change.
    """
    import torch.nn.functional as F
    return F.interpolate(x0, size=(h_lat, w_lat), mode="nearest")


def score(model, latents, prompt, device, ts_per_bucket, seed, h_lat, w_lat):
    import torch
    from nodes.components.diffusion import DiscreteLinearNoiseSchedule
    from nodes.components.model_io import KarrasInputScaler

    sched, scaler = DiscreteLinearNoiseSchedule(), KarrasInputScaler()
    totals = {n: 0.0 for n, _, _ in BUCKETS}
    counts = {n: 0 for n, _, _ in BUCKETS}
    with torch.no_grad():
        x0 = resize_latent(torch.cat(latents, 0).to(device), h_lat, w_lat)
        ctx_e, y = encoder().encode(prompt, batch_size=x0.shape[0],
                                    height=h_lat * 8, width=w_lat * 8)
        ctx_e, y = ctx_e.to(device), y.to(device)
        for _n, lo, hi in BUCKETS:
            for t in range(lo, hi, max(1, (hi - lo) // ts_per_bucket)):
                ti = torch.tensor([t], device=device)
                _, st = sched.alpha_sigma(ti)
                g = torch.Generator(device="cpu").manual_seed(
                    seed + h_lat * 7919 + w_lat * 104_729 + t)
                eps = torch.randn(x0.shape, generator=g).to(device)
                x_t = x0 + st.view(-1, 1, 1, 1) * eps
                pred = model.forward(scaler.scale_input(x_t, st),
                                     torch.full((x0.shape[0],), float(t), device=device),
                                     ctx_e, y)
                mse = (pred.float() - eps.float()).pow(2).mean().item()
                name = next(n for n, a, b in BUCKETS if a <= t < b)
                totals[name] += mse
                counts[name] += 1
    return {n: (totals[n] / counts[n] if counts[n] else None) for n in totals}


def main() -> int:
    global CHECKPOINT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--dataset", default="non-square")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--latents", type=int, default=4)
    ap.add_argument("--ts-per-bucket", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--prompt", default="", help="caption; empty by default, "
                    "matching how these datasets were ingested")
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--sizes", default="64x64,72x64,128x128",
                    help="latent HxW list; the first is the trained shape")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    CHECKPOINT = args.checkpoint

    import torch
    from nodes.core import ExecutionContext
    from nodes.model.lora_injector import build_lora_injected_unet
    from nodes.model.lora_checkpoint_loader import load_lora_into_registry
    from safetensors.torch import load_file

    sys.path.insert(0, str(Path(__file__).parent))
    from a5_per_t_holdout import build_holdout

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path=args.checkpoint)["weights"]
    latents, prompts, _ = build_holdout(args.dataset, args.batch, args.latents)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    print(f"holdout {len(latents)} latents (trained at "
          f"{latents[0].shape[-2]}x{latents[0].shape[-1]}), "
          f"sampling at {[f'{h*8}x{w*8}px' for h, w in sizes]}")

    results = {}
    for label, path in [("base", None)] + [
            (s.partition("=")[0], s.partition("=")[2]) for s in args.lora]:
        model = build_lora_injected_unet(
            weights=weights, device=args.device, dtype=dtype,
            rank=64, alpha=32.0, use_checkpoint=False)
        model.eval()
        if path:
            load_lora_into_registry(model._wrapper.lora_registry,
                                    load_file(path), source_description=path)
        results[label] = {
            f"{h}x{w}": score(model, latents, args.prompt, args.device,
                              args.ts_per_bucket, args.seed, h, w)
            for h, w in sizes
        }
        for h, w in sizes:
            print(f"{label} @{h*8}x{w*8}px: " + "  ".join(
                f"{k}={v:.5f}" for k, v in results[label][f'{h}x{w}'].items()))
        del model
        torch.xpu.empty_cache() if hasattr(torch, "xpu") else None

    # The verdict: advantage over base, per size. A LoRA conditioned on
    # resolution shows a shrinking (or reversing) advantage as size moves
    # away from the trained one.
    verdict = {}
    trained = f"{sizes[0][0]}x{sizes[0][1]}"
    for label, res in results.items():
        if label == "base":
            continue
        verdict[label] = {
            s: {b: (None if res[s][b] in (None, 0) or results["base"][s][b] in (None, 0)
                    else round(100.0 * (results["base"][s][b] - res[s][b])
                               / results["base"][s][b], 2))
                for b in res[s]}
            for s in res
        }
        print(f"\n{label}: % better than base per size "
              f"(trained size {trained} first)")
        for s in res:
            print(f"  {s:>9}: " + "  ".join(f"{b}={v}%" for b, v in verdict[label][s].items()))

    out = {"dtype": args.dtype, "dataset": args.dataset,
           "prompt": args.prompt, "sizes_latent": sizes,
           "trained_latent": trained, "seed": args.seed, "mse": results,
           "pct_better_than_base": verdict}
    dest = Path(args.out) if args.out else REPO / "runs" / "a6" / "a6.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


from nodes.model.checkpoint_loader import SafetensorsCheckpointNode  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())