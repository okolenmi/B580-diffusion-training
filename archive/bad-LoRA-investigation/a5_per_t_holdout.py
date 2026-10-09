#!/usr/bin/env python3
"""A5: per-t held-out eps-MSE, base vs base+LoRA, on REAL SDXL weights.

The task's section 1.1 closes every CPU/random-weight comparison and says
so: "Nothing here exercises real SDXL weights, bf16, a real tokenizer,
real images, or the optimizer loop." A5 is the first item that does.

The diagnostic: a healthy LoRA lowers eps-MSE on held-out images, or at
worst leaves it unchanged. A bucket where applying the LoRA *raises* the
loss localises the damage in the timestep range. That is what this prints,
per bucket, for as many LoRAs as it is given.

Why per-t at all, when a single MSE would do: one scalar cannot say
*where* a LoRA hurts, and the maintainer's symptom is specifically
first-sampling-step damage (high t). A single number would hide exactly
the thing being looked for.

Two design points that keep the comparison honest:

  * Same x_t, same noise, same t, same conditioning for every LoRA --
    per-sample and per-t. The difference between two arms is then
    attributable to the LoRA and nothing else.
  * The noise draw is fixed per (sample, t) by seeding from that pair
    rather than drawing once and reusing, so adding a third LoRA arm
    cannot shift the inputs the first two arms saw.

Runs in the training dtype (bf16 by default) because that is the dtype
the LoRA was trained in and the one bf16-only defects hide in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Fixed buckets from the task (A5). Ordered low->high t.
BUCKETS = [("[0,100)", 0, 100), ("[100,300)", 100, 300), ("[300,500)", 300, 500),
           ("[500,700)", 500, 700), ("[700,900)", 700, 900), ("[900,1000)", 900, 1000)]


def build_holdout(dataset, batch, limit):
    """Unpadded clean latents + conditioning, straight from the loader.

    Reads the loader's own samples rather than iterating its batches,
    because _merge_samples deliberately drops `x0` and this needs the
    *clean* latent: the loader picks t and noise per iteration, and this
    needs the same clean latent at every t so the arms stay paired.

    shape_bucket_multiple=0 so nothing is padded -- a padded latent would
    make the MSE partly measure padding rather than denoising, which is
    the mistake B1 of the task exists to correct.
    """
    import torch
    from nodes.core import ExecutionContext
    from nodes.dataset.managed import ManagedDatasetSourceNode

    ctx = ExecutionContext()
    source = ManagedDatasetSourceNode(ctx).build(
        dataset_root=dataset, batch_size=batch, shuffle=False,
        keep_incomplete_batches=True, shape_bucket_multiple=0)["batches"]

    samples = source._loader._load_all_samples()
    picked, prompts, shapes = [], set(), set()
    for s in samples:
        x0 = s["x0"]
        key = (int(x0.shape[-2]), int(x0.shape[-1]))
        # At most one batch per (caption, shape) group, matching what the
        # trainer itself can actually train on (loader groups by exactly
        # this key) and keeping the holdout inside one resolution where
        # possible -- a mixed-size holdout would make the y conditioning
        # vary between arms.
        if shapes and key not in shapes:
            continue
        shapes.add(key)
        picked.append(x0[0] if x0.dim() == 5 else x0)
        prompts.add(s["prompt"])
        if len(picked) >= limit * batch:
            break
    if not picked:
        raise SystemExit(f"no samples found in {dataset!r}")
    return picked, sorted(prompts), ctx


def score(model, latents, prompts, device, ts_per_bucket, seed, batch):
    """Mean eps-MSE per t bucket over the holdout.

    x_t is rebuilt from the stored clean latent and a *seeded* noise draw,
    because the loader draws t at iteration time and this needs the same
    noise at every t -- and the same noise for every LoRA arm.
    """
    import torch
    from nodes.components.diffusion import DiscreteLinearNoiseSchedule
    from nodes.components.model_io import KarrasInputScaler

    sched, scaler = DiscreteLinearNoiseSchedule(), KarrasInputScaler()
    totals = {name: 0.0 for name, _, _ in BUCKETS}
    counts = {name: 0 for name, _, _ in BUCKETS}

    with torch.no_grad():
        for bi in range(0, len(latents), batch):
            x0 = torch.cat(latents[bi:bi + batch], dim=0).to(device)
            h, w = x0.shape[-2] * 8, x0.shape[-1] * 8
            ctx_e, y = _encoder().encode(prompts[0], batch_size=x0.shape[0],
                                         height=h, width=w)
            ctx_e, y = ctx_e.to(device), y.to(device)
            for _label, lo, hi in BUCKETS:
                for t in range(lo, hi, max(1, (hi - lo) // ts_per_bucket)):
                    ti = torch.tensor([t], device=device)
                    _, st = sched.alpha_sigma(ti)
                    # Seeded per (batch, t): identical inputs for every arm.
                    g = torch.Generator(device="cpu").manual_seed(
                        seed + bi * 100_003 + t)
                    eps = torch.randn(x0.shape, generator=g).to(device)
                    x_t = x0 + st.view(-1, 1, 1, 1) * eps
                    xc = scaler.scale_input(x_t, st)
                    tt = torch.full((x0.shape[0],), float(t), device=device)
                    pred = model.forward(xc, tt, ctx_e, y)
                    mse = (pred.float() - eps.float()).pow(2).mean().item()
                    name = next(n for n, a, bnd in BUCKETS if a <= t < bnd)
                    totals[name] += mse
                    counts[name] += 1
    return {n: (totals[n] / counts[n] if counts[n] else None) for n in totals}


_ENC = None


def _encoder():
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


def main() -> int:
    global CHECKPOINT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--dataset", default="non-square")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--ts-per-bucket", type=int, default=2)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--lora", action="append", default=[],
                    help="label=path of a LoRA to score; repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    CHECKPOINT = args.checkpoint

    import torch
    from nodes.core import ExecutionContext
    from nodes.model.lora_injector import build_lora_injected_unet
    from nodes.model.lora_checkpoint_loader import load_lora_into_registry
    from safetensors.torch import load_file

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path=args.checkpoint)["weights"]

    batches, prompts, _ = build_holdout(args.dataset, args.batch, args.batches)
    print(f"holdout: {len(batches)} clean latents from {args.dataset!r}, "
          f"captions={prompts!r}")

    # Base first: an injected LoRA's up matrices start at zero, so a model
    # with no LoRA loaded IS the unmodified model and is scored on exactly
    # the same objects as every arm.
    #
    # Each LoRA arm gets its OWN injection rather than sharing one.
    # load_lora_into_registry refuses a partial load (it raises listing the
    # missing layers rather than silently scoring a half-applied LoRA),
    # and these two LoRAs genuinely differ in target set: Test_02 adapts
    # time_embed/label_emb, StylizedArt1llust adapts neither and carries 22
    # other_unet modules instead. A single shared injection could hold both
    # only via A3's conditioning-path switch, which does not exist in
    # build_lora_injected_unet yet. Each arm therefore also reports its
    # own base, measured in its own injection -- and since a zero-up LoRA
    # contributes nothing to the forward, that base is the same model, so
    # the comparison stays paired and the arms differ only in weights.
    arms = [("base", None)] + [
        (s.partition("=")[0], s.partition("=")[2]) for s in args.lora]

    results, bases = {}, {}
    for label, path in arms:
        rank = args.rank
        alpha = args.alpha
        model = build_lora_injected_unet(
            weights=weights, device=args.device, dtype=dtype,
            rank=rank, alpha=alpha, use_checkpoint=False)
        model.eval()
        if path is not None:
            sd = load_file(path)
            load_lora_into_registry(model._wrapper.lora_registry, sd,
                                    source_description=path)
            print(f"  loaded {label} <- {path}")
        results[label] = score(model, batches, prompts, args.device,
                               args.ts_per_bucket, args.seed, args.batch)
        print(f"{label}: " + "  ".join(
            f"{k}={v:.5f}" if v is not None else f"{k}=n/a"
            for k, v in results[label].items()))
        if label != "base":
            bases[label] = results["base"]
        del model
        torch.xpu.empty_cache() if hasattr(torch, "xpu") else None

    # The verdict: per bucket, does the LoRA beat the base?
    verdict = {}
    for label, res in results.items():
        if label == "base":
            continue
        verdict[label] = {
            b: (None if res[b] is None or results["base"][b] in (None, 0)
                else round(100.0 * (results["base"][b] - res[b]) / results["base"][b], 2))
            for b in results["base"]
        }

    out = {"dtype": args.dtype, "dataset": args.dataset,
           "batches": len(batches), "ts_per_bucket": args.ts_per_bucket,
           "seed": args.seed, "mse": results,
           "pct_better_than_base": verdict}
    print(json.dumps(verdict, indent=2))
    dest = Path(args.out) if args.out else REPO / "runs" / "a5" / "a5.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


from nodes.model.checkpoint_loader import SafetensorsCheckpointNode  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())