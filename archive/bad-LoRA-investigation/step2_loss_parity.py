#!/usr/bin/env python3
"""Step 2: step-0 loss parity on IDENTICAL tensors, real SDXL weights.

Why: kohya's average training loss was ~50% HIGHER than this trainer's on
the same images, yet kohya's LoRA is better. Equal data + equal model must
give about equal loss at step 0 (LoRA starts at zero contribution). A 50%
gap means the two trainers solve different problems, and the direction
(this trainer's is easier) predicts over-denoising, i.e. blur.

Step 1 (latent_stats.py) already exonerated the latents: project shards ==
diffusers reference (std/rough/hf all x1.000), and the VAE posterior std is
~0 (max 0.001 on real weights), so mode-vs-sample is also dead.

This tests what is left: the (x_t, target, t) pairing and the loss path.
Fixed (x0, eps, t) triples, empty-caption conditioning, LoRA off:

  (a) this project's UNet + input scaling, on XPU in bf16 (the training dtype)
  (b) diffusers UNet2DConditionModel + DDPMScheduler.add_noise, CPU fp32

(a)=(b) within ~2% per t bucket => the loss path is exonerated and the gap
is in training dynamics (optimizer/schedule/weighting), not in what the
model is asked to predict. Any disagreement localises the bug to the first
component that differs.

The pairing argument, stated so it can be checked: this project's
x_t = x0 + sigma*eps, scaled by 1/sqrt(sigma^2+1). Since sigma^2+1 =
1/alpha^2, that equals alpha*x0 + alpha*sigma*eps with
alpha*sigma = sqrt(1-ac) -- exactly diffusers' add_noise output. If (a)
and (b) still disagree, the pairing is NOT what differs; the timestep
embedding, the conditioning, or the UNet call itself is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CKPT = "/home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors"
N_TRIPLES = 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="a1_one")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from nodes.core import ExecutionContext
    from nodes.components.diffusion import DiscreteLinearNoiseSchedule
    from nodes.components.model_io import KarrasInputScaler
    from nodes.dataset.managed import ManagedDatasetSourceNode
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.lora_injector import build_lora_injected_unet
    from nodes.model.text_encoder import SDXLTextEncoderNode

    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path="div_4.safetensors")["weights"]
    source = ManagedDatasetSourceNode(ctx).build(
        dataset_root=args.dataset, batch_size=1, shuffle=False,
        keep_incomplete_batches=True, shape_bucket_multiple=0)["batches"]
    samples = source._loader._load_all_samples()
    assert len(samples) == 1, f"expected the 1-sample A1 dataset, got {len(samples)}"
    x0 = samples[0]["x0"]
    prompt = samples[0]["prompt"]
    print(f"x0 {tuple(x0.shape)} std={float(x0.std()):.4f} prompt={prompt!r}")

    enc = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    h, w = x0.shape[-2] * 8, x0.shape[-1] * 8
    ctx_e, y = enc.encode(prompt, batch_size=1, height=h, width=w)
    print(f"ctx {tuple(ctx_e.shape)} y {tuple(y.shape)}")

    # The pooled half of y is the first 1280 columns (pooled text), the rest
    # is the resolution time-embedding. diffusers wants them split: pooled as
    # text_embeds, raw size values as time_ids.
    pooled = y[:, :1280]
    time_ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=torch.long)

    sched = DiscreteLinearNoiseSchedule()
    g = torch.Generator().manual_seed(args.seed)
    triples = []
    for _ in range(N_TRIPLES):
        t = int(torch.randint(1, 1000, (1,), generator=g).item())
        eps = torch.randn(x0.shape, generator=g)
        triples.append((t, eps))

    # --- arm (a): this project's path, XPU bf16, LoRA injected but zero ---
    model = build_lora_injected_unet(
        weights=weights, device=args.device, dtype=torch.bfloat16,
        rank=16, alpha=16.0, use_checkpoint=False)
    model.eval()
    mses_a, per_t_a = [], []
    with torch.no_grad():
        for t, eps in triples:
            ti = torch.tensor([t])
            _, st = sched.alpha_sigma(ti)
            x_t = x0 + st.view(-1, 1, 1, 1) * eps
            xc = KarrasInputScaler().scale_input(x_t, st).to(args.device)
            # tt must ride along: the wrapper casts it to float32 but does
            # not move it, so a CPU tt against XPU activations dies in
            # time_embed's first linear.
            tt = torch.full((1,), float(t), device=args.device)
            pred = model.forward(xc, tt, ctx_e.to(args.device), y.to(args.device))
            mses_a.append(float((pred.float().cpu() - eps).pow(2).mean()))
            per_t_a.append(t)
    del model
    if hasattr(torch, "xpu"):
        torch.xpu.empty_cache()

    # --- arm (b): diffusers, CPU fp32 ---
    from diffusers import DDPMScheduler, UNet2DConditionModel
    # .to("cpu") explicitly: from_single_file may leave the model on the
    # accelerator via accelerate hooks, and arm (b) is defined as the CPU
    # fp32 reference -- its tensors are all CPU.
    d_unet = UNet2DConditionModel.from_single_file(CKPT).eval().to("cpu")
    d_sched = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085,
                            beta_end=0.012, beta_schedule="scaled_linear",
                            clip_sample=False)
    mses_b = []
    # .cpu() on every input: the project's encoder returns ctx/y already on
    # XPU, and passing an XPU tensor into the CPU reference dies in
    # get_aug_embed's concat -- the same device bug, one arm later.
    ctx_cpu, pooled_cpu = ctx_e.cpu().float(), pooled.cpu().float()
    x0_cpu = x0.cpu()
    with torch.no_grad():
        for t, eps in triples:
            tt = torch.full((1,), t, dtype=torch.long)
            noisy = d_sched.add_noise(x0_cpu, eps.cpu(), tt)
            out = d_unet(sample=noisy.float(),
                         timestep=tt,
                         encoder_hidden_states=ctx_cpu,
                         added_cond_kwargs={"text_embeds": pooled_cpu,
                                            "time_ids": time_ids}).sample
            mses_b.append(float((out - eps.cpu()).pow(2).mean()))

    buckets = [(0, 100), (100, 300), (300, 500), (500, 700), (700, 900), (900, 1000)]
    print(f"\n{'bucket':>12}{'n':>4}{'project':>10}{'diffusers':>10}{'rel diff':>10}")
    verdict = {}
    for lo, hi in buckets:
        ia = [m for m, t in zip(mses_a, per_t_a) if lo <= t < hi]
        ib = [m for m, t in zip(mses_b, per_t_a) if lo <= t < hi]
        if not ia:
            print(f"[{lo:>4},{hi:<5}){'0':>4}  (no samples)")
            continue
        ma = sum(ia) / len(ia)
        mb = sum(ib) / len(ib)
        rel = abs(ma - mb) / mb if mb else float("nan")
        verdict[f"[{lo},{hi})"] = {"n": len(ia), "project": ma,
                                   "diffusers": mb, "rel": rel}
        print(f"[{lo:>4},{hi:<5}){len(ia):>4}{ma:>10.5f}{mb:>10.5f}{rel:>9.1%}")

    out = {"seed": args.seed, "dataset": args.dataset, "n": N_TRIPLES,
           "buckets": verdict}
    dest = Path(args.out) if args.out else REPO / "runs" / "step2" / "step2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())