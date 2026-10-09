#!/usr/bin/env python3
"""latent_stats.py -- are the training latents the same kind of data in two trainers?

Why: on the same images and base model, kohya's average loss was ~50% HIGHER than this
trainer's, yet kohya's LoRA looked better (sharper backgrounds). Equal data + equal model
must give about equal loss. If it does not, the data (or the loss path) differs. A smoother,
lower-variance, lower-scale, or less noisy latent set is an *easier* denoising problem:
lower loss, and a model that learns to over-denoise (blur).

Compares, side by side, statistics of latents (all in SCALED units, i.e. x0 = vae * 0.13025,
which is what the UNet is trained on) from any of:

  --shards DIR      this project's dataset shards (*.safetensors with keys x0_<i>), already scaled
  --kohya DIR       kohya cache (*.npz with key 'latents'); kohya stores them UNSCALED, so x0.13025
  --images DIR      reference encode of an image folder with diffusers' AutoencoderKL taken from
                    --ckpt (SDXL single-file checkpoint), Lanczos resize to --px then center crop;
                    both the posterior MODE and a SAMPLE are reported (kohya uses sample, this
                    project uses mode)

Example (same images in all three):
  python latent_stats.py --shards datasets/aes/shards --kohya kohya_cache/ \
         --images imgs/ --ckpt div_4.safetensors --px 512

What to look at (one row per source):
  std            global std of x0. A source with a clearly smaller std makes the SAME sigma
                 schedule noisier relative to the signal -> easier eps prediction -> lower loss.
  rough          mean |neighbour difference| / std : local roughness. Lower = smoother data.
  hf_energy      fraction of spectral power above half-Nyquist. Lower = blurrier data.
  ch_mean/ch_std per-channel statistics; they should agree between sources to ~a few percent.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

SCALE = 0.13025


def stats_of(latents: list[np.ndarray]) -> dict:
    """latents: list of (4,H,W) float arrays in scaled units."""
    allv = np.concatenate([l.reshape(l.shape[0], -1) for l in latents], axis=1)   # (4, N)
    ch_mean, ch_std = allv.mean(1), allv.std(1)
    rough, hf = [], []
    for l in latents:
        s = l.std() + 1e-8
        dx = np.abs(np.diff(l, axis=2)).mean()
        dy = np.abs(np.diff(l, axis=1)).mean()
        rough.append(0.5 * (dx + dy) / s)
        P = np.abs(np.fft.fft2(l - l.mean(axis=(1, 2), keepdims=True))) ** 2   # (4,H,W)
        h, w = l.shape[1:]
        fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
        r = np.sqrt(fx ** 2 + fy ** 2) / 0.5                                      # 1.0 = Nyquist
        hf.append(float(P[:, r > 0.5].sum() / (P.sum() + 1e-12)))
    return {
        "n": len(latents),
        "shape0": tuple(latents[0].shape),
        "mean": float(allv.mean()), "std": float(allv.std()),
        "p01": float(np.percentile(allv, 1)), "p99": float(np.percentile(allv, 99)),
        "ch_mean": ch_mean, "ch_std": ch_std,
        "rough": float(np.mean(rough)), "hf_energy": float(np.mean(hf)),
    }


def load_shards(d: str) -> list[np.ndarray]:
    from safetensors import safe_open
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)):
        with safe_open(p, framework="numpy") as f:
            for k in f.keys():
                if k.startswith("x0_"):
                    out.append(np.asarray(f.get_tensor(k), dtype=np.float32).reshape(-1, *f.get_tensor(k).shape[-3:])[0])
    return out


def load_kohya(d: str) -> list[np.ndarray]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
        z = np.load(p)
        if "latents" in z:
            out.append(np.asarray(z["latents"], dtype=np.float32) * SCALE)
    return out


def encode_images(d: str, ckpt: str, px: int, device: str):
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image
    vae = AutoencoderKL.from_single_file(ckpt, torch_dtype=torch.float32).to(device).eval()
    modes, samples = [], []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.splitext(p)[1].lower() not in exts:
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size; s = px / min(w, h)
        im = im.resize((max(px, round(w * s)), max(px, round(h * s))), Image.Resampling.LANCZOS)
        w, h = im.size; l, t = (w - px) // 2, (h - px) // 2
        im = im.crop((l, t, l + px, t + px))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = vae.encode(x).latent_dist
            modes.append((dist.mode() * SCALE)[0].cpu().numpy())
            samples.append((dist.sample() * SCALE)[0].cpu().numpy())
    return modes, samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards"); ap.add_argument("--kohya"); ap.add_argument("--images")
    ap.add_argument("--ckpt"); ap.add_argument("--px", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    sources: dict[str, list[np.ndarray]] = {}
    if a.shards: sources["project(shards)"] = load_shards(a.shards)
    if a.kohya:  sources["kohya(npz x0.13025)"] = load_kohya(a.kohya)
    if a.images:
        if not a.ckpt:
            print("--images needs --ckpt", file=sys.stderr); return 2
        m, s = encode_images(a.images, a.ckpt, a.px, a.device)
        sources["ref diffusers MODE"] = m; sources["ref diffusers SAMPLE"] = s
    if not sources:
        ap.print_help(); return 2

    res = {k: stats_of(v) for k, v in sources.items() if v}
    for k, v in sources.items():
        if not v: print(f"[warn] no latents found for {k}")
    hdr = f"{'source':<24}{'n':>5}{'std':>8}{'mean':>8}{'p01':>8}{'p99':>8}{'rough':>8}{'hf_energy':>11}"
    print(hdr); print("-" * len(hdr))
    for k, s in res.items():
        print(f"{k:<24}{s['n']:>5}{s['std']:>8.3f}{s['mean']:>8.3f}{s['p01']:>8.2f}{s['p99']:>8.2f}{s['rough']:>8.3f}{s['hf_energy']:>11.4f}")
    print("\nper-channel (mean | std):")
    for k, s in res.items():
        print(f"{k:<24}" + "  ".join(f"c{i}:{s['ch_mean'][i]:+.2f}|{s['ch_std'][i]:.2f}" for i in range(4)))
    base = next(iter(res.values()))
    if len(res) > 1:
        print("\nratios vs first source (project should be ~1.0 against the reference/kohya):")
        for k, s in list(res.items())[1:]:
            print(f"{k:<24} std x{s['std']/base['std']:.3f}  rough x{s['rough']/base['rough']:.3f}  hf_energy x{s['hf_energy']/max(base['hf_energy'],1e-12):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
