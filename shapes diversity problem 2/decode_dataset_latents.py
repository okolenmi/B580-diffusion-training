#!/usr/bin/env python3
"""Decode the dataset's own stored latents and look at them.

The mosaic the user sees in LoRA output has been chased through the checkpoint
(a hand-assembled block merge -- unusual but reported working), the LoRA's
adapter coverage (attention-only, 3.6% of parameters), and the VAE scaling
factor (wrongly suspected; SDXL really is 0.13025). None of those is the layer
this looks at.

**The latents are frozen in the dataset and every LoRA ever trained on it saw
exactly these numbers.** So if they decode to mosaic images, the mosaic is in
the data, it is resolution-independent, it is consistent across different LoRA
attempts without anything having to be reproducible, and no amount of training
will remove it -- because the training objective would be reproducing the mosaic.

That is one hypothesis among several and this is the cheapest test of it: three
latents, decoded, no training, no sampler, no GPU needed (CPU VAE decode while
the validation sweep holds the card).

Also reported, because they are free and they would each be their own finding:
  * the raw latent std per channel -- a per-channel imbalance would show up here
    and would mean the VAE encode was wrong even if the decode looks fine;
  * the latent's spatial autocorrelation along each axis -- a blocky latent has
    low correlation between neighbouring columns/rows at block boundaries, which
    is measurable without decoding anything at all.

Decoding the largest and smallest stored shapes rather than three arbitrary
ones, so a shape-dependent effect would show up rather than average out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def load_samples(dataset: str):
    from manager.loader import ManagedDatasetLoader
    from paths import resolve_safe_dataset_path
    loader = ManagedDatasetLoader(
        dataset_root=resolve_safe_dataset_path(dataset), batch_size=1,
        shuffle=False, shape_bucket_multiple=0)
    return loader._load_all_samples()


def latent_stats(x0: torch.Tensor) -> dict:
    """Per-channel std and neighbour correlation along H and W.

    Both are decode-free signals for blockiness: if the VAE encode were wrong,
    the latent itself carries the structure, and a block pattern shows up as
    reduced correlation across block boundaries before any image exists.
    """
    t = x0.float()[0]                       # (C, h, w)
    out = {"per_channel_std": t.std(dim=(1, 2)).tolist(),
           "overall_std": float(t.std())}
    flat = t.mean(dim=0)                    # (h, w), channels averaged
    dh = float(torch.corrcoef(torch.stack([
        flat[:-1].reshape(-1), flat[1:].reshape(-1)]))[0, 1])
    dw = float(torch.corrcoef(torch.stack([
        flat[:, :-1].reshape(-1), flat[:, 1:].reshape(-1)]))[0, 1])
    out["neighbour_corr_h"] = dh
    out["neighbour_corr_w"] = dw
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="non-square")
    ap.add_argument("--checkpoint",
                    default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/opencode/latent_decode")
    args = ap.parse_args(argv)

    samples = load_samples(args.dataset)
    shapes = [(int(s["x0"].shape[-2]), int(s["x0"].shape[-1])) for s in samples]
    uniq = sorted(set(shapes))
    # Spread across the shape range rather than taking the first N, so a
    # shape-dependent problem cannot hide behind a convenient sample.
    picks = [uniq[0], uniq[len(uniq) // 2], uniq[-1]][:args.count]
    print(f"{len(samples)} samples, {len(uniq)} distinct shapes; "
          f"decoding {len(picks)}: {picks}  ({args.device})\n")

    print("latent statistics, before decoding anything:")
    for shape in picks:
        s = next(x for x in samples
                 if (int(x["x0"].shape[-2]), int(x["x0"].shape[-1])) == shape)
        st = latent_stats(s["x0"])
        pc = " ".join(f"{v:.3f}" for v in st["per_channel_std"])
        print(f"  {shape[0]:>3}x{shape[1]:<3} latent  std {st['overall_std']:.3f}  "
              f"per-channel [{pc}]  neighbour corr h {st['neighbour_corr_h']:.4f} "
              f"w {st['neighbour_corr_w']:.4f}")
    print("  (per-channel std that is far from uniform would mean the encode "
          "was wrong;\n   neighbour correlation well below 1 is what blockiness "
          "looks like in a latent)")

    from safetensors import safe_open
    from nodes.model.vae_decode import VAEDecoder
    with safe_open(args.checkpoint, "pt") as f:
        vae_sd = {k[len("first_stage_model."):]: f.get_tensor(k)
                  for k in f.keys() if k.startswith("first_stage_model.")}
    vae = VAEDecoder.from_vae_sd(vae_sd, device=args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    print()
    for shape in picks:
        s = next(x for x in samples
                 if (int(x["x0"].shape[-2]), int(x["x0"].shape[-1])) == shape)
        img = vae.decode(s["x0"].float())[0]        # (3, H, W) uint8
        path = out_dir / f"latent_{shape[0]}x{shape[1]}.png"
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(path)
        print(f"  decoded {shape[0]}x{shape[1]} latent -> "
              f"{tuple(img.shape[-2:])} px  saved {path}")
    print()
    print("Look at these. If they are mosaic, the dataset is mosaic and every "
          "LoRA trained\non it was learning to reproduce mosaic -- which "
          "would explain the symptom being\nconsistent across different "
          "attempts with nothing reproducible in common.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
