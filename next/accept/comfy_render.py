"""Renders recipes with ComfyUI itself (its loader, LoRA loader, text encoders, KSampler code and VAE), for comparison with this project's renderer.

Standalone (nothing here imports this repository), to run in the Python environment of a ComfyUI install:

    <comfy python> comfy_render.py --comfy <ComfyUI folder> --checkpoint <file> --recipes recipes.json --out <folder> [--lora <file>]

``recipes.json`` is a list of objects with ``stem`` (the PNG name), ``prompt``, ``negative``, ``seed``, ``steps``, ``cfg``, ``sampler``, ``scheduler``,
``width``, ``height`` and ``strength``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"))
    parser.add_argument("--unet-dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, os.path.abspath(args.comfy))
    sys.argv = ["comfy"] + (["--cpu"] if args.device == "cpu" else []) + ([] if args.unet_dtype == "auto" else [f"--{args.unet_dtype}-unet"])
    import comfy.options
    comfy.options.enable_args_parsing()
    import numpy as np
    import torch
    from PIL import Image

    import comfy.sample
    import comfy.sd
    import comfy.utils

    with open(args.recipes, encoding="utf8") as handle:
        recipes = json.loads(handle.read())
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(args.checkpoint, output_vae=True, output_clip=True)[:3]
    lora = comfy.utils.load_torch_file(args.lora, safe_load=True) if args.lora else None
    os.makedirs(args.out, exist_ok=True)

    def encode(prompt: str):
        return clip.encode_from_tokens_scheduled(clip.tokenize(prompt))

    for n, recipe in enumerate(recipes):
        patched = model
        if lora is not None and recipe["strength"] != 0:
            patched, _ = comfy.sd.load_lora_for_models(model, None, lora, float(recipe["strength"]), 0.0)
        latent = comfy.sample.fix_empty_latent_channels(patched, torch.zeros(1, 4, recipe["height"] // 8, recipe["width"] // 8))
        noise = comfy.sample.prepare_noise(latent, recipe["seed"])
        samples = comfy.sample.sample(patched, noise, recipe["steps"], recipe["cfg"], recipe["sampler"], recipe["scheduler"], encode(recipe["prompt"]), encode(recipe["negative"]),
                                      latent, denoise=1.0)
        pixels = vae.decode(samples.to(torch.float32) if samples.dtype == torch.float16 else samples)
        image = (pixels[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(np.array(image)).save(os.path.join(args.out, recipe["stem"] + ".png"))
        print(f"ComfyUI: {n + 1}/{len(recipes)} {recipe['stem']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
