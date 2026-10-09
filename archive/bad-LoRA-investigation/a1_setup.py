#!/usr/bin/env python3
"""A1: overfit ONE image, 500 steps, both trainers on identical pixels.

Why one image. The maintainer's point is the right one and it is also the
task's own A1 ("do first, about 1 hour"): 200 images over 200 steps cannot
show whether a LoRA is learning, because the objective is being averaged
over 200 different targets. One image over 500 steps has exactly one
target, so any movement in the output is movement in the fit -- and
whether the model converges on it is the cleanest available statement
about whether the training loop is sound.

This also removes the VRAM blocker. Kohya could not start at 1024x1024
(10,268 MB, DEVICE_LOST at step 0) or 768x768 on this 12,216 MB card,
while this trainer completed 1024x1024 at 10,635 MB. At 512x512 with one
image both fit comfortably, so the memory difference stops being the
experiment and the loop becomes it.

Identical pixels, deliberately. The padded 512x512 PNG is written ONCE
here and both trainers consume that same file:
  - kohya VAE-encodes the PNG directly
  - this trainer ingests the same PNG (resize_mode="pad" is a no-op on an
    already-square image, so its latent is of the same pixels)
Anything else -- letting each trainer resize independently -- reintroduces
the exact geometry mismatch that made the first kohya attempt train on
up-to-1408px canvases.

Caption is empty, matching how every dataset in this repo was ingested
(273/273, 204/204, 201/201 empty, counted in the sqlite trajectories
table) and the maintainer's stated goal of a style LoRA whose changes are
not bound to tags.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
SRC = Path("/home/okolenmi/Downloads/datas")
PX = 512
STEPS = 500
PY = "/home/okolenmi/comfy/venv/bin/python"

# One image, chosen deterministically: the first source by sorted name, so
# re-running reproduces the same A1 rather than silently picking another.
def pick() -> Path:
    files = [p for p in sorted(SRC.iterdir())
             if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    return files[0]


def main() -> int:
    src = pick()
    work = REPO / "runs" / "a1_one_image"
    imgs = work / "images"
    imgs.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    with Image.open(src) as im:
        img = im.convert("RGB")
        w, h = img.size
        scale = PX / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (PX, PX), (0, 0, 0))
        canvas.paste(resized, ((PX - nw) // 2, (PX - nh) // 2))
        out_png = imgs / "a1.png"
        canvas.save(out_png)
    with Image.open(out_png) as im:
        got = im.size
    if got != (PX, PX):
        print(f"FAIL: wrote {got}, expected {(PX, PX)}")
        return 1

    meta = work / "metadata.jsonl"
    meta.write_text(json.dumps({"image_path": str(out_png), "caption": ""}) + "\n")

    info = {
        "source": str(src), "source_px": [w, h],
        "prepared": str(out_png), "prepared_px": list(got),
        "caption": "", "steps": STEPS, "resolution": PX,
        "pad_fraction": round(1 - (nw * nh) / (PX * PX), 4),
    }
    (work / "a1_setup.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nkohya --in_json {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())