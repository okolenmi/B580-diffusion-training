#!/usr/bin/env python3
"""Regenerate the dataset at SDXL-native scale, matching what Kohya will see.

Why this exists: every one of the 273 samples in `non-square` is 384-768px
on the long side, i.e. entirely below SDXL's native ~1024px (0/273 in the
819-1280 band). The two LoRAs under comparison also saw different data --
201 uncapped source images (kohya) vs 273 aspect-capped samples (this
trainer) -- so the comparison cannot answer "which trainer is better".

Geometry, chosen rather than guessed:

  resize_mode="pad"  scales by px/max(w,h) and pads to px x px. Already
                     implemented (manager/builder.py, the "pad" branch) --
                     no builder change.
  latent_size=128    -> px = 128*8 = 1024.
  max_aspect_ratio   a no-op for pad (the builder's own docstring says so:
                     only "fit" consults the cap), so it is left at the
                     default and its irrelevance is stated rather than
                     relied upon silently.

Result: a fixed 1024x1024 canvas for every image, ~30% average padding,
short side 439-1024px. Nothing for kohya to bucket, so both trainers see
identical pixels and identical geometry -- which is the whole point.

The old dataset is NOT touched. Output goes to a separate directory so the
existing 273-sample set stays available for comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

IMAGES = Path("/home/okolenmi/Downloads/datas")
CKPT = "div_4.safetensors"
OUT_NAME = "pad1024"


def main() -> int:
    from manager.builder import DataTaskRunner
    from manager.dataset import ManagedDataset

    out_root = REPO / "datasets" / OUT_NAME
    if out_root.exists():
        raise SystemExit(f"{out_root} already exists; remove it or pick another name")

    # The dataset must exist and be v2 BEFORE ingestion: builder.py calls
    # ensure_v2() on dataset_root/metadata.db first, and a directory with
    # no DB reads as user_version=0 -> "legacy format, run
    # migrate_dataset_format.py". DatasetManager is what creates the
    # directory and stamps user_version=2 (manager/dataset.py, its
    # __init__ -> init_local_db). Creating the dir by hand looks
    # identical to a legacy dataset and trips the same check.
    ManagedDataset(out_root).name

    # Positional, matching backend/infrastructure/dataset_task_worker.py's
    # call: (dataset_root, model_path, image_dir, ...). model_path is the
    # checkpoint, read for its VAE only.
    ckpt = Path("/home/okolenmi/comfy/ComfyUI/models/checkpoints") / CKPT

    runner = DataTaskRunner(device="xpu")  # this project is B580-only
    t0 = time.monotonic()
    print(f"ingesting {IMAGES} -> {out_root}  "
          f"(pad, latent_size=128 -> px=1024)", flush=True)
    runner.run_lora_ingestion_task(
        out_root,
        ckpt,
        IMAGES,
        latent_size=128,
        recursive=True,
        resize_mode="pad",
        neg_prompt="low quality",
        model_type="eps",
        seed=42,
        max_aspect_ratio=2.0,   # no-op for pad; see module docstring
        task_id=None,
    )
    print(f"done in {time.monotonic()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())