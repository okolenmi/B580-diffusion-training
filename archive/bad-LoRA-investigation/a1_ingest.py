#!/usr/bin/env python3
"""Ingest the single A1 image into a 1-sample dataset for THIS trainer.

a1_setup.py wrote one padded 512x512 PNG. This puts that same file into a
dataset so the trainer can consume it, which is what makes the A1 arms
comparable: kohya VAE-encodes the PNG, and this trainer VAE-encodes the
same PNG. resize_mode="pad" at latent_size=64 is a no-op on an
already-square 512x512 image, so the latent is of those exact pixels.

Kept separate from a1_setup.py because ingestion needs the dataset to
already exist as v2 (builder.py calls ensure_v2() before anything else)
and because it touches the GPU, so it wants its own process and its own
clean card.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "datasets" / "a1_one"
SRC_PNG = REPO / "runs" / "a1_one_image" / "images" / "a1.png"
CKPT = Path("/home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors")
# Kohya reads a directory of images; the builder reads one directory too.
IMG_DIR = REPO / "runs" / "a1_one_image" / "images"


def main() -> int:
    from manager.builder import DataTaskRunner
    from manager.dataset import ManagedDataset

    if not SRC_PNG.exists():
        raise SystemExit(f"{SRC_PNG} missing; run a1_setup.py first")
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; remove it to re-ingest")
    # Must exist and be v2 BEFORE ingestion -- builder.py calls ensure_v2()
    # on dataset_root/metadata.db, and a hand-made empty directory reads as
    # user_version=0 ("legacy format"). ManagedDataset is what creates it
    # and stamps v2.
    ManagedDataset(OUT).name

    runner = DataTaskRunner(device="xpu")
    t0 = time.monotonic()
    print(f"ingesting 1 image -> {OUT} (pad, latent_size=64 -> px=512)", flush=True)
    runner.run_lora_ingestion_task(
        OUT, CKPT, IMG_DIR,
        latent_size=64, recursive=True, resize_mode="pad",
        neg_prompt="low quality", model_type="eps", seed=42,
        max_aspect_ratio=2.0,   # no-op for pad
        task_id=None,
    )
    print(f"done in {time.monotonic()-t0:.0f}s")

    import sqlite3
    c = sqlite3.connect(OUT / "metadata.db")
    rows = list(c.execute("select latent_h, latent_w from trajectories"))
    print(f"samples: {len(rows)}  shapes: {sorted(set(rows))}")
    assert len(rows) == 1, f"expected exactly 1 sample, got {len(rows)}"
    assert set(rows) == {(64, 64)}, f"expected a 64x64 latent, got {set(rows)}"
    print("verified: 1 sample, latent 64x64 = 512x512 px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())