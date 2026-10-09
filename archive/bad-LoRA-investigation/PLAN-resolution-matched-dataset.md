# PLAN: resolution-matched dataset for a fair A2 comparison

Status: **plan, not measured.** Every number in "measured" below was read off
the dataset DB or the source images, not from a training run.

## What is wrong with the current comparison

The two LoRAs in ComfyUI's `loras/` (`A2_kohya.safetensors`,
`a2match_mine.safetensors`, `a2match_nocond.safetensors`) were trained on
**different data**, so they cannot answer "which trainer is better".

| | Kohya arm | this trainer's arm |
|---|---|---|
| images | 201 source images | 273 dataset samples |
| geometry | resized into 384x384 buckets | 63 native odd shapes (392x512, 704x512, ...) |
| modules | 722 + 22 (incl. conv/other) | 560 attention-only |

## The resolution finding (measured, from the dataset)

Every one of the 273 samples is **384-768 px on the long side**. Counted
against SDXL's native ~1024 px:

    samples with long side in 819-1280 px:  0 / 273

So both LoRAs were trained entirely below the base model's native scale. 63
distinct latent shapes, most common 90x64, 49x64, 64x56.

This is a larger effect than the module-count difference and is a plausible
cause of the reported background blur: a LoRA trained on 63 inconsistent odd
shapes learns a blurrier average than one trained on a consistent canvas.
Reported by the maintainer as "x3 with kohya = sharp backgrounds, x3 with ours
= less detailed / blurrier".

## Why scale-by-short-side is the wrong geometry here

`resize_mode="fit"` (what the dataset used) scales by **short** side to `px`.
On portrait sources that puts the long side at 1300-1500 px and averages
1.53 Mpx per image:

    fit   @1024 (short side):  5.84x the activation memory of a 512 square
    pad   @1024 (long side):   2.79x
    source aspect ratios: min 1.00, median 1.41, MAX 2.33

Max aspect 2.33, so a cap of 2.0 keeps 198/201 images whole (99%). The
1024x2048 case feared for kohya does not occur in this data.

## The fix: `resize_mode="pad"` at px=1024

Already implemented (`manager/builder.py`, the `elif resize_mode == "pad"`
branch) -- no builder change needed. It scales by `px / max(w, h)` and pads
to `px x px`:

    canvas            1024 x 1024 for every image (latent 128x128)
    avg pad fraction  29.8%
    short side        439-1024 px, median 724 (inside SDXL's range)

This resolves the bucketing objection too: with a fixed 1024x1024 canvas
there is nothing for kohya to bucket, so both trainers see identical pixels
and identical geometry. Padding is what this project's shape bucketing
already handles, including the L1 true-size conditioning fix.

## Rank is not the lever (measured arithmetic)

Dropping rank shrinks only optimizer state, while activations at 1024 px are
what bind:

    rank  8:  14.3M params, 0.16 GB optimizer state
    rank 16:  28.7M params, 0.32 GB
    rank 32:  57.3M params, 0.64 GB
    rank 64: 114.7M params, 1.28 GB

64 -> 8 saves ~1.1 GB. Not enough to buy 1024 px. Rank is still worth
lowering for the comparison if both arms use the same value, but it does not
resolve the memory question.

## VRAM: the open risk

Not yet measured. The current 512 px run peaked 8,810 MB **while ComfyUI held
7.8 GB**, so it was contended and is not a clean baseline. 1024 px is ~2.8x
the activation memory. The maintainer reports batch 4 at 1024 with
checkpointing + CLIP unloading and batch 2 as optimal, but that was for a
square 1024 dataset, not this 2.8x one.

Plan: regenerate, then run batch 1 first and only step up if it fits.

## Not yet done

- dataset not regenerated
- no LoRA trained on the new data
- step timings still contended and should be re-measured on an idle card
- no image rendered by the harness in any arm

## Reproduce the measured parts

    # resolution distribution of the current dataset
    python -c "import sqlite3; c=sqlite3.connect('datasets/non-square/metadata.db'); \
      rows=[(h*8,w*8) for h,w in c.execute('select latent_h,latent_w from trajectories')]; \
      print(len(rows), 'samples,', len(set(rows)), 'shapes,', \
            sum(1 for h,w in rows if 819<=max(h,w)<=1280), 'near native')"

    # source aspect ratios and the fit-vs-pad memory comparison
    python -c "import glob; from PIL import Image; \
      rows=[Image.open(p).size for p in glob.glob('/home/okolenmi/Downloads/datas/*')]; \
      print('max ar', max(max(w,h)/min(w,h) for w,h in rows))"