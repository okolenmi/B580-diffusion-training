# Measured: the shape stall is oneDNN primitive-cache thrash — and the fix is 1.54x on real training

**Status: cause identified, fix measured on the real training path, nothing landed in
code yet.** Intel Arc B580 (12,216 MB), torch 2.12.1+xpu.
`scripts/hw_validate.py main --dataset non-square --batch 2 --steps 300`.

Supersedes `REPORT-multires-stall.md` section 2, which framed the question
(one-time vs recurring) correctly but had no hardware to answer it.

## The result

`ONEDNN_PRIMITIVE_CACHE_CAPACITY` defaults to **1024**. That is too small to
hold the primitives for a multi-resolution dataset, so every shape *transition*
re-creates them.

| 300-step run | repeat | revisit | first sighting | revisit/steady | steps/s |
|---|---|---|---|---|---|
| capacity 1024 (default) | 0.873 s | **3.790 s** | 3.989 s | **4.34x** | **0.480** |
| `ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536` | 0.953 s | **0.945 s** | 3.853 s | **0.99x** | **0.741** |
| single-shape `1024 aes` (control) | 1.283 s | — | — | — | 0.769 |

**1.54x throughput**, closing the multi-shape gap from **1.60x slower than
single-shape to 1.04x**. One environment variable, no dataset change, no
change to training numerics.

The mechanism is exactly what it should look like: revisits stop paying
(4.34x -> 0.99x), first sightings keep paying (3.99s -> 3.85s), because those
genuinely are new code. `repeat` is unchanged within noise. Peak reserved is
**identical to the byte** (8774.0 MB) — the oneDNN primitive cache is host
memory and cannot compete for VRAM.

## The cost is per shape TRANSITION, not per shape

Classifying each step by its shape (`latent_shape`, now recorded in
`steps.jsonl`):

| step kind | n | capacity 1024 | capacity 65536 |
|---|---|---|---|
| repeat — same shape as previous step | 161 | 0.873 s | 0.953 s |
| revisit — seen earlier, not previous | 94 | 3.790 s | 0.945 s |
| first sighting | 44 | 3.989 s | 3.853 s |

So a step is fast exactly when it repeats the previous shape, and slow on
*every* transition regardless of whether the shape was seen before. There are
44 first sightings and 118 slow steps: the extra 74 are revisits, and they are
the ones the cache fixes. The cost also explains the bimodal step-time
distribution — 40% of steps take >2s and hold 74% of wall time, while the
**median step (0.873s) is faster than the single-shape control's (1.283s)**.
Small latents really are cheaper; the loss is entirely in transitions.

## Why an earlier revision of this file said the opposite

An earlier draft concluded "revisits are already at steady state, it is a
one-time per-shape cost" and that the cache did nothing. **Both were wrong**,
for two separate reasons, and both are worth recording because each would have
sent the next person the wrong way:

1. **A misaligned reconstruction.** `steps.jsonl` recorded no per-step shape, so
   the shape order was rebuilt by replaying the seeded loader stream and zipping
   it against recorded step times. That join was wrong — it reported revisits at
   0.95x when they are 4.34x. The join is now unnecessary: `latent_shape` is
   recorded per step.
2. **A 40-step run had 3 revisits.** The first training-path test
   (`SHAPE_default` vs `SHAPE_cap65536`, 40 steps) came out 0.413 vs 0.411
   steps/s and I read that as "the fix does not work on training". Three
   revisits cannot detect a 4.34x effect. **The probe was right and my training
   test was too short to refute it.**

The probe's `--repeat 2` (two steps per shape, back to back) means most of its
"revisits" were in fact the second visit of a shape being compiled for the first
time — which is why the probe found the effect so strongly (6.34x). Real
training clumps shapes at mean 2.16 and switches often, so it thrashes the
small cache far more than the probe's own mix suggested at first glance. The
probe and training agree once measured with enough samples.

## Also corrected: the dataset has 44 distinct shapes

An earlier draft claimed 63 and that `docs/known-issues/open.md`'s 44 was
stale. **44 is right.** `SELECT DISTINCT latent_h, latent_w` over the metadata
table gives 63, but the loader emits 44 — it drops 19 samples sitting in
incomplete `(prompt, size)` groups and groups by prompt as well as size.
Reading the table counts shapes the trainer never sees.

## What landed in the repo to make this measurable

`MonitoringPhase` (both routes) now passes the step's latent shape to
`on_step`, and `hw_validate.py` records it as `latent_shape`. Without it, none
of the above is measurable from a run's output — which is why two wrong
conclusions were reachable here. `analyze_steps.py` in this folder, written
before that change, expects `shapes.jsonl` and is superseded by reading
`latent_shape` from `steps.jsonl` directly.

## What to do next

1. **Land `ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536`** (or the smallest value that
   works — see below) in `nodes/xpu_env.py`. Its own comment currently says the
   cache variables are off because they are "not confirmed on real hardware",
   which is now false for this one.
2. **Find the smallest sufficient capacity.** Only 1024 (broken) and 65536 (works)
   are measured; a sweep over 2048/4096/8192/16384/32768 would pin it down. A
   mechanical run (`find_capacity.py`, ~11 min per point on a 300-step run) and
   not a question.
3. **Persistent disk cache** (`SYCL_CACHE_PERSISTENT=1`, `SYCL_CACHE_DIR`) is the
   bigger remaining win, because **every graph run is a fresh child process**
   (MEM-05) — so the 44 first sightings are repaid in full on every run, which
   is ~3.9s x 44 = **~170 s**. Not measured.
4. **Pre-warm** is the same ~170 s paid once per run instead of being amortised
   over a long run; the two are alternatives, not complements.
5. Shape bucketing (44 -> 3 with padding, +15% compute) is *not* needed for the
   throughput problem and would change what the loss is computed over. It was
   the recommendation before this measurement; it is now clearly second-order.

## Method notes

- One run at a time on this card. Two concurrent probe processes caused
  `DEVICE_LOST` (each holds ~10 GB of 12,216 MB). The probe is not the training
  operating point, so its VRAM ceiling says nothing about training.
- `num_alloc_retries` does not exist in this torch build, so the probe's
  "allocator retries" verdict branch is unreachable and was never exercised.
- Per-step memory is identical on fast and slow steps (8774 MB reserved both),
  so this is not memory pressure. Recorded to rule the hypothesis out, not to
  support the conclusion.
