# Multi-resolution training on the Arc B580: stalls on new latent shapes

For the project's developers. Status: **diagnosis tooling and a ranked list of
fixes; no fix is claimed to work until the probe has run on the B580.** I had
no XPU. Everything below that is not from the project's own measurements is
marked as a hypothesis.

## 1. What the project has already established
From `docs/known-issues/open.md` (multi-resolution entry):
* A real non-square dataset: 44 distinct latent shapes (H 48-95, W 48-93).
  Training ran at **0.412 steps/s against 0.813 for a single shape (~2x slower)**.
* GPU utilisation oscillates between ~0% and ~90%; 73% of step time is spent
  blocked in `run_backward` / `conv2d` waiting on the device.
* Grouping same-shape batches ("clumps") barely helps: the mean same-shape run
  was 2.12 batches.
* Rounding each side up to a multiple of 32 gives 3 shapes at +15% compute;
  it was not adopted because padding changes what the loss is computed over.
* `nodes/xpu_env.py` sets `SYCL_CACHE_IN_MEM=1` and
  `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD=0`; its own comment says this is
  **"not confirmed on real hardware"** and copied from the older route.

New observation from the user: during the stall **the GPU is at ~30% and only
one CPU thread is busy.** That is the signature of CPU-side, single-threaded
work (kernel generation / primitive creation) with the device waiting.

## 2. What is NOT established, and why it decides everything
Nobody has measured whether the cost is **one-time per shape** or **recurring**.
The 2x figure comes from a run where most steps are first sightings of a shape,
so it cannot tell them apart. The right fix differs completely:

| If the cost is... | then the fix is... |
|---|---|
| one-time per shape (a revisit is fast) | persistent caches + a pre-warm pass; **no dataset change** |
| recurring, no allocator retries | a cache is too small or evicted; raise it, or reduce distinct shapes |
| recurring, with allocator retries | memory pressure / fragmentation, not compile |

A fourth possibility the probe does not cover: `AdaptiveResidencyController`
re-calibrating or offloading when a larger shape than the calibration step
arrives (the project measured a 4.7x slowdown from unnecessary offloading
before). Check its log lines against the shape sequence.

**Hypothesis worth testing first (not verified).** SYCL's own JIT compiles per
*kernel*, not per tensor shape, so `SYCL_CACHE_*` should not make shapes
cheaper. The shape-specific work on Intel GPUs most plausibly happens in
**oneDNN** (convolutions, matmuls, attention), whose primitive creation is
cached separately (`ONEDNN_PRIMITIVE_CACHE_CAPACITY`, default 1024). A SDXL
UNet needs several hundred distinct primitives per resolution (forward and
backward, convolutions and the LoRA matmuls), so a 1024-entry cache holds only
a couple of resolutions and a 44-shape dataset would thrash it. If true, the
variables the project currently sets are aimed at the wrong cache.

## 3. The probe: `scripts/probe_shape_stall.py`
Runs the project's own SDXL UNet (random weights, bf16) with LoRA-like trainable
adapters on all 700 attention/FF Linear layers (verified: 2,567,463,684 base
parameters, 41.9M adapter parameters at rank 16) over the dataset's real shapes
in mixed order for several passes, `--repeat 2` steps per visit. It compares:
* **warm repeat** (same shape as the previous step): the steady state;
* **first sighting**: pays any one-time cost;
* **revisit** (seen earlier, not the previous step): this is the discriminator.
It also logs CPU time / wall time per step (about 1.0 in slow steps = one thread
busy), allocator counters, and prints a verdict. CPU logic check passed
(including the `--warm-threads` path); it has **not** run on XPU.

```
python3 scripts/probe_shape_stall.py --dataset datasets/<name> --batch 2 --passes 3 --csv probe_A.csv
```
Experiments (run each, keep the CSV and the printed environment line):

| | command | what it settles |
|---|---|---|
| A | baseline | the verdict and the numbers above |
| B | `ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536 ...` | revisits become fast => the primitive cache was thrashing |
| C | `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=~/.cache/sycl_kernels ...` run **twice** | second run's pass 1 shows how much of the first-sighting cost the disk cache removes |
| D | `--warm-threads 1` then `--warm-threads 2` | does compiling shapes concurrently help? (watch device memory: each concurrent step holds 1.6-3.3 GB of workspace on top of the 7.5 GB resident floor, so about 2 threads is the ceiling on 12 GB) |
| E | `ONEDNN_VERBOSE=1 ... 2> verbose.txt` | count primitive creations per pass; a pass-2 count near pass-1's confirms thrash (check the exact verbose syntax for your oneDNN version) |

Also record: torch and oneDNN versions, driver version, `ONEDNN_PRIMITIVE_CACHE_CAPACITY`
default, and host RSS growth in B (a bigger primitive cache costs host RAM).

## 4. Fixes, in order, each conditional on section 3
1. **Always cheap: show the cost.** Have the trainer log, per step, the latent
   shape, whether it is new in this process, and the step time, and print
   "warming N shapes" progress. Today the stall is invisible in the monitor.
2. **If one-time:** enable the persistent kernel caches, and **pre-warm** every
   shape the dataset contains before the timed run (largest first so the
   allocator reaches its high-water mark early; use `torch.autograd.grad` into the
   adapters, not `.backward()`, so concurrent threads never touch `.grad`).
   Because each graph run is now a fresh child process, nothing is retained
   between runs today, so persistence is the only way a second run is cheap.
3. **If primitive cache thrash:** raise `ONEDNN_PRIMITIVE_CACHE_CAPACITY` by
   default, after measuring host RAM in experiment B.
4. **Reduce distinct shapes (helps in every case, since fewer shapes means
   fewer things to compile, cache and fragment).** Use the simple rule, not a
   planner: round each side to a multiple of 16 or 32 latent px.
   * Prefer **cropping down** with a random offset per epoch. It needs **no loss
     mask** and does not change what the loss is computed over, which was the
     objection to padding. SDXL's micro-conditioning supports it directly:
     `resolution_embedding(height, width, crop_h, crop_w, ...)` already takes
     crop coordinates; pass the uncropped size as the original size and the drawn
     offsets as the crop. Over many epochs every pixel is seen.
   * **Padding** (to a multiple of 32: 3 shapes, +15% in the project's measurement)
     is the fallback and needs a loss mask normalised by valid elements.
   * A dataset's distinct shapes also set its memory peak (the largest bucket),
     which the memory-admission fingerprint already needs.
5. Do **not** build more elaborate shape planning until step 2 or 3 has been
   tried: if the cost is one-time, no data change is needed at all.

## 5. A note on my own earlier tool
I also wrote `shape_policy.py`, which solves the choose-the-best-shapes problem
exactly (verified optimal against brute force). It is more machinery than the
problem needs; the simple multiple-of-N rule is nearly as good, and it was in
the project's own table already. Only its `apply_policy` (random crop to a
canonical shape, optional masked pad) and `crop_conditioning` helpers are worth
keeping, and only if step 4 is needed.

## 6. Separate finding: text-encoder prewarm is dispatch-bound
Not the shape stall, but it has the same symptom (low GPU use, one busy CPU
thread) and the project's own numbers in `nodes/model/clip.py` show it:
* production CLIP is fp16, where batching is deliberately disabled, so each
  prompt is encoded alone at **32.2 ms**. My estimate (not measured): roughly
  44 transformer layers x ~15 small kernels is several hundred launches per
  prompt, i.e. launch-bound, not compute-bound, which matches 30% GPU and one busy thread;
* batched fp16 would be **2.53 ms/prompt (12.7x)**, rejected because it
  "diverges 19%" from serial fp16; batched fp32 is **10.99 ms/prompt (2.9x)** and
  agrees with serial fp32 to 1.5e-6 of the largest activation.
Two suggestions: (a) run the prewarm in **fp32 batched** (the encoder is dropped
afterwards anyway; 2.9x faster and the more accurate result, with misses also
encoded in fp32 so warm and missed prompts stay coherent); (b) before accepting
"19%", measure the reference that matters, **both fp16 variants against fp32
serial**. If serial fp16 is equally far from fp32, batched fp16 is not worse,
only different noise, and the 12.7x path is safe. The 19% is normalised by mean
activation while CLIP hidden states have a few very large outlier dimensions (the docstring
itself cites a maximum far above the mean), which can exaggerate the figure.
This was not measured by me.

## 7. What to send back
The CSVs for A-E, the printed environment line of each, versions, and
the verdict lines. With those the right fix is a configuration change, not a
design discussion.
