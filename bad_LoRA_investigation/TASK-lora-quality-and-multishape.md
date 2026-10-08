# TASK: LoRA quality bug, then multi-shape training

Written for the repo's AI agent. Read all of it before changing anything.
Order matters: **Part A gates Part B.** Do not spend effort making a trainer faster
until it is known to produce good LoRAs.

## 0. Ground rules

1. **Reference = independent implementations, not ComfyUI and not this repo's docs.**
   ComfyUI is what the LoRA is *used* in, but its code is not a trusted oracle.
   Use `diffusers`, HuggingFace `transformers`, and kohya `sd-scripts`
   (`train_network.py` / `sdxl_train_network.py`) as references. When you claim a
   component is correct, say *against what* and with what tolerance.
2. **"Everything was checked, no problems found" is not an acceptable report.**
   Every report states: what was run, on what weights (random tiny / real SDXL),
   in what dtype, the numbers, and what remains untested.
3. **Report negative results as results.** A test that finds nothing is still a
   `MEASURED-*.md` entry. Never weaken or delete a test to make it pass.
4. **Mark every claim as one of:** *measured on real SDXL weights* / *measured on
   tiny random weights* / *read in code only*. Only the first counts for "fixed".
5. One commit per item, measurements in the commit body. Stop and report if an
   experiment's outcome changes the plan.

## 1. Situation (what is already established)

Symptom, from the maintainer: a LoRA trained by this trainer changes outputs
strongly and breaks denoising. Final images look like a collage of badly processed
patches, with unrelated objects that do not belong together, like an early
sampling step instead of a finished image. Checkpoint is fine, LoRAs from other
trainers are fine, optimizer choice does not matter, settings were default.
Per-block ablation (applying the LoRA block by block): **no single broken block.**
Many blocks each contribute a little damage, which adds up. `label_emb` and
`input_blocks.8` are the cleanest.

### 1.1 Verified correct (independent references, `independent_reference_tests/`)

| Component | Reference | Result |
|---|---|---|
| UNet forward (timestep embedding, `y`, attention, norms, up/down) | diffusers `UNet2DConditionModel`, weights moved by diffusers' own converter | bit-exact at t = 0, 1, 100, 500, 900, 999; outputs depend strongly on t, ctx, y (not vacuous) |
| LoRA forward + export (kohya keys, alpha, A/B orientation, incl. `time_embed`/`label_emb`) | independent merge using ComfyUI's key rule into a diffusers model | max diff ~1e-5 vs effect size ~1.6 |
| Gradients through the custom activation checkpointing | plain autograd | rel diff 1.4e-6 on all adapters, incl. cond-path |
| VAE encode (mode) + decode | diffusers `AutoencoderKL` | ~1e-6 / 1e-5 |
| CLIP-L-like and CLIP-G-like towers, penultimate layer, pooled, OpenCLIP key conversion (qkv split, `text_projection` transpose) | HF `CLIPTextModelWithProjection` | bit-exact |
| Noising, sigma schedule, UNet input scaling | diffusers `DDPMScheduler.add_noise` (scaled_linear) | 3e-3, bf16 quantisation |
| "Reversed t" | schedule table | sigma rises with t; sigma[0]=0.0292, sigma[999]=14.6146 |

Also read and found consistent: loss (uniform weighting by default), caption/image
pairing in `manager/loader.py` (batches are grouped by identical caption), LoRA
timestep gate (**off by default**), effective-alpha handling.

**Limit of all of the above:** random tiny weights, fp32, CPU. Nothing here exercises
real SDXL weights, bf16, a real tokenizer, real images, or the optimizer loop.
The bug, if it is in the trainer, is therefore most likely in one of:
real-weight/bf16 behaviour, the data/caption/resolution side, training dynamics
(optimizer/LR/accumulation), or in what the LoRA is applied to at inference.

### 1.2 What the maintainer's LoRA statistics say

`lora_forensics.py` (`||dW||/||W||` per module, rank 64, alpha 48):

* attention median ratio 0.029 for the bad LoRA vs 0.034 for a good rank-4 LoRA:
  **overall strength is not abnormal.**
* `time_embed` median 0.108 (max 0.141), `label_emb` 0.042: elevated (about 3.7x and
  1.4x the attention median) but not extreme. Combined with the per-block ablation,
  **the conditioning-path theory is downgraded to "unproven, cheap to rule out".**
* `attn_output` ratio is 2.3x `attn_input` (the good LoRA: 1.5x). The 8 most
  over-strong modules are all `attn2.to_out.0` (cross-attention output projection) in
  `output_blocks.0/1/5`, i.e. the lowest-resolution, most semantic part of the net.
  That pattern fits "the LoRA is trying to change what the text means / ignore the
  text", which points to **data/caption/conditioning** as much as to the optimizer.

Working conclusion: the damage is a *direction* problem, not a *magnitude* problem,
and it is spread over many modules. That is what you get when the training signal does
not match the function the LoRA is later used in. Find where train-time and
inference-time conditions differ.

## 2. Part A: find the LoRA quality problem

All experiments need a GPU and the real SDXL checkpoint. Use a fixed prompt, fixed seed,
fixed resolution for every comparison and **keep the images**.

### A1. Overfit-one-image sanity (do first, about 1 hour)
Dataset: **one** image at the model's native resolution (1024 px area), **one** caption.
Rank 16-64, 300-600 steps, default optimizer, `t_mode=uniform`.
Generate with that caption, same size, strength 1.0, in ComfyUI **and** with the
trainer's own sampler.
* Reproduces the image coherently: the training loop is sound; the problem comes from
  the dataset, hyperparameters or resolution. Go to A5, A6, A8.
* Collage/garbage even here: a real trainer-side defect. Go to A3, A4, A7.
Record the caption token count (above 77 tokens is a separate risk, see A4c).

### A2. Same data, reference trainer
Train the *same* dataset with kohya sd-scripts, matching rank, alpha, LR, steps,
resolution, caption handling. Record **how** the maintainer's "confirmed it's the
trainer" was established. If this has not been done on identical data, it is the
single most informative missing fact.

### A3. Conditioning-path switch (cheap, directly tests the earlier suspect)
`conditioning_path_switch.patch` adds `target_conditioning_path` (default `True`,
behaviour unchanged; `False` gives 560 attention modules like kohya/diffusers, instead
of 564). Retrain A1 and the real dataset with `False`, same seed. Also run, with no
training:
`python lora_forensics.py bad.safetensors --write-without time_embed,label_emb out.safetensors`
and load `out` in ComfyUI. (A recovery is strong evidence. No recovery is *not*
conclusive, because the other modules trained jointly with those four.)
Extend `lora_forensics.py` to also split `attn1` vs `attn2` and by resolution level, and
test "cross-attention only" and "self-attention only" files.

### A4. Real-weights parity (the CPU tests cannot see these)
On the maintainer's machine, with the real checkpoint, compare the project against
diffusers/transformers, **in the dtype used for training (bf16)** and in fp32:
* a. UNet eps prediction at t in {0, 10, 100, 300, 500, 700, 900, 999} on the same
  `(x_t, ctx, y)`. Report per-t relative error; expect ~1e-2 or better in bf16.
* b. Text encoders: ctx `[B,77,2048]` and pooled `[B,1280]` for ~20 real captions
  (short, long, empty, non-ASCII). Cosine per token position.
* c. **Captions longer than 77 tokens**: kohya concatenates 75-token chunks with BOS/EOS
  stripped; ComfyUI concatenates full 77-token chunks. Check what *this* trainer does
  and what the *inference* UI does for the same caption.
* d. VAE encode of 5 real training images: latents vs diffusers `AutoencoderKL`
  (mode and sample), and the exact image preprocessing (EXIF orientation, alpha
  channel, colour profile, resize/crop mode, 512 vs 1024).
* e. `size/crop` conditioning vector `y`: for 3 image shapes, compare against
  diffusers' `_get_add_time_ids` order (orig h,w; crop top,left; target h,w).
* f. The saved LoRA loaded two ways (ComfyUI patching vs diffusers `load_lora_weights`
  / manual merge): outputs must agree.

### A5. Per-t held-out loss, with and without the LoRA
Using the verified pipeline from A4, on **held-out** images (not in the training set),
compute eps-MSE per t bucket: `[0,100) [100,300) [300,500) [500,700) [700,900) [900,1000)`
for base vs base+LoRA. A healthy LoRA lowers it (or at worst leaves it unchanged) in
every bucket. A bucket where the LoRA **raises** it identifies the damaged region, which
answers the timestep-region question with data. Repeat for LoRAs trained with t pinned
(`t_mode=exact` at 100 / 500 / 900) and see how far each leaks into other buckets.

### A6. Resolution and conditioning consistency
Record, for the failing run: training latent size (`latent_size` default **64 = 512 px**,
far below SDXL's native ~1024 px), generation size, `y` size values used during training
and at inference. A LoRA trained at 512 px and sampled at 1024 px is a known way to get
duplicated objects and patchwork compositions on SDXL. If sizes differ, rerun A1 with
matching sizes before anything else.

### A7. Precision
Run 200 steps with identical seed in (i) bf16 UNet (current), (ii) fp32 UNet if memory
allows (or the smallest bucket), compare loss curves and the resulting LoRA in ComfyUI.
Also run inference in bf16 and fp32 to see whether the LoRA is precision-sensitive.
Known: `KarrasInputScaler` rounds the UNet input to bf16, so at t<30 (sigma <0.1)
the injected noise is coarsely quantised; this also happens at bf16 inference, but a
user running ComfyUI in fp16/fp32 sees a cleaner input than the LoRA was trained on.

### A8. The maintainer's timestep hypothesis, explicitly
(Reversed t is already excluded by the schedule table.) Test "region leakage" with A5's
pinned-t runs. Also confirm the gate (`gate_enabled`) is off in the failing run, and that
`t_low/t_high` were 1/999.

### A9. Training dynamics
Log per step: loss, grad-norm, LoRA `||B A||/||W||` per group, effective LR actually
applied (fused optimizers apply updates inside backward hooks: confirm grad
accumulation and clipping interact correctly there). Compare with kohya's numbers for the
same data over the first 200 steps. Look for the step where the ratio of cross-attention
`to_out` modules departs from the rest.

### Reporting for Part A
For each item: command, config, seed, image grid (base / bad / patched), numbers, verdict
(*confirmed cause / ruled out / inconclusive*), and what to do next. If no item finds
the cause, say so and list which hypotheses remain.

## 3. Part B: multi-shape training (start only after Part A has a conclusion)

State of play (from the repo's own docs): the stall was the oneDNN primitive cache
(2048 now lands in `nodes/xpu_env.py`; the entry in `docs/known-issues/open.md` saying
it has not landed is **stale**, fix it). `shape_bucket_multiple` x32 gives ~1.5x
throughput at ~13% padding; x48 (50% padding) measured worse. L1 (true-size
conditioning) and L3 (cache-thrash detector) are done. Per-sample captions in one
batch (L4) are not built. XPU graph capture works in isolation (~2x) but is not integrated.

B1. **Fix the evaluation first.** The quality sweep used 31 images drawn from the
training set. Build a real held-out split; report per-t-third MSE on unpadded images;
add a ComfyUI render comparison (same seed/prompts, bucketed vs unbucketed vs
baseline LoRA). MSE differences of 0.0001-0.002 on 0.166 cannot show a quality problem.

B2. **Padding semantics, one ablation each, on the held-out metric:**
pad content (zeros+noise = current vs edge-replicate vs reflect); whether pad tokens
should be masked in attention (currently they take part); pad-fraction cap per sample
(the mean is 13%, but the worst-case per-sample pad is what hurts).

B3. **Per-sample captions in a batch (L4).** Measure on a dataset with many distinct
captions. Per-sample ctx/`y` also subsumes L1. Note `keep_incomplete_batches=False`
silently drops every (caption, size) group smaller than the batch, so with
per-image captions a large part of the dataset may never be trained on: make this an
error or a loud default, not a warning.

B4. **Graph capture:** only after B1-B3 and Part A. One graph per bucket shape
(~460-520 MB each), MATH/EFFICIENT SDPA (FLASH cannot be captured), optimizer step,
static conditioning buffers. Acceptance: loss parity over 200 steps.

B5. **Defaults:** make `shape_bucket_multiple=32` default only if B1 shows no quality
loss on held-out data, and Part A is closed.

## 4. Smaller findings to fix or document

* `LoRALinear.scaling = alpha/rank * block_weight`, but the exported `.alpha` omits
  `block_weight`: with `block_weights` set, training strength and inference strength
  differ. Export the effective alpha or document it.
* `load_state_dict(strict=False)` in `unet_wrapper.py` prints only *missing* keys; make
  *unexpected* keys fail loudly too, and fail on any missing key not on a short allowlist.
* Ingestion defaults: `resize_mode="resize"` stretches non-square images to a square;
  `latent_size=64` is 512 px for SDXL. Both are surprising defaults: warn in the UI.
* VAE encode takes the posterior *mode*; ComfyUI-style pipelines sample. Minor; document.
* Quality gates rely on 300-step MSE. Add an automatic ComfyUI-free image smoke test
  (fixed seed render with LoRA gate 0 vs 1, saved to the run folder) so a broken LoRA is
  visible in the first 500 steps, not after a long run.

## 5. Files delivered with this plan

* `independent_reference_tests/` : six CPU tests vs diffusers/transformers
  (`REPO=<repo> sh run_all.sh`; needs torch, diffusers, transformers, safetensors).
* `lora_forensics.py` : offline LoRA group/strength inspector and key stripper.
* `conditioning_path_switch.patch` : A3 flag (default behaviour unchanged; affected
  smoke tests pass).
