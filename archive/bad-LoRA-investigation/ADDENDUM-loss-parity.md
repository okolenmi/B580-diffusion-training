# ADDENDUM to TASK-lora-quality-and-multishape.md: the loss gap

New facts from the maintainer (supersede the corresponding guesses in the task):

* All `t_mode` settings (uniform, low, mid, high, logit; 3000 steps each) show the problem.
  **So the t distribution is not the cause.** Do not spend more time on t_mode.
* With the 4 conditioning-path modules removed, the LoRA is still degraded.
  **That suspect is closed.**
* Resolution does not matter (512 / 1024 / mixed all show the same symptom; 512 a bit worse).
* The dataset has empty captions. The kohya run on the same data is fine, so this is not the cause.
* **Kohya's training loss was ~50% HIGHER than this trainer's on the same images, and kohya's
  LoRA is the better one** (sharper backgrounds; this trainer's LoRA gives blurry,
  less-detailed backgrounds after only 200 steps).

## Why the loss gap is the most important fact so far

Same images + same base model => the average eps-MSE at step 0 must be about the same in both
trainers (kohya samples latents and this trainer uses the mode, which can account for some of
it, see item 2). A 50% gap means the two trainers are solving different problems, and the
direction (this trainer's is easier) predicts the symptom: if the data looks "less noisy
relative to the signal" than it should, the model learns to attribute too much of the input to
noise, i.e. it over-denoises, which is blur. Candidates, in order of how cheaply they are tested:

1. **Latent statistics differ** (scale, smoothness, high-frequency content). Test first.
2. Latent *sampling noise*: kohya trains on `latent_dist.sample()`, this project on the mode.
   The SDXL posterior std is not negligible next to sigma at t < 100, so sampled latents carry
   an irreducible component that raises loss at low t. A real but probably partial explanation.
3. Pairing/scale of (x_t, target, t) inside this trainer's step, or the loss path. (Already
   verified against diffusers on random tiny weights; **not** on real weights.)
4. `t_mode` mismatch when the two totals were compared (compare uniform vs uniform only, and
   always per t bucket, never totals; base MSE falls from ~0.5 at t<100 to ~0.007 at t>900).

## Step 1: `latent_stats.py` (offline, minutes)

```
python latent_stats.py --shards <dataset>/shards --kohya <kohya latent cache dir> \
       --images <the same image folder> --ckpt div_4.safetensors --px 512
```
(kohya: run with `--cache_latents_to_disk` so `*.npz` exist; they are unscaled, the script scales.)
Read the table: project std / roughness / hf_energy vs the diffusers reference. Verdict rules:
* std ratio outside 0.95-1.05 => latent scale problem in ingestion. Stop; this is the bug.
* roughness or hf_energy clearly lower than the reference => ingestion makes data blurrier
  (preprocessing or VAE encode); compare decode(latent) vs the Lanczos-resized source.
* all ~1.0 for mode vs project, and kohya's SAMPLE differs by the posterior noise => item 2.
Note the `--images` path of the script was not run by the author (no checkpoint available);
fix it if it needs adapting, keep the metrics.

## Step 2: step-0 loss parity on identical tensors, real weights

Dump 16 (x0, eps, t) triples from the project's loader (fixed seed, uniform t), plus the empty-
caption context. Compute the eps-MSE at LoRA-off with:
 (a) this project's UNet + LossPhase (the real step, B=0),
 (b) diffusers `UNet2DConditionModel.from_single_file` + `DDPMScheduler.add_noise`,
 (c) kohya's code path (`get_noise_pred_and_target`) fed the same tensors.
Report per t bucket. (a)=(b)=(c) within 2% => the loss path is exonerated and the gap is data
(Step 1). Any disagreement localises the bug to the first component that differs.

## Step 3: per-t training curves, first 200 steps

Log (t, per-sample loss) in both trainers (patch kohya's loop; one line). Plot mean loss per t
bucket vs step for both. Healthy: equal at step 0, both falling similarly. Report where they
diverge (bucket and step).

## Step 4: still-open experiments from the main task, now with higher priority

* **Targets A/B** (`lora_targets_and_cond_switch.patch`; `target_modules="kohya_default"` = 722
  modules incl. feed-forward, vs the historical attention-only 560). `target_modules` was dead
  code before this patch. Same data, seed, 200 steps.
* **Blur metric**: Laplacian variance on background crops, same seeds, base vs kohya vs
  project. Turns "blurrier" into a number.
* **Windowed LoRA at sampling** (existing LoRA gate): LoRA active only for t<200 / 200-600 / >600,
  to see which range introduces the blur.
* **1-image 500-step run** in both trainers (the maintainer is doing this).

## Reporting reminder

Say what was run, on which weights, in which dtype, and what was not. Do not use relative
percentages of a tiny base loss as evidence of effect size (the previous report's "largest at
high t" was one: the absolute improvement is largest at mid t).
