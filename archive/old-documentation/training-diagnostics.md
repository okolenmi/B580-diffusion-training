*[← README](../README.md)*

# Training diagnostics: is this LoRA damaging a t region?

The per-bucket `loss_t_low/mid/high` the monitor charts are measured on
whatever one or two random samples each step happened to contain. They show
whether the *training* loss is falling; they cannot show whether a region got
**worse than the frozen base**, because (a) batch-to-batch sampling noise is
as large as the effect, and (b) a fit-the-data loss is blind to collateral
damage on timesteps the batch didn't cover.

`ManagedLoRATrainerNode`'s probe ports answer that question. Implementation:
`nodes/train/t_probe.py`; tests: `nodes/smoke_tests/smoke_test_t_probe.py`.

## Turning it on

| Port | Default | Meaning |
|---|---|---|
| `probe_every_n_steps` | 0 (off) | Run the probe every N optimizer steps, and once as soon as the probe images are captured. 0 = zero cost, zero behavior change. |
| `probe_items` | 2 | Training images captured (element 0 of the first batches). |
| `probe_points_per_bucket` | 2 | Fixed t's per low/mid/high third. |
| `probe_grad_alignment` | False | Also measure per-bucket gradient norms/cosines. Non-fused optimizers only (build-time error otherwise). |

Cost of the forward-only probe: `probe_items * 3 * probe_points_per_bucket`
batch-1 forwards (twice that on the first call, which also computes and caches
the frozen-base reference). Gradient alignment adds one forward+backward per
probe point and holds roughly 4-6 fp32 copies of the trainable parameters in
host RAM while it runs, so use a coarse `probe_every_n_steps` (e.g. 100-250)
with it on.

Output: a console line per probe, and `probe_*` / `gc_*` keys in that step's
monitor report (not charted by the dashboard, but included in its CSV export).

## What the numbers mean

The probe re-noises the *same* clean latents with the *same* seeded noise at
the *same* t's every time, so a difference between two probe records is the
model's change, not sampling noise. Each point is evaluated with the LoRA live
and with the LoRA gated to exactly zero (the frozen base; `gate=0` is tested to
give the exact base output for LoRA, DoRA and NF4 layers). The probe ignores
the training-time timestep gate on purpose: it judges the LoRA the way an
exported file would be applied at inference, at full strength.

- `probe_rel_t_*` = LoRA loss / base loss on identical inputs. **1.0 = no
  change, < 1 = better than base on this probe, > 1 = this run made that
  region worse than not using the LoRA.**
- `probe_worst_rel` = max over the three buckets. If one damaged region is
  enough to ruin an image, this is the number to watch.
- `probe_drift_t_*` = `||pred_lora - pred_base||^2 / ||pred_base||^2`: how far
  the LoRA moved the prediction, whether or not the movement helped. Large
  drift with `rel` near or above 1 = the LoRA is changing the model a lot
  without improving the fit.

**Important caveat:** the probe images are training images. A LoRA that
memorizes them will show `rel < 1` on all buckets while still being unusable
on other content, so a healthy probe is *necessary, not sufficient*. What the
probe reliably catches is the failure you are chasing: a region whose `rel`
climbs above 1 while another falls.

Reading order I'd suggest for a first run:

1. Step 1 record: `rel` should be ~1.000 and `drift` ~0 (LoRA B is zero-
   initialized). If not, the base reference and the live path disagree and
   nothing else is trustworthy.
2. Watch which bucket's `rel` first crosses 1.0, and at what step. Compare
   with LR warmup and with when `loss_t_*` stops improving.
3. Per-t line (`per-t (lora/base)`): shows whether damage is at an edge of a
   bucket or throughout.

## Gradient alignment (`gc_*`)

For each t bucket, the gradient of the probe loss over that bucket's points,
w.r.t. the trainable parameters:

- `gc_norm_t_*`, `gc_share_t_*`: gradient magnitude per bucket, and each
  bucket's share of the summed norms. A bucket with a much larger norm
  dominates the step direction under a plain mean loss.
- `gc_cos_a_b`: cosine between two buckets' gradients. Strongly negative =
  a step that helps one hurts the other (to first order).
- `gc_align_t_*`: cosine of a bucket's gradient with the *sum* of all bucket
  gradients (what a uniform step follows). **<= 0 means the combined update
  does not help that bucket to first order.**
- `gc_self_t_*`: split-half self-cosine, the noise floor. Each bucket's
  gradient is estimated from only `probe_items * probe_points_per_bucket`
  samples; if two independent halves of the *same* bucket agree with cosine
  0.1, then cross-bucket cosines of +-0.1 mean nothing. Interpret cross-bucket
  numbers relative to `gc_self_*`, not to +-1. Raise `probe_items` /
  `probe_points_per_bucket` to lift it.

These are first-order, local, probe-sample statistics: they say what one
small step would do now, not what a whole run does. They do not replace the
`probe_rel_*` trajectory; they are for explaining it.

## Not validated on real SDXL

The probe math is unit-tested on toy models with closed-form answers (no-op
LoRA gives rel=1/drift=0; constructed conflicting buckets give cosine -1;
bucket-private parameters give cosine 0). It has **not** been run on a real
SDXL UNet or on XPU hardware. Any numeric thresholds beyond the structural
ones above (1.0 = unchanged, self-cosine as a noise floor) would be guesses,
so none are given here.
