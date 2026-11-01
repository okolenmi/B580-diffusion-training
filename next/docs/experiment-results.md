# Experiment results (SDXL LoRA, 12 GB B580, 300 steps, regions 3, lr 1e-4 unless noted)

Newest last. What was tried, what came out, what it means. Hardware runs are the owner's; numbers come from the lab reports.

## Summary of the runs so far (what is settled and what is not)
| Question | Answer | Confidence |
|---|---|---|
| Several timesteps per image (3 regions) vs one | Better composition, less damage to other fields; 3 passes per step cost about 2.5x the time | Seen by eye on 3 rounds |
| Which gradient route | `equal` (unit-length region gradients) best for structure; `project` and `common` give the same, since region gradients are nearly orthogonal; `strata-common` was liked early | Good |
| Gradient accumulation 4 | Same results as 1 route-wise, but 3x fewer optimizer steps | Not comparable at equal lr |
| Input perturbation | Clean end only: more detail; with more noise or at the noisy end: composition breaks | Weak (within noise) |
| Epsilon scaling | More detail everywhere, much of it out of place | Clear: not a fix |
| Self-perceptual loss at 0.3 | Destroyed the model; diverged | Clear for this weight only |
| Learning rate 3e-5 / 1e-4 / 3e-4 | 3e-5 undertrained; 1e-4 and 3e-4 equal within noise | Good |
| Per-timestep loss vs base | About 0.96 at low and mid t, 0.91 at t=850, 0.78-0.80 at t=950 in every arm: it does not tell the arms apart | Good |
| Run-to-run noise | Same settings give the same loss to four digits but image metrics differ by about 10% | Good |
| Memory | 1024 px batch 1 fits (deterministic mode now uses blockwise attention); 3 regions about 10.5 GB | Measured |
| Cheap `equal` (weights instead of three passes, item 47) | `gradnorm` single-timestep steps with weights do not reach `equal`'s structure, even with 3x the steps; `equal` preferred by eye | Seen by eye |
| One pass instead of three (item 48) | `equal` still wins by eye. `mean` 11.4 min, `calibrated-10` 13.1, `calibrated-50` 12.1, `equal` 18.4: one batch of three costs 62% of three passes, not a third. `calibrated-50` is closest to `equal` in structure but softer; `calibrated-10` is good in small details | Seen by eye, 2 rounds |
| Not yet run | Weights instead of three forwards (47), one-pass calibration (48) | |

## Timesteps per image (regions)
- Several timesteps per image in one step (3 regions) beat one timestep per image on composition. The `strata` layout with per-region gradient normalisation (`equal`) is the best plain route so far: fewest structural errors.
- Gradients of timestep regions are nearly orthogonal (cosine within +-0.13), so `project` and `common` (MGDA) behave like `equal`. Normalising each region's gradient to unit length is the lever; region losses differ 20-70x.
- Gradient accumulation of 4 gives 3x fewer optimizer steps and is not comparable at the same learning rate.
- `equal` costs three separate forward-backward passes per step; `mean` is one batch and one pass but does not scale the regions.

## Input perturbation
- Extra noise in the network input only, target unchanged. At the noisy end it hurts (t=950 adapter/base ratio 0.80 -> 0.95). At the clean end only it adds detail; whether it beats plain `equal` is inside the noise.

## Epsilon scaling (sampling time, no training)
- Dividing the predicted noise by 1.005-1.04 raises the finest-octave detail 3.0x -> 22.6x, but the coarse change grows too (0.092 -> 0.140). Details look out of place. Not a fix; "the model under-predicts the noise magnitude" is not the cause.

## Self-perceptual loss (Lin and Yang, arXiv 2401.00110)
- Optional extra term `trainer.perceptual_weight`: frozen base network (adapter off), mid-block features of the re-noised image estimate against the true image.
- Weight 0.3 destroyed the result (adapter/base loss 2-14 at every t, finest octave 31x, coarse change 0.41). Its gradient was about 15x the plain loss's at step 1 and, after per-region normalisation, decided the direction; gradient norms then diverged (1e5 to 1e10). Weights 1 and 3 were not run.
- Likely causes: weight far too large for a LoRA; the estimate `x - sigma*eps_hat` is far outside what the network sees at high noise. The paper trains full weights, v prediction, batch 896, after MSE pretraining. Untried: weight 0.003-0.03, a bounded estimate, gradient-norm-matched weighting.

## Learning rate (resolution 768, rendered at 1024)
- Mean loss over the last 100 steps: 0.1028 (1e-4), 0.1028 (1e-4 again), 0.1036 (3e-5), 0.1026 (3e-4). Adapter/base loss ratio per t differs by at most 0.01 between 1e-4 and 3e-4; all arms improve the plain loss by about 5% (23% at t=950).
- The two 1e-4 controls match to four digits in loss but differ in image metrics: finest octave 1.77 vs 1.60, spectral slope -2.56 vs -2.60, fine-power share 0.115 vs 0.103. So image-metric differences under about 10-15% are not results, even when the training loss is identical (a sampler amplifies tiny weight differences).
- 3e-5: less coarse change (0.066 vs 0.086), more fine detail (3.26), judged worse by eye: undertrained. 1e-4 and 3e-4: each better on different images, within the noise. The learning rate is not the cause of the painted look in this range.
- Speed: 768 px gave about 3.2 s per step against about 3.5 s at 1024 (0.6x the pixels): batch-1 passes are probably launch-bound, not compute-bound (not measured).

## One pass instead of three (resolution 768, rendered at 1024)
- Minutes for 300 steps: equal 18.4, mean 11.4, calibrated-10 13.1, calibrated-50 12.1. One batch of three timesteps costs about 62% of three separate passes: the saving is a third at most, not two thirds.
- Coarse change (the least noisy image metric; the two identical 1e-4 controls differed by 1.5%): equal 0.0853, calibrated-10 0.0801, calibrated-50 0.0792, mean 0.0652 (about what the undertrained 3e-5 run gave, 0.0662). It follows the order the owner judged by eye: `equal` learns the most structure, `mean` the least. Fine/coarse is the reverse: equal 0.24, calibrated 0.27, mean 0.34, since in `mean` the clean-end region, with a loss 20x larger, dominates the gradient.
- Calibrated weights are fixed ratios measured every 10 or 50 steps; `equal` normalises each step's own gradient, and the norms vary several-fold from step to step, so a fixed weight cannot follow them.
- Conclusion: `equal` stays the base. The calibrated option stays in the code (`policy_calibrate_every`) but costs quality for a third less time. The logged loss of the calibrated arms (0.0497, 0.0424) was the balance-weighted loss, which the code no longer reports.
