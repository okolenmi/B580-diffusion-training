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
| First steps (item 49) | Warmup 50 steps and a ladder from 6 regions to 3 over 100 steps: `ladder-6-warmup` judged best by eye (small details of the composition more alive); coarse change 0.080 against 0.084 for plain `equal` (the metrics barely separate them). `ladder-9` ran out of memory | Seen by eye, 1 round |
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

## First steps (resolution 768, rendered at 1024)
- `equal` 16.7 min, `warmup-50` 17.7, `ladder-6-warmup` 19.6 (the ladder's early steps run up to 6 passes). The logged losses and per-t tables are the same to the digit in all three.
- Metrics: coarse change 0.0839 / 0.0797 / 0.0801, finest octave 1.80 / 1.78 / 1.57. The warmup arms change slightly less at the coarse level and are less detailed at the finest; the owner prefers `ladder-6-warmup` by eye for the small details of the composition.
- `ladder-9` ran out of memory at step 1: the gradient of every region (162 MB each) was kept on the card, nine of them plus the stacked matrix. Fixed by keeping them in host memory.
- Owner's idea for later: 2 and 4 regions. His merge experience: with 2 examples progress is faster than with 3-4, since more examples show more errors to reject.
- Owner's reading: for `ladder-6-warmup` the improvement is in composition, almost with no trade-off; details can be fixed later, fixing them at the start of training is useless.

## Region count (resolution 768; run before the streaming change, so the minutes include the copy and pause costs)
- Minutes for 300 steps: regions-2 11.7, ladder-6 21.3, regions-4 21.6, ladder-9 22.3. Time follows the number of passes: about 5.5 min per region per 300 steps.
- Coarse change: regions-2 0.0738, ladder-9 0.0776, ladder-6 0.0800, regions-4 0.0815 (noise about 2%): more regions change the structure more, saturating at 3-4; 2 changes it about 8% less. Regions-2 also keeps the spectrum nearest the base model (slope -2.33, fine-power share 0.289 against 0.287 for the base).
- Logged losses are the same to the digit across arms with the same region count because the noise and timestep stream is seeded the same way; they differ between region counts because the stream differs, so loss columns compare only arms with equal region counts.
- The owner could not name a clear winner by eye; he is running 600 steps and a heavier ladder (warmup 100, ladder 9, 4 regions).

## Region count at 600 steps (after the streaming change)
- Minutes for 600 steps: regions-2 22.9, ladder-6 33.7, ladder-9 35.9, regions-4 40.4, ladder-9-r4 49.0. A step of 3 regions takes about 2.8 s at the end (3.3 s before the streaming change), 4 regions 3.8 s (4.3 s), 2 regions 2.1 s: about 0.93 s per pass.
- Coarse change: regions-2 0.0856, ladder-9 0.0940, ladder-6 0.0967, regions-4 0.0981, ladder-9-r4 0.0994. Finest octave: ladder-6 3.02, ladder-9 3.03, ladder-9-r4 2.40, regions-4 2.02, regions-2 1.96; sharpness against the base 4.24 / 3.97 / 3.02 / 2.70 / 3.20. The 3-region ladders keep the most fine detail; 4 regions change the structure most and lose detail.
- Mean pairwise cosine of the regions' gradients (`cos_mean`): at the start 0.13 for ladder 6 and 0.22 for ladder 9, falling to about 0.04 after 200 steps; with a final count of 4 it stays higher (0.05-0.07); with 2 it is 0.03-0.04. Adjacent timesteps pull the adapter in more similar directions, so a 4th region adds less independent signal and fits the one image harder.
- Owner's reading by eye: 3 regions is the best count, ladder 9 better than ladder 6 on the long run (a good initial directive helps), 4 regions gives more issues, ladder-9-r4 (span 200, final 4) was a failure. That arm differs from ladder 9 in two ways (final count 4, span 200), and 4 regions alone also did worse, so the final count is the likelier cause. The metrics do not separate ladder 6 from ladder 9 (within the noise).
- Owner's main finding: with 3 regions (and the ladder) more steps now improve the pictures (300 -> 600: chaos turns into real forms, e.g. a crowd in one room in sheet pd41d8c_s3), where plain training gave no progress and only more destruction beyond a few hundred steps.
