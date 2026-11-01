# Experiment results (SDXL LoRA, 12 GB B580, 300 steps, regions 3, lr 1e-4 unless noted)

Newest last. What was tried, what came out, what it means. Hardware runs are the owner's; numbers come from the lab reports.

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
