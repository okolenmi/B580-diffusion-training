# Experiments to run when you want the answer

Research runs of the lab (`python -m lab ...`), hours each. They are not checks of the product: nothing is broken if they stay unrun, and none of them blocks anything.
Pick one when its question interests you, queue several with `python -m lab run a.json b.json ...` (a power cut is survived: finished arms are skipped and a cut arm
continues from its state written every 200 steps). Send the zip; findings go to `experiment-results.md` and the entry here is deleted. Item numbers are kept so the
results table and the lab templates still point at the right entry.

## 61. Continue your best adapter (about 1.5 hours)
Put your saved adapter (batch 1, lr 2e-4, 600 steps) into `lab-setup.json` under `loras` as `best`. The template renders it as it is and continues it three ways with a new `--init-adapter` option: 300 more steps at 2e-4, 300 at 5e-5, 600 at 2e-4 (fresh optimizer, 50 step warmup, no region ladder).
```
python -m lab template continue-best > more.json
python -m lab run more.json
```
Send the zip. Does more training improve it or damage it? Compare each arm with `best` on the same sheet and check the holdout column of the per-timestep table (above 1 = worse than the base model). If the continuation fails to start with a message about rank or targets, the adapter was trained with other settings than the template's (rank 16 default, all default targets); tell me which.

## 62. Queue run and the cache warning (check while running the above)
Run two files in one command (for example `python -m lab run aligned2.json more.json`). Each file now runs in its own process; the second file's training should start without the out-of-memory error that chained runs gave. Send the console of any failure. Also: the `oneDNN primitive cache is evicting shapes` warning should no longer appear (the cache floor is 4096 and counts four batch sizes per shape); tell me if it does.

## 65. LoRA+ ratios 2, 4, 8, 12 (about 3 hours for five arms; pick with `--arms`)
Both 4 and 16 looked good in the first test; this narrows it down on the same recipe, with the baseline rerun for one sheet.
```
python -m lab template lora-plus-values > loraplus2.json
python -m lab run loraplus2.json
```
Send the zip. Which ratio gives the best textures without painted or unstable pictures, and is there a visible step between neighbours or only noise?

## 66. After the cleaning: a much lower learning rate (about 1.5 hours)
Follow-up of 64: the cleaned and frozen 15% plus a cleared 85% that trains at 2e-5 and 4e-6 instead of 2e-4 (your overshoot idea). Same cleaning settings as in 64 (the cleaning is repeated in each arm, about 11 minutes).
```
python -m lab template clean-lr > cleanlr.json
python -m lab run cleanlr.json
```
Send the zip. Compare with `best` and `element-15-only` on the same sheet, and the holdout column of the per-timestep table (a lower ratio than `best` = better than the base model there).

## 67. The cleaner with other anchor weights (about 1 hour)
Result of 64: anchor 5 removed nearly all of the dataset's effect, no anchor kept most of it. The cleaned-only arms with anchor 0.5, 1 and 2 (element) and 1 (component) show where it changes.
```
python -m lab template clean-anchor > cleananchor.json
python -m lab run cleananchor.json
```
Send the zip. Look at the portrait and red house pictures against `best` (how much of the base model's look comes back) and at the empty prompt (how much of the dataset's look stays).

## 68. Magnitude cleaning variants (about 1.5 hours; no training in the arms)
Your verdict on 64: magnitude (the largest 15% of the weights) is the best of the cleaners. Variants: 8% and 30%; 15% counted per layer instead of over the whole adapter (`--clean-scope layer`); ranking by what a weight contributes to the layer's change (`--clean-mode product`); scaling every layer's change back to its old size after the cut (`--clean-rescale`); and the largest singular directions per layer (`--clean-mode singular`).
```
python -m lab template clean-magnitude > cleanmag.json
python -m lab run cleanmag.json
```
Send the zip. Which keeps the effect of `best` with the fewest issues? Does `rescale` give back the power without bringing the problems back?

## 69. Which parts of the adapter matter (about 1.5 hours; no training)
`best` with one part or region removed completely (feed-forward, attn1, attn2, projections, input, middle or output blocks) with 99% of the rest kept.
```
python -m lab template clean-parts > cleanparts.json
python -m lab run cleanparts.json
```
Send the zip. Which removal changes the pictures most, and which one removes an issue without losing the effect? That tells which parts to protect or to train less.

## 70. Train without cross-attention and without the sensitive blocks (about 2.5 hours for four trained arms; pick with `--arms`)
Result of 69: removing attn2, the middle block or the input blocks from your best adapter did not change the loss at all and the pictures got cleaner. This trains with those parts excluded from the start (your best recipe, 600 steps): no attn2; no middle block; both; and both plus input blocks 6-8 and output blocks 0-2. `best` is rendered for the same sheet.
```
python -m lab template targets > targets.json
python -m lab run targets.json
```
Send the zip. Do the cleaner pictures hold up when the parts are never trained rather than cut afterwards? The console prints the number of adapted layers; fewer layers should also mean a little faster steps and less memory.

## 71. Staged training in one run (about 5 hours for four trained arms; pick with `--arms`)
Your plan: train, clean to 8% and freeze, train the rest at a lower rate, clean to 15%, and so on, all in one process. Arms: `8-15-30` (600 steps, then 300 at 5e-5 and 300 at 2e-5, cleaned to 8%, 15%, 30%), `8-15`, `8-slow` (cleaned to 8%, then 600 steps at 2e-5) and `15-30`. `best` is rendered for the same sheet.
```
python -m lab template stages > stages.json
python -m lab run stages.json
```
Check on the way: the console prints a "cleaning before step 600 (magnitude)" line at every boundary and the loss does not jump up after it (a short rise while the rate ramps is fine); `<output>/clean` holds one `lora_cleaned_stepN.safetensors` per cleaning. Send the zip. Do the later stages add effect without the problems coming back, and which share and rate are enough? The cleaned adapter of the last cleaning is a better candidate than the last saved step.
You can also run it by hand: `--stage 600:0.08 --stage 300@5e-5:0.15 --stage 300@2e-5:0.3` with your usual arguments (`--stage` replaces `--steps`).

## 72. A learning rate per part of the network (about 3 hours for four trained arms; pick with `--arms`)
Your estimate: the middle block is the most sensitive, input blocks 6-8 and output blocks 0-2 the second tier. Arms: `mild`, `tiers`, `strong` (lower and lower rates for those parts, `--lr-region`) and `middle-only`. `best` is rendered for the same sheet.
```
python -m lab template block-rates > blockrates.json
python -m lab run blockrates.json
```
Send the zip. Does the loss stay the same while the pictures get cleaner? No warning about an unmatched region should appear at the start of a run (a misspelt prefix is reported there).

## 75. Is a run that was stopped and started again the same run? (about 3 hours: six arms of 600 steps, the lab stops twice)
The anchor run's baseline was cut and continued from step 400 and gave the best pictures yet. The CPU says a cut run is identical to an uninterrupted one to the last bit (`check_resume`); the first card try (200 steps, cut at 100) agreed for the deterministic arms and differed for the ordinary one, but was too short to compare with the baseline. This version has the baseline's recipe and length.
```
python -m lab template resume-proof > resume.json
python -m lab run resume.json
```
The lab trains `full-a`, then trains `cut-400` up to step 400 and **stops itself** (state saved, nothing left running). Start the same `python -m lab run resume.json` again: it continues `cut-400` as a new lab and a new process (close the window first; reboot if you want it as hard as the power cut). It stops once more at `cut-det-400`; start it again. After that the rest trains and the pictures are made. Arms: `full-a`, `full-b` and `cut-400` are ordinary runs (`full-a` repeats the anchor baseline); `full-det-a`, `full-det-b` and `cut-det-400` are the same with deterministic kernels. Drop `full-b` or the `-det` arms from the file if time is short, but keep `full-det-b`: without it a difference of `cut-det-400` proves nothing.
Send the zip. How to read it: the report's table of adapters against the first arm. Deterministic arms must read `yes` between themselves (the two full ones first: if they differ, the kernels are not deterministic on the card and the cut test says nothing). If `cut-400` is no farther from `full-a` than `full-b` is, a restart changes nothing the ordinary noise does not. Also look at the pictures of the ordinary and the deterministic arms side by side on every prompt: in the first try the deterministic ones looked better on most images, which, if it holds at 600 steps, says the kernels' noise costs quality.
