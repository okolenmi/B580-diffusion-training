# To test on the B580

Things written in the sandbox (no XPU) that need hardware. Delete an entry once it passed; if one fails, keep it
and paste the console lines. Feature background: `next/MANUAL.md`.

## Everything that can run unattended: one command

```
python -m next.accept run --out accept --models <models folder> --datasets <datasets folder> --dataset <dataset name> --file <checkpoint file name> \
  --output <output folder> --batch 2 --resolution 512 -- --param buckets.max_shapes=4
```

Runs six suites in separate processes (about 34 short runs, an hour or two; finished scenarios are kept, so rerun the same command after a stop) and writes
`accept/verdict.md`: a pass/fail row per check with the reason, LOOK for what needs your eyes. Send `accept/verdict.md` (and the `accept/matrix/*/run.log` of
anything marked FAIL). `--suites features` (or `speed`, `memory`, `nf4`, `determinism`, `allocator`) runs one suite; `python -m next.accept list` shows every check.

| suite | covers | what it checks |
|---|---|---|
| speed | int8 and native, eager and graph, both int8 gradient paths, attention modes, LoRA in bf16 | every run finishes with a finite loss, int8 saves memory and stays near native graph speed, graph decisions, attention modes beat `math`, losses close |
| memory | one-margin budget at 10240 and 9000 | stays under the limit, uses most of the room, prints its budget |
| nf4 | NF4 frozen weights | runs, saves memory, loss near int8's |
| determinism | two runs with the same seed | identical losses and adapter tensors |
| allocator | expandable segments on and off | the peaks side by side |
| features | probe and gradient alignment, blend, a 9216 cap without prewarm, three `--set`s at once, a typo in `--set`, mirror against zero padding, int8 with the torch backend, int8 with DoRA | each runs and prints what its item expects |

For speed numbers with power and a kernel profile (not pass/fail): `python -m next.perf hwcheck --out perf-bundle ...` as in `next/manual/performance.md`.

## Still manual

Experiments of the lab can be queued: `python -m next.lab run long.json loraplus.json aligned2.json ema.json more.json loraplus2.json` runs them one after the other. Training states are written every 200 steps, so after a power cut the same command skips what is finished and continues the arm that was cut (TESTING 58 checks this once).

## 6b. Merge Adapter
Add a `Merge Adapter` node (denoiser from Load Model -> Merge Adapter -> Inject Adapter) with a kohya LoRA you have, `strength` 1 (the CLI starters have no switch for it; edit the graph).
- Task line "Merging <file>"; training starts and the step-1 loss differs from the unmerged run; a LoRA of another family or a wrong key format stops with "nothing ... matches".
- Strength 0 gives the same step-1 loss as no merge node.
Pass: all three. Fail: paste the message.

## 6d. Auto caption
On a dataset where some images have no caption: `--auto-caption --set autocaption.limit=5` first, then without the limit.
- Needs `transformers` installed and network once for the model. Task lines "Loading the caption model" and "Captioning"; a `captions:` summary line; the model memory is gone before training (peak not higher than without it).
- Rerun: "0 captioned ... kept as they were": nothing replaced. Captions look sane in the dataset (and training prompts use them).
Pass: captions written, resumed, no memory left behind. Fail: paste the console lines.

## 17. Mirror padding: compare two LoRAs
Two full runs with the same seed, dataset and settings (packing on, `--max-shapes` as you like): one with `--pad-fill zero`, one with `--pad-fill mirror`.
- Quick check first (20 steps of each): both run, speed and peak are the same.
- Full runs: compare the loss curves (mirror should not be lower by a suspicious amount at high noise) and, more importantly, generations from both LoRAs with the same prompts and seeds. Look at borders/edges, symmetry artifacts, and style strength.
- With a low `--pack` (0.1) the difference should shrink; with a higher one (0.5) it should grow, which tells you padding matters on your data.
Pass: you pick a default; tell me which, and I will set it.

## 18. Epochs
Run `--epochs 2 --save-every-epoch 1 --save-every 100000` on a small dataset.
- Step lines show `epoch 1`, then `epoch 2`; the last step of each says `epoch N end`.
- The run stops after the second epoch (no step limit given); an adapter file is saved at each epoch end (its step number is the epoch's last step).
- `--epochs 3 --steps 10` stops at 10 steps. A cosine schedule node with an empty `total_steps` decays over the whole run.
Pass: all of the above.

## 21. Experiment: kernel compile cache between runs
You saw a default run after a graph run start with almost no new-shape penalty, and the next run slow again. The driver keeps a compiled-kernel cache on disk that may fill up and evict.
Try `NEO_CACHE_MAX_SIZE=4294967296` (4 GB) before two default runs in a row, and compare the first-sighting step times (`[... new]` markers) of the second run with a run without it.
Pass: tell me whether first sightings of shapes are faster on the second run. No fix needed either way.

## 29. NF4 frozen weights
First, no training: `python -m next.accept quant-probe --models <models> --file <checkpoint> --precision nf4 int8 --resolution 512 --out probe` (needs the card; each precision loads the checkpoint again). Send `probe/quant-nf4.md` and `quant-int8.md`: the whole-network error and the table by region show where the quantization hurts (the worst block is named). Then training: `--param adapter.base_precision=nf4` (once with `--param adapter.nf4_double_quant=true`), eager and `--set trainer.graph=true`, short run with `--perf`: step time, peak memory, the `quantization` line (layers and MB saved), and the loss over 100 steps against native and int8.
Known risks: the fused dequantizer's first call verifies itself (a few seconds, once); `xpu-kernels` older than 0.3 has no NF4 kernel and `auto` then uses the slow torch chain (a warning says so).
Pass: no out of memory at a batch or resolution that int8 cannot hold; loss within the noise of int8's at the same seed.

## 30. Padding decision by eye (mirror against zero)
`python -m next.accept padding-sheet --images <a few of your dataset images> --canvas 128x128 --count 4 --out sheet.png`, then look at `sheet.png`: per sample, the clean placement, zero padding (the strip is noise alone after noising) and mirror padding (a reflection of the noisy sample). Tell me which you would train on; the default stays zero until you say.

## Lab setup (once, before the lab items below)
```
python -m next.lab setup > lab-setup.json
```
Fill in `lab-setup.json` once: models, datasets, dataset, checkpoint file, training resolution, output folder, device, the image folders for the per-timestep loss, and under `loras` the adapters you saved (name: path). Every `python -m next.lab template KIND > file.json` reads it (or `--setup FILE`) and writes it into the file; a template that needs a saved adapter (`{lora:best}`) stops with a message when the name is missing or the path is still `/path/to/...`. `python -m next.lab run a.json b.json` checks all files before it starts (placeholder paths, missing adapter files) and runs each file in its own process.

## 53. A long run with and without learning-rate decay (about 2.4 hours)
Your finding is that more steps now improve the pictures. A constant learning rate leaves the final weights noisy at the end of a long run; a decay to a low rate usually cleans that up. 1200 steps with the best recipe, constant against cosine to 5%.
```
python -m next.lab template long-run > long.json
```
`python -m next.lab run long.json`, send the zip. Compare the final pictures, and with the 600-step ones you have: does 1200 keep improving, and does the decay add anything?

## 54. LoRA+ learning-rate ratio (about 2.5 hours)
B starts at zero and has to grow from nothing while A is already random, so one shared rate under-trains B. LoRA+ gives B a higher rate (`--lora-plus 4` means 4 times). Baseline, 4 and 16 on the current recipe, 600 steps.
```
python -m next.lab template lora-plus > loraplus.json
```
Run it, send the zip. Look at the pictures at equal steps and at how fast the loss falls early; a ratio that makes the images painted or unstable is too high.

## 55. Averaged adapter weights (EMA) (about 1 hour)
The weights jump around between steps; an average over the last few percent of the run is usually cleaner and costs nothing at sampling time. The template trains one run with the best recipe and `--ema 0.05,0.1` (one extra copy of the adapter on the card per width), then renders the trained adapter and both averages from that same run.
```
python -m next.lab template ema > ema.json
python -m next.lab run ema.json
```
Send the zip. Look for cleaner details and fewer blotches with the same composition. (The earlier version had placeholder paths and rendered nothing; that report does not count.)

## 56. Gradient-aligned adapter start (about 2.5 hours)
A fresh LoRA starts with random A rows, and B first moves along whatever the loss gradient has in common with them. This sets A to the gradient's main directions, measured on the first N batches before step 1 (two passes over them with no update, so a little slower to start; the model itself is unchanged). The baseline is the lora-plus recipe; 4 and 16 batches. Check first that the run does not run out of memory in the measuring step (it holds the same gradients as a normal step).
```
python -m next.lab template aligned-init > aligned.json
```
Run it, send the zip. Look at the early pictures (do they show the subject sooner?) and at the final ones at 600 steps; if they look the same, the start does not matter at this length.

## 57. Batch 1 first, a bigger batch at the end (about 2 hours)
Your batch test: 2 images per pass cost no extra time, 4 cost 1.7 times, a bigger batch gives more detail and less variety, lr 2e-4 suits it. The ramp trains the first batches with a small batch and the rest with the full one (`--batch 4 --batch-first 1 --batch-first-steps 400`). The control with batch 4 from the start is `batch-4-lr-2e-4` of the batch test.
```
python -m next.lab template batch-ramp > ramp.json
```
Run it, send the zip. Look at the minutes (a ramp should land between batch 1 and batch 4), the composition, and the details at the end. Note: the batch size changes mid-run, so the first big batch is a new shape for the card (one more compile), and the logged loss is noisier while the batch is small.

## 58. Resumable training: cut and continue (about 15 minutes)
```
python -m next.run sdxl-lora ... --steps 60 --resume-every 20 --output /tmp/r
```
Kill it (the process, or the power cord if you like) after step 30. Run the same command again: it should print `resuming from step 20 (N batches already used)`, continue with step 21 and finish at 60, and `/tmp/r/resume` holds `resume_00000040`/`resume_00000060` (`.json` + `.safetensors`). Compare the final picture or loss with an uninterrupted 60-step run (equal on a deterministic setup, close otherwise). Then a lab run: start any experiment, pull the plug mid-arm, run `python -m next.lab run exp.json` again; the arm should say "continuing an earlier session" and the report should list it as resumed. If a state is damaged (copy a resume file and flip a byte), the log says `resume state ... is not usable` and the run continues from the one before.

## 59. Trend weighting of the timestep regions (about 3 hours for five arms; pick with `--arms`)
Your idea, in code: each region of a step gets a weight from how the loss of its part of the timestep range is doing (12 smooth bins, a fast average against a slow one), and a region doing better than usual counts more. `trend` reads the bins' recent trend, `current` the region's own loss against the bins' average (noisier), `trend-1-penalty` also lowers a region doing worse. The weights act after each region's gradient is scaled to unit length, so they change the direction and, by the average weight, the length of the update.
```
python -m next.lab template trend > trend.json
```
Run it (`--arms baseline trend-1 current-1` if the time is short), send the zip. Step lines show `trend w a b c` per region: check that the weights move (a column stuck at 1.00 means no history yet or no success) and are not wild (above about 3 for many steps in a row). Then the usual: composition, details, and above all whether the individual styles of the dataset survive better than in the baseline at the same step. The loss itself will not tell the arms apart.

## 60. Gradient-aligned start: other values, and with LoRA+ (about 3 hours for five arms; pick with `--arms`)
Your result: 4 batches gave very good textures, 16 failed. This brackets 4 with 2 and 6, and puts 4 together with LoRA+ 4 and 8 (A is already a good direction, B has to grow from zero). `aligned-4` is run again so all pictures are on one sheet.
```
python -m next.lab template aligned-values > aligned2.json
python -m next.lab run aligned2.json
```
Send the zip. Look at textures first, then whether LoRA+ adds to the aligned start or fights it.

## 61. Continue your best adapter (about 1.5 hours)
Put your saved adapter (batch 1, lr 2e-4, 600 steps) into `lab-setup.json` under `loras` as `best`. The template renders it as it is and continues it three ways with a new `--init-adapter` option: 300 more steps at 2e-4, 300 at 5e-5, 600 at 2e-4 (fresh optimizer, 50 step warmup, no region ladder).
```
python -m next.lab template continue-best > more.json
python -m next.lab run more.json
```
Send the zip. Does more training improve it or damage it? Compare each arm with `best` on the same sheet and check the holdout column of the per-timestep table (above 1 = worse than the base model). If the continuation fails to start with a message about rank or targets, the adapter was trained with other settings than the template's (rank 16 default, all default targets); tell me which.

## 62. Queue run and the cache warning (check while running the above)
Run two files in one command (for example `python -m next.lab run aligned2.json more.json`). Each file now runs in its own process; the second file's training should start without the out-of-memory error that chained runs gave. Send the console of any failure. Also: the `oneDNN primitive cache is evicting shapes` warning should no longer appear (the cache floor is 4096 and counts four batch sizes per shape); tell me if it does.

## 63. Anchor loss: protect what the model already does (about 3 hours for four arms; pick with `--arms`)
New: `--anchor W` (with `--anchor-every N`, default 2, and `--anchor-prompts "a|b"`). Every Nth step the network is also asked about the batch's noisy images under a random prompt unrelated to the dataset, with and without the adapter; the squared difference of the two predictions times W is added to the gradient. The step line shows `loss_anchor` (the difference before the weight: it should stay small and not climb steadily). Costs about one more forward and backward on those steps; plain optimizer, no graph.
```
python -m next.lab template anchor > anchor.json
python -m next.lab run anchor.json
```
Send the zip. Look at the `portrait` and `red house` pictures first (the dataset is not in them): do they stay closer to the base model's rendering as the weight grows (1, 5, 25 against baseline)? Then the empty prompt: is the dataset's look still learned? If 25 changes nothing and 1 changes nothing, the weight range is too low and I will raise it; if even 1 flattens the dataset look, it is too strong.

## 64. LoRA cleaner (about 1.5 hours; the cleaning is 200 gate steps per arm)
Your idea, built: every weight of a trained adapter (`element`) or every singular direction of each layer (`component`) gets a gate between 0 and 1. The adapter is frozen and only the gates train, with the training loss and, with `--anchor`, the damage to other prompts; the 15% with the highest gates stay and, unless `--clean-no-freeze`, are frozen while the cleared rest trains from zero (a removed direction from a fresh start). In one command: `--init-adapter best.safetensors --clean 0.15 --anchor 5 [--clean-mode component] [--clean-steps 200] [--clean-at N]` (`--clean-at N` cleans an adapter trained in the same run, before step N). It needs the plain optimizer and no graph.
```
python -m next.lab template clean > clean.json
python -m next.lab run clean.json
```
Needs `best` in `lab-setup.json` (rank 16 default targets as in the template). Send the zip. `magnitude-15-only` is the control without learning (the largest 15% of the weights stay): the gates are worth it only if they look better than it. Look first at the `-only` arms against `best` (is the dataset's look still there with 15%?), then at the portrait and red house pictures (closer to the base model than `best`?), then at `element-15-more`. In each arm's `train/clean/` folder, `summary.json` says what share was kept per part of the network (input, middle, output blocks; attention, feed-forward): send it too. The console prints the gate loss at the start and the end of the cleaning; if the cleaning step is slow or runs out of memory, tell me (the gates are one float per weight).

## 65. LoRA+ ratios 2, 4, 8, 12 (about 3 hours for five arms; pick with `--arms`)
Both 4 and 16 looked good in the first test; this narrows it down on the same recipe, with the baseline rerun for one sheet.
```
python -m next.lab template lora-plus-values > loraplus2.json
python -m next.lab run loraplus2.json
```
Send the zip. Which ratio gives the best textures without painted or unstable pictures, and is there a visible step between neighbours or only noise?

## 66. After the cleaning: a much lower learning rate (about 1.5 hours)
Follow-up of 64: the cleaned and frozen 15% plus a cleared 85% that trains at 2e-5 and 4e-6 instead of 2e-4 (your overshoot idea). Same cleaning settings as in 64 (the cleaning is repeated in each arm, about 11 minutes).
```
python -m next.lab template clean-lr > cleanlr.json
python -m next.lab run cleanlr.json
```
Send the zip. Compare with `best` and `element-15-only` on the same sheet, and the holdout column of the per-timestep table (a lower ratio than `best` = better than the base model there).

## 67. The cleaner with other anchor weights (about 1 hour)
Result of 64: anchor 5 removed nearly all of the dataset's effect, no anchor kept most of it. The cleaned-only arms with anchor 0.5, 1 and 2 (element) and 1 (component) show where it changes.
```
python -m next.lab template clean-anchor > cleananchor.json
python -m next.lab run cleananchor.json
```
Send the zip. Look at the portrait and red house pictures against `best` (how much of the base model's look comes back) and at the empty prompt (how much of the dataset's look stays).

## 68. Magnitude cleaning variants (about 1.5 hours; no training in the arms)
Your verdict on 64: magnitude (the largest 15% of the weights) is the best of the cleaners. Variants: 8% and 30%; 15% counted per layer instead of over the whole adapter (`--clean-scope layer`); ranking by what a weight contributes to the layer's change (`--clean-mode product`); scaling every layer's change back to its old size after the cut (`--clean-rescale`); and the largest singular directions per layer (`--clean-mode singular`).
```
python -m next.lab template clean-magnitude > cleanmag.json
python -m next.lab run cleanmag.json
```
Send the zip. Which keeps the effect of `best` with the fewest issues? Does `rescale` give back the power without bringing the problems back?

## 69. Which parts of the adapter matter (about 1.5 hours; no training)
`best` with one part or region removed completely (feed-forward, attn1, attn2, projections, input, middle or output blocks) with 99% of the rest kept.
```
python -m next.lab template clean-parts > cleanparts.json
python -m next.lab run cleanparts.json
```
Send the zip. Which removal changes the pictures most, and which one removes an issue without losing the effect? That tells which parts to protect or to train less.

## 70. Train without cross-attention and without the sensitive blocks (about 2.5 hours for four trained arms; pick with `--arms`)
Result of 69: removing attn2, the middle block or the input blocks from your best adapter did not change the loss at all and the pictures got cleaner. This trains with those parts excluded from the start (your best recipe, 600 steps): no attn2; no middle block; both; and both plus input blocks 6-8 and output blocks 0-2. `best` is rendered for the same sheet.
```
python -m next.lab template targets > targets.json
python -m next.lab run targets.json
```
Send the zip. Do the cleaner pictures hold up when the parts are never trained rather than cut afterwards? The console prints the number of adapted layers; fewer layers should also mean a little faster steps and less memory.

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
