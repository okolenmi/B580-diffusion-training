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

## 46. Noise floor: how different are two runs that should be the same (about 1 hour; do this first)
In item 45 the `equal` arm trained to the same numbers as the `equal` arm of item 44 (mean loss, per-t table and probe identical to 3 digits, last step lines identical), yet its images scored differently (finest octave 2.57 against 3.03, coarse change 0.0868 against 0.0924) and looked worse to you. Either the kernels are not bit-exact and small differences grow, or the renders differ. This runs the same training twice, and twice with deterministic kernels (slower).
```
python -m next.lab template noise-floor > floor.json
```
Edit (`images`), `python -m next.lab run floor.json`, send the zip. Reading: `det-a` against `det-b` (identical pictures: the training is exact and the rest is rendering; different: something in our pipeline is not deterministic), `equal-a` against `equal-b` (the size of the normal run-to-run difference in the detail columns; any arm difference below it is noise).

## 47. Weights instead of three forwards (about 1 hour)
The gradient norms you sent: region 0 and region 1 about 0.0095 on average, region 2 (noisy end) about 0.0023, so `equal` mostly boosts the noisy end by about 4. `--time-balance gradnorm` measures the same norms on ordinary one-timestep steps (batch 1, no accumulation) and weights each third by the inverse; the step line prints the weights (`weights a/b/c`), expect about 0.5 / 0.5 / 2 once all thirds have 10 steps.
```
python -m next.lab template cheap-equal > cheap.json
```
Arms: `equal` (3 forwards per step), `gradnorm-300` (a third of the compute), `gradnorm-900` (the same compute as `equal`), `single-900` (same compute, no weights). Send the zip. If `gradnorm-900` looks like `equal`, per-step cost drops by three at the same quality, and `gradnorm-300` shows whether it is that or the step count.

## 48. Epsilon scaling on an adapter you already have (no training, about 30 minutes)
Tests whether the painted look or the composition errors are partly exposure bias (the model sees its own slightly-wrong inputs while sampling), by shrinking the predicted noise a little at every step. Needs no training and works on any LoRA from the earlier experiments.
```
python -m next.lab template eps-scale > eps.json
```
Edit `images` and the first arm's `lora` (path of, for example, the `equal` LoRA from item 45: `lab-out/regions-perturb/equal/train/lora_step000300.safetensors`), `python -m next.lab run eps.json`, send the zip. The arms render the same adapter at factors 1, 1.005, 1.01, 1.02, 1.04 (same seeds, so the pictures differ only by the factor). Look at the contact sheets: structure errors and the painted look first; the report's detail columns second (coarse change should move little, the fine columns show whether detail came back or went). Pass: a factor between 1.005 and 1.02 gives visibly better pictures than 1 (then exposure bias is part of the cause and the perturbation placement of item 45 is the training-side fix); no change up to 1.02 and a worse picture at 1.04 means it is not. Also worth one try in ComfyUI if a node for it exists (not checked).

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
