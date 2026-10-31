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


## 40. Lab: is our renderer ComfyUI's? (no training, about 15 minutes)
Before trusting images from the lab, check them against ComfyUI on the same settings:
```
python -m next.lab engines --models /home/okolenmi/comfy/ComfyUI/models/ --file divA.safetensors --comfy /home/okolenmi/comfy --comfy-python /home/okolenmi/comfy/venv/bin/python --lora <your lora> --out lab-engines
```
(`--comfy` is the folder that holds `comfy/` and `main.py`; without `--lora` it compares the base model.) Add `--comfy-dtype bf16` for a second run: ComfyUI picks fp16, we run bf16. Send `lab-engines/engines.md` and `lab-engines/engines-sheet.png` (left ours, right ComfyUI).
Pass: the same pictures up to small differences (mean PSNR above about 30 dB; fine and coarse change small and similar); a clearly different composition or a washed-out or noisy result is a bug in the renderer.

## 41. Lab: the first experiment (range ablation)
`python -m next.lab template > range.json`, then edit: put **your own usual training command** (the whole `python -m next.run sdxl-lora ...` line, with the resolution, batch and memory settings you really train with) into `train_command` (the lab takes the paths out, drops `--output` and `--name` and keeps the rest for every arm), set `images.train` (a folder with some of your training images) and `images.holdout` (ten images that are not in the dataset), and run `python -m next.lab run range.json`. The template's `--resolution 512 --batch 2` are placeholders: the first attempt used the trainer's defaults (1024 px) and ran out of memory before step 1. Four arms train on all timesteps, t 0-250, 250-600 and 600-999 (t=0 is a clean image, t=999 pure noise; σ 0.03-0.70, 0.70-2.29, 2.29-14.6). The report now warns when the training resolution differs from the render size; if the first arm fails the lab stops and shows the end of its log. The first run prints seconds per image; lower `grid.steps`, `seeds` or the prompt list if it is too slow (finished arms are kept). Send `lab-out/range-ablation/range-ablation-share.zip`.

## 42. Memory at the defaults: 1024 px, batch 1
First round (done): defaults fit (step peak 6.4 GB), batch 4 fits through chunking, no checkpointing does not fit (expected), and `--set runtime.deterministic=true` ran out of memory in the first measuring pass (2.5 GiB allocation: the deterministic form of the stock attention materialises the score matrix). Deterministic mode now runs blockwise bf16 attention. Repeat only this one and send the console output:
```
python -m next.run sdxl-lora <your paths> --resolution 1024 --batch 1 --steps 5 --set runtime.deterministic=true --output t42b
```
Pass: it runs, the step peak is close to the default run's (about 6.4 GB), and the first lines warn that attention runs blockwise. Check also that the loss is of the same size as without the flag.

## 43. Several timesteps per image and gradient policies
First round (done): A single < batch 1 loses to everything; project and common came out equal; the floor run (0.45) barely moved. Reading: with regions 3 and unrelated gradients `region_shared` is about 0.33, so a floor of 0.45 skipped most steps (my mistake); and project/common on near-orthogonal gradients both end up as the average of the three unit-length gradients, so what C and D share is the per-region normalisation, not conflict handling. Second round, one lab experiment that separates these (about 1.5 hours):
```
python -m next.lab template regions > regions.json
```
Edit as in item 41 (your command is already in it, 1024 px batch 1, 300 steps, no deterministic flag; set `images`), `python -m next.lab run regions.json`, send the zip and the console of the `strata-*` arms (step lines now print `regions r0:.. r1:.. r2:.. cos_min .. shared ..`). Arms: single; `batch3` (fair control, three images per step); `regions3-mean` (old B); `strata-equal` (unit-length region gradients, averaged); `strata-project`; `strata-common`.
Reading: if equal looks like project/common, the conflict handling does nothing at this batch size and normalisation is the lever; if batch3 beats regions3-mean, three images beat three timesteps of one image; `cos_min` near 0 and `shared` near 0.33 confirm the gradients are unrelated.

## 44. Input perturbation, and regions with less noisy gradients (about 2 hours)
Item 43 showed (step lines of the three policy runs): `cos_min` between -0.08 and 0.06 and `shared` 0.33-0.41, so with one image per region the regions' gradients are unrelated noise and project/common/equal are the same method. The lab now averages each region's gradient over the accumulation window before the policy looks at it, and the noiser can add an input perturbation per region.
```
python -m next.lab template regions-signal > signal.json
```
Edit as before (`images`), `python -m next.lab run signal.json`, send the zip and the step lines (last 15) of `equal-acc4`, `project-acc4`, `common-acc4`. Reading: if `cos_min` there is clearly below 0, the regions really conflict and project/common can differ from equal; if it stays near 0, the conflict idea is finished at these batch sizes and equal is the method. `equal-perturb*` against `equal`: perturbation is the sampler-diversity idea (the model answers from slightly wrong inputs, as it does while sampling); the per-t table should stay as it is and the pictures are the verdict.

## 45. Input perturbation at the clean end of the range only (about 1.2 hours)
Item 44: project and common are the same as equal even with 12 images per step (`cos_min` within +-0.13), and the strong perturbation hurt structure. The perturbation list goes to the regions in turn, and region 2 is the noisy end, so `0,0.05,0.1` gave the strongest perturbation exactly where the structure is made. This run puts it at region 0 and none at region 2.
```
python -m next.lab template regions-perturb > perturb.json
```
Edit (`images`), `python -m next.lab run perturb.json`, send the zip. Pass: `equal` and the `perturb-low*` arms keep the same structure (the t=850-950 ratios in the per-t table stay at about 0.80-0.91) while the detail columns (finest octave, fine / coarse) and your eyes say whether details improved. `--regions 3` is the short form of `noiser.regions=3 noiser.layout=strata trainer.gradient_policy=equal`; the step line also prints the mean gradient norm of every region (`grad norm`), which I need for the next idea (weights that equalise regions without three forwards per step).

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

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
