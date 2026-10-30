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

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
