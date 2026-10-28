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

## 31. Pictures for the faster defaults
`adapter.lora_compute=activation` and `trainer.fused_geglu=true` passed every number (steps 8% to 40% faster, loss over the last 10 steps the same to four digits; with the baselines pinned to float32 LoRA the activation dtype is 10% faster eager and 24% faster in a math_bf16 graph). They become defaults after a look at pictures. Two LoRAs are enough: today's defaults against both options together (the options are judged as one package, because that is what would become the default; fused GEGLU is checked against the eager result when it starts, so activation is the one that can change a picture).

Same command for both, with your paths; `--steps 600` or more, at the resolution you train at if the time allows (512 is fine for a first look). `--set runtime.deterministic=true` makes the default run repeatable, so any difference below comes from the options and not from the run.
```
python -m next.run sdxl-lora --models <models> --datasets <datasets> --dataset <name> --file <checkpoint> --output <output> --resolution 512 --batch 2 \
  --steps 600 --save-every 600 --set runtime.deterministic=true --name lora_default
python -m next.run sdxl-lora ...the same arguments... --name lora_fast --param adapter.lora_compute=activation --param trainer.fused_geglu=true
```
Add nothing else to either (no graph, no int8, so the only difference is the two options). Optional third run for scale: the default again with `--set seed.base=1 --name lora_other_seed` (a different noise and order, which is the kind of difference you would never notice as a flaw).

Then in ComfyUI: the same checkpoint, the same 4 or more prompts (some from your dataset's style, some off it), 3 seeds each, the same sampler and strength, once per LoRA. Look at the pairs side by side and answer: can you tell `lora_default` from `lora_fast`, and is the gap larger than the one to `lora_other_seed`? Also send both console loss lines at steps 100, 300, 600.
Pass: the pairs look alike (differences of the seed-change kind). Fail: describe what changed (washed out, noisier, style weaker), and I will keep the defaults and look at which option causes it (one more LoRA with `activation` only).

## 37. Numerics of the device, the moves and the adapter path
Why: two LoRAs trained a minute apart differ by chance, so what matters is whether anything we do adds a *systematic* error, or the numbers are not repeatable. This checks the numbers directly, without training. About 10 minutes; needs about 3 GB of free device memory besides the checkpoint's and about 5 GB of free host memory (`merge-effect` keeps a copy of the adapted weights while it merges).
```
python -m next.accept numerics --device xpu --models <models> --file <checkpoint> --lora <one of your trained LoRA files> --resolution 256 --out numerics
```
Send `numerics/numerics.md` (and `numerics.json` if it is small). A check that crashes is reported as a FAIL with its traceback, so the run always finishes.
Pass: no FAIL. A FAIL means a move or a repeat is not bit-exact (host and back, memory pattern, repeated forward output, optimizer or weights offload, graph replay, adapter file) and is the real finding. The rest I read myself.

## 38. Does the merge explain the pictures? (ComfyUI precision)
`numerics` item 37 measured that a trained LoRA merged into bf16 weights loses a large part of its update (the update is about as small as the weight's rounding step), while running it beside the weights, as the trainer does, loses nothing. ComfyUI merges a LoRA into the model's weights when it loads it, in the dtype the model runs in. So the same LoRA can look different there than in training, and by a different amount per LoRA. The test: the same LoRA files (`lora_default`, `lora_fast`), the same prompts and seeds, strengths 0.5 and 1.0, with ComfyUI started with `--fp16-unet` (finer rounding: about 8 times smaller loss) and then with `--bf16-unet`; `--fp32-unet` as well if it fits in memory (no loss at all). Send the pictures side by side per dtype.
Pass (the merge is the cause): the fp16/fp32 pictures look like what the training loss promised (less destroyed at 1.0, the two LoRAs closer to each other) and the bf16 ones are noisier or weaker. If all dtypes look alike the merge is not the cause and I look elsewhere.

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
