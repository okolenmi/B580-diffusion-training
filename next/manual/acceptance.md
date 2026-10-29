# Acceptance lab

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `untested` = unit-tested on a CPU in the sandbox only. Design: `../../rework/75-acceptance-lab.md`.

The hardware runs listed in `TESTING.md` are mostly numeric judgements (a step time within a margin of another, a peak under a limit, a loss curve inside a band). The lab restates them as suites of scenarios plus checks, runs or reads the runs, and writes one `verdict.md`. What only eyes can decide stays on your list, with the file to open.

## Commands
| Goal | Command |
|---|---|
| List suites and checks | `python -m next.accept list` |
| Run suites (each scenario in its own process) and judge | `python -m next.accept run --suites speed nf4 --out accept --steps 28 --warmup 20 --models D:/models --datasets D:/datasets --dataset my_set --file sd_xl_base_1.0.safetensors --output D:/out --batch 2 --resolution 512` |
| Judge folders that exist (a `hwcheck` bundle's `matrix/`, a rerun) | `python -m next.accept judge DIR/matrix speed allocator --out DIR/verdict` |
| NF4 or int8 quality by region of the network | `python -m next.accept quant-probe --models M --file CKPT --precision nf4 int8 --out DIR` |
| What zero and mirror padding put around samples | `python -m next.accept padding-sheet --images FOLDER --canvas 128x128 --count 4 --out sheet.png` |

`python -m next.perf hwcheck` runs the `speed` and `allocator` judgement itself at the end (`verdict/verdict.md` in the bundle). Suites: `speed`, `memory`, `nf4`, `determinism`, `allocator`, and `features` (probe and alignment, blend, a tight cap without prewarm, several `--set`s and a typo, zero against mirror padding, int8 with the torch backend and with DoRA). `run` reuses finished scenario folders, so an interrupted run continues.

## Your options in the command
Everything after `--` and the train options before it reach every scenario. The scenarios of the `speed` suite state the options they compare (LoRA compute dtype, fused GEGLU) themselves, so a `--set adapter.lora_compute=activation` of yours changes the `features` runs but not the baselines the speed checks compare against.

## Verdicts
Each check ends in PASS, FAIL, LOOK (a judgement for eyes, with the file to open) or SKIP (the data is not there: the scenario did not run, or the log has no such lines). A scenario that failed to run is not an exception: its checks fail with the last log lines. A suite's threshold carries its source (the measurement or TESTING item) in the table's detail.

## Instruments
- **Regions.** `Regions` maps a module path to `down.N`, `mid`, `up.N`, `embed`, `out` (the perf lab's module table uses the same split).
- **Block recorder.** Records every block's output and the gradient that reaches it, on the host in fp32, within a memory limit.
- **Quantization probe.** Dense network against its quantized copy on fixed inputs: weight error per layer, then output and gradient error per block, summarised by region with the worst block named. `quant-probe` runs it on a real checkpoint with random text conditioning (a fixed seed); it measures how far the network moves, not how good samples are.
- **Visual aids.** `ImageSheet` (labelled grid) and `PaddingSheet` (clean placement, zero padding where the strip is noise, mirror padding, side by side; pixels stand in for latents).

## Numerics
`python -m next.accept numerics` (no training) writes `numerics.md` and `numerics.json`. Checks that must be bit-exact (a move to the host and back, the device's memory, repeating the same forward, an optimizer state or the weights offloaded and reloaded, a graph replay against eager, an adapter written to a file and read back) are PASS or FAIL. Gradients of a whole model are not bit-exact on every device, so the model checks measure instead: each comparison (float32 against activation LoRA, kept against recomputed activations, chunked against whole batches, int8 against bf16, every attention path) is INFO with its difference and a factor against the *reference*, what one bf16 step in the input does (rounding-level noise); WARN is more than twice the reference. `model-repeat` runs the same backward again with default kernels, math attention and deterministic algorithms; `backward-bisect` names the first module, going backwards, whose gradient differs between two identical runs. `merge` and `merge-effect` (with `--lora`) measure what is lost when the adapter is added into bf16 or fp16 weights, per weight and on the model's output. `--tiny` runs the model checks on a small random UNet (no checkpoint, also what the CPU test does); `--only` picks checks (see `list`).

## Training signal
- `python -m next.accept prompts --datasets D --dataset NAME [--tag-swap] [--epochs 3]` writes `prompts.md`: per caption channel (`text`, `tags`, `desc`) how many are empty and how long, which channel the trainer reads, and the prompts the trainer feeds per epoch with and without Tag Swap, with examples. An image whose read channel is empty trains with the empty prompt.
- `python -m next.accept lora-map --lora A.safetensors [B.safetensors ...] [--models M --file CKPT]` writes `lora-map.md`: how large each layer's update is, by part of the network (down, mid, up), kind of layer and block, the ten layers that changed most, layers that did not change, and (with the checkpoint) the update relative to the weight. With two or more files, the cosine between the updates layer by layer: how much of what the adapters learned is shared. It reads files only and runs on the CPU.

- `--save-probe` on a training run makes the saver run one forward with the live adapter before each file is written and keep it in `<file>.probe.safetensors` (inputs, conditioning, prediction with the adapter and with it switched off, for a random prompt and the empty prompt at three timesteps). `python -m next.accept save-probe --probe P --models M --file CKPT [--comfy DIR --comfy-python PY]` loads the saved file into a fresh model of ours and, with `--comfy`, into ComfyUI (`accept/comfy_forward.py`, standalone), and writes `save-fidelity.md`: per comparison a verdict from the adapter's *effect* (prediction with minus without it): PASS when the effect's cosine with the training one is at least 0.98 and its size within 5%, FAIL below 0.8 or off by 30%; the text conditioning and the two engines' difference without any adapter are listed beside.

## Adding a check
A check reads `RunArtifacts` (console log as step records and named lines, `report.json`, output files) and returns a `Result`; add it to `accept/checks.py`, then to a suite in `accept/suites.py` with the scenarios it reads and the reason for its threshold. A new log line the checks need is one regex in `accept/artifacts.py`.
