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
`python -m next.accept numerics` (no training) writes `numerics.md` and `numerics.json`. Checks that must be bit-exact (a move to the host and back, the device's memory, repeating the same forward, an optimizer state or the weights offloaded and reloaded, a graph replay against eager, an adapter written to a file and read back) are PASS or FAIL. Where two valid computations may differ (float32 against activation LoRA, kept against recomputed activations, chunked against whole batches, int8 against bf16) the line is INFO with the measured difference, next to a reference of the same size made by changing the input by 1/256 (one bf16 rounding step); WARN means more than rounding explains. `--lora` inspects a trained adapter file and measures what a merge into bf16 weights loses on the real weights; `--tiny` runs the model checks on a small random UNet (no checkpoint, also what the CPU test does); `--only` picks checks (see `list`).

## Adding a check
A check reads `RunArtifacts` (console log as step records and named lines, `report.json`, output files) and returns a `Result`; add it to `accept/checks.py`, then to a suite in `accept/suites.py` with the scenarios it reads and the reason for its threshold. A new log line the checks need is one regex in `accept/artifacts.py`.
