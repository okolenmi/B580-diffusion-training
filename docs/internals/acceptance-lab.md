# 75. Acceptance lab: tests the owner does not have to read, and instruments that make the next test cheaper

Status: **T1 to T3 built (CPU-tested, not run on the card).** Manual: `docs/manual/acceptance.md`. `accept/`, beside `perf/`. NF4 frozen weights (docs/72) are built with it as the first user: whether a quantized network is
good enough is a question about *regions* of the network, and nothing answered it per region.

## Why
`TESTING.md` is a list of runs the owner does by hand and judges by reading logs. Most of those judgements are numeric (a step time within a margin of another,
a peak under a limit, a loss curve inside noise) and can be decided by a program from the run's artifacts; the owner's time should go to what only eyes can
decide (does this padding look right, is this caption sane). So: automate the numeric verdicts, and make the visual ones cheap to look at.

## Parts
| # | Delivers | Check here |
|---|---|---|
| T1 | **Regions and probes.** `Regions` (a path -> region/level/role map for the UNet, shared with the perf lab's module table), `ModelProbe` (per region statistics of activations and of gradients while a model runs: rms, max, non-finite count), `QuantizationProbe` (a dense network against its quantized copy on fixed inputs: weight error per layer, output and input-gradient error per region; works for int8 and NF4), `python -m accept quant-probe` on a real checkpoint | tiny UNet on a CPU: int8 and NF4 errors by region, a planted bad layer is the worst one |
| T2 | **Acceptance.** `RunArtifacts` (console log parsed into step records, memory csv, `perf/report.json`, output files), `Check` classes (`StepTimeWithin`, `PeakBelow`, `LossCurveClose`, `LinesPresent`, `AdapterFileHolds`, ...), `Verdict` (pass, fail, look: a check that needs eyes names the file to open), suites that restate the `TESTING.md` items as scenarios plus checks, `python -m accept run SUITE` using the perf lab's matrix runner, one `verdict.md` | fake artifacts: each check passes and fails when it should; the runner with a fake command |
| T3 | **Visual aids.** `ImageSheet` (a labelled grid of images), a padding sheet (the same samples padded with zeros and mirrored, side by side: decides the pending padding question by eye), a checklist the verdict ends with | a sheet from synthetic images |

## Rules
- A check reads artifacts only; it never imports the trainer. A suite names the commands, the checks and the thresholds in one place, and the threshold carries the
  reason (the measurement it comes from).
- A failed run is a verdict ("did not finish: last log lines"), not an exception.
- The instruments are generic (regions, probes, sheets); a suite is the only place that knows a particular `TESTING.md` item.
- Probes attach hooks only inside a window and detach after, like the perf lab's `ModuleTimer`.

## As built
T1: `regions.py`, `recorder.py` (`BlockRecorder`), `quantization.py` (`QuantizationProbe`, `ProbeResult`), `fixtures.py` (`DenoiserInputs`), `loading.py`, `quant-probe`.
T2: `artifacts.py` (`RunArtifacts` parses the console format of `run.py`; a change to those lines needs the regexes changed with it), `checks.py`, `suites.py` (`SuiteBook`:
speed, memory, nf4, determinism, allocator; scenario names are the ones `perf.hwcheck` runs, plus `{folder}` in an argument for per-scenario output paths), `verdict.py`
(`Judge`, `write_verdict`), `acceptance.py` (`Acceptance`), commands `run`, `judge`, `list`. `hwcheck` ends by running `judge` as a command (perf does not import accept).
T3: `visual.py` (`ImageSheet`, `PaddingSheet`: bucket as ingestion makes it, four fills side by side), `padding-sheet`.
Not done: checks over the memory csv (`--memory-log`) and the quantization probe's numbers (thresholds need a first real measurement); a suite for adapter file formats.

## First run on the card (2026-11-24, B580, Python 3.14.8, torch 2.12.1+xpu, xpu-kernels 0.4.0, SDXL, 512 px, batch 2)

Suites: memory 11/11, nf4 17/17, features 29/29 passed (a `mirror-signal` run was added since); speed and determinism failed on thresholds that were wrong or checks that were too blunt. All changed since.

| what | measured | consequence |
|---|---|---|
| native graph step | 630 ms (eager 1086 ms) | |
| `math_bf16` graph attention | 689 ms, x1.09 of the default fused kernel | it beat the old `math` default, not the fused one; limit now x1.15 |
| `chunked_bf16` graph attention | 1006 ms, x1.49-1.60 at 512 px; x1.66 at 1024 px (4897 against 2954 ms) | limit now x1.75; it exists for memory |
| `chunked_bf16` against `math_bf16`, 1024 px, batch 1 | allocator peak 7348 against 9734 MB (-2386 MB), step peak 6912 against 8456 MB; the `math_bf16` graph could not be captured (752 MB short) and ran eagerly | the memory side holds; `chunked_bf16` is the mode for large sizes |
| int8 / native peak | 4244 / 6303 MB (2059 MB saved) | |
| NF4 | 662 layers, 3012 MB saved (3110 with double quantization), peak 3154 MB against int8 4244 MB, step 1328 against 1678 ms, last-10 loss equal to int8's | |
| memory suite | native 9210 MB, int8 9390 MB under 10240; a 9000 MB cap held at 6179 MB | |
| allocator | expandable segments 8124 MB against 8612 MB (-488 MB) | |
| mirror padding cost | x0.99 of zero | |
| epochs | `--epochs 2 --save-every-epoch 1`: a file and an "epoch N end" line at each epoch's last step; `--epochs 3 --steps 10` stops at 10 | |
| kernel compile cache | `NEO_CACHE_MAX_SIZE=4 GB` changed nothing (all three runs within 2%); every new shape costs 3.5-5.4 s again on every run | no fix: fewer shapes (`max_shapes`) is the only lever |
| numerics (`accept numerics --tiny`) | 52 PASS, 1 WARN, 40 INFO. The WARN: the default attention kernel repeats 25 of 100 backward results differently, the math backend 0 of 100. INFO: int8 output error 1.9e-2, NF4 0.133 (bf16 step 25%); merging a LoRA delivers 97% of its effect; fused GEGLU 2.8e-7 | |

Open problems:

* **Determinism**: one of two runs of the determinism scenario differed (1444 of 2166 adapter tensors, largest 2e-3), the rerun was exact. The scenario's eager path already uses `chunked_bf16` attention, so the default kernel's repeat WARN is a suspect, not the proven cause. The suite now saves every 7 steps and the check names the first adapter file that differs, which dates the divergence; a result is still needed.
* **Profiler crash**: two scenario processes (and once the parent) ended with SIGSEGV when the profiler's `KinetoEvent` objects were freed (torch 2.12 on Python 3.14). Both reran fine. The trace no longer frees them.
* **Quantization probe** ran out of memory on the card: the dense network was recorded with gradients for its own weights (as much as the weights) and the allocator kept its blocks before the conversion. It now freezes the parameters while recording and empties the cache between phases; to be retested.
* **Padding**: the old sheet shrank each sample to 55-90 % of the canvas, which showed a black band on all four sides that training never has. With the trainer's defaults (`multiple` 4, `max_shapes` 16, `pack` 0.3) padding is 3 % of a canvas on average and 26 % at worst (204 samples, 72 shapes into 16); with `max_shapes` 4 and batch 2, 14 % and 34 %. The sheet now draws the real bucket. The third fill `mirror-signal` (reflect the clean sample, noise it separately) exists for a three-arm comparison; `zero` stays the default.
* The tester's machine had no `node`: the frontend suite and every browser check were not run (items 96-107 of `TESTING.md` are untested).
