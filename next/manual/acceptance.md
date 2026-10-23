# Acceptance lab

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `untested` = unit-tested on a CPU in the sandbox only. Design: `../../rework/75-acceptance-lab.md`.

The hardware runs listed in `TESTING.md` are mostly numeric judgements (a step time within a margin of another, a peak under a limit, a loss curve inside a band). The lab restates them as suites of scenarios plus checks, runs or reads the runs, and writes one `verdict.md`. What only eyes can decide stays on your list, with the file to open.

## Commands
| Goal | Command |
|---|---|
| List suites and checks | `python -m next.accept list` |
| Run suites (each scenario in its own process) and judge | `python -m next.accept run speed nf4 --out DIR --steps 28 --warmup 20 -- --models M --datasets D --file F --dataset NAME --output OUT --batch 2 --resolution 512` |
| Judge folders that exist (a `hwcheck` bundle's `matrix/`, a rerun) | `python -m next.accept judge DIR/matrix speed allocator --out DIR/verdict` |
| NF4 or int8 quality by region of the network | `python -m next.accept quant-probe --models M --file CKPT --precision nf4 int8 --out DIR` |
| What zero and mirror padding put around samples | `python -m next.accept padding-sheet --images FOLDER --canvas 128x128 --count 4 --out sheet.png` |

`python -m next.perf hwcheck` runs the `speed` and `allocator` judgement itself at the end (`verdict/verdict.md` in the bundle). Suites: `speed` (TESTING 23, 25, 26, 28), `memory` (24), `nf4` (29), `determinism` (22), `allocator` (24). `run` reuses finished scenario folders, so an interrupted run continues.

## Verdicts
Each check ends in PASS, FAIL, LOOK (a judgement for eyes, with the file to open) or SKIP (the data is not there: the scenario did not run, or the log has no such lines). A scenario that failed to run is not an exception: its checks fail with the last log lines. A suite's threshold carries its source (the measurement or TESTING item) in the table's detail.

## Instruments
- **Regions.** `Regions` maps a module path to `down.N`, `mid`, `up.N`, `embed`, `out` (the perf lab's module table uses the same split).
- **Block recorder.** Records every block's output and the gradient that reaches it, on the host in fp32, within a memory limit.
- **Quantization probe.** Dense network against its quantized copy on fixed inputs: weight error per layer, then output and gradient error per block, summarised by region with the worst block named. `quant-probe` runs it on a real checkpoint with random text conditioning (a fixed seed); it measures how far the network moves, not how good samples are.
- **Visual aids.** `ImageSheet` (labelled grid) and `PaddingSheet` (clean placement, zero padding where the strip is noise, mirror padding, side by side; pixels stand in for latents).

## Adding a check
A check reads `RunArtifacts` (console log as step records and named lines, `report.json`, output files) and returns a `Result`; add it to `accept/checks.py`, then to a suite in `accept/suites.py` with the scenarios it reads and the reason for its threshold. A new log line the checks need is one regex in `accept/artifacts.py`.
