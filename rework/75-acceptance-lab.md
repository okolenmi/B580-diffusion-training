# 75. Acceptance lab: tests the owner does not have to read, and instruments that make the next test cheaper

Status: **T1 to T3 built (CPU-tested, not run on the card).** Manual: `next/manual/acceptance.md`. `next/accept/`, beside `next/perf/`. NF4 frozen weights (rework/72) are built with it as the first user: whether a quantized network is
good enough is a question about *regions* of the network, and nothing answered it per region.

## Why
`TESTING.md` is a list of runs the owner does by hand and judges by reading logs. Most of those judgements are numeric (a step time within a margin of another,
a peak under a limit, a loss curve inside noise) and can be decided by a program from the run's artifacts; the owner's time should go to what only eyes can
decide (does this padding look right, is this caption sane). So: automate the numeric verdicts, and make the visual ones cheap to look at.

## Parts
| # | Delivers | Check here |
|---|---|---|
| T1 | **Regions and probes.** `Regions` (a path -> region/level/role map for the UNet, shared with the perf lab's module table), `ModelProbe` (per region statistics of activations and of gradients while a model runs: rms, max, non-finite count), `QuantizationProbe` (a dense network against its quantized copy on fixed inputs: weight error per layer, output and input-gradient error per region; works for int8 and NF4), `python -m next.accept quant-probe` on a real checkpoint | tiny UNet on a CPU: int8 and NF4 errors by region, a planted bad layer is the worst one |
| T2 | **Acceptance.** `RunArtifacts` (console log parsed into step records, memory csv, `perf/report.json`, output files), `Check` classes (`StepTimeWithin`, `PeakBelow`, `LossCurveClose`, `LinesPresent`, `AdapterFileHolds`, ...), `Verdict` (pass, fail, look: a check that needs eyes names the file to open), suites that restate the `TESTING.md` items as scenarios plus checks, `python -m next.accept run SUITE` using the perf lab's matrix runner, one `verdict.md` | fake artifacts: each check passes and fails when it should; the runner with a fake command |
| T3 | **Visual aids.** `ImageSheet` (a labelled grid of images), a padding sheet (the same samples padded with zeros and mirrored, side by side: decides the pending padding question by eye), a checklist the verdict ends with | a sheet from synthetic images |

## Rules
- A check reads artifacts only; it never imports the trainer. A suite names the commands, the checks and the thresholds in one place, and the threshold carries the
  reason (the measurement it comes from).
- A failed run is a verdict ("did not finish: last log lines"), not an exception.
- The instruments are generic (regions, probes, sheets); a suite is the only place that knows a particular `TESTING.md` item.
- Probes attach hooks only inside a window and detach after, like the perf lab's `ModuleTimer`.

## As built
T1: `regions.py`, `recorder.py` (`BlockRecorder`), `quantization.py` (`QuantizationProbe`, `ProbeResult`), `fixtures.py` (`DenoiserInputs`), `loading.py`, `quant-probe`.
T2: `artifacts.py` (`RunArtifacts` parses the console format of `next/run.py`; a change to those lines needs the regexes changed with it), `checks.py`, `suites.py` (`SuiteBook`:
speed, memory, nf4, determinism, allocator; scenario names are the ones `perf.hwcheck` runs, plus `{folder}` in an argument for per-scenario output paths), `verdict.py`
(`Judge`, `write_verdict`), `acceptance.py` (`Acceptance`), commands `run`, `judge`, `list`. `hwcheck` ends by running `judge` as a command (perf does not import accept).
T3: `visual.py` (`ImageSheet`, `PaddingSheet`), `padding-sheet`.
Not done: checks over the memory csv (`--memory-log`) and the quantization probe's numbers (thresholds need a first real measurement); a suite for adapter file formats.
