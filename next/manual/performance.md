# Performance lab

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `untested` = unit-tested on a CPU in the sandbox only. Design and rules: `../../rework/74-perf-lab.md`.

The lab answers "where does a step's time go, and how much of it could be recovered". It measures on the device (kernel table, device events), compares every piece with what this card measures it can do (matmul and int8 throughput, copy bandwidth, launch cost), and ranks by **recoverable milliseconds** (measured minus ideal), so a big, efficient matmul ranks below a small slow chain of elementwise kernels.

## Commands
| Goal | Command | Writes |
|---|---|---|
| Profile a few steps of a real run | `python -m next.run train ... --perf DIR` (or `--set perf.dir=DIR`) | `DIR/report.md`, `report.json`, `ceilings.json` |
| Everything pending in one go | `python -m next.perf hwcheck --out DIR -- --models M --datasets D --file F --dataset NAME --output OUT [--batch 2 --resolution 512 ...]` | `DIR/README.md`, `bundle.zip` (text only), the pieces below |
| A set of runs you define | `python -m next.perf matrix scenarios.toml --out DIR [--resume]` | `DIR/summary.md`, one folder per scenario |
| What changed between two runs | `python -m next.perf compare DIR_A DIR_B` | stdout |
| One operation at a time | `python -m next.perf bench --out DIR [--suites gemm attention layers ...] [--batch 2 --size 768]` | `DIR/bench.md`, `bench.json` |
| This machine | `python -m next.perf hardware --out DIR` | `hardware.md`, `hardware.json` |
| Allocator A/B | `PYTORCH_ALLOC_CONF=expandable_segments:True python -m next.perf allocator` (then `:False`) | one JSON line |
| Re-render a report | `python -m next.perf report DIR` | `report.md` with the current rules |

`hwcheck` runs every step in its own process (a fresh allocator, nothing held by the coordinator), keeps going when a step fails, and puts the text files in `bundle.zip`: send that file back. It uses 28 steps per scenario with a 20-step warm-up (graph capture and its timing need several sightings per shape); ten scenarios: native and int8, eager and graph, both int8 gradient paths, measured checkpoint placement, and the allocator setting both ways. Pass your own `--steps`/`--warmup` to `hwcheck`, not inside the train arguments.

## Settings
`perf.dir` (empty = off), `perf.warmup_steps` (8), `perf.steps` (3), `perf.modules` (true), `perf.trace` (false), `perf.ceilings` (file shared between runs). Steps before the window run untouched, the window's steps run under the profiler with the device idle at both ends, so they are slower than normal steps; nothing else changes.

## Reading a report
1. **Step budget.** Device busy time against wall time. A low utilisation (under 60%) means launch bound; the finding names graph replay and whether replay was already running.
2. **Findings**, largest recoverable time first. Each carries its evidence and the lever; levers are suggestions, not changes the trainer makes.
3. **Categories.** Device time by kernel family (gemm, convolution, attention, norm, elementwise, copy, int8, ...). The `other` row lists names the table does not know: add a line to `perf/categories.py`.
4. **Piece groups and modules.** Layers grouped by kind, role and region (down, mid, up), rated against the ceiling; the module table shows forward, recompute and backward separately. Backward costs are estimates (`perf/rating.py`).
5. **Reconciliation.** Module time against device time. A large gap is work outside the hooked modules (loss, optimizer, data movement) or host waits. A window with replayed steps has no module table (a replayed graph runs no Python); profile an eager run for modules.

## Limits to know
- On a CPU the "device" times are host operator times; the report says so and does not judge utilisation.
- Module backward hooks wrap module outputs; the model code has no in-place operation on them, but if a hook warns or fails, run with `perf.modules=false` and say which module.
- The XPU profiler may print a registration warning; it is noise.
- Cost for a module comes from the first shape it saw in the window.
