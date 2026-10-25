# Performance lab

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `untested` = unit-tested on a CPU in the sandbox only. Design and rules: `../../rework/74-perf-lab.md`.

The lab answers "where does a step's time go, and how much of it could be recovered". It measures on the device (kernel table, device events), compares every piece with what this card measures it can do (matmul and int8 throughput, copy bandwidth, launch cost), and ranks by **recoverable milliseconds** (measured minus ideal), so a big, efficient matmul ranks below a small slow chain of elementwise kernels.

## Commands
| Goal | Command | Writes |
|---|---|---|
| Profile a few steps of a real run | `python -m next.run sdxl-lora ... --perf DIR` (or `--set perf.dir=DIR`) | `DIR/report.md`, `report.json`, `ceilings.json` |
| Everything pending in one go | `python -m next.perf hwcheck --out DIR --models D:/models --datasets D:/datasets --dataset my_set --file sd_xl_base_1.0.safetensors --output D:/out --batch 2 --resolution 512 [-- more sdxl-lora options]` | `DIR/README.md`, `bundle.zip` (text only), the pieces below |
| A set of runs you define | `python -m next.perf matrix scenarios.toml --out DIR [--resume]` | `DIR/summary.md`, one folder per scenario |
| What changed between two runs | `python -m next.perf compare DIR_A DIR_B` | stdout |
| One operation at a time | `python -m next.perf bench --out DIR [--suites gemm attention layers ...] [--batch 2 --size 768]` | `DIR/bench.md`, `bench.json` |
| This machine | `python -m next.perf hardware --out DIR` | `hardware.md`, `hardware.json` |
| Allocator A/B | `PYTORCH_ALLOC_CONF=expandable_segments:True python -m next.perf allocator` (then `:False`) | one JSON line |
| Do launches link to operators on this build | `python -m next.perf events --out DIR` | `events.md`, `events.json` (raw event fields, kernel attribution of a few known operations) |
| Re-render a report | `python -m next.perf report DIR` | `report.md` with the current rules |

`hwcheck` runs every step in its own process (a fresh allocator, nothing held by the coordinator), keeps going when a step fails, and puts the text files in `bundle.zip`: send that file back. It uses 28 steps per scenario with a 20-step warm-up (graph capture and its timing need several sightings per shape); thirteen scenarios: native and int8, eager and graph, both int8 gradient paths, measured checkpoint placement, the two bf16 graph attention modes, and the allocator setting both ways (they measure speed and memory; losses are compared separately, see TESTING 28). Pass your own `--steps`/`--warmup` to `hwcheck`, not inside the train arguments.

## Settings
`perf.dir` (empty = off), `perf.warmup_steps` (8), `perf.steps` (3), `perf.modules` (true), `perf.trace` (false), `perf.provenance` (false; `hwcheck` turns it on), `perf.ceilings` (file shared between runs). Steps before the window run untouched; the next `perf.steps` steps are timed alone (device idle at both ends) and give the step time; the `perf.steps` after them run under the profiler and the module hooks, which gives device and module times and makes those steps several times slower than normal. The report is then built from the profiler's raw events; the console line `perf: report written ... (stage seconds)` and `report_seconds` in `report.json` say what each stage cost. A run needs at least `warmup + 2 * steps` steps.

## Provenance
With `perf.provenance` every kernel of the profiled window is tied, through the correlation id the profiler gives both the launch and the kernel, to the operator chain that launched it (`aten::to > aten::copy_`), the operator's input shapes and dtypes, and the module whose call it ran in (the module hooks open a profiler range per call). The report then has: module times that are device times (spans between two timestamps mostly measure waiting while the host is slow under the profiler), a table by model region, the non-matmul kernels grouped by operator, dtypes, shapes and module (what the casts and elementwise kernels are), and matrix products by shape with achieved TFLOPS against the measured ceiling. Kernels of replayed graph steps launch nothing and stay unattributed; the coverage line says how much was tied. If the coverage is low the profiler does not link launches on that build; `events` shows the raw fields.

## Reading a report
1. **Step budget.** Device busy time against wall time. The report also gives the average GPU power over the unprofiled steps (Linux, Intel card, hwmon energy counter; absent elsewhere) and the matrix summary has a power column: a device that is busy at low power is stalled or launch bound, saturated matrix work draws much more. A low utilisation (under 60%) means launch bound; the finding names graph replay and whether replay was already running.
2. **Findings**, largest recoverable time first. Each carries its evidence and the lever; levers are suggestions, not changes the trainer makes.
3. **Categories.** Device time by kernel family (gemm, convolution, attention, norm, elementwise, copy, int8, ...). The `other` row lists names the table does not know: add a line to `perf/categories.py`.
4. **Piece groups and modules.** Layers grouped by kind, role and region (down, mid, up), rated against the ceiling; the module table shows forward, recompute and backward separately. Backward costs are estimates (`perf/rating.py`).
5. **Reconciliation.** Module time against device time. A large gap is work outside the hooked modules (loss, optimizer, data movement) or host waits. A window with replayed steps has no module table (a replayed graph runs no Python); profile an eager run for modules.

## Limits to know
- On a CPU the "device" times are host operator times; the report says so and does not judge utilisation.
- Module backward hooks wrap module outputs; the model code has no in-place operation on them, but if a hook warns or fails, run with `perf.modules=false` and say which module.
- The XPU profiler may print a registration warning; it is noise.
- Cost for a module comes from the first shape it saw in the window.
