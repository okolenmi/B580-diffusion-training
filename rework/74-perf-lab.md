# 74. Performance lab: measuring where a step's time goes, and what is recoverable

Status: **P1 built (CPU-tested, not yet run on the card); P2 next.** A sub-project inside `next/perf/`, beside the trainer, that turns one run (or a matrix of runs) into a report that names the
slowest pieces, says how far each is from what this card can do, and says which known cause it looks like. It exists because the sandbox has no XPU:
every run on the owner's card must return everything needed to understand it, in one file.

## Principles
- **Device time, not wall time.** Host timers measure enqueue. The primary numbers come from the device (events, the profiler's kernel table); wall time
  is only used for the step and for host cost.
- **Rate against what the card can do, not against zero.** Each piece gets an *ideal* time from measured ceilings (large bf16 GEMM TFLOPS, int8 TOPS, copy
  bandwidth, launch cost) and its own shape. The ranking is by **recoverable time** (measured minus ideal), so a big but efficient GEMM ranks below a
  small, slow elementwise chain.
- **Attribute at three levels, and reconcile them.** Step phases (fetch, conditioning, forward, loss, backward, optimizer); modules (per block and per
  layer kind, forward and backward); kernels (by name, grouped into categories). The totals of each level must match the step's device time within a stated
  tolerance, and the report says by how much they do not.
- **Never change what is measured.** Probes add events and hooks only (no synchronisation inside the step except in the explicitly timed phase variant);
  anything that does is named in the report and kept to the profiled window.
- **One command in, one file out.** A run with `--set perf.dir=DIR` writes `DIR/report.md` (read by a person or an agent), `report.json` (compared by
  tools) and optionally `trace.json` (Chrome trace). Everything else (versions, driver, settings, ceilings) is in the report header.
- **Runs on a CPU too**, at tiny scale, so the pipeline (categorising, rating, comparing, rules) is tested here; only the device probes need the card.

## What a report contains
1. **Header**: command, settings, torch/xpu-kernels/driver versions, device, ceilings used (measured now or loaded from the cache for this driver).
2. **Step budget**: wall per step, device busy time (union of kernel intervals), device idle gap, **device utilisation** (busy / wall: low means launch bound),
   launches per step, host time per launch; split by phase; eager and replayed steps listed separately; first-sighting outliers (new shape compile cost).
3. **Categories** (device time share, calls, mean per call): GEMM, convolution, attention, normalisation, elementwise, cast/copy, reduction, optimizer,
   int8 (rotate, quantize, dequantize, int matmul), memory operations, other. Each with achieved bandwidth or TFLOPS against its ceiling.
4. **Modules**: per layer kind (attention q/k/v/out, feed-forward in/out, convolutions, norms) and per region (down, mid, up, resolution level), forward and
   backward device time, FLOPs and bytes by an analytic cost model, achieved against ideal, **recoverable ms**. Top N slowest by recoverable time.
5. **Memory**: peak per phase, the allocator's free and stuck memory, the budget lines.
6. **Findings**: rules (below) that turn the numbers into named issues with the evidence and the lever that applies, ordered by recoverable ms.
7. **Comparison** (for two or more runs): per category and per module delta, so "why is int8 slower" is a table of which pieces grew.

## Findings rules (each a small class with `check(report)`)
- Device utilisation below 60% of the step: launch bound; names the lever (graph replay) and whether replay was tried and dropped, and why.
- Attention kernels in a replayed step are the math ones while the eager step ran flash: the capture limitation (rework/73).
- Cast and copy kernels above a share: dtype round trips (autocast, `.to` chains), listing the modules that issue them.
- Int8 path: rotate+quantize+dequantize share against the int matmul share; dequantize per backward against the layer's GEMM.
- Elementwise and norm tail: many small kernels with low achieved bandwidth: fusion candidates, grouped by module.
- Recompute: share of forward time spent twice (checkpointed blocks), and the memory it buys (from the placement notes).
- New shapes: steps that are outliers because a latent shape appeared for the first time (kernel and primitive creation), the count of distinct shapes against
  the primitive cache.
- Synchronisation points inside the step (host reads of device values): counted from the trace's wait events.
- Convolutions share against their efficiency: the part no current kernel route reaches.
- Anything in a module whose achieved/ideal is below a threshold, ranked; the generic "slowest piece" list.

## Architecture (`next/perf/`, one concern per class)
| Class | Job |
|---|---|
| `Ceilings`, `CeilingProbe` | Measure bf16 GEMM TFLOPS, int8 TOPS, copy GB/s, empty-kernel launch cost, host op dispatch cost, sync cost; cache by device and driver |
| `DeviceClock` | Synchronised wall time, and device events where the backend has them |
| `Categorizer` | Kernel name to category by an ordered rule table (one place to teach it a new kernel); unknown names are listed, never dropped |
| `KernelTrace` | Wraps the profiler for a step window; returns kernel rows (name, device time, calls, start/end), host op counts; degrades to host events on a CPU |
| `CostModel` | FLOPs and bytes per module kind from the shapes seen (Linear, Conv2d, attention, norms, Int8Linear) |
| `ModuleTimer` | Forward and backward hooks with device events around chosen modules; attaches only inside the window; records the shapes for the cost model |
| `StepBudget` | Combines the phase times, trace and counts into the budget above |
| `Rating` | Ideal time from the ceilings and the cost model; recoverable time; ranking |
| `Rule` subclasses, `Findings` | The rules above |
| `ReportWriter`, `Comparison` | Markdown and JSON out; deltas between two reports |
| `PerfSession` | What the trainer calls: `step(n)` context manager that decides the window, starts and stops the probes, and returns the data at the end |
| `Scenario`, `Matrix` | A named set of settings and arguments; runs each in its own process (fresh allocator) and collects the reports |
| `HardwareReport` | Versions, driver, environment variables, the allocator's config, what the card reports |

Layering: `perf` may import `engine`, `data`, `graph`, `families.api`, `families.registry`; `nodes` may import `perf` (the trainer node builds the session);
`engine` never imports `perf` (the trainer takes a duck-typed observer).

## Phases
| # | Delivers | Check here |
|---|---|---|
| P1 | `Ceilings`, `DeviceClock`, `Categorizer`, `KernelTrace`, `CostModel`, `ModuleTimer`, `StepBudget`, `Rating`, first rules, report writer, `PerfSession`, trainer hook, settings `perf.dir` `perf.warmup_steps` `perf.steps` `perf.modules` `perf.trace`, `--perf` flag | CPU run of a tiny model produces `report.md` and `report.json`; categoriser, rating and rules unit tests |
| P2 | `Scenario` and `Matrix` runner (`python -m next.perf matrix FILE`), `Comparison`, a scenario file for the open questions (native/int8 x graph x checkpointing x attention x batch) | comparison of two synthetic reports; matrix with a fake runner |
| P3 | Micro-benchmark suite: layer shapes (native/int8, both gradient paths), attention backends (what captures, per-backend device time), GEMM/conv/norm efficiency vs ceilings, launch-cost probe; results feed `Ceilings` and the rating | scripts run on CPU at small size |
| P4 | `python -m next.perf hwcheck`: the pending `TESTING.md` runs in one go, one process each, one bundle (`HardwareReport` + reports + matrix); `next/manual/performance.md` | the runner against a fake command |

## Owner's workflow
`python -m next.run ... --perf` (or `--set perf.dir=DIR`), then push `DIR/report.md` and `report.json`; later `python -m next.perf matrix scenarios/open.toml`
and push the bundle. Reports are small text files on purpose.

## Not in scope
Changing what the trainer does because a rule fired (the rules name levers; changing defaults is a separate decision per lever), and anything needing a
second GPU. Profiler caveats that are known: the XPU profiler prints a registration warning that is noise; an out-of-memory inside a profiled step loses the
device and the process dies (the memory budget is the protection, and the window never starts when the budget has no room).
