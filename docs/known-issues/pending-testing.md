*[← docs/known-issues index](README.md)*

# Pending user testing

Fixes that are believed correct but have only been exercised on CPU, or
whose *effect* has not been measured. Each entry says what is unproven.

A fix leaves this file by being run on real hardware -- Intel Arc B580,
12 GB, torch 2.12.1+xpu -- and moving to [`resolved.md`](resolved.md)
with the numbers that confirmed it. `runs/` and `datasets/` are
gitignored, so those measurements exist nowhere else, which is the whole
reason the resolved entries keep them.

## Confirming one

- `scripts/hw_validate.py` -- a single-experiment harness that builds
  and runs a real training graph on the real XPU (main or managed
  route, real checkpoint, real dataset), capturing per-step loss,
  wall time, and torch-reported reserved/allocated/peak memory to
  `runs/hw_validation/<label>/{steps.jsonl,summary.json}`. One process
  per experiment; an OOM or a `strict=True` raise is recorded as a
  *result* (exit 2 / outcome classification), not a harness failure.
- `scripts/hw_validation_batch.sh` -- the standard set of experiments
  (attention-checkpointing before/after, non-square ratchet,
  budget-pressure + strict, managed-route perf, managed-route
  escalation), runnable as a batch or one label at a time.

A fix that needs hardware confirmation should get its own experiment
there, and an entry here until it has been run.

## MEM-08: the memory-rework hardware protocol

Partly run on the B580 (12,216 MB, torch 2.12.1+xpu, desktop running) on
2026-10-05. What is measured is measured; what is not is named. Each
measurement below is reproducible with the snippet quoted beside it.

### (a) The allocator cap exists, and breaking it is a normal exception

`torch.xpu.set_per_process_memory_fraction` **is present** on this build.
Setting 0.02 (a 244.3 MB ceiling on a 12,216 MB card) and then allocating
256 MiB raises:

```
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 256.00 MiB.
GPU 0 has a total capacity of 11.93 GiB of which 10.92 GiB is free.
244.32 MiB allowed; ...
```

Two things that matter for the design: the cap is genuinely enforced, and
the failure is a **catchable Python exception naming the allowed amount**,
not a process-killing device fault. So a backstop over-budget allocation
can be handled like any other OOM.

```python
torch.xpu.set_per_process_memory_fraction(0.02)
torch.zeros(64 * 2**20, dtype=torch.float32, device="xpu")  # raises
```

### (b) Per-process overhead: ~19 MB idle, ~530 MB once a model is loaded

Driver movement (`mem_get_info`) minus allocator movement
(`memory_stats`) is what "overhead" means here:

| Stage | driver moved | allocator moved | overhead |
|---|---|---|---|
| idle context (warmed) | 20.9 MB | 2.0 MB | **18.9 MB** |
| allocating 2,048 MB of plain tensors | 2,048.0 MB | 2,048.0 MB | **0.0 MB** |
| after `empty_cache()` | -2,048.0 MB | -2,050.0 MB | 2.0 MB |

For an idle context and for plain contiguous allocations the overhead is a
**fixed one-time cost of a live context (~19 MB)**, not a per-allocation
tax: once the context exists, the allocator and the driver move together
exactly.

**A loaded training process is a different figure, and much larger.** From
the archived run `A_after_mon` (rank-64 LoRA, 1024 px, batch 1, 40 steps),
pairing each step's `reserved_mb` from `steps.jsonl` against the nearest
`driver_used_mb` sample in `A_after_mon_vram_monitor.jsonl`, minus the
idle foreign baseline (951.8 MB, sampled before this process was up):

| step | reserved_mb | process driver usage | overhead |
|---|---|---|---|
| 36 | 8592.0 | 9130.9 | **538.9 MB** |
| 37 | 8592.0 | 9131.0 | **539.0 MB** |
| 38 | 8592.0 | 9117.0 | **525.0 MB** |
| 39 | 8592.0 | 9122.9 | **530.9 MB** |

So **~530 MB steady-state**, against ADR 0005's "~600 MB" and
`DEFAULT_PROCESS_OVERHEAD_MB = 600`. The default is therefore sound and
slightly conservative (~13% above measured) — **not** the 30x overstatement
an idle-context reading alone would suggest. The ~511 MB gap between idle
and loaded is the model, kernels, shader cache and XMP state, which exist
only once real weights are resident; the "0.0 MB" row is that same
allocator/driver agreement, in a process with none of that.

*Method note:* these are two artifacts of one run, paired by timestamp
rather than read from a single instrumented sample, and the monitor could
not read the training process's allocator itself (`procs: []` throughout),
so the allocator side comes from `steps.jsonl`. The steady-state rows
agree to within ~2.5%, which is tighter than the run-to-run variation,
but a single-instrumented run would be the clean form. No checkpoint is
present on this machine to re-run it.

### (c) `mem_get_info` against the allocator, desktop running

At rest with the desktop up: `total = 12,216 MB`, `free = 11,187 MB`, so
foreign use is **~1,029 MB** against `DEFAULT_FOREIGN_RESERVE_MB = 1024`.
The reserve is therefore about right at idle and *marginal* under load --
an earlier measurement in the same session saw foreign use at ~1,045 MB,
i.e. the reserve was exceeded by ~21 MB. That margin is why an exploratory
run (whose grant is capacity = total - 1024) gets refused by MEM-05's
physical check on an idle-looking machine.

### (d) Eviction cost per resident (median of 5, real tensors)

| Size | offload (device->host) | reload (host->device) | driver MB actually freed |
|---|---|---|---|
| 256 MB | 48.2 ms | 52.0 ms | 0.3 MB |
| 512 MB | 132.8 ms | 108.4 ms | 0.0 MB |
| 1,024 MB | 265.8 ms | 213.5 ms | 0.2 MB |
| 2,048 MB | 531.0 ms | 427.0 ms | 19.8 MB |

Roughly **0.26 ms per MB** in each direction, i.e. ~3.9 GB/s offloading and
~4.8 GB/s reloading on contiguous tensors. That is the cost the MEM-06
eviction ordering sorts on, and it is what makes the ordering worth having
at all. It is *not* a measurement of a UNet round trip or any other
module-tree `.to("cpu")` move, which carries far more per-parameter
overhead than a contiguous copy; the "2,594 ms" figure in
`control_handle.py`'s docstring is its own separate measurement and this
one neither confirms nor refutes it.

**The "driver MB actually freed" column is the finding.** It is ~zero,
because an offload does not return memory to the driver (see below).

### (d') An offload does not move `reserved_mb`; only `empty_cache()` does

This one is worth reading twice, because a whole component reads the
number in question. A 1,024 MB block:

```
  resident live                reserved=  1024.0 allocated=  1024.0 driver_free= 10217.6
  after .to(cpu), no cache     reserved=  1024.0 allocated=     0.0 driver_free= 10217.7
  after empty_cache()          reserved=     0.0 allocated=     0.0 driver_free= 11241.7
```

`BudgetedResourceControlHandle._make_room()` reads
`memory_stats()['reserved_mb']` after every offload and keeps going until
it drops below the ceiling. On this hardware **an offload does not move
that number at all** -- the freed segments stay in the allocator's cache
and only `empty_cache()` returns them to the driver.
`nodes/model/text_encoder.py`'s `offload()` is a bare `.cpu()` by design
(its own docstring: the trainer owns `empty_cache_every_n_steps` and
forcing a reclaim on every per-step offload was a measured cost).

Consequence, not yet acted on: the handle's relief loop will keep
offloading residents that cannot relieve the reading, and under
`strict=True` can then raise even though the memory genuinely was not
needed -- it was merely cached. The memory was still available for
*reuse* the whole time, so nothing was actually at risk. Not fixed here:
it is a change to the handle's measurement semantics, and
`smoke_test_resource_control_strict.py` scripts `reserved_mb` directly,
so it is a real behavioural change with its own evidence.

### Not yet run

- **(e) the deliberate collision** -- a dataset task and a graph that do
  not fit together, the second refused with the full breakdown.
- **(f) default demand per dataset task type.**

## Pending

- **[2026-10-04] `test_graph_task_gateway.py`'s cooperative-stop test failed
  about one gate run in eight under load. RESOLVED 2026-10-05.** It was
  reproduced and rooted: both stop tests (`test_stopping_a_run_actually_stops_it`
  and its SIGTERM twin) waited for the child's first node record under the
  ordinary 90 s `WAIT` budget, but reaching a node record means exec +
  torch import + node discovery, and under a full 40-file gate that
  exceeded 90 s. The child's log was empty in every failing run -- it was
  never slow enough to log anything, just not far enough along. Fixed by
  giving those two startup waits their own `STARTUP_WAIT = 300.0` while
  leaving the post-stop waits at `WAIT`, so a genuinely hung child still
  fails rather than stalling. Verified: **5 consecutive clean full
  `run_all.py` runs** after the fix, where it previously failed about one
  run in three.

- **[2026-10-05] the handle's relief loop reads a number its own actions do
  not move. Reads like a memory leak, and is not one.** See MEM-08 (d')
  above: on this card an offload leaves `reserved_mb` unchanged and frees no
  driver memory; only `empty_cache()` does.
  `BudgetedResourceControlHandle._make_room()` reads `reserved_mb` after
  every offload and loops until it falls.

  **Why it looks like a leak.** From the handle's side `reserved_mb` only
  ever rises and never comes back down, which is the signature of one. It
  is not: `empty_cache()` returns the segments immediately (1,024 MB
  resident -> `reserved` 1024.0 -> `.to("cpu")` still 1024.0 ->
  `empty_cache()` 0.0). Nothing is lost; the allocator is holding segments
  it is entitled to reuse.

  **Why it still matters.** The memory *was* available for reuse the whole
  time, so nothing was ever at risk of OOM. But the loop cannot see that,
  so it keeps offloading residents that cannot relieve the reading, and
  under `strict=True` it can raise a hard failure about a condition that
  was never real.

  **A second-order consequence, which is the sharper problem.** MEM-06's
  lease API accounts by *declared footprint*, and a footprint **does** drop
  the moment a resident is offloaded. The handle accounts by `reserved_mb`,
  which does not. So the two halves of the memory rework disagree about
  whether the same offload freed anything: the lease will grant a request
  on the strength of room the handle still shows as held. Whichever way
  this is resolved, the two accountings need to be reconciled deliberately
  -- they currently answer "did that free memory?" differently, from the
  same event.

  Not fixed: it is a change to what the handle measures, its own smoke test
  scripts `reserved_mb` directly, and the right fix (measure allocator
  availability, or reclaim on demand) is its own piece of work. Note the
  practical exposure is low today -- offloading is driven by pressure, and
  for the current dataset shapes there is little reason to offload per
  step; it becomes live with preview generation and with caching a large
  pack of prompt embeddings, where both this loop and the two accountings
  will actually be exercised.

The two entries this section used to hold are now in
[`resolved.md`](resolved.md) with the hardware numbers that closed them:
`keep_incomplete_batches`, and the training diagnostics.
