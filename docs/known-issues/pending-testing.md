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

### (d'') What the offload's failure to move `reserved_mb` costs, and the fix

The consequence named in (d') is real and was measured directly, by
running the loop's two possible remedies side by side on the card. Two
offloadable residents (512 MB and 256 MB, `reserved` at rest 768 MB)
against a 600 MB usable ceiling, both arms in one process, best of 3:

```
arm 1  offload, re-measure, offload again   177.6 ms   offloaded [a, b]
        still over budget: reserved=768 > 600   <- True
arm 2  offload, reclaim, re-measure         130.0 ms   offloaded [a]
        still over budget: reserved=256 > 600   <- False
```

Arm 1 evicted **both** residents and still missed the ceiling, because
neither eviction moved the reading its own exit condition tests. Under
`strict=True` that is a hard failure about a condition a single reclaim
clears. The cost is not the 178 ms: an offloadable resident is
offloadable precisely because it gets reloaded, so every resident arm 1
moved is a host-to-device transfer charged again on each subsequent use.

So the loop now reclaims instead. `BudgetedResourceControlHandle._reclaim()`
hands the cache back, re-reads, and logs one line naming the reading it was
chasing. Three properties make it a fix rather than a new tax:

- **At most once per relief attempt.** A second reclaim with nothing
  allocated in between returns exactly what the first returned, so it
  cannot lower the reading further.
- **Never when under budget.** `text_encoder.py`'s offload() docstring
  records that forcing a reclaim per per-step offload was a measured cost,
  which is why the trainer owns `empty_cache_every_n_steps`. That reasoning
  is about a *recurring* reclaim; this one fires only when an eviction has
  already proved inert.
- **Safe with live tensors.** `empty_cache()` returns only segments
  nothing is allocated from. Measured with a 1024 MB tensor live:
  `reserved` 1024.0 and `allocated` 1024.0 both before and after.

Measured cost of the action, against the thing it replaces: a reclaim after
one offload was **132 ms** (512 MB resident) and **267 ms** (1024 MB), versus
**132 ms** / **263 ms** for one offload-and-reload round trip of the same
resident. Reclaiming instead of evicting is not the expensive branch, and it
leaves a resident resident rather than requiring it back.

`strict=True`'s message now names the reclaim, so a reader who hits it knows
the cache was already tried and was not the answer.

Two notes on the evidence, both about the measurement rather than the finding.
First, the reclaim cost and the round-trip cost came out within 1% of each
other at both sizes, which is suspiciously close for two operations that do
different work; the reclaim figure is dominated by the preceding offload's
transfer, since both arms include one. Second, an earlier version of the
arm script hung, and the cause was a transcription error worth recording:
it called `offload()` unconditionally, without the real loop's
`next_to_evict() is None` exit. On an already-offloaded resident the offload
is a host-side no-op, so `reserved` never moved and neither did the loop's
exit condition. The real loop terminates *only* by exhausting candidates,
which is itself the thing being demonstrated -- an eviction-only loop's
termination depends on there being something left to evict, not on having
achieved anything.

The `reserved_mb` reading is unchanged, deliberately. It is a correct
reading of a number this process holds and may spend again; what was wrong
was the action chosen in response to it, not the measurement.

`smoke_test_resource_control_strict.py`'s eight original checks pass
unchanged. Six new checks cover this, on a `_CachingDeviceContext` that
models the real allocator's contract (an offload moves live MB into cache,
so `reserved` holds; only `empty_cache()` moves it out) -- the scripted
readings-based fake could not have caught this, since it returns whatever
numbers it was handed and so makes an eviction-only loop look correct.
Verified failing without the fix: reverting the reclaim block fails on
"b must not be evicted once an eviction proved unable to move the reading".

### (e) The deliberate collision, on the real card

Production `MemoryLedger` + production `admit()`, a real `TorchDeviceProbe`
(total 12,216 MB, so capacity 11,192 MB after the 1,024 MB foreign
reserve), and the demands measured above:

```
task  demand=9424 MB -> granted 9424 MB
      held now: 9424 MB, free: 1768 MB

graph demand=9192 MB -> REFUSED (MemoryUnavailableError)

  a graph run cannot be admitted: asked for 9192 MB but only 1768 MB is
  free on the device (capacity 11192 MB, free 1768 MB, held by
  task:ingest_lora)

ledger after: held=9424 free=1768
  holder: task:ingest_lora
```

The refusal names what was asked, what is free, the capacity, and **who
holds the card** -- and the ledger still holds the task's claim afterwards,
so a refused start leaks nothing.

### (f) Default demand per dataset task type

Measured by running each kind's real path (`DataTaskRunner`, the same
call the child makes) in its own process against
`div_4.safetensors`, sampling `memory_stats()` and `mem_get_info()`
in-process. `TASK_DEMAND_MB` is in **allocator MB** (the ledger adds the
600 MB overhead to reach device MB).

| kind | input | peak reserved (allocator MB) | overhead | device MB |
|---|---|---|---|---|
| `ingest_lora` | 8 square images @ 512 and @ 768px | **2,268.0** | ~465 | 2,868 |
| `ingest_lora` | 8 **non-square** images @ 512 and @ 768px | **4,882.0** | ~100 | 5,482 |
| `generate_teacher` | 2 conditions, 1 step @ 1024px | **8,824.0** | 525.0 | 9,424 |

`generate_teacher`'s 525 MB overhead matches the training figure (~530),
as it must -- it loads the full UNet and VAE. `ingest_lora`'s is far lower
because it loads only the VAE.

**Correction, and it retracts an earlier claim here.** A first pass measured
ingest by forcing `latent_size=128` (i.e. a 1024px *output*) over the
512--768px source images, and reported 6,914 MB square / 11,736 MB
non-square, concluding that "an ingest of non-square 1024px images does not
fit on this card". **That was wrong.** It was upscaling 512--768px sources
to 1024px, not measuring the real workload: at native resolution the same
code peaks at 2,268 / 4,882 MB, and 5,482 device MB sits comfortably inside
the 11,192 MB capacity. The non-square case is the heavier one -- the crops
`_preprocess_image_crops` takes to cover a square area -- and its own
docstring already records the VRAM ratchet an unbounded `fit` long side
caused, but it fits the card.

**`TASK_DEMAND_MB` is deliberately still empty.** Now that the measurement
is representative, the reason is narrower but still real: ingest's demand is
a function of the *input* (2,268 vs 4,882 MB, a 2.15x spread on identical
code), and resolution moves it too -- the 1024px-upscaling run was 3x the
native one. A single per-kind constant cannot honestly cover that. The
empty map is the safe default and stays: a kind with no number is UNKNOWN,
which the ledger treats as an exploratory exclusive claim -- admitted when
nothing else holds the card, refused with a breakdown otherwise. For an
ingest that wants 5,482 device MB that is already the right behaviour in
practice. Wiring the map wants one decision first -- whether demand is
stated per *kind* or per kind-and-input-shape -- and that belongs with the
resolution and aspect work, not ahead of it.

All six items of this protocol have now been run on the B580. What remains
is not measurement but two decisions, both recorded above: whether
`TASK_DEMAND_MB` should be per kind or per kind-and-input-shape (f), and
whether the handle should keep measuring `reserved_mb` (d', above).

## Pending

- **[2026-10-07] shape bucketing's default, now that (c) has been reported.**
  `shapes diversity problem 2/MEASURED-l2-bucketing.md`.

  Measured on a fixed unpadded holdout, with a seed-to-seed control to read
  against: a multiple of 16 and a multiple of 32 are **within noise** of not
  bucketing, and a multiple of **24 is measurably worse** (+0.000748, 2.9x the
  0.000258 control). x32 is 1.56x faster with peak memory unchanged.

  Not defaulted on, deliberately. "Within noise at 300 steps" is not "no
  difference", the measurement is one dataset with one caption, and the
  remaining arguments for off are unchanged. The x24 result is the durable
  lesson: **pad fraction predicts the cost, bucket count does not** — x24
  reaches fewer shapes than x32 while padding nearly twice as much, so a
  policy chosen by shape count would pick the worst of the three.

- **[2026-10-07] `--attn-ckpt-fraction`, the real version of the
  checkpointing lever.** Turning activation checkpointing **off** is ~1.2x
  faster and **OOMs at step 4** in a real run (peak 10,282 MB against a
  12,216 MB card), so the on/off switch is not a choice on this card. The
  fraction sweep (0.5, 0.25, …) trades a fraction of that 1.2x for a fraction
  of the +3 GB and is unrun. Note the correction this records: an isolated
  probe measured checkpointing-off at **1.56x** and that figure was reported
  as a trade to weigh before the full run was executed. Correct for a bare
  UNet with no optimizer, text encoder or residency controller; wrong for
  training, because what the probe omitted costs memory rather than time.

- **[2026-10-07] `test_memory_peak_store.py`'s concurrency test reports a
  lost update when the real failure is an I/O error.** Pre-existing and
  load-dependent; reproduced on an unmodified tree, so it is not caused by
  whatever ran before it. Left unfixed here deliberately — it is not this
  work's to change, and a green gate that was made green by editing a test
  is worth less than the flake.

  **Mechanism, precisely.** `test_concurrent_writers_max_semantics` starts
  six processes that each construct a `SqlitePeakStore` over one path on
  tmpfs, meet at a `threading.Barrier`, then `record()`. Under contention one
  worker's `_ensure_table()` raises
  `sqlite3.OperationalError: disk I/O error`. That worker therefore never
  reaches the barrier, so the other five wait out `_BARRIER_TIMEOUT_S` (120 s)
  and get `BrokenBarrierError`. The parent then reads back a value written by
  only five writers and asserts
  `stored == max(values) (lost update!)` — which is the symptom, not the
  cause. The lost update it names did not happen; a writer failed to start.

  Observed: 1 failure in 12 concurrent runs of that file on a clean tree, and
  1 in 4 `run_all.py` runs. `/tmp` was not full (20 GB tmpfs, 57 MB used,
  5.1 M free inodes), so it is contention on tmpfs rather than exhaustion.

  Two things are wrong and both are in the test, not the store:

  1. **A worker that dies before the barrier is indistinguishable from a
     lost update.** The parent's assertion names the wrong failure. A
     non-zero child exit, or a `BrokenBarrierError` from a worker, should be
     reported as itself — that is the diagnosis the log already contains and
     the assertion discards.
  2. **The barrier amplifies one slow start into a 120 s stall.** Nothing
     waits for the barrier with any awareness that a peer has already failed,
     so a single I/O error costs two minutes of wall time before the real
     cause is read.

  Reusable rule: a concurrency test that asserts on the *result* rather than
  on the *participants'* exit status will convert any participant failure into
  a plausible-looking correctness failure. Collect child failures first and
  assert on those.

- **[2026-10-06] the lease API and the handle still account for the same
  offload differently.** The sharper half of the `reserved_mb` issue, and
  deliberately left open by MEM-09.

  `BudgetedResourceControlHandle._make_room()` reads
  `memory_stats()['reserved_mb']` and, on this backend, an offload does not
  move it -- the freed segments stay in the allocator's cache. MEM-06's
  lease API accounts by *declared footprint*, and a footprint **does** drop
  the moment a resident is offloaded. So the two halves of the memory rework
  answer "did that offload free anything?" differently, from the same event,
  and a lease can be granted on the strength of room the handle still shows
  as held.

  MEM-09 fixed the loop's *response* to that reading (it reclaims once
  instead of evicting residents that cannot move it) and left the reading
  itself alone. It deliberately did not reconcile the two accountings, because
  they answer different questions on purpose: a lease asks what this run
  declared it needs, the handle asks what the allocator holds. Reconciling
  them means deciding which of those the lease's grant should be bounded by,
  which is a design call rather than a bug fix -- and doing it inside
  `GraphMemory.request()` would touch the admission path that MEM-05 #2's
  physical check already depends on.

  Worth doing before preview generation and large prompt-embedding caches
  land, since those are what actually make per-step offloading happen.
  Until then exposure is low: offloading is pressure-driven, and current
  dataset shapes rarely trigger it.

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
  not move. Reads like a memory leak, and is not one. FIXED 2026-10-06.**
  See MEM-08 (d') for the measurement and (d'') for the fix.

  **Why it looked like a leak.** From the handle's side `reserved_mb` only
  ever rises and never comes back down, which is the signature of one. It
  is not: `empty_cache()` returns the segments immediately (1,024 MB
  resident -> `reserved` 1024.0 -> `.to("cpu")` still 1024.0 ->
  `empty_cache()` 0.0). Nothing is lost; the allocator is holding segments
  it is entitled to reuse.

  **What it actually cost.** Measured by running both remedies side by side
  (d''): the eviction-only loop offloaded *both* registered residents, still
  missed its own ceiling, and under `strict=True` would have raised about a
  condition one reclaim clears. Each resident it moved is a host-to-device
  transfer charged again on every subsequent use.

  **The fix.** `_make_room()` now notices that a move did not lower the
  reading, and reclaims the allocator cache once instead of evicting
  another resident. At most once per relief attempt, and never when under
  budget -- so it is not the per-step tax `text_encoder.py`'s offload()
  docstring exists to avoid. The `reserved_mb` reading itself is unchanged;
  it was the response to it that was wrong.

  **Left open, deliberately: the two accountings still disagree.** MEM-06's
  lease API accounts by *declared footprint*, which **does** drop when a
  resident is offloaded; the handle accounts by `reserved_mb`, which does
  not. So a lease can be granted on the strength of room the handle still
  shows as held. This fix does not touch that, because the two answer
  different questions on purpose -- the lease asks what *this run* declared
  it needs, the handle asks what the allocator holds -- but the pair needs
  reconciling deliberately rather than left to disagree, and that is its own
  piece of work. Practical exposure remains low today (offloading is
  pressure-driven, and current dataset shapes rarely trigger it); it becomes
  live with preview generation and large prompt-embedding caches.

The two entries this section used to hold are now in
[`resolved.md`](resolved.md) with the hardware numbers that closed them:
`keep_incomplete_batches`, and the training diagnostics.
