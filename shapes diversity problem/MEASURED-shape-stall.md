# Measured: the shape stall is oneDNN primitive-cache thrash — and the fix is 1.54x on real training

**Status: cause identified, fix measured on the real training path, nothing landed in
code yet.** Intel Arc B580 (12,216 MB), torch 2.12.1+xpu.
`scripts/hw_validate.py main --dataset non-square --batch 2 --steps 300`.

Supersedes `REPORT-multires-stall.md` section 2, which framed the question
(one-time vs recurring) correctly but had no hardware to answer it.

## The result

`ONEDNN_PRIMITIVE_CACHE_CAPACITY` defaults to **1024**. That is too small to
hold the primitives for a multi-resolution dataset, so every shape *transition*
re-creates them.

**In production configuration** — the five variables
`nodes/xpu_env.py:set_xpu_perf_env_vars()` sets, which is what a graph child
actually gets (`backend/infrastructure/graph_task_worker.py:432` calls it
before importing torch), 300 steps at batch 2:

| production-env run | repeat | revisit | first sighting | revisit/steady | steps/s |
|---|---|---|---|---|---|
| capacity 1024 (default) | 0.884 s | **3.935 s** | 3.849 s | **4.45x** | **0.466** |
| `ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536` | 0.925 s | **0.925 s** | 3.847 s | **1.00x** | **0.757** |
| single-shape `1024 aes` (control, no env vars) | 1.283 s | — | — | — | 0.769 |

**1.62x throughput, reaching 98% of single-shape training speed.** One
environment variable, no dataset change, no change to training numerics.

The same comparison *without* the production variables, which is what
`scripts/hw_validate.py` runs (it never calls `set_xpu_perf_env_vars()`):

| | revisit/steady | steps/s |
|---|---|---|
| capacity 1024 | 4.34x | 0.480 |
| capacity 65536 | 0.99x | 0.741 |

The fix holds either way (1.54x without the SYCL variables, 1.62x with), so the
project's own SYCL cache settings neither cause nor mask this. Worth knowing,
because `SYCL_CACHE_IN_MEM=1` and `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD=0` look
like they should have addressed it — an unlimited in-memory *kernel* cache does
not help a oneDNN *primitive* cache, which is a different layer.

The mechanism is exactly what it should look like: revisits stop paying
(4.45x -> 1.00x), first sightings keep paying (3.85s -> 3.85s), because those
genuinely are new code. `repeat` is unchanged within noise. Peak reserved is
**identical to the byte** (8,774 MB) — the oneDNN primitive cache is host
memory and cannot compete for VRAM.

## The cost is per shape TRANSITION, not per shape

Classifying each step by its shape (`latent_shape`, now recorded in
`steps.jsonl`):

| step kind | n | capacity 1024 | capacity 65536 |
|---|---|---|---|
| repeat — same shape as previous step | 161 | 0.873 s | 0.953 s |
| revisit — seen earlier, not previous | 94 | 3.790 s | 0.945 s |
| first sighting | 44 | 3.989 s | 3.853 s |

So a step is fast exactly when it repeats the previous shape, and slow on
*every* transition regardless of whether the shape was seen before. There are
44 first sightings and 118 slow steps: the extra 74 are revisits, and they are
the ones the cache fixes. The cost also explains the bimodal step-time
distribution — 40% of steps take >2s and hold 74% of wall time, while the
**median step (0.873s) is faster than the single-shape control's (1.283s)**.
Small latents really are cheaper; the loss is entirely in transitions.

## Why an earlier revision of this file said the opposite

An earlier draft concluded "revisits are already at steady state, it is a
one-time per-shape cost" and that the cache did nothing. **Both were wrong**,
for two separate reasons, and both are worth recording because each would have
sent the next person the wrong way:

1. **A misaligned reconstruction.** `steps.jsonl` recorded no per-step shape, so
   the shape order was rebuilt by replaying the seeded loader stream and zipping
   it against recorded step times. That join was wrong — it reported revisits at
   0.95x when they are 4.34x. The join is now unnecessary: `latent_shape` is
   recorded per step.
2. **A 40-step run had 3 revisits.** The first training-path test
   (`SHAPE_default` vs `SHAPE_cap65536`, 40 steps) came out 0.413 vs 0.411
   steps/s and I read that as "the fix does not work on training". Three
   revisits cannot detect a 4.34x effect. **The probe was right and my training
   test was too short to refute it.**

The probe's `--repeat 2` (two steps per shape, back to back) means most of its
"revisits" were in fact the second visit of a shape being compiled for the first
time — which is why the probe found the effect so strongly (6.34x). Real
training clumps shapes at mean 2.16 and switches often, so it thrashes the
small cache far more than the probe's own mix suggested at first glance. The
probe and training agree once measured with enough samples.

## Also corrected: the dataset has 44 distinct shapes

An earlier draft claimed 63 and that `docs/known-issues/open.md`'s 44 was
stale. **44 is right.** `SELECT DISTINCT latent_h, latent_w` over the metadata
table gives 63, but the loader emits 44 — it drops 19 samples sitting in
incomplete `(prompt, size)` groups and groups by prompt as well as size.
Reading the table counts shapes the trainer never sees.

## What landed in the repo

- **`nodes/xpu_env.py` sets `ONEDNN_PRIMITIVE_CACHE_CAPACITY=2048`.** Placed
  there because it is the one place both entry points already call before torch
  is imported — `backend/cli.py` for the server, and each graph child in
  `backend/infrastructure/graph_task_worker.py`. Since MEM-05 made every graph
  run a fresh process, a setting that lives in the server is not a setting the
  child gets. `nodes/smoke_tests/smoke_test_xpu_env.py` (5 checks) pins it,
  including that the value cannot silently fall back to oneDNN's 1024.
- **`MonitoringPhase` (both routes) passes the step's latent shape to
  `on_step`**, and `hw_validate.py` records it as `latent_shape`. Without it,
  none of the above is measurable from a run's output — which is why two wrong
  conclusions were reachable here. The arity-tolerant dispatch is
  `nodes/train/step_notify.py` (13 checks); passing a third argument
  unconditionally broke every existing two-argument `on_step` caller.
  `analyze_steps.py` in this folder, written before that change, expects
  `shapes.jsonl` and is superseded by reading `latent_shape` directly.

## The landed capacity: 2048, not 65536

A capacity sweep, production env, 150 steps per point (44 shapes end their
first sightings by ~step 44, so the remaining ~100 steps are revisits — ample
to see whether revisits are fast, and half the cost of a 300-step run):

| capacity | revisit/steady | steps/s | peak host RSS |
|---|---|---|---|
| 1024 (oneDNN default) | **3.85x** | 0.488 | 15,567 MB |
| **2048** | **0.99x** | 0.599 | 15,567 MB |
| 4096 | 1.00x | 0.620 | 15,567 MB |
| 8192 | 1.00x | 0.591 | 15,567 MB |
| 16384 | 0.99x | 0.605 | 15,567 MB |
| 32768 | 1.01x | 0.583 | 15,567 MB |
| 65536 | 1.00x | 0.591 | 15,567 MB |

**2048 is enough, and 65536 buys nothing over it** (0.599 vs 0.591 steps/s at
equal steps, identical ratios) — so the value first landed was 32x larger than
needed. Corrected to 2048. The spread across 2048–65536 (0.583–0.620) is
run-to-run noise; revisit/steady is flat at ~1.00 throughout and is the measure
that discriminates, being independent of step count.

**Host RAM cost: none.** Peak RSS is byte-identical at all seven capacities.
`VmHWM` is a kernel-maintained high-water mark, so unlike a sampler it cannot
miss a peak — this is not a measurement gap. The primitive descriptors are
simply small.

The bracket is **(1024, 2048]** — so 2048 is the smallest *measured* sufficient
value for *this* dataset, not a proven minimum, and the requirement scales with
the shape count. A dataset with several hundred distinct shapes needs more.

## Sizing the cache to a dataset, and where that stops being the answer

Three further measurements, because "set it to 65536" was an answer to a
question nobody had asked yet.

**The variable is read lazily.** `set_xpu_perf_env_vars()` is documented as
having to run before anything touches an XPU device, and that is true for the
five SYCL variables — but not for this one. Setting the capacity *after*
importing torch **and** touching the device gives revisit/steady **0.99x**,
identical to setting it before:

| capacity 2048 set... | revisit/steady | steps/s |
|---|---|---|
| before torch import | 0.99x | 0.599 |
| **after device init** | **0.99x** | 0.577 |

So the dataset's shape count does not have to be known before torch loads. That
matters because `set_xpu_perf_env_vars()` runs in the graph child *before the
graph is parsed*, so a dataset-sized capacity could not have been set there;
this is what makes the option open at all. (The 0.577 vs 0.599 is run-to-run
noise — the sweep spread across working capacities was 0.583–0.620.)

**How many primitives per shape.** Narrowing the 44-shape bracket:

| capacity | revisit/steady |
|---|---|
| 1024 | 3.85x |
| 1280 | 2.22x |
| **1536** | **1.05x** |
| 2048 | 0.99x |

44 shapes need **(1280, 1536]**, i.e. **29.1 to 34.9 primitives per shape**.
Sizing uses the wide end, so a dataset is never sized to the optimistic end of
a 1.2x-wide measurement.

**Capacity is a ceiling, not an allocation.** Peak host RSS is **15,567 MB at
every capacity measured: 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536,
262144** — byte-identical across a 256x range, and `VmHWM` is a kernel
high-water mark, so unlike a sampler it cannot miss a peak. Unused capacity is
free. That is why the shipped default is a round 2048 rather than the measured
1536: the cost of headroom is zero and the cost of being stingy is a 4x revisit
penalty.

**Landed:** `nodes/xpu_env.py` exposes `set_xpu_perf_env_vars(onednn_primitive_
cache_capacity=N)` and `primitive_cache_capacity_for_shapes(n)`. Resolution
order is explicit argument → exported `ONEDNN_PRIMITIVE_CACHE_CAPACITY` → the
measured default, because both gateways spawn children via `os.environ.copy()`
(`backend/infrastructure/graph_task_gateway.py:98`), so one export covers the
server and every child it starts. An unusable value is refused with a warning
*and replaced in the environment* — oneDNN reads `os.environ` directly, so
leaving `"junk"` there is what makes it parse as 0, not the warning.
17 checks across `smoke_test_xpu_env.py` and
`smoke_test_primitive_cache_sizing.py`.

**Not wired into the trainer yet.** The sizing function is ready and tested;
calling it needs a distinct-shape count, and the two available sources are both
awkward. Iterating the batch source would advance loader state
(`bucket_balance.observe`, epoch counters) and can skip data. A `DISTINCT`
query on the loader's own `metadata.db` is the safe source and is cheap, but it
is a change to the data path and belongs in its own commit with its own test.
Not taken here because this file is a measurement record, not a change set.

## Where it is triggered in a graph

**`ManagedDatasetSourceNode.build()` — the dataset-loading node.** Not a
trainer hook and not a server-side spawn argument, because of the two facts
already established:

- **Order.** Nodes build in topological order
  (`backend/infrastructure/graph/runtime.py:385`), so the dataset is built
  before the trainer that consumes its batches. The dataset is also the first
  node that knows how many distinct latent shapes the run will see.
- **Timing.** The variable is read lazily, so "before the first training step"
  is early enough. That is what frees this from the constraint that made the
  `graph_task_worker` placement impossible.

The count comes from `loader.trajectories`, which `ManagedDatasetLoader` has
already fetched in its constructor — no extra query, and nothing that advances
loader state. It **over-counts**: 63 rows for `non-square`, where the loader
emits 44. That is deliberate and load-bearing. Samples in incomplete
`(prompt, size)` groups never reach training, so the trajectory table contains
shapes this run will never compile a primitive for. Over-counting is the only
safe direction here, because capacity is a ceiling and unused entries cost
nothing: `non-square` gets 2205 (63 x 35) rather than 1540 (44 x 35), which is
free, whereas an under-count is the 4x revisit penalty.

Three properties, each with a check:

- **The dataset node cannot become the reason a dataset fails to load.** The
  whole sizing block is best-effort; anything unexpected is logged and the
  measured default stands, which is correct for `non-square` and merely
  conservative for anything larger.
- **An operator's exported value survives**, because a knob that cannot be
  deliberately set smaller is not a knob. The shortfall is logged when the
  computed requirement is larger, since the symptom it causes looks exactly
  like a bug.
- **Sizing does not cost the node its batches.** Checked explicitly: the
  dataset node still yields a usable batch afterwards. A mis-sized cache that
  broke data loading would be a worse bug than the slow steps it prevents.

## Where grouping becomes the answer instead

The 4096-shape worst case (`[64-128]x[64-128]` unstandardised) needs ~143,000
entries at 35 primitives per shape. **The cache would hold it, free.** So the
boundary where "raise the cache" stops being the answer is *not* the cache
running out — it is compute:

| dataset | shapes | capacity needed | first-sighting cost/run | verdict |
|---|---|---|---|---|
| `non-square` | 44 | 1,536 | ~184 s (44 x 4.18 s) | cache is enough |
| moderate | 512 | 17,920 | ~1,900 s | **grouping wins** |
| worst case | 4,096 | 143,360 | ~15,000 s | **grouping wins** |

Grouping 44 shapes to 3 cuts *both* the transition cost and the
first-sighting cost, and the second is the one no cache reaches. It is
implemented (`shape_bucket_multiple`, default off) and measured at **1.83x** —
see "Shape bucketing: implemented, and 1.83x on the real path" below, which
also reports that it trains 273/273 samples instead of 242/273.
`SHAPES_WHERE_GROUPING_WINS = 512` names that
boundary, expressed in shapes because that is what a user can act on. It is a
recommendation, not a cliff: above it, grouping is cheaper, not required.

## The persistent disk cache does not work — measured at two run lengths

`SYCL_CACHE_PERSISTENT=1` + `SYCL_CACHE_DIR`, with the primitive fix in place so
only first-sighting cost remains, 150 steps per run:

| | first sighting | steps/s | disk cache after |
|---|---|---|---|
| persistent cache off | 3.718 s | 0.598 | 0 files |
| cold (cache empty) | 3.680 s | 0.611 | 108 files / 1,138 MB |
| **warm (second process)** | **3.604 s** | 0.624 | 108 files / 1,138 MB |

**A warm cache saves 3 s of a 164 s cost — 2% throughput, for 1.1 GB written to
disk.** The reason is the layer mismatch this file keeps running into: it
persists SYCL's SPIR-V kernel binaries, while the first-sighting cost is
oneDNN *primitive* creation, which has no supported persistent form.

Re-measured at **300 steps** to test whether a longer run amortises it better.
It does not, and the reason is structural rather than incidental — first
sightings are capped at 44 *per process* because there are only 44 unique
shapes, so the fixed cost stays ~166 s while the denominator grows:

| | first sighting | total per run | steps/s |
|---|---|---|---|
| off, 300 steps | 3.862 s | 170 s | 0.735 |
| cold, 300 steps | 3.851 s | 169 s | 0.746 |
| **warm, 300 steps** | **3.767 s** | **166 s** | 0.761 |

**4 s of 166 s — the same 2% as at 150 steps.** A longer run makes the saving a
*smaller* fraction, not a larger one.

So the ~166 s per run is **not removable by configuration**. It was measured,
not inferred, and it is the reason the recommendation below changed.

## The remaining cost is ~40% of every run, and only shape count touches it

With the fix landed, a run still pays ~3.85 s x 44 = **~164 s** of first
sightings. A 300-step run takes ~400 s, so that fixed cost is **~40% of
wall time** — paid in full on *every* run, because MEM-05 made every graph run a
fresh child process.

This reverses an earlier statement in this file, which called shape bucketing
"clearly second-order". That was wrong, and it was wrong because it reasoned
only about the *transition* cost the cache fix removes. The first-sighting cost
is a per-run fixed cost, it is not amortizable, no cache reaches it, and it
dominates short runs — which is the common case for a graph execution.

Bucketing 44 shapes to 3 cuts first sightings from 44 to ~3, i.e. ~164 s to
~12 s. That is the remaining win, and it is now the *only* lever on it. Its
cost is unchanged and still unadopted: +15% compute, and padding changes what
the loss is computed over, so the true size must survive into the loss and into
any preview or VAE decode.

## Shape bucketing: implemented, and 1.83x on the real path

Optional (`shape_bucket_multiple`, default 0). Pads each latent up to the next
multiple of N so a multi-resolution dataset trains on few shapes, and excludes
the padded region from the loss with a validity mask. 150 steps, batch 2, the
primitive-cache fix in both arms:

| | shapes | first sightings | steady step | steps/s | wall |
|---|---|---|---|---|---|
| unbucketed | 44 | 159 s | 0.855 s | 0.599 | 249 s |
| **bucketed, multiple of 32** | **3** | **12 s** | 0.865 s | **1.096** | **136 s** |

**1.83x**, from 147 s of first-sighting cost removed. Peak reserved is
unchanged (8,762 vs 8,774 MB) and drift drops from 14 MB to 2 MB.

**The real cost is 14x smaller than the pixel count suggests.** The +15% in
this project's earlier table is a latent-*pixel* figure, and at these sizes
compute is not proportional to pixel count: the measured penalty is **+0.010 s
per step (+1%)**, not +0.140 s.

That corrects advice given earlier in this investigation, which put the
break-even at ~430 steps (batch 4) to ~1130 (batch 2) on the strength of that
+15% and called bucketing a short-run setting. **Measured break-even is ~14,700
steps** — 147 s saved against 0.010 s per step — so bucketing pays for itself in
any run that trains at all. The port's docstring still says "short-run
setting", and that is now wrong; it is left uncorrected only in the sense that
the docstring is a conservative statement of a feature whose default is off.

### A second benefit nobody was looking for

Bucketing groups by *bucketed* size, so samples that were landing in
incomplete `(prompt, size)` groups now share a bucket:

```
unbucketed: 19 of 273 samples sit in groups smaller than a batch and are
            NEVER trained on -- only 242 of 273 are used per epoch
bucketed:    0 of 273 samples sit in groups smaller than a batch
```

**All 273 samples now train, instead of 242.** That is a data-correctness
improvement rather than a speed one, and it is a stronger argument for the
feature than the throughput is.

### Why padding and not cropping

Measured on `non-square` (63 stored shapes, 273 samples):

| multiple | PAD | CROP |
|---|---|---|
| 32 | 3 shapes, +15% latent px | 4 shapes, **−29% pixels** |
| 64 | 3 shapes, +36% | 1 shape, **−52% pixels** |

Cropping reaches fewer shapes by discarding half of every image. Padding keeps
the data and pays compute, so it rounds **up** and the pad offset is drawn per
sample so borders do not become systematically real.

### What the mask has to get right

Two properties, both checked in `nodes/smoke_tests/smoke_test_shape_bucketing.py`
against analytically-known cases rather than recorded numbers:

- **Padding must not move the loss.** Corrupting the padded region with garbage
  leaves the loss at 1.000000. Without the mask this is the failure the earlier
  report warned about: training on elements that were never image content.
- **Masking must not rescale the loss.** Masking the squared error and taking
  a plain mean reports ~87% of the true loss for a padded batch, which trains
  ~13% too slowly and looks like a bad learning rate. Dividing by the valid
  element count instead gives **padded 1.000000 == unpadded 1.000000**.

One implementation trap worth recording: a bucket mixes samples that need
padding with samples already at a multiple of 32, so emitting a mask only for
padded samples would drop it for the **whole batch** — training on padding in
exactly the batches that have it. Every bucketed batch therefore carries a
mask, all-ones where no padding was needed, and an all-ones mask is provably a
no-op.

## The single-threaded stall: what it actually is

The remaining cost, and the one the user reported independently: during a new
shape, **one CPU thread runs at 100% while the GPU sits at 25-30%**.

Measured with kernel thread accounting (`/proc/<pid>/task/*/stat`, no profiler
needed), correlating each step window from `steps.jsonl` against per-sample CPU
deltas:

| | n | step | CPU/step | **cores busy** | `python` | `pt_autograd_0` |
|---|---|---|---|---|---|---|
| slow (>2 s) | 18 | 4.18 s | 4.17 s | **1.00** | 1.64 s | 2.53 s |
| fast (≤2 s) | 21 | 1.05 s | 1.06 s | **1.01** | 0.35 s | 0.70 s |

Three things follow, and the first two are not what the symptom suggests:

1. **The whole run is single-core, not just the stalls.** `cores busy` is 1.00
   in both columns. Nothing about this is specific to a new shape — the GPU is
   under-fed for the entire run, and a new shape just adds ~3.1 s more of the
   same serial CPU work. That reframes the problem: it is not "compiles are
   slow", it is "one thread cannot keep the device fed".

2. **The extra time is split across two threads, and most of it is in the
   autograd worker.** A slow step spends +1.29 s more on `python` and +1.83 s
   more on `pt_autograd_0` than a fast one. `pt_autograd_0` is the thread that
   drives backward kernel launches, so the work is **inline in the op-launch
   path during backward** — not a separate compile pool, and not something
   running on a runtime-owned worker thread. Only one other thread exists
   (`p:clctxworker0`, Level Zero) and it does no measurable CPU at all.

3. **So the cost is per-shape primitive creation, and it was already
   identified as such by the cache experiment.** Raising
   `ONEDNN_PRIMITIVE_CACHE_CAPACITY` removed the cost from revisits while
   leaving first sightings untouched (4.34x -> 0.99x, first 3.99 s -> 3.85 s).
   That cache stores oneDNN *primitives*, so a new shape's ~4 s is descriptor
   construction plus JIT, happening on the caller's thread. No profiler is
   needed to know this; the earlier experiment is the evidence.

### What could not be measured here, and why

- **`ONEDNN_VERBOSE=2` produces no oneDNN output on this torch build** (only
  the Rusticl/Mesa warning). So per-primitive `jit` timings are unavailable,
  and `analyze_onednn_verbose.py` in this folder cannot run against this build
  at all — it is written for a build that honours the variable.
- **Native stack traces are blocked.** `kernel.yama.ptrace_scope` is 1, so gdb
  cannot attach to the running trainer, and no py-spy is installed. Frames
  naming oneDNN's JIT internals would confirm point 3 rather than establish it.

Both gaps are tooling, not findings: the thread profile and the cache
experiment together already say what the work is and where it runs.

### A bug this analysis found in itself

The first version of the thread sampler ranked the "busiest" thread by
**cumulative** CPU. That makes the lifetime leader the busiest in *every*
sample, so it reported `python` at **99% of samples** — a confident, plausible,
wrong answer for a run using 1.00 cores across two threads. Fixed to rank by
per-sample delta, and `check_thread_ranking.py` exercises both rankings on
synthetic input so the difference is demonstrated rather than asserted:
cumulative says one thread 100%, per-sample correctly says 50/50.

### What is left to try, and its size

The only lever not yet used is **compiling shapes concurrently** — the report's
experiment D. The ceiling is memory: each concurrent step holds 1.6-3.3 GB of
workspace on top of a ~7.5 GB floor, so about **two** threads on this card.

But it is a much smaller prize than it was, and the reason is the two sections
above this one:

- **Bucketing already removed 41 of the 44 compiles** (159 s -> 12 s).
- **A persistent SYCL kernel cache does not touch it** (2%).

So after bucketing, three serial compiles cost ~12 s per run, and parallelising
them saves at most ~6 s. The lever matters for datasets where bucketing is off
— 44 shapes, 184 s of serial compile, ~92 s with two threads — and for the
first run on any new machine, which pays every shape's compile from nothing.
Not implemented: it needs a pre-warm phase inside the trainer, and the honest
version of that has to respect the memory ceiling rather than assume a thread
count.

## What to do next

1. **The single-core bottleneck is the real remaining limit**, and it is not
   shape-specific: the run uses 1.00 cores throughout with the GPU at 25-30%.
   Bucketing removes 41 of the 44 compiles, but the serial
   descriptor-plus-JIT work per shape remains, inline in the op-launch path.
2. **Concurrent shape compilation** is the untried lever (2 threads max on this
   card). Worth ~6 s per run after bucketing, ~92 s without it. Needs a
   pre-warm phase in the trainer that respects the memory ceiling.
3. **OneDNN JIT is single-threaded per primitive** and serial across primitives
   within a step, so there is no environment variable for it. A cross-process
   primitive cache would be the real fix and oneDNN has no supported one; that
   is a platform limit rather than a setting nobody found.
3. **Pre-warm** is *not* an alternative to bucketing: it pays the same ~166 s
   per run, just earlier and in one lump, and a graph run is a fresh process
   every time so there is nothing to amortize against. Earlier drafts of this
   file listed it as a fix; it only moves the bill.
4. Re-measure the per-shape ratio if the model changes — 35 is a property of
   this UNet on this backend, not a constant of oneDNN.

## Method notes

- One run at a time on this card. Two concurrent probe processes caused
  `DEVICE_LOST` (each holds ~10 GB of 12,216 MB). The probe is not the training
  operating point, so its VRAM ceiling says nothing about training.
- `num_alloc_retries` does not exist in this torch build, so the probe's
  "allocator retries" verdict branch is unreachable and was never exercised.
- Per-step memory is identical on fast and slow steps (8774 MB reserved both),
  so this is not memory pressure. Recorded to rule the hypothesis out, not to
  support the conclusion.
