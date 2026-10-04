# TASK: memory admission and per-graph memory (merged design)

Repository `okolenmi/B580-diffusion-training`, start from `main` at `5f767e5` or
later. Read first, in this order: `docs/design/09-prioritized-backlog.md`
(the "Memory admission belongs to the graph" item and the three entries after
it), `nodes/memory/device_reservations.py`, `nodes/memory/peak_record.py`,
`nodes/memory/control_handle.py`, `docs/design/13-process-isolation.md`,
`docs/decisions/`. If they exist in the repo, also read `memory-plan-mine.md`
and `memory-comparison.md` (the reasoning behind this task).

You are merging two designs. **Keep** the measurement discipline and the
semantics already pushed (fingerprint, monotonic peak, stated never lowered,
refused run holds nothing). **Fix and extend** where the work stops at the
process boundary. Nothing here changes training numerics.

Working rules are the same as the earlier task files: one commit per work
package (`MEM-NN: <summary>`, finding and proving tests in the body), minimal
diffs, never weaken a test, run `python3 backend/tests/run_all.py`,
`python3 run_tests.py` and `python3 scripts/check_quality.py` before each
commit, every `except Exception` logs, comments that say "safe/never/atomic"
need a test that fails if untrue. Python 3.14+.

---------------------------------------------------------------------------

## 0. The design in one page

**Units.** `allocator MB` = what `torch.xpu.memory_reserved` reports.
`device MB` = what the card sees = allocator budget + `process_overhead_mb`.
A *grant* is in **device MB**. Capacity is
`total_mb - foreign_reserve_mb` (the desktop and other applications; the
measured gap was ~950 MB desktop + ~600 MB overhead on a 12,216 MB card).
Both constants are settings with the measured values as defaults, recorded in
an ADR.

**Two levels, two owners.**
* **Level 1, admission (between processes): the server owns it.** A
  `MemoryLedger` in `backend/application` decides, once, at start, inside the
  existing start lock, for **every GPU-using child**: graph executions,
  dataset tasks (note: today a graph and a dataset task, or two datasets'
  tasks, can share the card; the dataset check is per dataset), and the
  installer's device probe. No preemption, no mid-run negotiation.
* **Level 2, arbitration (inside one graph): the child owns it.** A
  `GraphMemory` object on `ExecutionContext.memory`, enforcing the grant and
  letting nodes ask for room, freeing only *this graph's* residents.

**Demand has three sources and one mechanism** (already established upstream):
*stated* (a person typed it), *observed* (the persistent peak for this
configuration's fingerprint, plus a pillow), *unknown* (nothing known).
Unknown is never a zero claim. Policy for unknown: admit **only if nothing else
holds the card**, as an *exploratory exclusive* run that claims **all free
capacity**, and record its peak; otherwise refuse with the breakdown.

**Peaks are written by one writer: the server.** Children report peaks through
the event file; the server applies them with `MAX` semantics in SQLite. Never
by several processes read-modify-writing one JSON file (see MEM-04).

**Not a node.** `VRAMBudgetControllerNode` becomes a deprecated shim whose
values are lifted into graph settings on load.

---------------------------------------------------------------------------

## 1. Rules specific to this task (each comes from a verified defect)

1. **Every property claimed across processes is tested with real processes**
   (`multiprocessing` / the real gateway), never only with threads or fakes.
   The pushed registry was "per device" and thread-safe and still admitted two
   graphs onto one card, because each child had its own copy.
2. **No `Optional` on a safety check.** A parameter, total or estimate that can
   be `None` must either be required, or make the check return an explicit
   `UNKNOWN`/refusal. `admit()` currently records a claim unchecked when
   `total_mb is None`; do not copy that shape.
3. **Read-modify-write needs one atomic step** (SQL `UPSERT ... MAX`, or a lock
   that every writer takes). "Atomic replace of the file" prevents a torn file,
   not a lost update. Measured: 69 of 150 trials with six concurrent writers
   ended with a stored peak lower than the highest recorded.
4. **Account in device MB**, not allocator MB. A ledger that sums allocator
   peaks against the raw card total over-admits by about a gigabyte plus one
   overhead per process.
5. **Release exactly once, on every path** (finish, fail, cancel, crash,
   reconcile, reaper). Add fault-injection tests for reserve -> spawn -> finish.
6. **A refusal explains itself**: capacity, foreign reserve, every holder and
   its size, what is free, what was asked, and what would fit.
7. **Offloading is demand-driven.** No eviction without a request or measured
   pressure (an always-offload design measured 4.7x slower). The negative
   result stands: evicting the model inside the trainer rescues no budget;
   do not build eviction inside the trainer.
8. **Measure before migrating** any node to the new API; record before/after in
   the commit body. Nothing is wired into a trainer without a measurement.
9. **Layering.** `nodes/` never imports `backend/`. The ledger, peak store and
   settings live in `backend/`; the child receives numbers (spawn arguments,
   `ExecutionContext.memory`) and sends numbers (event-file records).
10. If something cannot be verified on the B580, say so in
    `docs/known-issues/pending-testing.md` and implement a clear fallback with a
    logged warning; do not pretend it was enforced.

---------------------------------------------------------------------------

## 2. Work packages (in order; each leaves the repo green)

### MEM-00 Decisions and ADR
Write `docs/decisions/0005-memory-admission.md`: units; `foreign_reserve_mb`
(default 1024) and `process_overhead_mb` (default 600) with the measurements
quoted; the unknown policy above; single writer for peaks; the two-level split
and what "no in-flight eviction" means (cross-graph only); what is deprecated.
Update the backlog item to point at it. No code.

### MEM-01 The fingerprint, computed without a GPU
Their steps 1-3 say estimators first; do the cheaper thing first. The server
needs a fingerprint **before spawning**, and must not import torch.
1. Find how the node catalog obtains class metadata without importing torch in
   the server (`grep -rn "introspect\|NodeCatalog" backend`). Add a class-level,
   declarative `memory_fields: tuple[str, ...]` (names of the params that change
   the peak) to the nodes that matter: the trainer(s), the UNet holder, the
   LoRA config node, the optimizer nodes, the dataset source node. Default
   empty. This is data, not code, so the server can read it.
2. A pure function `graph_fingerprint(graph, dataset_stats) -> Fingerprint` in
   `backend/application`: the existing `Fingerprint` fields (model,
   batch_size, latent_h, latent_w, rank, checkpointing, optimizer) taken from
   the declared fields; latent h/w from the **largest bucket** of the dataset
   (the peak is set by the largest shape; check `DatasetStats` for bucket
   sizes). If any required field cannot be found the fingerprint is **unknown**,
   not defaulted.
3. A graph with no node that declares fields has an unknown fingerprint.
**Tests.** Same graph, different dataset shuffle/captions/paths -> same
fingerprint. Batch 2 vs 4, checkpointing on vs off -> different. A graph with
a missing field -> unknown. Torch is not imported (assert
`"torch" not in sys.modules` after computing a fingerprint in a clean process).

### MEM-02 Graph settings in the graph format
1. `MemorySettings` value object: `vram_min_mb`, `vram_max_mb | "auto"`,
   `strict` (default true), `policy` (reserved for MEM-06), and
   `ram_max_mb | "auto"` (carried and validated now, enforced later).
   `GraphSettings` holds it. Bump the graph format to 2; format 1 loads with
   defaults. Storage, API schema and the saved-graph library round-trip it.
2. An execution request may carry overrides; the effective values are stored
   on the execution row (`memory_json`).
3. A pure `effective_memory(graph, request, peak_record, capacity)`; the one
   place defaults are computed: stated beats observed (held = max of the two,
   as in `Reservation.held_mb`); observed = recorded peak + pillow.
**Tests.** Old graph JSON loads and re-saves unchanged except for the format
number; overrides win; unknown settings are rejected with a clear message.

### MEM-03 The ledger and admission (the core)
1. Port `MemoryLedger` (`reserve(owner, demand) -> Grant | Refusal`,
   `release(owner)`, `snapshot()`), adapter derived from DB rows: add
   `reserved_mb` to graph executions and dataset tasks; capacity from settings
   and the device total reported by the existing cached device probe. **The
   ledger holds no state of its own** beyond the lock; on startup it is
   rebuilt from rows of adopted/running children.
2. Call it inside `StartGraphExecution` and `StartDatasetTask` (and the probe),
   before the row is spawned, under the existing lock. Single-active-graph
   stays as a policy (`max_concurrent_graphs`, default 1); the ledger is now
   what keeps a graph and a dataset task apart. A dataset task's demand:
   stated per task type with a measured default (add to the hardware protocol).
3. Refusal: `409 memory_unavailable` carrying the breakdown (rule 6).
4. Unknown fingerprint: exclusive exploratory run per section 0, otherwise
   refuse.
5. Release exactly once from `_finish`, reconcile, the sweeper and the reaper
   paths; `/health` shows the snapshot (holders, free).
**Tests (real concurrency, not just fakes).**
```python
# N parallel starts that fit exactly K -> exactly K admitted, N-K refused with a breakdown
# Seed for the process-level claim (this is the case the pushed registry fails):
def child(name, q):                       # two OS processes ask the SERVER-SIDE ledger via the real start use case
    ...
# expected: the second start is refused; sum of grants <= capacity at all times
```
plus: crash between reserve and spawn releases the claim; a refused run holds
nothing; a restart reproduces the same held total from the rows; release is
idempotent; a dataset task blocks a graph that does not fit with it and vice
versa; the breakdown names every holder (assert on text).

### MEM-04 One writer for peaks (SQLite), replacing the lost-update file
1. Table `memory_peaks(fingerprint TEXT PRIMARY KEY, peak_mb REAL, samples
   INTEGER, updated_at TEXT)`; write only through
   `INSERT ... ON CONFLICT(fingerprint) DO UPDATE SET peak_mb = MAX(peak_mb,
   excluded.peak_mb)`. Reads return `None` for unknown.
2. The child already streams monitor records; add a `memory` record kind
   (reserved, allocated, peak, budget) to the event file, the reader, the event
   schema and the contract test. The server's watcher applies each record's
   peak to the table (cheap: one statement, only when the peak rose).
3. Keep `nodes/memory/peak_record.py` as the in-process/offline implementation
   **only if** it is made correct: take an `fcntl` lock around the whole
   read-modify-write and fsync before replace. Otherwise remove it and port its
   14 checks to the SQLite adapter. Both implementations must pass one shared
   contract test (monotonic, never lowered, corrupt/missing reads as unknown,
   re-record keeps the max).
**Test (seed; this is the experiment that failed upstream: 69/150):**
```python
def worker(db_path, value, barrier):
    store = SqlitePeakStore(db_path); barrier.wait(); store.record(FP, value)
# 6 processes x 150 trials, distinct values 1000..1500 at the same instant:
# stored == max(values) in EVERY trial.
```

### MEM-05 The child: `GraphMemory`
1. Spawn arguments `--memory-budget-mb` (allocator MB) and
   `--memory-grant-mb`; the worker builds `GraphMemory` first, before loading
   anything, and exposes it as `ExecutionContext.memory`.
2. **Physical check:** after the device context exists and before any load,
   compare `mem_get_info().free` with the grant; if the card cannot give it,
   write an `outcome` with the numbers and exit cleanly (foreign users such as
   ComfyUI or the desktop are not in the ledger).
3. **Backstop:** `set_per_process_memory_fraction(budget / total)` when the
   attribute exists on the installed xpu build; otherwise log one warning
   ("budget not enforced by the allocator") and continue. Record which case in
   the telemetry so the UI can say it.
4. Telemetry: the `memory` record of MEM-04 at a fixed interval and on every
   lease/eviction (eviction decisions with MB and measured cost in ms).
5. `ResourceControlHandle` becomes a thin adapter over `GraphMemory`: keep the
   three registration states exactly (`never`, `offloadable`, `sacrificable`)
   and their semantics; existing trainers and tests must pass **unchanged**.
   `VRAMBudgetControllerNode` becomes a shim: if a graph has no memory settings
   its values are lifted into them on load, with a warning shown in the editor.
**Tests.** On a deterministic `FakeDevice` (capacity, allocation, overhead, a
foreign user that can appear): physical check refuses when the foreign user
holds the room; budgets are enforced in the fake; the adapter reproduces the
current `ResourceControlHandle` behaviour (run its whole existing test file
against the adapter); a node module cannot import `backend` (add to the layering
test).

### MEM-06 Inter-node requests (build only with a real consumer)
The user's requirement: *nodes may ask the graph to release some memory for
their needs*. Find a concrete consumer first (`grep` for a node that runs after
the trainer and loads something large: sampling/preview, VAE decode). If there
is none today, implement the API and its tests on the fake device and stop;
do not migrate anything speculatively.
```
with memory.request(mb, why="vae decode") as lease: ...
```
Granted if `budget - in_use >= mb`; otherwise evict **this graph's** evictable
residents until it is, in this order: never `pinned`; then `priority`; then
lowest measured reload cost per MB freed; then least recently used. If it still
cannot be satisfied raise `MemoryRequestDenied` carrying needed, available and
the evictable list, leaving state unchanged. Thread-safe. Every decision is one
log line and one telemetry record.
**Tests.** A request never pushes in-use above budget in the fake; pinned never
evicted; policy order; a denied request changes nothing; concurrent requests
serialise; reload restores state; a lease releases on exception.

### MEM-07 Editor UI
A **graph settings panel** (not a canvas node): min/max/auto for VRAM, the
learned peak for this fingerprint ("last time: 7,666 MB at batch 2"), a "will
it fit" indicator from `/health`'s ledger snapshot, and the full refusal
breakdown when a start is refused. Pure logic goes into `frontend/js/lib/` with
`node:test` tests; the panel must handle unknown fingerprints and non-finite
values. No HTML-writing APIs (`textContent` only).

### MEM-08 Hardware validation protocol (write it, run it on the B580)
`docs/known-issues/pending-testing.md`: (a) is
`set_per_process_memory_fraction` present on the 2.12 xpu build and what
exception does an over-budget allocation raise; (b) per-process overhead
(driver reading minus allocator reserved) for an idle context and a training
step; (c) `mem_get_info` against the driver reading with the desktop running;
(d) eviction cost per resident in ms and MB; (e) the deliberate collision: a
dataset task and a graph that do not fit together, the second is refused with
the breakdown; (f) default demand for each dataset task type. Put the measured
numbers into the ADR.

---------------------------------------------------------------------------

## 3. Do not do
* Do not add an estimator per node first; the graph-level observed peak keyed by
  the fingerprint comes first, estimators are a later refinement (and must fail
  loudly outside their fitted range).
* Do not make the first run of every configuration refuse; use the exploratory
  exclusive rule.
* Do not keep the registry per process; do not write peaks from children.
* Do not touch the trainers except through the adapter.
* Do not store the budget in a global setting; it is a graph property.

## 4. Final report
One line per work package (`done | partial | skipped`), commit, proving test,
doubts. List every behaviour that only the B580 can confirm. Do not mark a
package done unless its cross-process tests use real processes.
