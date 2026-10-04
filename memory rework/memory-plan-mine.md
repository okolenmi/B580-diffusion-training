# Memory handling rework: my plan (written BEFORE fetching upstream)

Written against local checkout `f4e5a91` ("settle the memory-admission TODO:
reservation at graph start, graph as the object"). I had read the design doc
(`docs/design/09-prioritized-backlog.md`), `nodes/memory/control_handle.py`,
`ExecutionContext`, `GraphDefinition`, the start use cases, and the measured
numbers in `docs/known-issues/resolved.md`. I had NOT looked at any
implementation pushed after that commit.

## 0. How I read the requirement
"The graph owns a specific amount of memory; nodes may ask to release some for
their needs; this sits one level above nodes, not as a node."

That is two different mechanisms with different owners, and I think the design
is cleaner if they are kept apart:

* **Level 1, admission (between processes).** Owned by the *server*. Decided
  once, at graph start. Settled in the doc: reservation, no preemption, a
  graph that does not fit does not start.
* **Level 2, arbitration (inside one graph).** Owned by the graph's *own
  process*. A node asks the graph's memory manager for room; the manager frees
  the graph's *own* residents to make it. This is the "ask to release" part. It
  never touches another graph. The doc's "no in-flight eviction" is, in my
  reading, about Level 1 only; I would confirm that reading with the author.

## 1. Facts from the code that shape the plan
1. Training runs as a **child process per graph** (design doc 13). The
   scarce resource is shared between processes, so the only party that can
   see all claims is the server.
2. `StartGraphExecution` checks only other graph executions;
   `StartDatasetTask` checks only tasks **of the same dataset**. Nothing keeps a
   graph and a dataset task (or two datasets' tasks) off the card together.
   Level 1 must cover every GPU-using child kind, not only graphs.
3. `DeviceContext` exposes this process's allocator stats and the total. It has
   no device-wide free memory. torch exposes `mem_get_info` and
   `set_per_process_memory_fraction` for xpu (checked on 2.14; must be
   re-checked on the 2.12 xpu build).
4. Measured: a bf16 SDXL LoRA, batch 2 at 1024 px peaks near 9.2 GB reserved;
   the device-level reading was 10,140 MB = 8,592 reserved + ~950 desktop +
   ~600 other. So about a gigabyte of the card belongs to the desktop, and
   there is per-process overhead the allocator does not report. A budget in
   "allocator MB" is not a budget in "device MB".
5. An earlier always-offload design was measured at ~4.7x slower with the
   offloading never once necessary. Offloading must be demand-driven.
6. `VRAMBudgetControllerNode` must be wired into one trainer; each trainer
   believes it is alone on the card.
7. Host RAM is a second pool with the same shape (warm cache now has a RAM
   budget; pinned memory is not really pinned on this build).

## 2. Design

### 2.1 Settings: a graph property, not a node
`GraphSettings` lives next to `GraphDefinition` (graph format 2; absent means
defaults, so old graphs load unchanged):
```
memory:
  vram:  { min_mb, max_mb | "auto", policy, strict }
  ram:   { max_mb | "auto" }            # same machinery, second pool
```
`min_mb` is the least the graph can run in (refuse to start below it);
`max_mb` is the ceiling it wants; `"auto"` means "everything that is free".
The *execution request* may override them, so one saved graph can run with
different budgets; the effective values are stored on the execution row.
A pure function `effective_settings(graph, request, history)` is the single
place defaults are computed. If the graph has no settings, `min_mb` defaults to
**1.05 x the peak of the last successful execution of the same graph** (stored
on the row), so the system learns its own numbers.

### 2.2 Level 1: the ledger (server, application layer)
* Port `MemoryLedger` with `reserve(owner, min_mb, max_mb) -> Grant | Refusal`,
  `release(owner)`, `snapshot()`.
* Owners are executions and dataset tasks (and the device probe as a small
  ephemeral owner). Capacity = `total - foreign_reserve_mb`
  (`foreign_reserve_mb` is the one genuinely global setting: a fact about the
  card and the desktop, default ~1,024).
* `Grant.mb` is **device-visible**: allocator budget + `process_overhead_mb`
  (measured constant, calibrated on the B580, stored as a setting).
* Called inside the existing start lock, before the row is spawned. Refusal is
  `409 memory_unavailable` with a *breakdown*: capacity, foreign reserve, each
  holder and its size, free, what the graph asked, and what would fit
  ("lower batch size", "stop task X"). Refusal is the complete answer.
* No separate persisted state: the ledger is **derived** from rows
  (`reserved_mb` on running executions/tasks). On startup it is rebuilt from
  adopted children; the orphan path releases a reservation exactly once.
* Single-active-graph stays as a *policy* (`max_concurrent_graphs = 1`), no
  longer the only protection.
* Physical double-check: the child, as its first act after the device context
  exists and before loading anything, compares `mem_get_info().free` with its
  grant and fails fast with a clear outcome error if foreign users (ComfyUI,
  the desktop) have taken the room. The ledger covers cooperating processes;
  this covers everyone else.

### 2.3 Level 2: `GraphMemory` (child process), on `ExecutionContext.memory`
Built by the child from the effective settings; enforces the grant:
`set_per_process_memory_fraction(budget / total)` as a backstop so a runaway
allocation fails inside this process, not the card.
```
memory.register(name, resident, pool="vram", pinned=False, priority=0)
with memory.lease(mb, why="vae decode") as lease: ...   # transient claim
memory.release(name) / memory.ensure_loaded(name)       # today's semantics
memory.stats() / memory.events()
```
* `lease` succeeds if `budget - in_use >= mb`; otherwise evicts the graph's own
  evictable residents by policy until it does; otherwise raises
  `MemoryRequestDenied` carrying needed, available and the evictable list. A
  node may catch it and fall back (tile, smaller batch) or let the graph fail
  with that message.
* Policy: never evict `pinned`; order by `priority`, then cheapest
  reload-per-MB-freed (measured on first eviction), then LRU. Every decision
  is logged as one line and counted (the cost of evictions goes to the
  monitor) because of fact 5.
* **Demand-driven only.** No eviction unless a lease or measured pressure
  requires it. Planned `release()` hints stay (they are today's deterministic
  mode) but are never inferred.
* Thread-safe (data-loader threads, monitor).
* Enforcement is `strict` by default under reservation: exceeding the budget
  is a bug in a node, and the message names the node that held the lease.

### 2.4 Compatibility and removal of the node
`ResourceControlHandle` becomes a thin adapter over `GraphMemory` so existing
trainers (`managed.py`, `supervised.py`, `CachingTextEncoder`,
`AdaptiveResidencyController`, `state_store.py`) keep working unchanged in the
first step. `VRAMBudgetControllerNode` becomes a deprecated shim: if present,
its parameters are *lifted into graph settings on load* and a warning is
shown; it is removed one release later. Migrating the big transient users
(VAE decode, text-encoder cache, optimizer state) to `lease` is a separate,
later, individually measured step.

### 2.5 Observability and UI
The child writes a `memory` record (reserved, allocated, peak, budget, active
leases, evictions with cost) into the event file at a fixed interval and at
every lease/eviction; the server stores `peak_reserved_mb` on the row at the
end. `GET /health` (or a `/memory` endpoint) shows the ledger. The editor gets
a **graph settings panel** (not a canvas node) with min/max/auto, the learned
last-peak, and a live "will it fit" indicator driven by the ledger snapshot.

## 3. Phases (each independently shippable and green)
0. ADR + this spec + a **hardware validation protocol** (below).
1. Domain: `GraphSettings`, graph format 2 + loader, effective-settings
   function, schema/API, round-trip tests with old graphs.
2. Ledger + admission for graphs **and dataset tasks**, refusal breakdown,
   startup rebuild, `reserved_mb` columns, health snapshot.
3. Child: `GraphMemory`, fraction backstop, physical check, telemetry,
   `ExecutionContext.memory`, adapter under `ResourceControlHandle`.
4. `lease` API; migrate text-encoder cache, then optimizer state, then VAE
   decode, each with a before/after measurement.
5. UI settings panel; lifting the old node's parameters; deprecation.
6. Learning from peak (`min_mb` default).
7. Real-hardware validation on the B580.

## 4. Tests I would require
* **Ledger invariants, property-based:** under random interleavings of
  reserve / release / child crash / restart, the sum of grants never exceeds
  capacity, no owner is released twice, and a restart reproduces the same sum
  from the rows. Concurrency test with N parallel `reserve` calls on a ledger
  that fits exactly K.
* **Fake device:** a deterministic device model (capacity, allocation,
  overhead, a foreign user that can appear) used by both levels, so no test
  needs a GPU.
* **GraphMemory:** a lease never pushes in-use above budget in the fake; every
  request is granted or denied with reasons; pinned residents are never
  evicted; eviction order follows policy; reload restores state; a denied
  lease leaves state unchanged; thread-safety under contention.
* **Fault injection:** raise at each collaborator call during
  reserve -> spawn -> finish; the reservation is released exactly once on every
  path (this is the class of bug the supervisor had three times).
* **Refusal message test:** the breakdown names every holder.
* **Back-compat:** old graph JSON loads with defaults; a saved graph with the
  budget node is lifted to settings and behaves the same.
* **Gate:** a test that no node module imports the ledger (nodes see only
  `ExecutionContext.memory`).

## 5. Hardware validation protocol (cannot be done in CI)
Measure on the B580 and write the numbers into the ADR: (a) does
`set_per_process_memory_fraction` exist on the 2.12 xpu build and what error
does an over-budget allocation raise; (b) per-process overhead (driver reading
minus allocator reserved) for an idle context and for a training step;
(c) `mem_get_info` accuracy against the driver reading with the desktop
running; (d) eviction cost per resident (text encoder, optimizer state) in ms
and MB; (e) a deliberate collision: start a dataset task and a graph that do
not fit together and confirm the second is refused with the breakdown.

## 6. Open questions I could not answer from the code
1. Is intra-graph eviction allowed under the settled "no in-flight eviction"?
   (I assume yes: it is the point of Level 2.)
2. Should a graph be allowed to *shrink* its own reservation mid-run to let a
   waiting task start? I say no (keeps the ledger monotonic within a run).
3. `foreign_reserve_mb`: static setting or measured at admission? I propose a
   static default plus the child-side physical check.
4. Are host-RAM budgets in scope now, or only designed for?
