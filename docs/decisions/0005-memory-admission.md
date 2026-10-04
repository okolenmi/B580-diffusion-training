# ADR-0005: Memory admission and per-graph memory

**Status:** accepted
**Date:** 2026-10-04

## Context

The B580 has 12,216 MB. A rank-64 LoRA at 1024 px peaks at 7,666 MB
(batch 2) or 8,954 MB (batch 4). The card is shared: the desktop holds
~950 MB, and each process carries ~600 MB of driver overhead the allocator
does not report. Two graphs on one card each believe they are the only
tenant, each stay under their own ceiling, and together overrun it.

The existing `DeviceReservations` (nodes/memory/device_reservations.py) is
an in-process, `threading.Lock`-protected dict. It cannot coordinate
processes: two children each see 0 MB held by the other, so both are
admitted. The existing `PeakRecord` (nodes/memory/peak_record.py) writes
atomically (temp file + rename) but loses updates across processes: six
processes recording distinct peaks at the same instant ended with a stored
peak lower than the highest recorded in 69 of 150 trials.

## Decision

### Units

- **allocator MB** = what `torch.xpu.memory_reserved` reports.
- **device MB** = what the card sees = allocator budget + `process_overhead_mb`.
- A *grant* is in **device MB**. Capacity is `total_mb - foreign_reserve_mb`.

### Two levels, two owners

- **Level 1, admission (between processes): the server owns it.** A
  `MemoryLedger` in `backend/application` decides, once, at start, inside
  the existing start lock, for every GPU-using child: graph executions,
  dataset tasks, and the installer's device probe. No preemption, no mid-run
  negotiation.
- **Level 2, arbitration (inside one graph): the child owns it.** A
  `GraphMemory` object on `ExecutionContext.memory`, enforcing the grant and
  letting nodes ask for room, freeing only *this graph's* residents.

### Demand has three sources and one mechanism

- **stated** (a person typed it),
- **observed** (the persistent peak for this configuration's fingerprint,
  plus a pillow),
- **unknown** (nothing known).

Unknown is never a zero claim. Policy for unknown: admit **only if nothing
else holds the card**, as an *exploratory exclusive* run that claims **all
free capacity**, and record its peak; otherwise refuse with the breakdown.

### Peaks are written by one writer: the server

Children report peaks through the event file; the server applies them with
`MAX` semantics in SQLite. Never by several processes read-modify-writing
one JSON file.

### Not a node

`VRAMBudgetControllerNode` becomes a deprecated shim whose values are lifted
into graph settings on load.

## Consequences

- A graph that does not fit does not start. The refusal explains itself:
  capacity, foreign reserve, every holder and its size, what is free, what
  was asked, and what would fit.
- The ledger holds no state of its own beyond the lock; on startup it is
  rebuilt from rows of adopted/running children.
- Release exactly once, on every path (finish, fail, cancel, crash,
  reconcile, reaper).
- Offloading is demand-driven. No eviction without a request or measured
  pressure.
- Layering: `nodes/` never imports `backend/`. The ledger, peak store and
  settings live in `backend/`; the child receives numbers (spawn arguments,
  `ExecutionContext.memory`) and sends numbers (event-file records).

## What is deprecated

- `DeviceReservations` as a per-process dict: replaced by the server-side
  ledger for cross-process claims.
- `PeakRecord` as a JSON file: replaced by the SQLite peak store for
  cross-process persistence. The in-process implementation is kept only if
  made correct (fcntl lock around read-modify-write + fsync).
- `VRAMBudgetControllerNode`: lifted into graph settings on load, then
  removed.
