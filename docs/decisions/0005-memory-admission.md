# ADR-0005: Memory admission and per-graph memory

**Status:** accepted
**Date:** 2026-10-04

## Context

The B580 has 12,216 MB. A rank-64 LoRA at 1024 px peaks at 7,666 MB
(batch 2) or 8,954 MB (batch 4). The card is shared: the desktop holds
~950 MB, and each process carries ~600 MB of driver overhead the allocator
does not report. Two graphs on one card each believe they are the only
tenant, each stay under their own ceiling, and together overrun it.

**The ~600 MB overhead figure is confirmed, and it is specific to a loaded
process.** Measured on this card on 2026-10-05
([`docs/known-issues/pending-testing.md`](../known-issues/pending-testing.md),
MEM-08 (b)): steady-state overhead during a rank-64 LoRA run at 1024 px is
**~530 MB** (538.9 / 539.0 / 525.0 / 530.9 MB across the last four steps,
from timestamp-paired driver and allocator readings of one run). So
`DEFAULT_PROCESS_OVERHEAD_MB = 600` stands, ~13% conservative.

The distinction matters and is easy to get backwards: an *idle* context
overhead is only ~19 MB, and plain contiguous allocations add **0.0 MB** --
the allocator and driver move together exactly. The ~511 MB difference
appears only once real weights are resident (model, kernels, shader cache,
XMP state). A reading taken from an idle process therefore understates this
constant by more than an order of magnitude; it is the loaded case that
admission has to reserve for.

The existing `DeviceReservations` (nodes/memory/device_reservations.py) is
an in-process, `threading.Lock`-protected dict. It cannot coordinate
processes: two children each see 0 MB held by the other, so both are
admitted. The `PeakRecord` JSON file (nodes/memory/peak_record.py, since
removed) wrote atomically (temp file + rename) but lost updates across processes:
six processes recording distinct peaks at the same instant ended with a stored
peak lower than the highest recorded in 69 of 150 trials.

## Decision

### Units

- **allocator MB** = what `torch.xpu.memory_reserved` reports.
- **device MB** = what the card sees = allocator budget + `process_overhead_mb`.
- A *grant* is in **device MB**. Capacity is `total_mb - foreign_reserve_mb`.

Measured dataset-task demands on this card (allocator MB; the ledger adds
the 600 MB overhead to reach device MB) --
[`docs/known-issues/pending-testing.md`](../known-issues/pending-testing.md),
MEM-08 (f):

| kind | input | allocator MB | device MB |
|---|---|---|---|
| `ingest_lora` | 8 square images @ 1024px | 6,914.0 | 7,514 |
| `ingest_lora` | 8 non-square images @ 1024px | 11,736.0 | 12,336 |
| `generate_teacher` | 2 conditions @ 1024px | 8,824.0 | 9,424 |

`TASK_DEMAND_MB` is deliberately **still empty**, and these numbers are not
in it. Ingest's peak is dominated by the input's *aspect ratio* -- a 1.7x
spread on identical code and resolution -- and the non-square figure
(12,336 device MB) exceeds the 11,192 MB capacity outright, so no single
per-kind number is honest for both. An unknown demand already behaves
safely (exploratory exclusive: admitted only when the card is otherwise
free, refused with a breakdown otherwise), and that is the better default
than a figure that is wrong by 70% in one direction and unusable in the
other.

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

### An unknown device total names what the rows still claim (MEM-03H-03)

"Unknown total" has two causes with different stories: nothing has probed
yet (a fresh install, no torch), and *the probe cannot answer* -- a server
restarted while an adopted child holds the card, or properties unreadable
because the device is busy. The unfinished rows are the record of the
second case, so both surfaces say it instead of a bare unknown:

- the 409 `memory_unavailable` refusal (`details.reason =
  device_total_unknown`) carries `details.holders` recovered from the
  graph/task rows -- owner and size -- and names them in the message;
- `/health` answers `memory: {"total_mb": null, "holders": {...}}` while
  no ledger exists: still an explicit unknown (a null total, never a
  fabricated zero), now with the row-recovered holders beside it.

With no such rows both say `holders: {}`, which is the truth there, not
a default. A source with no rows wired to it (a bare test lambda) also
answers empty.

### What counts as GPU-using (decided MEM-03H-03; not implemented)

A node class is GPU-using when it sets `uses_accelerator = True`; the
default is `False`, which is right for pure-data nodes (scale, reshape,
metadata). A graph is GPU-using iff one of its node classes is.

**Decided:** a graph with no GPU-using node takes no device claim -- it
skips level-1 admission entirely, so an unknown total cannot block it.
**Not implemented:** no class carries the attribute yet, and every
graph -- pure-data included -- still passes through admission, so today
an unknown total refuses all starts (the safe side). The classification
has to arrive with discovery (ComfyUI's own node classes are not ours
to edit), and the admission skip lands with it. Until then this section
is the decided rule, not the behavior.

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
  cross-process persistence. Removed (MEM-04 #3) rather than kept and made
  correct with an fcntl lock: its 14 checks are ported to the SQLite store's
  tests, including corruption reads as unknown and re-records rewrite cleanly.
- `VRAMBudgetControllerNode`: lifted into graph settings on load, then
  removed.
