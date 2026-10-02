# 0002 — Graph training runs in the API process

## Context

A saved graph runs arbitrary node code: LoRA merge, dataset ingest,
teacher generation. Some of that is heavy — torch imports, XPU allocation,
minutes of compute.

Training *runs* (the `/runs` family) already get a child process: a
subprocess, a pid, a supervisor, adoption across restarts. Graph execution
does not. It runs in a worker thread inside the server process.

The reason it was built that way is delivery order: the graph surface
needed to observe node execution live, and node events ride the same
in-process event bus as everything else. Moving it behind a process
boundary means node events have to cross it.

There is exactly one GPU, so a second concurrent graph execution is refused
regardless — this decision is not what enforces that.

## Decision

`backend/infrastructure/graph/runtime.py` executes node code in the API
process, driven by a supervisor thread.

Node events go through the ordinary domain event bus, so the editor sees
the same `/events` stream for graph execution that it sees for runs. There
is no serialisation step between a node finishing and the browser hearing
about it, and no question about what a child process's event queue is.

## Consequences

* **A device fault, an OOM kill, or a segfault in a node takes the server
  down with it.** A user sees the backend go away mid-run, not one run
  fail. This is the cost, and it is a real one — on an XPU device a
  recoverable fault inside a node has no way to be contained.
* **A node competes with the event loop for the GIL and the device.** Slow
  nodes make the UI feel slow, because they are in the same process.
* Training runs are unaffected: they are already a subprocess, and this
  asymmetry is worth knowing when reading a supervisor bug report.
* Porting this to a child process means re-deriving adoption, reconcile
  and the event contract for node events. That is tracked, not done.

## What pins it

* `backend/tests/test_graph_execution.py` — lifecycle, validation, the
  stop race, and that a second execution is refused while one is live.
* `backend/tests/test_api_graphs.py` — the HTTP surface over the same
  runtime.