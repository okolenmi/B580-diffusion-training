# Contract: runs

Implements plan 30 section 6.1, plan 10 section 6. Python: `next/graph/runtime.py`.

## Lifecycle

`validate` -> `run`. A run executes the data plane once in topological order (ties
broken by `uid`), retains reactive nodes, registers their event handlers, and ends
when every node has built. `run` never raises for graph or node problems; it returns a
`RunOutcome`.

## Node states

`pending` -> `running` -> `done` | `failed`; or `pending` -> `skipped` (after a stop, or
an upstream failure). Plus the editor-side `resolving` (see `resolve.md`) and `stale`
(editor: params changed since the last result).

Frame on stream `graph:run` for every transition:
`{"kind":"node_state","uid":"...","state":"running","t":...,"detail":{...}}`;
`detail` may carry `progress` (0..1), `message`, `duration_ms`, `error`.

Run frames: `run_started`, `run_finished`, `run_failed`, `run_stopped` as
`{"kind":"run","state":"...","t":...,"error":null}`.

## Step report

The trainer also reports one flat frame per optimizer step on its own stream `node:<uid>`: `{"step", "total", **metrics}` (the
keys of `StepFinished.metrics`: `loss`, `lr`, `step_ms`, memory and timing keys...). It is the shape the monitor boards read;
display modules (`MODULES` of a node class, in the catalog) name the keys they draw. The editor's Monitor panel
(`frontend/js/graph2/monitor.js`) shows the modules of every node of the open graph that has some.

## Result

```python
RunOutcome(results: tuple[NodeResult, ...], error: str | None, stopped: bool)
NodeResult(uid, ok, outputs: dict (JSON description), error: str | None,
           duration_ms: float, state: str)
```

Outputs are described, never serialised: plain JSON passes through; other objects
become `{"_type": name, "_repr": first 200 characters}`.

## Completion

`Completion` (complete | unfinished) travels beside values exactly as before
(doc 16): a node reports with `report_completion`, reads with `upstream_completion`,
and a node that sets `ACCEPTS_UNFINISHED` still runs after a stop when an input was
unfinished and every source ran.

## Observer

`RunObserver` (Protocol in `next/graph/runtime.py`) receives `node_state`, `node_done`,
`event` and `run` callbacks. The server turns them into stream frames and domain events.
A failing observer never aborts a run.

## Peak memory fingerprint

The server remembers how much memory a configuration peaked at and admits the next run of it against that number. It
must know the configuration before the child starts, so it reads **declarations, never values it would have to compute**:
a node class lists the params that change its memory peak in `memory_fields` (`BatcherNode`: `batch_size`;
`InjectAdapterNode`: `rank`, `kind`, `checkpointing`, ...; `LatentCacheNode`: the size policy). The key is the dataset
(the first `dataset` param in the graph) plus, per declaring node, its class and each declared value: the node's param,
else the port's default. A declared name must be a port of the class (`next/tests/check_node_surface.py`).

The fingerprint is unknown, never defaulted, when no node names a dataset, a declared port is fed by another node, a
required one is not set, or no node declares anything. An unknown fingerprint is not a zero peak: the run is admitted as
exploratory.

## Display streams

A node with `MODULES` reports its frames on `node:<uid>`; the monitor boards and the template editor read them there.
`graph:run` carries the run's own frames (node states, memory). The child writes every stream frame to the run's event
file and the server republishes it.
