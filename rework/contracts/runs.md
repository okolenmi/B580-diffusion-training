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
