# 05 -- Graph runtime (M4)

The nodegraph subsystem behind two application ports, replacing
`server/nodegraph_registry.py` + `nodegraph_introspect.py` +
`graph_executor.py` + `routes_nodegraph.py`. Legacy files stay untouched
(reference only); this is a clean break with deliberate divergences,
listed at the end.

The goal, per the milestone plan: *more adaptive for future changes*.
Every adaptivity mechanism below names the legacy pain point it removes.

## 1. What "adaptive" means here

| Legacy pain point | M4 mechanism |
|---|---|
| Hand-maintained import list of ~40 classes (`_load()`); drift bugs happened twice | **Auto-discovery**: `pkgutil` walk of `nodes.*`, every concrete `Node` subclass collected; per-module import failures surface as `load_errors` in the catalog response (loud, not invisible). Verified against the legacy list: exact same 36 classes, 0 errors, 1.3 s |
| Module-global `_CACHE` / `_registry` / `_ExecutionRegistry` | Instance-owned `NodeRegistry` + `GraphExecutionSupervisor`, built once in `bootstrap.py`; two app instances cannot share state; tests inject fakes |
| JS re-implements type knowledge (`typesCompatible` string matching) | Server is authoritative: `POST /graphs/validate` returns structured issues computed with real `issubclass`; the frontend's local check stays UX sugar only |
| `repr()`-string defaults round-tripped through the editor | Port metadata carries both `default` (JSON-native when representable) and `default_repr` (repr string); consumers pick |
| Validation scattered; unknown params explode mid-run as `TypeError` | One `GraphRuntime.validate()` producing a complete, structured issue list; `POST /graphs/run` rejects with 422 `graph_invalid` + `details.issues` *before* any thread starts |
| No params-aware shape resolution (documented gap, phase-3 doc) | `Node.resolve_inputs(params)` / `Node.resolve_outputs(params)` classmethod hooks (default: static `INPUTS`/`OUTPUTS`); validator and edge-compat checks call them, so a future dynamic-shape node needs zero backend changes |
| Memory-only execution state, unbounded, lost on restart | `graph_executions` table (migration 005): status machine with CAS, per-node results + timings, graph snapshot, startup reconcile |
| Poll-only progress | Lifecycle events on the existing `EventBus` (`/api/v1/events` SSE): started / progressed (per node) / finished / failed / stopped, alongside polling endpoints |
| Assets + monitor endpoints riding the nodegraph router | Assets already exist as `/api/v1/assets/*` (not duplicated). Monitor stream served by `GET /api/v1/monitor/{id}/stream` over the application `MonitorBus` port (M6, wrapping the repo-root bus) |

## 2. Ports (application-owned)

**`GraphCatalog`** -- read side, no execution:

```python
snapshot(refresh: bool = False) -> CatalogSnapshot   # nodes + domain + load_errors
diagnostics(class_name, params) -> dict[str, list[str]]  # raises NodeClassNotFoundError;
                                                          # node exceptions propagate to the use case
```

`PortInfo` / `PresetInfo` / `NodeInfo` / `CatalogLoadError` live in the
port module (they cross the boundary, like `DatasetTask` does).
`NodeInfo` includes `domain` (from module path, `nodes.optimizer.x` ->
`optimizer`), `node_kind`, `presets`, `has_diagnostics`, `bases`.

**`GraphRuntime`** -- the executor:

```python
validate(graph) -> tuple[GraphIssue, ...]            # complete report, never raises
execute(graph, *, cancel_event, on_node_done=None) -> GraphOutcome
release_memory() -> None                             # gc + xpu cache (lazy core bridge)
```

`GraphIssue(severity, code, message, node_id?, edge_index?, param?)` with
`severity in {"error", "warning"}`; errors block a run, warnings do not.
`NodeResult(node_id, ok, outputs, error, duration_ms)` lives in
`domain/graph.py` (the entity stores results; domain cannot import
application). `on_node_done` fires after every node so the supervisor can
persist partial results + publish progress.

Both ports are implemented by infrastructure (`infrastructure/graph/`)
over one shared, instance-owned `NodeRegistry`. The port carries *real*
`type` objects internally (validation needs `issubclass`), but nothing
JSON-facing ever leaves as a non-JSON value.

**`GraphExecutionRepository`** -- `RunRepository`'s shape: `add` (binds
id), `get`, `list(limit)` newest-first, `find_active`, `list_unfinished`,
`update`, `update_if_status` (the CAS), `delete_all`.

**`GraphLibrary`** -- named graph storage: `save` (upsert, returns
`(row, created)`), `get`, `list`, `delete`. Stores the submitted graph
JSON **verbatim** (plus a stamped `format: 1`): saving never validates
class names, so a graph saved while a node class is absent still loads
later -- validation happens at run time, not save time (forward
compatibility).

## 3. Domain

* `domain/graph.py`: `GraphNodeSpec`, `GraphEdgeSpec`, `GraphDefinition`
  (frozen; dumb data -- all checking goes through `validate()` so reports
  are complete rather than raise-on-first), `NodeResult`.
* `domain/entities/graph_execution.py`: `GraphExecution`, a state machine
  like `Run`:

  ```
  queued  -> running | stopped | error
  running -> finished | error | stopped
  terminal: finished, error, stopped (final)
  ```

  `record_result` only while running, emits no event (progress is
  published by the supervisor, mirroring `RunProgressed`); every
  lifecycle transition buffers its event for `collect_events()`.
* `GraphStatus` + `ExecutionId` in `value_objects.py`.
* Events (7, mirroring the run set): `GraphExecutionQueued`,
  `GraphExecutionStarted`, `GraphExecutionProgressed`,
  `GraphExecutionFinished`, `GraphExecutionFailed`,
  `GraphExecutionStopped`, `GraphExecutionsDeleted`.

## 4. Validation issue codes

Generated deterministically: nodes in submission order, then edges by
index; params in `INPUTS` order.

| severity | code | meaning |
|---|---|---|
| error | `invalid_node_id` | empty node id |
| error | `duplicate_node_id` | two nodes share an id |
| error | `unknown_class` | `class_name` not in the discovered registry |
| error | `edge_unknown_node` | edge endpoint references a missing node |
| error | `unknown_output_port` | edge source port absent on the class (uses `resolve_outputs`) |
| error | `unknown_input_port` | edge target port absent on the class (uses `resolve_inputs`) |
| error | `incompatible_types` | real `issubclass(out, in)` fails (`Any` passes) |
| error | `cycle` | topological order impossible (message lists remaining ids) |
| error | `shape_resolution_failed` | the node's `resolve_inputs/outputs` hook raised |
| error | `missing_required_input` | required port with no param and no edge feeding it |
| error | `unknown_param` | params key is not a port of this class |
| error | `invalid_choice` | value outside `Port.choices` (None allowed: "use default") |
| error | `type_mismatch` | wire-safe type check (below) rejected the literal value |
| (none) | `param_overridden` | *deliberately not emitted* -- edges overwrite params by design; a warning would fire on every wired widget that carries its editor default |

Wire-safe type check (only for `bool/int/float/str/list/dict/tuple/Path`
ports -- JSON can be checked meaningfully; handle/class/`Any`/generic
ports are skipped, presence is covered by `missing_required_input`):
`float` accepts `int` (not `bool`); `int`/`bool` reject `bool` mismatches
per declared type; `tuple` accepts list or tuple; `Path` accepts `str` or
`Path`; `None` on an optional port always passes.

## 5. Execution lifecycle

`StartGraphExecution` (under a lock):

1. `runtime.validate(graph)` -- any error severity -> 422
   `graph_invalid`, `details.issues` (issue dicts via
   `issue_to_dict`, one shared converter);
2. `find_active()` -- a live execution exists -> 409
   `graph_execution_active` (**deliberate divergence**: legacy allowed
   parallel runs; one B580 + in-process training nodes makes that an OOM
   waiting to happen, consistent with single-run/single-task rules);
3. insert `queued` row, publish `GraphExecutionQueued`;
4. `supervisor.launch(execution_id, graph)` -> daemon thread, returns
   201 `{execution_id, status: "queued"}`.

Supervisor thread: CAS `queued -> running` (loss = stop/reconcile won ->
exit silently), publish `Started`, then `runtime.execute` with a
per-node callback that CAS-persists partial results + publishes
`Progressed`. On completion: `error` if the outcome carries one, else
`stopped` if the cancel event is set, else `finished` -- CAS
`running -> final` (a lost CAS means the stop use case already wrote
`stopped`; its results stand, the thread's are discarded), publish the
buffered events (through the shared `ExecutionLifecycleWriter`, which
does the CAS-then-announce in one place). `GraphRuntime.execute` runs
`release_memory()` in its own `finally`, so the caller cannot forget it.
A crashed thread
best-effort fails the row (never leaves `running` stuck blocking the
next start).

`StopGraphExecution`: set the cancel event first (nodes poll
cooperatively), then CAS to `stopped` with the pre-read status as the
expected value (loop bounded to 3 refetches for the queued->running
race); terminal -> 409 `graph_execution_not_active` naming the winner.
Late progress/results from the thread then lose their CAS and are
dropped -- exactly one writer owns each transition.

**Where it runs, and what a restart costs.** Unlike `core.cli` training
(docs 03 §5), a node-graph run executes *inside the API server process*,
on a daemon thread. Two consequences are deliberate for now, and are
stated here rather than discovered later:

* **A server restart loses an in-flight graph run.** The thread dies
  with the process; startup reconcile fails the row ("died mid-flight",
  partial results kept) -- there is no process to re-attach to, because
  there is no separate process (docs 07 F-13). A `core.cli` trainer, by
  contrast, survives a restart and is re-adopted.
* **A device fault or OOM kill takes the server with the run.** The
  event loop also shares the machine with training threads.

Subprocess isolation for graph runs (the same "own session + re-adopt"
shape `core.cli` already uses) is recorded as follow-up work, not an
accident of the design.

Node-build failure now ends the run as `error` with the failed node's
message (**divergence**: legacy surfaced node failures as status
`finished` with an `ok:false` result -- a poller could not tell).

`ReconcileGraphExecutions` at startup: `queued` -> failed "server
stopped before the execution started"; `running` -> failed "server
restarted while the execution was in flight"; CAS-protected, publishes
`GraphExecutionFailed`.

History: unbounded table + `DELETE /graphs/executions` (mirrors runs --
no silent eviction).

## 6. API (`/api/v1/graphs`)

| Method | Path | Use case / behavior |
|---|---|---|
| GET | `/nodes?refresh=` | `ListNodeCatalog` -> `{count, domains: {domain: [node...]}, load_errors}` |
| POST | `/nodes/{class_name}/diagnostics` | `NodeDiagnostics` (body `{params}`) -> `{messages}`; 404 `node_class_not_found`, 400 `node_diagnostics_failed` |
| POST | `/validate` | `ValidateGraph` -> 200 always: `{ok, issues}` |
| POST | `/run` | `StartGraphExecution` -> 201; 422 `graph_invalid` + issues, 409 `graph_execution_active` |
| GET | `/executions?limit=` | `ListGraphExecutions` (1..500, newest first) -> summaries |
| GET | `/executions/{id}` | `GetGraphExecution` -> detail: status, results (+`duration_ms`), graph snapshot, timestamps; 404 `graph_execution_not_found` |
| POST | `/executions/{id}/stop` | `StopGraphExecution` -> detail; 409 `graph_execution_not_active` |
| DELETE | `/executions` | `DeleteGraphExecutions` -> `{deleted}` |
| GET | `/library` | `ListGraphs` -> summaries (+`node_count`) |
| PUT | `/library/{name}` | `SaveGraph` (upsert) -> 201 created / 200 replaced |
| GET | `/library/{name}` | `GetGraph` -> stored graph verbatim; 404 `graph_not_found` |
| DELETE | `/library/{name}` | `DeleteGraph` -> `{deleted}`; 404 |

Run payload keeps the legacy field names (`nodes[{id, class_name,
params}]`, `edges[{from_node, from_port, to_node, to_port}]`) to ease the
M5 frontend port. Saved-graph storage replaces the browser's
`localStorage` (`ng_graph_v1`) -- the client decides when to migrate.

All errors leave as the one envelope; codes added to
`presentation/errors.py`: `graph_invalid` (422),
`graph_execution_not_found` (404), `graph_execution_not_active` (409),
`graph_execution_active` (409), `node_class_not_found` (404),
`node_diagnostics_failed` (400), `graph_not_found` (404).

## 7. Introspection shape (per node)

```json
{"class_name": "...", "display_name": "...", "domain": "optimizer",
 "module": "...", "doc": "...", "bases": [...], "node_kind": "static",
 "has_diagnostics": true, "presets": null,
 "inputs":  [{"name", "type", "type_mro", "default", "default_repr",
              "required", "doc", "path_kind", "choices",
              "visible_when", "widget_only"}],
 "outputs": [the same keys with identity values -- default/default_repr
             null (outputs are wire-fed), widget hints null/false]}
```

`class_name` stays the stable identity (saved graphs resolve against
it); `display_name` stays presentation-only (auto-derived with the
curated token list, or `Node.DISPLAY_NAME`). Legacy-class guessing
(`introspect_legacy_class`) is dropped: everything discoverable *is* a
real `Node`.

## 8. Testability

* Discovery takes an injectable `scan` callable -- tests feed fixture
  node classes (defined in `tests/support.py`); production uses the
  `pkgutil` walk. One dedicated file (`test_graph_discovery.py`) runs the
  *real* scan: same 36 classes as the frozen legacy list, zero load
  errors, every `NodeInfo` JSON-serializable.
* `Node.resolve_*` hooks get a fixture class overriding them.
* The memory releaser is injectable (tests never touch `core/`/GPU).
* Fixture nodes come from `nodes.core` directly -- it is stdlib-only
  (no torch), so hermetic tests stay torch-free.

## 9. Deferrals

* **Monitor bus**: shipped in M6 -- application `MonitorBus` port,
  infrastructure adapter wrapping the repo-root bus, runtime passes it
  into `ExecutionContext`, stream at `GET /api/v1/monitor/{id}/stream`.
* **Assets**: already served by `/api/v1/assets/*`; not duplicated here.
* **Frontend**: monitor slice shipped in M6 (`frontend/`); the graph
  editor shipped in M7 (`/graph`): localStorage -> library import,
  local type check -> validate endpoint, lifecycle live via `/events`
  SSE with API polling as the fallback while executions are active.
  Dataset/config/run-history views are M8.
