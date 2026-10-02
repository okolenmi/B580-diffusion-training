# 05 -- Graph runtime (M4)

The nodegraph subsystem behind two application ports, replacing
`archive/server/nodegraph_registry.py` + `nodegraph_introspect.py` +
`graph_executor.py` + `routes_nodegraph.py`. Legacy files stay untouched
(reference only); this is a clean break with deliberate divergences.

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

`PortInfo` / `PresetInfo` / `NodeInfo` / `CatalogLoadError` live in the
port module (they cross the boundary, like `DatasetTask` does).
`NodeInfo` includes `domain` (from module path, `nodes.optimizer.x` ->
`optimizer`), `node_kind`, `presets`, `has_diagnostics`, `bases`.

**`GraphRuntime`** -- the executor:

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

**`GraphLibrary`** -- named graph storage: `save` (upsert, returns
`(row, created)`), `get`, `list`, `delete`. Stores the submitted graph
JSON **verbatim** (plus a stamped `format: 1`): saving never validates
class names, so a graph saved while a node class is absent still loads
later -- validation happens at run time, not save time (forward
compatibility).

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
4. `supervisor.launch(execution_id, graph)` -> spawn the run, return
   201 `{execution_id, status: "queued"}`.

The supervisor's watcher: CAS `queued -> running` (loss = stop/reconcile
won -> kill the run, exit silently), publish `Started`, then tail the
run's event file, CAS-persisting each node result and publishing
`Progressed`. On completion: `error` if the run's outcome carries one, no
outcome record at all if the process died, else `stopped` if a stop was
requested, else `finished` -- CAS `running -> final` (a lost CAS means the
stop use case already wrote `stopped`; its results stand, the watcher's
are discarded), publish the buffered events (through the shared
`ExecutionLifecycleWriter`, which does the CAS-then-announce in one place).
`GraphRuntime.execute` runs `release_memory()` in its own `finally`, in
whichever process ran the graph, so neither the supervisor nor the child
can forget it. A watcher that crashes best-effort fails the row (never
leaves `running` stuck blocking the next start).

**Where it runs, and what a restart costs** — see
[`13-process-isolation.md`](../13-process-isolation.md): the run lives in a
child process behind `GraphTaskGateway`, and a run that outlives the
server is re-adopted rather than failed.

**A scaling limit, measured.** Persisting a node result is a
compare-and-swap on the whole row, and `results` is one JSON column that
grows with the node count -- so the supervision cost is quadratic in node
count. Measured end to end, less the ~1.95 s of child startup:

| nodes | 100 | 400 | 1600 | 3200 |
|---|---|---|---|---|
| excess over startup | 0.1 s | 0.15 s | 11.9 s | 54.7 s |

Doubling the nodes quadruples the time. It is invisible at real graph
sizes -- a hand-built graph against a 35-class palette is tens of nodes,
where the whole run is the child's startup -- and it was only found by
deliberately feeding a 3200-node graph of trivial nodes, which is not
something the system is used for.

Not fixed deliberately. The cheap fix is to batch result writes, and that
trades away the property the round-2 review asked for: partial results
surviving a crash. The real fix is a different storage shape -- results as
rows rather than a JSON blob — which is a migration and would change the
entity's `results` tuple invariant. Both are larger than the problem they
solve, so the limit is recorded here instead.

`StopGraphExecution`: set the cancel event first (nodes poll
cooperatively), then CAS to `stopped` with the pre-read status as the
expected value (loop bounded to 3 refetches for the queued->running
race); terminal -> 409 `graph_execution_not_active` naming the winner.
Late progress/results from the thread then lose their CAS and are
dropped -- exactly one writer owns each transition.


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

The endpoint list is served by the server itself: `/openapi.json`.

Run payload keeps the legacy field names (`nodes[{id, class_name,
params}]`, `edges[{from_node, from_port, to_node, to_port}]`) to ease the
M5 frontend port. Saved-graph storage replaces the browser's
`localStorage` (`ng_graph_v1`) -- the client decides when to migrate.

## 7. Introspection shape (per node)

`class_name` stays the stable identity (saved graphs resolve against
it); `display_name` stays presentation-only (auto-derived with the
curated token list, or `Node.DISPLAY_NAME`). Legacy-class guessing
(`introspect_legacy_class`) is dropped: everything discoverable *is* a
real `Node`.

## 8. Testability

* `Node.resolve_*` hooks get a fixture class overriding them.
* The memory releaser is injectable (tests never touch `core/`/GPU).
* Fixture nodes come from `nodes.core` directly -- it is stdlib-only
  (no torch), so hermetic tests stay torch-free.
