# Audit: backend graph engine (read from code and design docs, origin/main a7cc71e)

Sources: `docs/design/backend/05-graph-runtime.md`, `09-event-contract.md`,
`docs/design/13-process-isolation.md` (headings), `15`, `16`,
`docs/decisions/0005-memory-admission.md`, `docs/design/graph-rework-inputs.md`,
`nodes/core.py`, `backend/infrastructure/graph/{catalog,runtime,discovery}.py`,
`backend/domain/modules.py`, `nodes/ui.py`.

## 1. Shape of the system

* `backend/` is layered (domain, application ports, infrastructure,
  presentation). `nodes/` is the user-facing node library and is imported only
  by `backend/infrastructure/graph/` (discovery, catalog, runtime).
* Discovery: `NodeRegistry.scan_nodes` walks `nodes.*`, collects every concrete
  `Node` subclass (38 classes), reports import failures as `load_errors`.
  `class_name` is the persistent identity of saved graphs.
* One `Node` base (`nodes/core.py`): `INPUTS`/`OUTPUTS` dicts of `Port`,
  `MODULES` (display modules), `build(**inputs)`, `diagnostics(params)`,
  `resolve_inputs(params)` / `resolve_outputs(params)` classmethods,
  `ExecutionContext` (cancel flag, bus, memory), `Completion` side channel.

## 2. Execution model

* **Forward-only DAG, whole graph per run.** `GraphRuntime.validate` (cycle,
  types, required inputs...), then `execute` in topological order. One run is
  active at a time (409 otherwise). The run lives in a **child process**
  (`GraphTaskGateway`); the server supervises, persists per-node results in
  `graph_executions`, and publishes lifecycle events (queued, started,
  progressed per node, finished, failed, stopped).
* A node's `build()` is the whole of its work. The trainer node's `build()` is
  the training loop. There is no "node is idle and wakes up later".
* **Stop**: cancel flag polled cooperatively; nodes with
  `ACCEPTS_UNFINISHED` still run afterwards if an input is unfinished
  (doc 16). This is a *backward-looking* signal in one narrow case (Stop), and
  the only one.
* **What breaks for backward events** (end of epoch re-triggering an earlier
  node): the topological executor calls each node once and moves on; a node
  that already returned cannot be re-entered; values on wires are plain
  objects handed forward. A cycle is a validation error. An event system must
  therefore be a *second plane* beside the data DAG, not a way to add edges.

## 3. Events that exist

| Mechanism | What | Who subscribes |
|---|---|---|
| Domain `EventBus` -> SSE `/api/v1/events` | numbered (`epoch:seq`), lifecycle ring buffer, replay or `resync_required`, delivery classes state/delta/lifecycle | frontend `lib/events.js` (editor, executions, other views) |
| `MonitorBus` -> SSE `/monitor/{stream}/stream` | per-stream report frames; streams `node:<id>` and `graph:run` | templates board, monitor windows |
| Child process event file | the run reports node results and peaks to the supervisor | `GraphExecutionSupervisor` |
| `Completion` | per-wire side channel about finished/unfinished | trainer and saver nodes |
| `ExecutionContext.cancel` | cooperative stop | all nodes |

All of these flow **from the run to observers**. None flows *between nodes*
during a run, and none lets a node subscribe to something.

## 4. Dynamic shape: what is already there

`resolve_inputs/resolve_outputs(params)` are classmethods that the validator
and the edge-compat check call with the node's params. Today:

* they receive **params only**: a loader that must look inside a file can do so
  (it is a classmethod; nothing forbids reading a header), but it is called on
  every validation and every templates resolve, so it must be cheap and cached;
* there is **no HTTP endpoint for resolved outputs** (`catalog.resolved_inputs`
  exists; outputs do not), and the editor does not call either;
* `catalog.py:48-49` swallows exceptions from a mid-edit params dict and falls
  back to the static shape, which is correct for typing in progress.

So the engine side of "Load Model spawns outputs dynamically" is half built:
the contract exists, the transport and the UI do not.

## 5. Graph settings and the memory budget

* Graph-scope fields (`backend/domain/modules.py`): `memory.vram_max_mb`
  (number or `auto`), `memory.vram_min_mb`, `memory.strict`,
  `memory.ram_max_mb`; graph-scope displays `run`, `memory`, `memory_now`.
* ADR 0005: memory admission, ledger, peaks written by the server.
  "`VRAMBudgetControllerNode` becomes a deprecated shim whose values are lifted
  into graph settings on load. **Not done**": the node's `vram_budget_mb` is an
  allocator ceiling minus a reserve; the graph's `vram_max_mb` is a device
  claim plus overhead. **Which wins when both exist is undecided.** The node is
  still in the palette (ports: `vram_budget_mb`, `vram_reserve_mb`, `device`,
  `strict`) and the managed trainer still has a `resource_control` port for it.
* Environment-like ports that live on nodes today (to move into graph
  settings, plan 20): `device` (VRAMBudgetController, ResourcesController,
  trainer), `unet_dtype` (ResourcesController), residency settings, text-encoder
  prewarm and "empty cache after", graph capture, `unet_weight_store`.

## 6. Handles

Port types are real Python classes; compatibility is `issubclass` on the
server (`type_mro` to the UI). Six handle types are consumed by ports but have
no producer node (component map). Handles are defined next to the nodes that
use them, not in one registry file.

## 7. Persistence

Graphs: library table, stored verbatim with `format: 1`. Templates: table with
graph snapshot inside. Executions: table with graph snapshot and per-node
results. No schema version per node class.

## 8. Engine gaps against the owner's wishes

| Wish | Gap |
|---|---|
| Events that go "backward" | no node-to-node channel; a second plane is needed (plan 10) |
| Dynamic outputs | resolve hooks exist; need an outputs endpoint, a cache, a pending state, and a cheap "inspect file" function in the model-family layer |
| Settings out of nodes | graph memory settings exist; the ADR's shim lift is undecided; device/dtype/residency have no graph-level home |
| Node field groups | `Port.group/advanced` exist in the catalog; only the UI lacks them |
| Live update | SSE and module streams exist; no per-node "state" event kind (pending, running, progress) |
| Single trainer | both trainers are independent `build()` loops |
