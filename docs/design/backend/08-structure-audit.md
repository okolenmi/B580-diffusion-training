# 08 -- Structure audit: strict-OOP worklist (2026-10-01)

The external review (doc 07) asked "is this *correct*". This document
asks "is this *well shaped*": the OOP and layering quality of
`backend/` itself, after 17 correctness findings had been closed.

Two audits were run over the tree (application layer, domain + ports)
and their findings are merged and renumbered here as **S-01..S-26** so
that progress is trackable across commits. Status values:

* **Fixed** -- landed, with the commit or batch named.
* **Deferred** -- deliberately not done, with the reason.
* **Open** -- not started.

Findings are ordered by leverage, not by severity alone: an item that
deletes twenty copies of a rule outranks an item that tidies one class.

## Summary

| ID | Area | Issue | Severity | Status |
|---|---|---|---|---|
| [S-01](#s-01) | wiring | Supervisors injected as concrete classes; `start_training.py` annotates a name it never imports | High | **Fixed** (S-batch 1) |
| [S-02](#s-02) | wiring | `ReconcileRuns` takes `supervisor`/`artifacts` as optional, silently reverting to "kill live trainers" | High | **Fixed** (S-batch 1) |
| [S-03](#s-03) | layering | `ListDatasetTasks` (a query) CAS-writes rows of *every* dataset while listing one | High | **Fixed** (S-batch 1) |
| [S-04](#s-04) | layering | The monitor stream endpoint reaches past the use cases to `services.monitor_bus` | Med | **Fixed** (S-batch 1) |
| [S-05](#s-05) | layering | `GraphExecutionSupervisor` owns device-memory reclamation for the runtime it holds | Med | **Fixed** (S-batch 1) |
| [S-06](#s-06) | duplication | `_resolve()` byte-identical in 6 use cases; "path is required" in 5 | High | **Fixed** (S-batch 2) |
| [S-07](#s-07) | duplication | "publish the entity's buffered events" copy-pasted 9 times | Med | **Fixed** (S-batch 2) |
| [S-08](#s-08) | duplication | Page-size / log-line / description bounds duplicated across use cases and routes | Med | **Fixed** (S-batch 2) |
| [S-09](#s-09) | duplication | Asset-kind and item-id validation boilerplate across 9 use cases | Low | **Fixed** (S-batch 2) |
| [S-10](#s-10) | layering | Error-code -> HTTP status table duplicated in two layers (26 strings) | High | **Fixed** (S-batch 3) |
| [S-11](#s-11) | domain | Both entities are anemic: 17 and 7 public mutable attributes bypass every invariant | High | **Fixed** (S-batch 4) |
| [S-12](#s-12) | domain | The terminal-state set is hand-maintained beside the transition table, twice per aggregate | Med | **Fixed** (S-batch 4) |
| [S-13](#s-13) | domain | Two lifecycle machines, two supervisors, two reconcilers, no shared abstraction | Med | **Fixed** (S-batch 4, S-batch 5) |
| [S-14](#s-14) | domain | `ProgressSample` is interrogated by field-name reflection in the supervisor | Med | **Fixed** (S-batch 5) |
| [S-15](#s-15) | domain | Two `Run` invariants enforced in the supervisor instead of the entity | Med | **Fixed** (S-batch 5) |
| [S-16](#s-16) | domain | Rehydration has no sanctioned factory, so a loaded row can violate cross-field rules | Med | **Fixed** (S-batch 5) |
| [S-17](#s-17) | domain | `DatasetTask` has no entity: its state machine lives in a port plus SQL literals | High | Deferred |
| [S-18](#s-18) | domain | `GraphDefinition` is shallow-frozen and its structural rules are enforced in an adapter | High | Deferred |
| [S-19](#s-19) | ports | `MonitorBus` hands the application layer an `asyncio.Queue` of pre-rendered SSE frames | High | Deferred |
| [S-20](#s-20) | ports | `DatasetLibrary` is 14 methods / five jobs and returns a `Path` | Med | Deferred |
| [S-21](#s-21) | ports | `continue_ids_above` puts a filesystem concern in a repository port | Med | Deferred |
| [S-22](#s-22) | ports | Four ports return bare `dict`; the contract lives in the pydantic layer | Med | Deferred |
| [S-23](#s-23) | ports | `RunRepository`/`GraphExecutionRepository`/`DatasetTasks` restate the same 8 methods | Med | Deferred |
| [S-24](#s-24) | domain | Closed vocabularies are `str` (`mode`, task status, severity, `start_from`) | Med | **Fixed** (S-batch 6) |
| [S-25](#s-25) | dead code | Six unused imports/constants/parameters, one unreachable branch | Low | **Fixed** (S-batch 1) |
| [S-26](#s-26) | housekeeping | Unresolvable forward references in `dto.py`; untyped collection annotations | Low | **Fixed** (S-batch 1) |

---

## Fixed

### S-01
Supervisors injected as concrete classes; `start_training.py` annotates
a name it never imports

```python
# application/use_cases/start_training.py:44 (before)
supervisor: "RunSupervisor",  # noqa: F821 -- application sibling
```

`RunSupervisor` was never imported in that module -- the `noqa`
suppressed the undefined name, so the annotation was a lie to any type
checker, and `reconcile_runs.py` importing the same class properly made
the two disagree. `GraphExecutionSupervisor` was injected concretely in
two more use cases.

Now two narrow ports carry exactly what the callers use
(`application/ports/run_watcher.py`, `application/ports/execution_launcher.py`);
the supervisors implement them, and `bootstrap.py` is the only module
that names the concrete classes.

### S-02
`ReconcileRuns` takes `supervisor`/`artifacts` as optional

```python
# application/use_cases/reconcile_runs.py:49 (before)
supervisor: RunSupervisor | None = None,
```

Both real call sites pass both, so the `None` branch was dead code that
advertised "kill every live trainer on startup" as a supported
configuration -- the data loss doc 07 F-11 exists to prevent. Both are
required now; `ReconcileRuns` takes the `RunWatcher` port (S-01).

### S-03
`ListDatasetTasks` (a query) wrote rows

`_sweep_dead()` iterated `tasks.list_unfinished()` -- *every* unfinished
task in *every* dataset -- and CASed them to `failed` while answering a
read for one dataset. It also duplicated `ReconcileDatasetTasks`.

The sweep moved into `ReconcileDatasetTasks.execute` (startup) and
`StartDatasetTask.execute` (the two points that own liveness), and
`ListDatasetTasks` is now a query again.

### S-04
The monitor endpoint reached past the use cases

`presentation/api/monitor.py` called `services.monitor_bus.subscribe()`
directly, making `ApplicationServices` a partially-open service locator
for exactly the two members it claimed to encapsulate. A
`SubscribeMonitor` use case now sits in front of it, so every
presentation entry point goes through a use case.

### S-05
The graph supervisor owned memory reclamation

`GraphExecutionSupervisor._supervise` called `runtime.release_memory()`
in its own `finally` -- a supervisor reaching into its executor for GPU
bookkeeping. `GraphRuntime.execute` now releases in its own `finally`,
where the allocation happened.

### S-06
`_resolve()` in six copies

```python
def _resolve(self, raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else self._root / candidate
```

Five config-path use cases also repeated `if not path: raise
InvalidQueryError("config path is required")`. `ProjectPaths`
(`application/project_paths.py`) owns both: `.require(raw)` raises with
the field name, `.resolve(raw)` resolves under the project root. The
five constructors lost their bare `project_root: Path`.

### S-07
The publish loop, nine times

```python
def _publish(self, run: Run) -> None:
    for event in run.collect_events():
        self._events.publish(event)
```

`EventPublisher` (`application/event_publisher.py`) takes anything with
`collect_events()`, so supervisors, stop use cases and reconcilers call
one collaborator instead of re-deriving the loop.

### S-08
Bounds duplicated across layers

`MAX_ITEM_PAGE = 500` existed in both a use case and a route; the runs
default of 50, the executions default of 50 and the log default of 100
were each written twice. `application/limits.py` is the single source;
routes import the use-case constants they document.

### S-09
Validation boilerplate across nine use cases

`if not kind: raise InvalidQueryError("asset kind is required")` (five
classes), `if not item_ids: ...` (three), `if all(v is None ...)` (two).
`AssetRequest`, `ItemSelection` and `ItemChanges.from_changes()` raise
once; the use cases delegate.

### S-10
Error-code -> status, twice

`ApplicationError.code` is a string that `presentation/errors.py`
re-maps through a 26-entry table; a typo in either place silently
produces a 400. Each error now carries `status_code` and the handler
reads it off the class -- adding an error is a one-file change.

### S-11
Anemic entities

`Run` advertised "the entity owns its invariants" and 17 of its
attributes were public and writable, so `run.done_steps = -5` or
`run.finished_at = None` on a completed run was legal Python that no
check could stop. Both entities now keep state private behind read-only
properties; the mutator methods remain the only writers, and the
persistence mappers read through the properties unchanged.

### S-12
Terminal states maintained by hand

`_TERMINAL` in `value_objects.py` restated, for each aggregate, the set
of `_ALLOWED` keys whose value is empty -- four coordinated edits per
new terminal state. Terminal-ness is now derived from the machine
(`LifecycleMachine.terminal`), so the table is the only source.

### S-13
Two of everything

`Run` and `GraphExecution` each carried their own `_ALLOWED`,
`_transition`, `_require_status`, `_require_id`, `_emit` and event
buffer; each had a supervisor with its own `_fail_leftover` and its own
reconciler. `domain/lifecycle.py` now owns the machine (transition
table, terminal derivation, event buffer, identifier requirement) as a
`StatusMachine[S]` the two aggregates hold; the application-side
"finalise through compare-and-swap, then publish" sequence is one
`RunLifecycleWriter`/`ExecutionLifecycleWriter` pair instead of six
hand-rolled copies.

### S-14
Reflection over `ProgressSample`

`supervisor.py` asked "is this sample only a terminal marker?" by
`getattr`-ing eight field names, so renaming a field would turn the
check into a silent always-`True`. `ProgressSample.is_terminal_only`
answers it on the type that knows.

### S-15
Invariants enforced in a background thread

`total_steps` monotonicity ("the plan never shrinks") lived in
`RunSupervisor._apply_sample`; a second writer calling
`record_progress` would have erased it. It is now an invariant of
`record_progress`, pinned in `test_domain_run.py` rather than through
the supervisor.

### S-16
Rehydration without a factory

The public constructor was the only way to load a row, and it checked
single fields only, so a loaded aggregate could be `running` with no
`started_at`, or carry `done_steps > total_steps`. `Run.restore()` and
`GraphExecution.restore()` are the sanctioned rehydration paths and
validate the cross-field rules once.

### S-24
Closed vocabularies as bare strings

`Run.mode` accepted any truthy string; dataset-task status, graph issue
severity, `start_from`, prompt/model/resize/text modes and the item
verdict were all `str`. `TrainingMode`, `TaskStatus`, `IssueSeverity`
and friends are `str`-valued enums in the domain now, so `mode="typo"`
is a type error and the active/terminal task sets are derived from the
enum rather than string tuples.

### S-25 / S-26
Dead code and unresolvable annotations

Six unused imports/constants/parameters and one unreachable branch
removed (including a `_fail_leftover(..., event)` parameter that was
never read). `dto.py`'s quoted annotations that named classes it never
imported now import them; every collection annotation carries its
element type.

---

## Deferred

These are real, and each is deferred for a stated reason rather than
by omission.

### S-17 -- `DatasetTask` has no entity
Its lifecycle lives in `ports/dataset_tasks.py` as status tuples plus
raw SQL in the adapter (`UPDATE ... SET status = 'running'`), with the
vocabulary imported *from the port* by a use case, an `assert` as the
only type check, and zero domain events. The right fix is an entity
mirroring `Run` plus task events, which changes the persisted
vocabulary and every dataset-task test. Deferred to a dedicated batch:
it is a data-shape change, not a refactor, and deserves its own commit
and migration note.

### S-18 -- `GraphDefinition` structural rules in an adapter
`GraphDefinition` is `frozen=True`, but `params: dict` is mutable
through the "immutable" object, and duplicate node ids / dangling edges
/ cycles are only caught by `GraphRuntime.validate()` in
`infrastructure/`. Pure graph algebra needs no registry. Deferred: the
runtime validator already answers 422 with a complete issue list, and
splitting it would split that contract across two layers with different
shapes.

### S-19 -- `MonitorBus` leaks asyncio and the wire format
The port hands out an `asyncio.Queue` of pre-rendered `data: {...}`
frames, so the SSE format is decided by the port and a second consumer
would have to parse it. The underlying repo-root `monitor_bus.py` is
shared with the legacy server and its frame format is pinned by its
smoke test. Deferred until that bridge is allowed to change; the
port-level defect is documented rather than papered over.

### S-20 -- `DatasetLibrary` is five jobs
`list/get/stats/root/create/delete`, the five curation methods, sets,
and `first_preview` (which duplicates `DatasetPreviews.resolve`).
`root() -> Path` also turns the port into a filesystem escape hatch.
Deferred: splitting it is mechanical but touches every dataset use case
and both bridges; better done with the dataset-task entity (S-17) so the
port split lands once.

### S-21 -- `continue_ids_above` in the repository port
A filesystem high-water mark (scanned in `bootstrap`) reaches a
SQL `sqlite_sequence` through a repository method. Deferred: the
method is the reason a colliding run id is impossible, and moving the
scan inside the adapter would make the persistence layer scan
directories.

### S-22 -- Four ports return bare `dict`
`ConfigOptions.schema()`, `AssetStore.inspect()`,
`NodeCatalog.diagnostics()`, `ConfigFiles.read()` -- the shape is only
described by the pydantic schemas in presentation. Each wants a frozen
value object. Deferred: mechanical, but four separate value objects
whose fields are UI-facing (and therefore change when the UI does).

### S-23 -- Three restatements of the same repository shape
`RunRepository` and `GraphExecutionRepository` are the same nine
methods (the second port's docstring says so), and `DatasetTasks`
renames half of them. A generic `LifecycleRepository[E, I, S]` protocol
would collapse them. Deferred: the ports are small, typed and
well-documented; a generic would trade three clear signatures for one
parameterised one, and the shared terminal-writer service (S-13)
already removed the duplication that actually caused bugs.

---

## How the batches map to commits

Each S-batch is one commit, gated by `scripts/full_gate.sh` (legacy +
22 backend files + `node --check`); the browser layer is unaffected by
every item here, so `visual_smoke.py` is not re-run except when a
frontend file changes.