# 08 -- Structure audit: index, and what was deliberately left undone

A strict-OOP pass over `backend/` itself, run after the 17 correctness
findings in doc 07 were closed. It ran as six commits (`3269a35` ..
`a159959`, 2026-10-01) and **finished nineteen of its twenty-six
findings**; seven are still open and listed below with their reasons.

Source comments cite these findings as `docs 08 S-NN`, so the ids are
kept stable even though the audit's own write-up is gone: **each fix now
lives in the docstring of the code it changed**, with the reasoning next
to the mechanism. The table is the index, not a description.

| ID | Finding, in one clause | The reasoning now lives in |
|---|---|---|
| S-01 | supervisors injected as concrete classes | `application/ports/run_watcher.py`, `execution_launcher.py` |
| S-02 | `ReconcileRuns` made adoption deps optional | `application/use_cases/reconcile_runs.py` |
| S-03 | listing one dataset's tasks rewrote rows of others | `application/dataset_task_sweeper.py` |
| S-04 | the monitor route reached past the use cases | `application/use_cases/subscribe_monitor.py` |
| S-05 | the supervisor owned someone else's VRAM cleanup | `infrastructure/graph/runtime.py` |
| S-06 | `_resolve()` duplicated in six use cases | `application/project_paths.py` |
| S-07 | "publish the entity's events" duplicated nine times | `application/event_publisher.py` |
| S-08 | bounds written twice, in two layers | `application/limits.py` |
| S-09 | validation boilerplate across nine use cases | `application/requests.py` |
| S-10 | error code -> status table kept in two layers | `application/errors.py`, `presentation/errors.py` |
| S-11 | both entities were fully mutable | `domain/entities/run.py`, `graph_execution.py` |
| S-12 | terminal states restated beside the table | `domain/value_objects.py` |
| S-13 | two copies of the lifecycle guard, and of the writer | `domain/lifecycle.py`, `application/lifecycle_writer.py` |
| S-14 | `ProgressSample` was reflected over by field name | `application/ports/progress_source.py` |
| S-15 | `total_steps` monotonicity lived in a thread | `domain/entities/run.py` |
| S-16 | rehydration had no sanctioned factory | `domain/entities/run.py` (`restore`) |
| S-24 | closed vocabularies were bare strings | `domain/value_objects.py`, `application/ports/dataset_tasks.py` |
| S-25, S-26 | dead code, unresolvable annotations | removed; `scripts/full_gate.sh` now lints for the first |

Two test files came out of it: `backend/tests/test_value_objects.py` (the
extracted rules) and `backend/tests/test_error_contract.py`, which
*parses* the error table out of doc 02 so code and documentation cannot
drift apart.

Three of the nineteen fixed findings were real defects rather than
only shape, which is the argument for having run the audit at all:

* `Run.record_progress` applied `done_steps` *before* validating
  `total_steps`, so a progress sample that was then rejected had still
  moved the run forward;
* `dataset_file_not_found` existed in code but was missing from the
  documented error table — found by the contract test that reads the
  doc;
* the task repository wrote its active-status SQL beside the tuple that
  named the same thing, and listing one dataset's tasks rewrote rows
  belonging to *other* datasets.

## S-17 -- `DatasetTask` has no entity

Its lifecycle lives in `application/ports/dataset_tasks.py` as status
enums plus raw SQL in the adapter, with zero domain events; the port is
the only place the vocabulary is declared. The right fix is an entity
mirroring `Run`, plus task events.

**Deferred** because it is a data-shape change, not a refactor: the
persisted vocabulary moves and every dataset-task test changes with it.
It deserves its own commit and a migration note, not a line in a
cleanup.

## S-18 -- `GraphDefinition`'s structural rules live in an adapter

`GraphDefinition` is `frozen=True`, but `params: dict` is mutable
through the "immutable" object, and duplicate node ids / dangling
edges / cycles are caught only by `GraphRuntime.validate()` in
`infrastructure/`. Pure graph algebra would need no registry at all.

**Deferred**: the runtime validator already answers 422 with a complete
issue list, and splitting the rules would split that one contract across
two layers with different issue shapes — a worse outcome than the
misplaced code, as long as nobody is fooled into thinking the dataclass
is immutable.

## S-19 -- the `MonitorBus` port leaks asyncio and the wire format

The port hands the application layer an `asyncio.Queue` of
pre-rendered `data: {...}` SSE frames, so the SSE format is decided by
the port and a second consumer would have to parse it.

**Deferred until that bridge is allowed to change**: the underlying
repo-root `monitor_bus.py` is shared with the retired `server/` and its
frame format is pinned by that project's own smoke test. The port-level
defect is documented here rather than papered over.

## S-20 -- `DatasetLibrary` is five jobs

`list`/`get`/`stats`/`root`/`create`/`delete`, the five curation
methods, sets, and `first_preview` (which duplicates
`DatasetPreviews.resolve`). `root() -> Path` also turns the port into a
filesystem escape hatch.

**Deferred**: splitting it is mechanical but touches every dataset use
case and both bridges. Better done together with S-17, so the port split
lands once instead of twice.

## S-21 -- `continue_ids_above` puts a filesystem concern in a repository port

A filesystem high-water mark (the highest existing `runs/run_*`
directory) reaches a SQL `sqlite_sequence` through a repository method.

**Deferred**: the method is the reason a colliding run id is
impossible — the first guard that makes F-04 unreachable rather than
merely unlikely. Moving the scan inside the adapter would make the
persistence layer scan directories, which is worse.

## S-22 -- four ports return bare `dict`

`ConfigOptions.schema()`, `AssetStore.inspect()`,
`NodeCatalog.diagnostics()`, `ConfigFiles.read()` — the shape is
described only by the pydantic schemas in presentation.

**Deferred**: mechanical, but four separate value objects whose fields
are UI-facing, and therefore change whenever the UI does. Worth doing
when there is a reason to want the type, not as tidying.

## S-23 -- three restatements of the same repository shape

`RunRepository` and `GraphExecutionRepository` are the same set of
methods (the second port's docstring says so), and `DatasetTasks`
renames half of them. A generic `LifecycleRepository[E, I, S]` protocol
would collapse them.

**Deferred deliberately**: the ports are small, typed and well
documented, and a generic would trade three clear signatures for one
parameterised one. The duplication that actually *caused* bugs — the
compare-and-swap-then-announce sequence — is already gone, replaced by
one `LifecycleWriter` per aggregate.

## Two decisions worth keeping visible

* **`StatusMachine` is held, not inherited.** A supervisor-like guard is
  a smaller thing than an aggregate, and inheriting would couple the run
  and graph-execution lifecycles for no gain.
* **Wire schemas keep plain `str` for closed vocabularies.** An unknown
  `kind` or `start_from` must arrive as a 422 that names the vocabulary,
  not as a `ValueError` from an enum constructor — so the domain uses
  `str`-valued enums and the request layer converts, deliberately.