# 01 -- Backend architecture

Status: **M1-M4 (API domains), M5 (frontend decision + migration
strategy docs), M6 (monitor frontend slice) and M7 (graph editor
slice) implemented and tested** (2026-10-01).
This doc is the blueprint `backend/` was built from and the contract
later milestones must keep.

## 1. Why this exists: the evaluation of `server/`

The legacy server is 4,542 lines of Python across 27 files (51 HTTP
endpoints, 6 smoke suites) plus 6,275 lines of frontend JS -- written
by many different AI passes (every cited legacy path now resolves
under `archive/server/` -- the layer was archived at M9). The
algorithms in it are mostly sound; the *ownership model* is the
disease. Ranked problems, with evidence:

1. **State lives in module-level singletons, not in the app.**
   Composition happens at *import* time: `settings = Settings()` is
   frozen at import (`config.py:157`), `get_training_service()`
   (`service.py:194`), `_registry = _ExecutionRegistry()`
   (`routes_nodegraph.py:149`), `_task_manager`/`_library`/`_runner`
   (`routes_datasets.py:61-66`), `_CACHE`
   (`nodegraph_registry.py:13`), a module-global `sse`. Consequence:
   tests poke globals, two instances are impossible, init order is
   implicit.
2. **Import-time side effects spread over four files**: XPU env vars
   (`server_cli.py:20`), `sys.path` mutation (`config.py:8`), `.env`
   loading (`paths.py:46`), then lifespan patches `paths` setters
   (`main.py:40-41`); settings resolution even reads the DB
   (`config.py:36`).
3. **Layering violations / god files**: `main.py` mixes wiring +
   logging policy + middleware + HTML pages; route files contain
   business logic and background threads (`routes_datasets.py` 359
   LOC, `routes_nodegraph.py` 355); SQL leaks outside `db.py`
   (`process_manager.py:210`).
4. **Inconsistent API contracts** across 51 endpoints: pydantic body
   models for some handlers, raw dicts for others, none for the rest;
   error handling split between `HTTPException`, bare returns, and a
   global handler; ad hoc response shapes.
5. **Training-orchestration ownership split three ways** (routes ->
   `TrainingService` singleton -> `process_manager` functions, with
   signal escalation in ad-hoc inner threads).
6. **Test coverage concentrated on the nodegraph layer**; config /
   history / settings / training / SSE endpoints are barely covered.
7. **Threading hazards**: blocking calls in async handlers, poll
   loops, per-call SQLite connections without WAL (side-by-side
   processes can collide with "database is locked").

**Verdict:** port the behavior, rebuild the structure. Do not port
the ownership model.

## 2. Decisions (user-approved, 2026-09-30)

* **Folder `backend/`**, parallel to `server/`; the old server keeps
  running untouched throughout development.
* **Full clean break**: the new API is designed from scratch, not
  byte-compatible with the old one. The current frontend will be
  adapted only after a migration strategy is ready (doc 02/03);
  conflicts with non-server code (shared `paths.py`, `monitor_bus`)
  are acceptable for now and resolved at migration time -- the
  backend bridges to them through ports/adapters instead of being
  shaped by them.
* **Full enterprise layering**: domain / application /
  infrastructure / presentation with ABC ports and a single
  composition root.

## 3. The layering and the dependency rule

```
                 ┌──────────────────────────┐
                 │       presentation       │  FastAPI, pydantic, SSE
                 └────────────┬─────────────┘
                              │ depends on
                 ┌────────────▼─────────────┐
                 │       application        │  use cases, DTOs,
                 │  ports (ABCs) live here  │  ApplicationServices
                 └────────────┬─────────────┘
                              │ depends on
                 ┌────────────▼─────────────┐
                 │         domain           │  Run entity, events,
                 │  imports NOTHING external│  value objects, rules
                 └──────────────────────────┘

infrastructure implements the application's ABCs ──▶ application
bootstrap (composition root) imports ALL layers and wires them
```

Rules (each is enforced by review and by the tests):

1. **One composition root.** `backend/bootstrap.py` is the only
   module allowed to import every layer. It constructs the database,
   repository, bus, clock, use cases, and returns a frozen
   `Container`. No `get_*()` accessors, no module-level mutable
   singletons anywhere else.
2. **Import purity.** Importing any backend module performs no I/O,
   sets no environment, opens no file. All env/arg handling is in
   `cli.main()`; `Settings.load()` is called explicitly.
3. **Ports are ABCs owned by the application** (`application/ports/`);
   only infrastructure implements them; tests substitute in-memory
   fakes. ABCs exist only for genuinely swappable capabilities
   (persistence, time, event bus, training gateway) -- no
   interface-spam for one-off code.
4. **Entities own invariants.** `Run.status` is read-only; every
   transition goes through the state table and raises
   `InvalidTransitionError` otherwise, without partial mutation.
   Mutations buffer domain events; the use case drains them
   (`collect_events`) after persisting.
5. **Use cases are one class per scenario** with
   `execute(dto) -> dto`. They validate input (single source of
   truth -- route handlers add no second validation layer), call
   ports, project entities to DTOs. Entities never leak to
   presentation.
6. **Presentation stays thin**: parse -> one use case -> shape. All
   failures leave as one envelope (section 5). No SQL, threads, file
   I/O, or env reads in handlers.
7. **All SQL lives in infrastructure**, in the repository modules,
   against a `SqliteDatabase` that owns WAL, busy_timeout, and
   versioned migrations (`infrastructure/persistence/migrations/`).
8. **Time is injected** through the `Clock` port; entities take
   `at=` parameters instead of reading clocks -- deterministic tests.
9. **Event semantics**: no history, no replay; a subscriber sees
   events published after it subscribes; handlers run on the
   publisher's thread and a failing handler is logged, not raised.
10. **Tests are standalone scripts** in the repo's `check()` style
    (no pytest dependency): in-memory fakes for units, a temp SQLite
    file for integration, raw-ASGI calls for end-to-end -- including
    the SSE stream, which must not deadlock through the loop hop.

## 4. File structure (as of M4)

```
backend/
├── __init__.py               # version stamp; package docstring
├── config.py                 # frozen Settings; explicit Settings.load(env)
├── json_safe.py              # non-finite float sanitizer for every JSON body
├── cli.py                    # entry: python -m backend.cli [--host --port --db]
├── bootstrap.py              # composition root -> Container (+ startup reconcile)
├── domain/                   # imports nothing
│   ├── value_objects.py      # RunId/RunStatus; ExecutionId/GraphStatus;
│   │                         # TrainingMode, StartFrom; the two transition tables
│   ├── lifecycle.py          # StatusMachine[S]: the guard both aggregates hold
│   ├── exceptions.py         # DomainError, InvalidTransitionError
│   ├── events.py             # DomainEvent + lifecycle + graph execution events
│   ├── graph.py              # GraphDefinition/GraphNodeSpec/GraphEdgeSpec/NodeResult
│   └── entities/             # run.py (Run state machine); graph_execution.py
│                             # (GraphExecution: queued->running->terminal + CAS)
├── application/
│   ├── errors.py             # every error declares its code + HTTP status
│   ├── dto.py                # Run/config/settings/asset/dataset/graph shapes
│   ├── project_paths.py      # ProjectPaths: "a path the client named"
│   ├── requests.py           # AssetRequest/ItemSelection/ItemChangesRequest
│   ├── limits.py             # the numbers the API and its clients agree on
│   ├── event_publisher.py    # drain an aggregate's buffer, or emit one event
│   ├── lifecycle_writer.py   # Run/ExecutionLifecycleWriter: CAS then announce
│   ├── dataset_task_sweeper.py  # one definition of "this task row is dead"
│   ├── services.py           # ApplicationServices + Config/Settings/Asset/
│   │                         # Dataset/Graph/Monitor groups
│   ├── supervisor.py         # RunSupervisor: one daemon thread per run
│   ├── graph_supervisor.py   # GraphExecutionSupervisor: one thread per graph run
│   ├── ports/                # ABCs: RunRepository, EventBus, Clock,
│   │                         # TrainingGateway, ConfigInspector, ConfigFiles,
│   │                         # ConfigOptions, SettingsStore, AssetStore,
│   │                         # DatasetLibrary, DatasetTasks, DatasetTaskGateway,
│   │                         # DatasetPreviews (M8f), RunArtifacts,
│   │                         # ProgressSource, GraphCatalog,
│   │                         # GraphRuntime, GraphExecutionRepository, GraphLibrary,
│   │                         # MonitorBus (M6), RunWatcher, ExecutionLauncher
│   │                         # (the supervisors behind interfaces)
│   └── use_cases/            # runs (ListRuns..ReconcileRuns); config (Get/Update/
│                             # raw x2/options/start-options); settings (Get/
│                             # Update); assets (List/Browse/MakeFolder/Upload/
│                             # Inspect); datasets (List/Get/Create/Delete, items
│                             # x4, sets/commit, tasks list/start/stop,
│                             # ReconcileDatasetTasks, SetDatasetPreview); graphs (catalog/diagnostics,
│                             # validate/start/list/get/stop/delete executions,
│                             # ReconcileGraphExecutions, library save/get/list/
│                             # delete)
├── infrastructure/
│   ├── clock.py              # SystemClock
│   ├── workspace.py          # WorkspaceLayout (settings_kv tier + paths bridge)
│   ├── path_tiers.py         # shared resolution policy (layout == settings view)
│   ├── subprocess_gateway.py # SubprocessTrainingGateway (spawn/signal/reap)
│   ├── dataset_library.py    # SqliteDatasetLibrary (own-SQL reads, manager bridges)
│   ├── dataset_previews.py   # SqliteDatasetPreviews (backend.db override row +
│   │                         # first-item fallback, M8f)
│   ├── dataset_tasks.py      # SqliteDatasetTasks (CAS row store for tasks)
│   ├── dataset_task_gateway.py   # SubprocessDatasetTaskGateway (fork gateway)
│   ├── dataset_task_worker.py    # child entry: reporter + DataTaskRunner
│   ├── core_config_inspector.py  # summarize/describe (core.config_io)
│   ├── core_config_files.py  # read/merge/replace (core.config_io/model)
│   ├── config_schema.py      # TrainingConfig introspection -> option metadata
│   ├── config_ui_data.py     # hand-authored labels/groups/visibility (data)
│   ├── config_options.py     # PydanticConfigOptions: schema + metadata merged
│   ├── settings_store.py     # SqliteSettingsStore (validate-then-write, atomic)
│   ├── file_asset_store.py   # FileSystemAssetStore (sandboxed paths; dataset
│   │                         # kind is catalog-only)
│   ├── directory_run_artifacts.py
│   ├── jsonl_progress_source.py  # offset-tailed progress reader
│   ├── graph/                # discovery.py (pkgutil walk -> NodeRegistry),
│   │                         # introspect.py (class -> NodeInfo), catalog.py
│   │                         # (GraphCatalog), runtime.py (GraphRuntime:
│   │                         # validate + execute over real Node classes)
│   ├── persistence/          # SqliteDatabase + SqliteRunRepository +
│   │                         # SqliteGraphExecutionRepository/SqliteGraphLibrary
│   │                         # + migrations/ (001..006_dataset_previews.sql)
│   ├── monitor_bus.py         # SharedMonitorBus: wraps repo-root monitor_bus.MonitorBus
│   └── events/               # CallbackEventBus (thread-safe)
├── presentation/
│   ├── app.py                # create_app(services, *, static_dir) factory
│   ├── deps.py               # get_services request dependency
│   ├── errors.py             # the one error envelope (4 handlers)
│   ├── security.py           # Host/Origin guard (DNS rebinding + cross-site writes)
│   ├── schemas.py            # pydantic response models + *_out mappers
│   ├── responses.py          # SanitizingJSONResponse (app-wide default)
│   ├── sse.py                # EventBus -> text/event-stream bridge
│   ├── frontend.py           # register_frontend: page routes + /ui mount
│   └── api/                  # health.py, runs.py, config.py, settings.py,
│                             # assets.py, datasets.py, graphs.py, events.py,
│                             # monitor.py (M6)
└── tests/                    # standalone check() scripts + run_all.py
```

Later milestones add: the frontend decision + migration strategy
(M5).

Deliberate bridges to the repo (adapter-owned, never leaked past
infrastructure): `workspace.py`/`path_tiers.py` import `paths` so
parent and *child* agree on `runs/run_<id>/log.progress.jsonl` exactly
(its documented .env fill-on-import equals what `run_server.sh` does
for the shell); the gateway/inspector/config-files adapters import
`core.config_io`/`core.config_model` for argv/config parsing; the
asset store imports `nodes.model.resource_inspection` lazily inside
`inspect()` (it pulls torch -- catalog/browse stay torch-free); the
dataset library bridges to `manager.db`/`manager.dataset` lazily inside
`create`/`commit` only (schema and membership semantics must stay
byte-identical to what the trainer writes -- torch loads on those two
calls, never on reads), and the dataset task worker imports
`manager.builder` in the child process only. The graph domain's one
lazy bridge is the memory releaser inside `ReflectedGraphRuntime`
(`core.comfy_setup.xpu_empty_cache`, pulled on the first *run*, never
at startup); discovery itself walks `nodes/` through `pkgutil` (the
modules are the project's own, stdlib-plus-torch as they come).
Import purity holds for every backend module itself.

## 5. API contract (v1, clean-break)

Endpoints as of M4:

| Method | Path | Use case |
|--------|------|----------|
| GET | `/api/v1/health` | liveness + version |
| GET | `/api/v1/runs?limit=&status=` | `ListRuns` |
| GET | `/api/v1/runs/active` | `GetActiveRun` (404 `no_active_run`) |
| GET | `/api/v1/runs/{id}` | `GetRun` |
| POST | `/api/v1/runs` | `StartTraining` (body: `config_path`, `start_from`, `reset_optimizer`) -> 201 |
| POST | `/api/v1/runs/{id}/stop` | `StopTraining` (body: `force`) |
| GET | `/api/v1/runs/{id}/log?lines=` | `GetRunLog` (1..500, tail text) |
| DELETE | `/api/v1/runs` | `DeleteRuns` |
| GET | `/api/v1/config?path=` | `GetConfig` -- validated config as nested JSON |
| PATCH | `/api/v1/config` | `UpdateConfig` (body: `path`, `overrides` deep-merge) |
| GET | `/api/v1/config/raw?path=` | `ReadConfigRaw` (`{content}`) |
| PUT | `/api/v1/config/raw` | `WriteConfigRaw` (body: `path`, `content`; create-or-replace) |
| GET | `/api/v1/config/options` | `GetConfigOptions` -- field schema, no config read |
| GET | `/api/v1/config/start-options?path=` | `GetStartOptions` |
| GET | `/api/v1/settings` | `GetSettings` (`{stored, resolved}`) |
| POST | `/api/v1/settings` | `UpdateSettings` (partial; `""` clears) |
| GET | `/api/v1/assets/{kind}` | `ListAssets` (kind: `checkpoint`, `lora`, `dataset` -- the last is catalog-only) |
| GET | `/api/v1/assets/{kind}/browse?path=` | `BrowseAssets` |
| GET | `/api/v1/assets/{kind}/inspect?path=` | `InspectAsset` (header-only safetensors) |
| PUT | `/api/v1/assets/{kind}/folders/{path}` | `MakeAssetFolder` -> 201 |
| PUT | `/api/v1/assets/{kind}/files/{path}` | `UploadAsset` (raw bytes body) -> 201 |
| GET | `/api/v1/datasets` | `ListDatasets` -- identity + stats (`stats: null` for legacy v1) |
| POST | `/api/v1/datasets` | `CreateDataset` (body: `name`, `description?`) -> 201 |
| GET | `/api/v1/datasets/{name}` | `GetDataset` (409 `dataset_not_migrated` on v1) |
| DELETE | `/api/v1/datasets/{name}` | `DeleteDataset` (any version; 409 while a task is active) |
| GET | `/api/v1/datasets/{name}/items?committed=` | `ListDatasetItems` |
| PATCH | `/api/v1/datasets/{name}/items` | `BulkUpdateDatasetItems` (body: `item_ids`, `prompt?`, `prompt_mode?`, `neg_prompt?`, `cfg?`) |
| PATCH | `/api/v1/datasets/{name}/items/{id}` | `UpdateDatasetItem` (single; explicit `type` replaces the legacy toggle) |
| POST | `/api/v1/datasets/{name}/items/discard` | `DiscardDatasetItems` (body: `item_ids`) |
| GET | `/api/v1/datasets/{name}/sets` | `ListDatasetSets` |
| POST | `/api/v1/datasets/{name}/sets` | `CommitDatasetItems` (body: `item_ids`, `name`) -> 201 |
| GET | `/api/v1/datasets/{name}/tasks?active_only=` | `ListDatasetTasks` (a pure query -- it writes nothing, not even rows of other datasets; sweeping belongs to startup and to the next start) |
| POST | `/api/v1/datasets/{name}/tasks` | `StartDatasetTask` (body: kind/image_dir/model/flags) -> 201; 409 `dataset_task_active` |
| POST | `/api/v1/datasets/{name}/tasks/{id}/stop` | `StopDatasetTask` (SIGKILL; 409 if terminal) |
| GET | `/api/v1/graphs/nodes?refresh=` | `ListNodeCatalog` -- palette (auto-discovered classes by domain; `refresh=true` re-walks `nodes/`) |
| POST | `/api/v1/graphs/nodes/{class}/diagnostics` | `NodeDiagnostics` (404 unknown class; 400 `node_diagnostics_failed`) |
| POST | `/api/v1/graphs/validate` | `ValidateGraph` -- full issue list, always 200 (`ok=false` = run would refuse) |
| POST | `/api/v1/graphs/run` | `StartGraphExecution` -> 201; 422 `graph_invalid` (all issues); 409 `graph_execution_active` (single-active) |
| GET | `/api/v1/graphs/executions?limit=` | `ListGraphExecutions` (1..500, newest first) |
| DELETE | `/api/v1/graphs/executions` | `DeleteGraphExecutions` (`{deleted}`) |
| GET | `/api/v1/graphs/executions/{id}` | `GetGraphExecution` (results + snapshot; 404) |
| POST | `/api/v1/graphs/executions/{id}/stop` | `StopGraphExecution` (409 `graph_execution_not_active` if terminal) |
| GET | `/api/v1/graphs/library` | `ListGraphs` (summaries, most recently updated first) |
| PUT | `/api/v1/graphs/library/{name}` | `SaveGraph` -> 201 first save / 200 replace (verbatim payload, no class validation; 422 `invalid_query` on bad name) |
| GET | `/api/v1/graphs/library/{name}` | `GetGraph` (404 `graph_not_found`) |
| DELETE | `/api/v1/graphs/library/{name}` | `DeleteGraph` (404 `graph_not_found`) |
| GET | `/api/v1/events` | SSE stream of domain events |
| GET | `/api/v1/monitor/{monitor_id}/stream` | SSE monitor telemetry -- legacy frame contract pinned in `03-migration-strategy.md` §4 (M6) |

Page routes (M6/M7, no schema): `GET /` serves `frontend/index.html`,
`GET /monitor/{monitor_id}` serves `frontend/monitor.html` (the id is
the page's own URL segment), `GET /graph` serves `frontend/graph.html`
(the editor), `/ui/*` mounts the frontend directory.
Registered after the API and never under `/api/`, so unknown API
routes keep the JSON error envelope. Pages and `/ui/*` assets always
send `Cache-Control: no-cache` (revalidate on every reload) -- ported
from legacy `server/main.py`'s middleware after the visual smoke found
heuristic freshness serving stale JS; `/api/*` responses are untouched
so the SSE streams keep their own semantics. Pinned by
`test_pages.py`.

`/runs/active` is registered before `/runs/{id}` so the path param
never swallows it. Request bodies are thin: validation that matters
(`start_from` values, `lines` range, config resolution, non-empty
path params) lives in the use cases, not in pydantic/`Query` -- one
source of truth.

**Config contract (M3a, clean break)**: `PATCH` merges a *nested*
partial object (dotted/flat keys and string coercion do not exist
here); the file must already exist (`PUT /raw` is the
create-or-replace path) and validation precedes every write, so a
rejected update never touches the file. Unknown override keys are
ignored (`TrainingConfig` is permissive by design). Saving never
mutates launch state and launching never mutates the config:
launch options (`start_from`, `reset_optimizer`) ride the
`POST /runs` body, not the config file. `GET .../options` is a pure
function of the config model + UI metadata (values come from
`GET .../`), and `start-options` reports availability as
configured-path + actual-existence, with `lora_checkpoint` absent
(not faked as unavailable) for non-LoRA configs; a broken config
propagates its error instead of degrading to an empty 200.

**Settings contract (M3a)**: `{stored, resolved}` -- raw KV values
vs. what the resolution policy currently points at (`null` only for
`comfy_dir`, which has no fallback). Updates are atomic: every
provided value validates first, then all persist in one transaction;
a rejection carries the full `{key: message}` map under `details`
and writes nothing. Tier order lives in exactly one place
(`path_tiers.py`), shared by the API view and `WorkspaceLayout`, and
a *configured* `venv_python` is used as-is -- a stale value fails
loudly at spawn rather than silently running another interpreter.

**Assets contract (M3a)**: client paths are untrusted -- sandboxed
against the kind's base dir (no absolute paths, no `..`, final
target inside the base); listings exclude `resume/` and dotfiles in
both catalog and browse; `inspect` returns the fixed per-kind shape
(`{kind, path, components}` for checkpoints,
`{kind, path, dtype, rank, key_count}` for LoRAs), never a raw
header dump. The `dataset` kind (M3b) is catalog-only: its options
are dataset names (same visibility rules as the datasets API) and
browse/upload/mkdir/inspect are refused with guidance.

**Datasets contract (M3b)**: storage is format v2
(`04-dataset-format.md`) and the API refuses anything else --
listings *show* legacy datasets with their real `format_version` and
`stats: null` (never fabricated counts), every other operation on one
returns 409 `dataset_not_migrated` with the migration command in
`details` (delete excepted: removing a legacy dataset must not require
migrating it). Reads are torch-free own-SQL over each dataset's
`metadata.db`; `create`/`commit` bridge to `manager` lazily (schema
and membership semantics have exactly one implementation, the
trainer's). Task state is *server* state: rows live in `backend.db`
(`dataset_tasks`, migration 004) and are written by three racing
actors -- start, the child's reporter, stop/reconcile -- through
compare-and-swap, so exactly one final outcome wins (`pending ->
running -> finished|failed|killed`, a late progress tick is a no-op,
never a resurrection). One active task per dataset (409
`dataset_task_active`); the fork gateway spawns
`venv_python -m backend.infrastructure.dataset_task_worker` with a
`/proc` cmdline marker guarding both kill (refuse strangers) and
liveness (a reused pid or a zombie reads as dead, a task that
outlives a server restart still reads as alive). Startup runs
`ReconcileDatasetTasks`, and `StartDatasetTask` sweeps before its
active-task check -- so a predecessor whose child died cannot answer
409 "a task is already active". Both call one
`DatasetTaskSweeper`, which fails rows whose child is gone (running +
dead immediately, pending with no pid: at startup always, mid-flight
only after 60 s). The list endpoint itself is a pure query: it used to
sweep, which meant reading dataset A rewrote rows of dataset B
(docs 08 S-03).

**Error envelope** -- every non-2xx response, no exceptions (unknown
routes, method-not-allowed, and framework validation included):

```json
{"error": {"code": "run_not_found", "message": "...", "details": [...]}}
```

Codes map to statuses centrally: `run_not_found` 404,
`invalid_query` 422, `validation_error` 422 (FastAPI/pydantic input),
`config_not_found` 404, `config_invalid` 422, `run_already_active`
409, `run_not_running` 409, `no_active_run` 404,
`training_launch_failed` 500, `settings_invalid` 400 (`details` is a
`{key: message}` map), graph codes `graph_invalid` 422 (its `details`
is the full issue list), `graph_execution_not_found` 404,
`graph_execution_active` 409, `graph_execution_not_active` 409,
`node_class_not_found` 404, `node_diagnostics_failed` 400,
`graph_not_found` 404, `http_{status}` for transport-level errors,
`internal_error` 500 (traceback logged, message generic).

**SSE**: `data: {json}` frames where json carries `type`
(`stream_opened`, `run_created`, `run_started`, `run_completed`,
`run_failed`, `run_cancelled`, `runs_deleted`, `run_progressed`,
plus the graph set `graph_execution_queued/started/progressed/
finished/failed/stopped`, `graph_executions_deleted`),
`occurred_at`, and the event's fields; `: ping` heartbeat every 15 s.
`run_progressed` is telemetry: published by the supervisor per
progress sample, never buffered by the entity (`Run.record_progress`
emits nothing -- that invariant is test-pinned); the graph twin
(`graph_execution_progressed`, one per completed node) is published by
the graph supervisor after the row CAS lands, not by the entity.

## 6. Concurrency model

* **SQLite**: one connection per thread (`threading.local`), WAL +
  `busy_timeout=5000` -- safe for a second process sharing the file,
  which the legacy server's design never allowed.
* **Handlers are sync `def`** for use cases (FastAPI runs them in its
  thread pool); async only where the transport is async (SSE).
* **Event flow**: use case / monitor thread publishes -> handlers run
  on that thread -> the SSE bridge hops to the event loop with
  `call_soon_threadsafe` into a bounded queue (256). Full queue drops
  the incoming event: SSE is a live tail, history belongs to the
  database. Subscription closes when the stream ends.
* **Single-writer status (M2)**: the supervisor, `StopTraining`, and
  `ReconcileRuns` all race to finalise a run, so every status write
  goes through `RunRepository.update_if_status(run, expected)` --
  `UPDATE ... WHERE status = expected`. Exactly one wins; losers
  discard their outcome and publish nothing. The supervisor is the
  sole writer of *final* outcome for a run it watches: it re-fetches
  the row every 0.5 s tick and exits without writing when the row is
  no longer `running` (someone stopped or swept it).
* **One run at a time**: `StartTraining` holds a lock across
  check-then-spawn (the process is single-process; the lock closes the
  TOCTOU a concurrent double-POST would open), and any row in
  `created`/`running` counts as active. `ReconcileRuns` runs at
  composition time, before the first request can race it.
* **Trainer subprocess**: spawned with `start_new_session=True`
  (own process group); stop = SIGINT to the group with SIGKILL
  escalation after 3 s (`force` skips ahead); orphan kill re-checks
  `/proc/<pid>/cmdline` for `core.cli` against PID reuse. Progress is
  read from the run's `log.progress.jsonl` by an offset-tailed reader
  (state per path; a truncated file resets its offset).
* **Graph executions (M4)**: `GraphExecutionSupervisor` runs one
  daemon thread per run against the same CAS rules
  (`update_if_status` on `graph_executions`); `StartGraphExecution`
  holds a lock across validate + active-check + insert, and **one
  execution may be active at a time** (409
  `graph_execution_active`) -- deliberate: one B580, and graph nodes
  can build in-process training loops. Cancellation is cooperative
  twice: the stop use case sets the thread's event first, then CASes
  `queued|running -> stopped` (3 refetch attempts for the claim
  race); the executor checks the event between nodes and passes it
  into every node's `ExecutionContext`. `ReconcileGraphExecutions`
  sweeps non-terminal rows at composition time with the same
  "server stopped/restarted" reasons.

## 7. Milestones

| # | Scope | Status |
|---|-------|--------|
| M1 | Skeleton: layering, settings, SQLite + migrations, runs read-side, delete + event, error envelope, SSE, tests, boot on own port | **done** |
| M2 | Training lifecycle: `TrainingGateway` port, command building, spawn/stop/kill, progress watching, `StartTraining`/`StopTraining` use cases, monitor telemetry on the bus | **done** |
| M3a | Config (read/PATCH/raw/options/start-options), settings store (atomic, tiered), assets (catalog/browse/mkdir/upload/inspect) | **done** |
| M3b | Datasets: `DatasetLibrary` + `DatasetTasks` ports, own-SQL reads, lazy `manager` bridges, fork task gateway, startup task reconcile (storage format already changed to v2 first -- see `04-dataset-format.md`) | **done** |
| M4 | Graph subsystem: auto-discovery (`pkgutil` -> `NodeRegistry`), reflection palette, authoritative `validate` (issue-code table), threaded executor + CAS history, single-active runs, server-side saved-graph library (`05-graph-runtime.md`) | **done** |
| M5 | Frontend decision + parity audit + migration strategy (doc 02/03), decommission plan for `server/` | **done** |
| M6 | Frontend slice 1: `frontend/` shell served by the backend, monitor-bus port + `GET /monitor/{id}/stream`, monitor dashboard + training controls | **done** |
| M7 | Frontend slice 2: graph editor against `/graphs` (palette, validate, run, executions, library + localStorage import) | **done** |
| M8 | Frontend slice 3: dataset manager + config editor + run history views | **done** |
| M8d | Shell redesign: icon rail on every page, floating persistent console, `/help` + `/settings` | **done** |
| M8e | Dataset add-data + edit modes: `generate_teacher` task kind (validated in `application/teacher_prompts.py`), bulk multi-edit (`neg_prompt_mode`, `type`, `prepend`/`append`), add-data dialog (generate/import), browse/edit item modes, advanced item editor | **done** |
| M8f | Dataset card previews + item context menu: resolved `preview_path` on list/detail (backend.db pointer, migration `006`; first non-bad item fallback; stale pointers degrade, never dead URLs), `PUT /datasets/{name}/preview` by item id, card thumb on `/datasets`, half-transparent `⋮` per item with the one-option "Set as dataset preview" menu | **done** |
| M9 | Flip: README/run entry point -> `backend`; decommission `server/` (per `03-migration-strategy.md` §6) | **done** |

## 8. Running it

```bash
# server (own port; old server keeps 8765)
python -m backend.cli --port 8766

# tests (21 files: domain, repositories, use cases, supervisors, event
# bus, training adapter, start/stop, end-to-end API+SSE, config,
# settings, assets, dataset library/tasks/API, graph discovery/catalog/
# runtime/execution/API, monitor stream, frontend pages (dashboard,
# monitor, graph editor + their assets))
python backend/tests/run_all.py
```

## 9. Non-goals (for now)

* No API/data compatibility with `server/` -- the clean break is
  deliberate; migration comes later.
* No frontend work in M1-M4; the current frontend keeps talking to
  the old server.
* No DI framework: manual wiring in the composition root *is* the
  pattern.
* No pytest: tests are plain scripts, consistent with the repo.
