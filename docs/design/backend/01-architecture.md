# 01 -- Backend architecture

Status: shipped through **M9** (the entry-point flip; `server/` is
archived). M1-M9 are all **done** -- the milestone log was removed in
the 2026-10-01 docs cleanup, since it recorded finished work; what
shipped is visible in the code and in `git log`.

This document is the *decisions and contracts* half: why the structure
is what it is, the layering rules a change has to keep, and the
behaviours a client or a neighbouring process depends on. File layout,
endpoint tables and error codes are **not** here -- open the code, or
`/openapi.json` for the API.

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

* **Folder `backend/`**, parallel to `server/`; the old server kept
  running untouched throughout development and was archived whole at
  M9 (now `archive/server/`, still launchable for reference).
* **Full clean break**: the new API is designed from scratch, not
  byte-compatible with the old one -- the frontend was adapted rather
  than preserved. Conflicts with non-server code (shared `paths.py`,
  `monitor_bus`) were accepted for now and resolved at migration time;
  the backend bridges to them through ports/adapters instead of being
  shaped by them. **A reader hitting those bridges should know the
  non-compatibility was a deliberate, approved trade, not an
  oversight** (section 9 repeats it as a non-goal).
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
4. **Entities own invariants.** Every field is private behind a
   read-only property -- `run.done_steps = -5` is not legal Python.
   Every transition goes through the state table and raises
   `InvalidTransitionError` otherwise, without partial mutation.
   Mutations buffer domain events; the writer drains them
   (`collect_events`) after persisting, and only the writer that won
   the status compare-and-swap may publish. A row loaded from the
   database goes through `restore()`, which checks the cross-field
   rules a single column cannot carry.
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
    `backend/tests/run_all.py` discovers them; `scripts/full_gate.sh`
    runs them plus a lint and a documentation-link check.

## 4. Bridges to the rest of this repository

Deliberate, adapter-owned, and never leaked past infrastructure. The
rule behind them: **reads stay torch-free, and a bridge exists only
where one implementation must exist**.

Concretely: `workspace.py`/`path_tiers.py` import `paths` so
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

## 5. API contract

Endpoint-by-endpoint detail is served as OpenAPI from the route
decorators themselves (`/openapi.json`), so it cannot drift; the error
table lives in `02-api-reference.md` and is parsed by
`backend/tests/test_error_contract.py`. What follows is the part the
schemas cannot express.

The nine page routes (`/`, `/monitor/{id}`, `/graph`, `/config`,
`/run/{id}`, `/datasets`, `/datasets/{name}`, `/help`, `/settings`,
plus the `/ui/*` asset mount) are registered with
`include_in_schema=False`, so they are absent from `/openapi.json` --
which is why they are listed here rather than looked up.

Two invariants: they are registered **after** the API and never under
`/api/`, so an unknown API path keeps the JSON error envelope instead
of falling through to the HTML shell; and pages and `/ui/*` assets
always send `Cache-Control: no-cache` (revalidate on every reload) --
ported from the legacy `server/main.py` after the visual smoke found
heuristic freshness serving stale JS. `/api/*` responses are untouched,
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
  composition time, before the first request can race it, and is what
  adopts a surviving trainer.
* **Trainer subprocess**: spawned with `start_new_session=True`
  (own process group, so the whole tree can be signalled); stop =
  SIGINT to the group with SIGKILL escalation after **15 s** (`force`
  skips ahead -- the grace was originally 3 s, which turned "saving a
  checkpoint" into "killed", docs 07 F-12). Orphan kill re-checks
  `/proc/<pid>/cmdline` for `core.cli` against PID reuse. Progress is
  read from the run's `log.progress.jsonl` by an offset-tailed reader
  (state per path; a truncated file resets its offset).
* **A trainer that outlives the server is adopted, not killed**
  (docs 07 F-11). Trainers start in their own session precisely so
  they survive a restart, and the legacy server killed those orphans
  anyway -- losing hours of work. Startup reconciliation re-attaches to
  a live, still-ours process instead. The honest consequence: an adopted
  process is not our child, so its exit code cannot be read, and the
  trainer's own last progress line decides (`finished` -> completed,
  `error` -> failed, **no line -> failed with that reason**, never a
  hopeful "completed").
* **Graph executions (M4)**: `GraphExecutionSupervisor` supervises one
  run at a time against the same CAS rules (`update_if_status` on
  `graph_executions`). The run itself executes in a **child process**
  behind `GraphTaskGateway`, reported through an append-only event file
  the supervisor tails -- see
  [`13-process-isolation.md`](../13-process-isolation.md) for the
  channel, the stop semantics, adoption across a restart, and the
  measured cost. `StartGraphExecution` holds a lock across validate +
  active-check + insert, and **one execution may be active at a time**
  (409 `graph_execution_active`) -- deliberate: one B580, and graph
  nodes can build real training loops. Cancellation is cooperative
  twice: the stop use case signals the child first (`SIGINT`, so its
  runtime notices between steps), then CASes `queued|running ->
  stopped` (3 refetch attempts for the claim race); a run that ignores
  the request is hard-killed after the grace period, and
  `ReconcileGraphExecutions` sweeps non-terminal rows at composition
  time -- adopting a still-running child rather than failing it, and
  failing the rest with the same "server stopped/restarted" reasons.

## 9. Non-goals

* No API/data compatibility with `server/` -- the clean break was
  deliberate, and `server/` is archived rather than migrated.
* No DI framework: manual wiring in the composition root *is* the
  pattern.
* No pytest: tests are plain scripts, consistent with the repo.
* No frontend test framework. The browser layer is covered by one
  Playwright script (`backend/tests/visual_smoke.py`) that needs a live
  server, which is why it is not part of `scripts/full_gate.sh`. That
  is a real gap, not a considered trade.
