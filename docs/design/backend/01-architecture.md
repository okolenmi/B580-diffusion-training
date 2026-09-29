# 01 -- Backend architecture

Status: **M1 implemented and tested** (2026-09-30). This doc is the
blueprint `backend/` was built from and the contract later milestones
must keep.

## 1. Why this exists: the evaluation of `server/`

The legacy server is 4,542 lines of Python across 27 files (51 HTTP
endpoints, 6 smoke suites) plus 6,275 lines of frontend JS -- written
by many different AI passes. The algorithms in it are mostly sound;
the *ownership model* is the disease. Ranked problems, with evidence:

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

## 4. File structure (as of M1)

```
backend/
├── __init__.py               # version stamp; package docstring
├── config.py                 # frozen Settings; explicit Settings.load(env)
├── cli.py                    # entry: python -m backend.cli [--host --port --db]
├── bootstrap.py              # composition root -> Container
├── domain/                   # imports nothing
│   ├── value_objects.py      # RunId, RunStatus (state enum)
│   ├── exceptions.py         # DomainError, InvalidTransitionError
│   ├── events.py             # DomainEvent + run lifecycle events
│   └── entities/run.py       # Run: state machine + event buffer
├── application/
│   ├── errors.py             # RunNotFoundError, InvalidQueryError (code -> HTTP)
│   ├── dto.py                # RunDTO + query/result shapes + mapping
│   ├── services.py           # ApplicationServices (frozen aggregate)
│   ├── ports/                # ABCs: RunRepository, EventBus, Clock
│   └── use_cases/            # ListRuns, GetRun, DeleteRuns
├── infrastructure/
│   ├── clock.py              # SystemClock
│   ├── persistence/          # SqliteDatabase + SqliteRunRepository + migrations/
│   └── events/               # CallbackEventBus (thread-safe)
├── presentation/
│   ├── app.py                # create_app(services) factory
│   ├── deps.py               # get_services request dependency
│   ├── errors.py             # the one error envelope (4 handlers)
│   ├── schemas.py            # pydantic response models + *_out mappers
│   ├── sse.py                # EventBus -> text/event-stream bridge
│   └── api/                  # health.py, runs.py, events.py
└── tests/                    # standalone check() scripts + run_all.py
```

Later milestones add: `application/ports/training_gateway.py` +
`use_cases/training/*` + `infrastructure/training/*` (M2),
config/settings/datasets domains (M3), the nodegraph subsystem
behind a `GraphRuntime` port (M4).

## 5. API contract (v1, clean-break)

Endpoints as of M1:

| Method | Path | Use case |
|--------|------|----------|
| GET | `/api/v1/health` | liveness + version |
| GET | `/api/v1/runs?limit=&status=` | `ListRuns` |
| GET | `/api/v1/runs/{id}` | `GetRun` |
| DELETE | `/api/v1/runs` | `DeleteRuns` |
| GET | `/api/v1/events` | SSE stream of domain events |

**Error envelope** -- every non-2xx response, no exceptions (unknown
routes, method-not-allowed, and framework validation included):

```json
{"error": {"code": "run_not_found", "message": "...", "details": [...]}}
```

Codes map to statuses centrally: `run_not_found` 404,
`invalid_query` 422, `validation_error` 422 (FastAPI/pydantic input),
`http_{status}` for transport-level errors, `internal_error` 500
(traceback logged, message generic).

**SSE**: `data: {json}` frames where json carries `type`
(`stream_opened`, `run_created`, `run_started`, `run_completed`,
`run_failed`, `run_cancelled`, `runs_deleted`), `occurred_at`, and
the event's fields; `: ping` heartbeat every 15 s.

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

## 7. Milestones

| # | Scope | Status |
|---|-------|--------|
| M1 | Skeleton: layering, settings, SQLite + migrations, runs read-side, delete + event, error envelope, SSE, tests, boot on own port | **done** |
| M2 | Training lifecycle: `TrainingGateway` port, command building, spawn/stop/kill, progress watching, `StartTraining`/`StopTraining` use cases, monitor telemetry on the bus | next |
| M3 | Config file, settings store, assets (checkpoints/loras), datasets domains | planned |
| M4 | Nodegraph subsystem (registry, introspect, executor, presets) behind a `GraphRuntime` port | planned |
| M5 | Frontend decision + parity audit + migration strategy (doc 02/03), decommission plan for `server/` | planned |

## 8. Running it

```bash
# server (own port; old server keeps 8765)
python -m backend.cli --port 8766

# tests (5 files: domain, repository, use cases, bus, end-to-end API+SSE)
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
