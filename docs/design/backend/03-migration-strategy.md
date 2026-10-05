# 03 -- Migration strategy: frontend, cutover, decommissioning `server/`

Status: **decisions made** (2026-09-30), implementation phased as in
section 6. Companion to `01-architecture.md` (backend contract) and
`02-api-reference.md` (the API the frontend targets).

## 1. Frontend decisions

| Question | Decision |
|---|---|
| Stack | **Vanilla ES modules, no framework, no build step.** Performance first, structured, easy to expand -- no npm/node toolchain enters this repo. The legacy frontend (7,576 lines across 18 JS + 6 HTML files in `archive/server/static/`) is the reference, not the base: it is
rewritten. |
| What carries over | The **monitor dashboard's visualization** -- charts, series selection, CSV export, replay-on-reload -- is good enough to keep. Its data contract (SSE stream shapes below) is treated as pinned; the code is restructured into modules while the visuals stay. |
| Who serves it | **The backend serves its own static files** (one origin on 8766: no CORS, no proxy, SSE and API same-origin). |
| Build order | **Monitor + training controls first** (the daily driver -- a usable tool after every step), then the graph editor, then dataset manager + config/history tabs. |

## 2. New frontend layout (planned)

Rules that keep it fast and expandable:

* zero globals -- each page boots from one ES module entry
  (``type=module`` is deferred, so DOM is ready at evaluation);
* no framework, no virtual DOM -- direct DOM with targeted re-renders
  (the monitor's charts already work this way and stay at 60fps with
  100k records);
* one API module -- handlers never hand-roll `fetch`, so the error
  envelope and status codes are decoded in exactly one place;
* state lives in one small store per view; SSE is the source of truth
  for anything live, polling only where the API is the source of truth.

## 3. Parity audit: legacy surface vs backend API

51 legacy endpoints vs 51 backend endpoints -- and they are not a
1:1 mapping. Families:

| Legacy family | Status |
|---|---|
| config (`GET/PUT /config`, `GET/PUT /config/raw`, `/options/tree`, `/control/options`) | **parity** (PATCH merges nested partials; tree shape improved) |
| installer (`/installer/readiness`, `/state`, `/manifest`, `/apply`) | **new**: first-run. Nothing in the legacy surface reported whether a machine could train, or wrote the paths before they were written by hand. `apply` is refused once configured. |
| training control (`/run/start`, `/run/stop`, `/run/status`, `/run/log`) | **dropped 2026-10-02**: the supervised-subprocess route they described spawned `python -m core.cli` and was removed with `core/`. Training is started as a graph execution -- see the nodegraph row and `docs/design/11-core-removal.md`. |
| `/run/reset` | **dropped**: legacy in-memory service reset; backend state lives in the DB |
| runs history (`/runs`, `/runs/{id}`, `/{id}/log`, `/{id}/events`, `/{id}/previews`, `/logs/clear`) | **dropped 2026-10-02**, with the route above. Graph executions have their own history (`/executions`, `/executions/{id}`), which is where "what ran, when, how did it end" now lives. |
| `/sse` | **parity+** (all domain events, generic encoder, heartbeat) |
| settings (`GET/POST /settings`, `/files/{kind}`) | **resolved** (3.1): `/files/{kind}` dropped -- the config editor uses a path input + datalists |
| datasets (trajectories CRUD, training-sets, tasks, checkpoints) | **parity** (toggle -> explicit `type`; pending -> `committed=false`; reject -> `discard`); both task kinds since M8e (`ingest_lora` + `generate_teacher`, vs legacy `type=lora`/`teacher`); bulk edit exceeds legacy (neg modes + verdict, M8e); **view shipped M8c, add-data + edit modes M8e, card preview + `⋮` menu M8f (beyond legacy)** |
| nodegraph (`registry`, `executions`, `run`+stop, `node/{class}/diagnostics`) | **parity** (plus `validate`, library, history wipe -- legacy had none) |
| nodegraph assets (`assets/{kind}`, `browse`, `inspect`, `mkdir`, `upload`) | **parity** |
| **monitor stream** (`GET /nodegraph/monitor/{id}/stream`) | **shipped M6** (section 4) |
| page routes (`/`, `/nodegraph`, `/nodegraph/monitor/{id}`, `/datasets`) | **shipped** M6 shell + monitor, M7 `/graph`, M8a `/config`, M8b `/run/{id}`, M8c `/datasets`, M8d `/help` + `/settings` |

### 3.1 Gaps and their resolutions

| Gap | Resolution |
|---|---|
| Monitor SSE stream | **Port it** (section 4) -- required for the monitor-first slice. **Shipped in M6.** |
| Run previews (`runs/run_{id}/previews/` manifest + images) | **Drop** (M8 decision): the producer was `core/preview_sampler.py`, which went to `archive/` with the rest of `core/`. |
| Per-run `events` history (`/runs/{id}/events`) | **Drop**: SSE gives live events and the log/DB carry state; a persisted per-event history has no consumer the new frontend needs (the M8b run detail shipped without one). |
| `/runs/logs/clear` | **Drop**: superseded, then dropped with the route. `DELETE /executions` wipes execution history. |
| `/files/{kind}` (settings file browser) | **Drop** (M8 decision): the config editor shipped with a path input + datalists and never needed a browser; revisit only if the settings tab's redesign asks for one. |
| Dataset preview images (legacy: static mount `/datasets/{name}/{preview_path}`) | **Port** (M8c): served through `GET /api/v1/datasets/{name}/files/{path}` with containment enforced by the adapter, and the preview allowlist on top. **Shipped in M8c.** See [02 §3](02-api-reference.md#what-a-client-named-path-may-point-at) for what each path-taking surface may point at. |
| `/run/reset` | **Dropped** (table above). |

## 4. Monitor data path (pinned contract)

Facts the port preserves (source: `monitor_bus.py`, `nodes/monitor/*`,
`archive/server/routes_monitor.py`,
`archive/server/static/monitor_dashboard.js`):

* `MonitorBus` (repo root): thread-safe pub-sub keyed by **`monitor_id`**,
  `report(monitor_id, dict)` appends *unwrapped* dicts to a per-id
  history (`HISTORY_LIMIT = 100000`, matched to the dashboard's
  `MAX_RECORDS`), `clear(monitor_id)` empties it and broadcasts a
  `{"type": "clear"}` frame, `subscribe()` replays history to a new
  queue (a reload restores the full chart). No module-level singleton:
  one instance per server process.
* **`monitor_id` is a node input param** (`mon-` + 8 base36 chars,
  generated by the editor), deliberately *not* the execution id -- it
  outlives runs; the monitor node clears it at the start of each run.
* Stream frames: `{"type": "connected"}` first, then raw step-report
  dicts (`step`, `total_steps`, `loss`, `lr`, `t`, `weight_t_*`,
  `prob_t_*`, `*_ms` timings, `vram_*`, `resident_*_mb`, ...),
  `{"type": "clear"}`, and terminal `{"type": "run_end", "step",
  "cancelled"}`. The dashboard ignores anything without `step` it
  doesn't recognize.
* Backend wiring (M6 slice 1): an application port over this contract;
  the infrastructure adapter wraps the repo-root `MonitorBus` (same
  class the legacy server and `nodes/` use -- payload and replay
  semantics stay byte-identical); `ReflectedGraphRuntime` passes the
  instance into `ExecutionContext(monitor_bus=...)` instead of
  `None`; presentation exposes
  `GET /api/v1/monitor/{monitor_id}/stream` mirroring the legacy
  frame sequence. Nodes are untouched -- they already duck-type
  `report`/`clear` and no-op on `None`.

## 5. Data cutover

* **Saved graphs**: legacy graphs live in browser `localStorage`
  (`ng_graph_v1`). The new editor offers a one-click *import* that
  POSTs them into `/api/v1/graphs/library` (payloads are
  shape-compatible; class names are validated at run time, not save).
  No server-side migration is possible or needed -- the data is
  per-browser.
* **Runs history**: not imported. Legacy history remains readable
  through the legacy server until decommission, and the new backend
  starts with an empty history. Log files on disk are never touched:
  run ids are numbered from the database, and a fresh one would start
  at 1 -- colliding with `runs/run_1/`. Startup therefore seeds the id
  sequence **above the highest existing `runs/run_*` directory**, and two
  guards refuse a collision rather than truncating it: `prepare` rejects
  a non-empty `runs/run_<id>/` (409 `run_directory_conflict`) and the
  gateway refuses to open a non-empty `log.txt`/`log.progress.jsonl` with
  `"w"` (docs 07 F-04). So the numbering continues past the legacy
  history and no old run is ever overwritten.
* **Datasets**: already server-side and format-versioned (`04`); both
  servers can read them -- nothing to move.
* **Settings/config**: backend reads the same project config files and
  settings tiers; no copy.

## 6. Decommission plan for `server/` (executed)

| Phase | Scope | Entry criterion |
|---|---|---|
| M9 | **Flip**: `README.md` + `run_server.sh` point at the backend; `server/` moves to archive (its 6 smoke tests retire with it; the 66 `nodes/` tests are unaffected -- one import repointed to `archive.server`); legacy `smoke_test_*` knowledge is preserved in this doc series | **executed 2026-10-01 on user instruction**: `server/` + `server_cli.py` -> `archive/`, entry-point docs (README, setup, architecture) flipped to `backend/`, `run_tests.py`/`full_gate.sh` down to nodes+manager (68 tests), `backend/cli.py` inherits the XPU-env entry-point contract. Entry criterion status: the real training cycle on the new frontend **remains the follow-up validation** |

The six `server/` smoke tests retired with the move; the files live on
under `archive/server/smoke_tests/` (still runnable for reference),
and what each one pinned is recorded here so the knowledge survives
the tree move:

| Retired test | What it pinned |
|---|---|
| `smoke_test_graph_executor.py` | topological execution + port compatibility (`graph_executor.py`, `nodegraph_registry.py`) |
| `smoke_test_execution_registry.py` | the real threaded `GraphExecutor` over a trivial one-node graph, execution registry |
| `smoke_test_nodegraph_introspect.py` | `display_name` / NodeInfo introspection (design doc §11.5: class name stays the stable registry key) |
| `smoke_test_node_presets.py` | `Node.NODE_KIND` / `NodePreset` / `list_presets()` and their introspection |
| `smoke_test_asset_inspect.py` | asset path sandboxing + `inspect()` real safetensors I/O |
| `smoke_test_static_caching.py` | `Cache-Control` policy for browser-facing responses (`archive/server/main.py`) |
