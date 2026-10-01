# 03 -- Migration strategy: frontend, cutover, decommissioning `server/`

Status: **decisions made** (2026-09-30), implementation phased as in
section 6. Companion to `01-architecture.md` (backend contract) and
`02-api-reference.md` (the API the frontend targets).

## 1. Frontend decisions

| Question | Decision |
|---|---|
| Stack | **Vanilla ES modules, no framework, no build step.** Performance first, structured, easy to expand -- no npm/node toolchain enters this repo. The legacy frontend (7,576 lines across 18 JS + 6 HTML files in `server/static/`) is the reference, not the base: it is rewritten. |
| What carries over | The **monitor dashboard's visualization** -- charts, series selection, CSV export, replay-on-reload -- is good enough to keep. Its data contract (SSE stream shapes below) is treated as pinned; the code is restructured into modules while the visuals stay. |
| Who serves it | **The backend serves its own static files** (one origin on 8766: no CORS, no proxy, SSE and API same-origin). The legacy server keeps 8765 untouched until decommission. |
| Build order | **Monitor + training controls first** (the daily driver -- a usable tool after every step), then the graph editor, then dataset manager + config/history tabs. |

## 2. New frontend layout (planned)

```
frontend/                     # shipped slices 1+2 (M6/M7) + M8 views + M8d shell + M8e dataset flows
├── index.html                # System tracker page (/): state hero, history, log
├── monitor.html              # standalone monitor dashboard (/monitor/{monitor_id})
├── graph.html                # graph editor page (/graph)
├── config.html               # config editor page (/config, M8a)
├── run.html                  # run detail page (/run/{id}, M8b)
├── datasets.html             # dataset manager (/datasets + /{name}, M8c);
│                             # add-data + item editor dialogs (M8e)
├── help.html                 # help skeleton (/help, M8d)
├── settings.html             # settings: design theme first (/settings, M8d)
├── css/
│   ├── style.css             # shared design system: tokens (incl. shell tokens),
│   │                         # buttons, inputs, badges, console-line primitive,
│   │                         # card, page chrome (topbar/body, tabs, state/error
│   │                         # blocks), log pane; dead weight removed
│   ├── shell.css             # THE application shell: icon rail (frame, items,
│   │                         # hover tips, active/disabled states) + floating
│   │                         # console window + FAB; layout contract documented
│   │                         # in its header (M8d)
│   ├── training.css          # system tracker page: state hero, history/log grid,
│   │                         # monitor hand-off strip
│   ├── config.css            # config page: path bar, grouped form, raw editor
│   ├── run.css               # run detail page: details grid, log card
│   ├── datasets.css          # dataset manager: card grid, stats, item cards,
│   │                         # bulk bar, task form
│   ├── help.css              # help page: where-things-live rows, stub grid
│   ├── settings.css          # settings page: theme picker, tag, link rows
│   ├── monitor.css           # monitor page block
│   └── editor.css            # editor layout, canvas plane, node visuals
└── js/
    ├── shell.js              # mounts rail + floating console on every page
    │                         # (loaded FIRST); active item from pathname;
    │                         # console geometry/minimized state in localStorage
    ├── api.js                # THE fetch wrapper: error envelope decoded once, + sse()
    ├── monitor.js            # monitor page entry (ported visual, new stream URL)
    ├── editor.js             # editor page entry: catalog, GraphDoc, toolbar, /events
    ├── views/dashboard.js    # training controls (runs REST + /events SSE)
    ├── views/config.js       # config editor: schema form + raw buffer (M8a)
    ├── views/run.js          # run detail: full RunOut grid + log tail (M8b)
    ├── views/datasets.js     # dataset manager: list/detail/items/sets/tasks,
    │                         # add-data dialog + browse/edit modes (M8c, M8e)
    ├── editor/               # state.js (GraphDoc + wire forms), canvas.js (render/
    │                         # drag/connect), inspector.js (params form), palette.js,
    │                         # executions.js (run lifecycle), library.js (+ legacy import)
    └── lib/loss_chart.js     # chart lib as an ES module (visuals untouched)
```

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

51 legacy endpoints vs 47 backend endpoints -- and they are not a
1:1 mapping. Families:

| Legacy family | Backend equivalent | Status |
|---|---|---|
| config (`GET/PUT /config`, `GET/PUT /config/raw`, `/options/tree`, `/control/options`) | `GET/PATCH /config`, `GET/PUT /config/raw`, `GET /config/options`, `GET /config/start-options` | **parity** (PATCH merges nested partials; tree shape improved) |
| training control (`/run/start`, `/run/stop`, `/run/status`, `/run/log`) | `POST /runs`, `POST /runs/{id}/stop`, `GET /runs/active`, `GET /runs/{id}/log` | **parity** (start is JSON, not multipart form) |
| `/run/reset` | -- | **dropped**: legacy in-memory service reset; backend state lives in the DB (delete + restart covers it) |
| runs history (`/runs`, `/runs/{id}`, `/{id}/log`, `/{id}/events`, `/{id}/previews`, `/logs/clear`) | list/get/log + `DELETE /runs` | **resolved** (3.1): `events`/`previews`/`logs/clear` all dropped; the rest shipped |
| `/sse` | `GET /events` | **parity+** (all domain events, generic encoder, heartbeat) |
| settings (`GET/POST /settings`, `/files/{kind}`) | `GET/POST /settings` | **resolved** (3.1): `/files/{kind}` dropped -- the config editor uses a path input + datalists |
| datasets (trajectories CRUD, training-sets, tasks, checkpoints) | items/sets/tasks endpoints + `GET /assets/checkpoint` + `GET /datasets/{name}/files/{path}` (preview bytes) | **parity** (toggle -> explicit `type`; pending -> `committed=false`; reject -> `discard`); both task kinds since M8e (`ingest_lora` + `generate_teacher`, vs legacy `type=lora`/`teacher`); bulk edit exceeds legacy (neg modes + verdict, M8e); **view shipped M8c, add-data + edit modes M8e** |
| nodegraph (`registry`, `executions`, `run`+stop, `node/{class}/diagnostics`) | `/graphs/nodes`, `/graphs/executions`, `/graphs/run`+stop, `/graphs/nodes/{class}/diagnostics` | **parity** (plus `validate`, library, history wipe -- legacy had none) |
| nodegraph assets (`assets/{kind}`, `browse`, `inspect`, `mkdir`, `upload`) | `GET/PUT /assets/{kind}...` | **parity** |
| **monitor stream** (`GET /nodegraph/monitor/{id}/stream`) | `GET /api/v1/monitor/{monitor_id}/stream` | **shipped M6** (section 4) |
| page routes (`/`, `/nodegraph`, `/nodegraph/monitor/{id}`, `/datasets`) | `GET /`, `GET /monitor/{monitor_id}`, `GET /graph`, `GET /config`, `GET /run/{id}`, `GET /datasets`, `GET /datasets/{name}`, `GET /help`, `GET /settings`, `/ui/*` (section 2) | **shipped** M6 shell + monitor, M7 `/graph`, M8a `/config`, M8b `/run/{id}`, M8c `/datasets`, M8d `/help` + `/settings` |

### 3.1 Gaps and their resolutions

| Gap | Resolution |
|---|---|
| Monitor SSE stream | **Port it** (section 4) -- required for the monitor-first slice. **Shipped in M6.** |
| Run previews (`runs/run_{id}/previews/` manifest + images) | **Drop** (M8 decision): the producer is `core/preview_sampler.py` on the unsupported `core/` route -- `nodes/` (the supported route) never writes previews, so no new run will ever have a manifest. The run detail page serves the full `RunOut` + log instead. |
| Per-run `events` history (`/runs/{id}/events`) | **Drop**: SSE gives live events and the log/DB carry state; a persisted per-event history has no consumer the new frontend needs (the M8b run detail shipped without one). |
| `/runs/logs/clear` | **Drop**: `DELETE /runs` (history wipe) plus per-run artifacts on disk cover the intent. |
| `/files/{kind}` (settings file browser) | **Drop** (M8 decision): the config editor shipped with a path input + datalists and never needed a browser; revisit only if the settings tab's redesign asks for one. |
| Dataset preview images (legacy: static mount `/datasets/{name}/{preview_path}`) | **Port** (M8c): served through `GET /api/v1/datasets/{name}/files/{path}` with containment enforced by the adapter (02 section 7). **Shipped in M8c.** |
| `/run/reset` | **Dropped** (table above). |

## 4. Monitor data path (pinned contract)

Facts the port preserves (source: `monitor_bus.py`, `nodes/monitor/*`,
`server/routes_monitor.py`, `server/static/monitor_dashboard.js`):

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
  `None` (doc 05's deferred item); presentation exposes
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
  through the legacy server until decommission; log files on disk stay
  untouched either way. The new backend starts with an empty history.
* **Datasets**: already server-side and format-versioned (`04`); both
  servers can read them -- nothing to move.
* **Settings/config**: backend reads the same project config files and
  settings tiers; no copy.

## 6. Decommission plan for `server/`

Phases, each with an entry criterion -- the legacy server keeps
serving until the criterion for flipping the default is met. Phase
numbers match the milestone table in `01-architecture.md`:

| Phase | Scope | Entry criterion |
|---|---|---|
| M6 | Monitor slice: backend monitor port + stream endpoint, static serving, frontend shell + monitor dashboard + training controls | **shipped 2026-09-30**: backend serves the monitor page on 8766; the stream's replay/live/clear frames are test-pinned (`test_api_monitor.py`) |
| M7 | Graph editor (palette/validate/run/executions/library) against `/api/v1/graphs` | **shipped 2026-10-01**: `/graph` page; validate/run/library round-trip (incl. `layout` extras) smoke-tested on 8766; every frontend module passes `node --check` |
| M8 | Dataset manager + config editor + run history views | **shipped 2026-10-01**: `/config` (M8a), `/run/{id}` (M8b), `/datasets` + preview bytes (M8c); each page is smoke-covered |
| M8d | Shell redesign: icon rail (all pages) + floating persistent console + `/help` + `/settings` | **shipped 2026-10-01**: `shell.js`/`shell.css` mounted on all six existing pages (rail order: Graph Editor, Datasets, Workflows (soon), System tracker, Help; Settings pinned last; monitor stays workflow-attached, no rail slot); console geometry + minimized state persist in localStorage; visual smoke: 132 checks incl. scenario G |
| M8e | Dataset add-data + edit modes | **shipped 2026-10-01**: `generate_teacher` task kind (legacy `type=teacher` parity: prompt/keyword sources, neg keyword mix, cfg/steps/t ranges, batch, conditions x samples); import kind hardened (resize/model-type enums validated); bulk multi-edit (`neg_prompt_mode`, `type`, `prepend`/`append`, idempotent joins); frontend: per-card "Add data" dialog (generate/import tabs with the full option sets), Browse/Edit item modes, advanced item editor (fields + metadata + prev/next walk), multi-edit panel; visual smoke scenario H |
| M9 | **Flip**: `README.md` + `run_server.sh` point at the backend; `server/` moves to archive (its 6 smoke tests retire with it; the 66 `nodes/` tests are unaffected); legacy `smoke_test_*` knowledge is preserved in this doc series | M8 complete and the new frontend used for a real training cycle |

Both servers run side by side until M8 (8765 legacy, 8766 backend) --
they share data files read-only, so there is no cutover day, only the
flip of the default entry point.
