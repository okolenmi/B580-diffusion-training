# 06 -- Frontend visual smoke checklist (browser-enabled session)

Status: **executed 2026-10-01** from a desktop-app browser session
against a scratch-DB backend on 8766, **automated the same day** -- the
checklist below now runs as `backend/tests/visual_smoke.py` (50
checks, Playwright). The API/SSE/pages coverage comes from
`backend/tests/run_all.py` and the full gate; this checklist is the
missing layer -- JavaScript actually running in a real browser.
Findings and fixes from that run are recorded at the bottom.

## Why this exists

`backend/tests/run_all.py` and the full gate cover the API, SSE frame
contracts, and that the pages/assets *serve*. They cannot cover
JavaScript that actually runs: module wiring, DOM lookups, runtime
exceptions in event handlers. This checklist is that missing layer.

## Automated run (Playwright)

```bash
# terminal 1: scratch backend on 8766 (see Setup)
# terminal 2:
~/.venvs/pw/bin/python backend/tests/visual_smoke.py
```

Covers: idle state (real backend), running state (API mocked with a
synthetic `RunOut` + history + log), interactions (start-form guard,
row-click -> log, wipe confirm dialog captured in code), elapsed
ticker, a monitor/graph regression pass (all three pages share
`style.css`), and the config editor (schema-driven form, visibility,
dirty tracking, E2E save against a throwaway copy). Console is
asserted clean across every page. Desktop-browser execution of the
same checklist below stays as the manual fallback.

## Prerequisites

* Browser tools connected in the session (desktop app, experimental
  browser setting on), with devtools/console access.
* `/home/okolenmi/comfy/venv/bin/python` (plain `python3` has no torch
  and the backend imports it).

## Setup

```bash
cd /home/okolenmi/Desktop/B580-diffusion-training
rm -f /tmp/opencode/smoke.db
BACKEND_DB_PATH=/tmp/opencode/smoke.db \
  /home/okolenmi/comfy/venv/bin/python -m backend.cli --port 8766
```

A scratch DB keeps the check independent of real training history.

## 1. App shell + training controls -- `http://127.0.0.1:8766/`

* [x] Page renders: sidebar nav, topbar + state hero (start form when
      idle, live run when active), history table (6 column headers) +
      log pane. Layout is state-driven: idle never shows empty metric
      cards, running never shows a start form that would 409.
* [x] Console clean (no errors/warnings from our modules; the lone
      `/runs/active` 404 line is the browser's network log for the
      expected `no_active_run` answer, rendered correctly as
      "no active run").
* [x] Start-options: with no config path chosen the page shows the
      placeholder (the API deliberately answers 422 `invalid_query`
      for an empty `path` -- that is contract, not a bug). Choosing a
      real path was **not** exercised (needs config files).
* [x] `GET /runs/active` 404 (`no_active_run`) renders as "no active
      run", not an exception.
* [x] Runs history lists rows (empty state: "No runs yet."; Wipe
      disabled while empty).
* [x] `/api/v1/events` SSE connects (System Console: "Connected to
      /api/v1/events.").
* [x] Running-state hero (automated with a mocked `RunOut`): phase in
      the badge, progress `412 / 1000 · 41%`, cache sub-bar
      (`.progress-sub.active` mechanism), stop/kill visibility,
      elapsed ticker advancing, row click selects + loads the log,
      wipe `window.confirm` captured by a dialog handler.
* [ ] **deferred with training tests**: the same controls against a
      *real* active run (process actually stopping/saving).

## 2. Monitor dashboard -- `http://127.0.0.1:8766/monitor/<any-id>`

* [x] Page renders: chart canvas, series list, controls; console clean.
* [x] Stream connects: status badge reaches `live`.
* [x] Empty-id history: absent series/stats show em dashes (Best loss,
      LR, Grad norm, VRAM, Peak all `--`) -- **never fabricated zeros**.
* [ ] **needs a live producer**: points appear live, mid-stream reload
      replays history, `clear` empties the chart, CSV export + series
      toggles on real data (the M6 harness covered the frame contract;
      real-data rendering waits for the training-test gate).
* [x] `Export CSV` button present and enabled.

## 3. Graph editor -- `http://127.0.0.1:8766/graph`

* [x] Palette lists the catalog grouped by domain (7 domains, 36
      nodes); `load_errors` section exists (catalog had none).
* [x] Add nodes from the palette; move by dragging the header (edge
      paths follow); connect output->input; click an edge to select,
      `Delete` removes it (Esc deselects); library save/load round-trip
      keeps positions bit-identical.
* [x] Params form: typed widgets (number, choices, bool) edit the
      node's `params`; wired inputs show "overridden by <src>".
* [x] Validate: clean graph -> "Graph is valid."; unknown class ->
      complete issue list with a node chip that focuses the node.
* [x] Run: execution appears in the list, per-node progress badges
      stream in via `/events` (ok + duration), terminal event logs
      "finished (3 nodes)", wipe history works (with confirm).
* [x] Type check: `int` -> `float` wire is rejected with an
      explanation (matches the backend's `incompatible_types`);
      `float` -> `float` connects, compatible sockets highlight green
      during the drag.
* [x] Library: save, load, delete path exercised; importing a legacy
      `ng_graph_v1` draft maps `connections`/`paramValues`/`x,y` and
      keeps unknown classes as placeholders with a console warning.
* [x] Network tab: the editor talks only to `/api/v1/graphs/*` (+ the
      shared `/events` stream).

## 4. Config editor -- `http://127.0.0.1:8766/config`

* [x] Honest empty state before a config is loaded (a form rendered
      without values would lie).
* [x] Form renders from `GET /config/options`: 78 fields in 6 numbered
      groups, subgroup headers, help text; `visible_when` toggling
      works (LoRA Rank shows for `lora`, hides for `distillation`).
* [x] Launch-only options (`start_from`, `reset_optimizer` -- both
      `persist_locally`) are excluded: they belong to the Training
      page's start form, are never written to the file (their contract
      says so), and carry deliberate duplicate ids.
* [x] Dirty tracking: edits enable Save + show the chip; Revert
      discards edits by re-reading the file.
* [x] E2E save against a throwaway copy (absolute path accepted):
      PATCH merges, the form keeps the saved value, the raw tab
      refreshes from the written file. Repo config files untouched.
* [x] Raw tab round-trips `GET/PUT /config/raw`; buffer resync rules:
      form saves refresh a clean raw buffer, raw writes reload the form.
* [x] Console clean across all four scenarios (A idle, B running,
      C monitor+graph regression, D config).
* [ ] `PUT /config/raw` rejection (invalid TOML -> 422, file
      untouched) is API-tested (`test_config.py`); not driven in the
      browser.

## 5. Record

Paste screenshots of each page into the session and list pass/fail
per checkbox. Fix regressions in the milestone that owns the code --
do not adjust the checklist to match broken behavior.

Caveat from the 2026-10-01 desktop run: the desktop app's `screenshot`
tool served stale frames (byte counts matched new captures, but the
images rendered earlier states). DOM inspection via `evaluate` and the
console reader were authoritative; treat desktop screenshots as
advisory. **Superseded for testing purposes**: the Playwright suite
above takes real frame-buffer screenshots from headless Chromium and
is the driver for this checklist now; the desktop browser is for
viewing only.

## Findings from the 2026-10-01 run (all fixed in the same session)

1. **`hidden` attribute did nothing on class-styled elements** --
   `STOP & SAVE`/`FORCE KILL` rendered with no active run, and palette
   search filtering silently failed. Root cause: the port switched from
   legacy's inline `style="display:none"` toggling to the `hidden`
   attribute, but author class rules (`.btn`, `.ed-palette-item`)
   legitimately outrank the UA sheet's `[hidden]`. Fixed with
   `html [hidden] { display: none; }` (specificity 0,1,1, position-
   independent, no `!important`).
2. **No `Cache-Control` on pages/assets** -- heuristic freshness served
   stale `style.css` across reloads (observed: `transferSize: 0`).
   Root cause: the M6 static-serving port dropped legacy
   `server/main.py`'s revalidation middleware (its comment documents a
   real mixed-version incident). Fixed by stamping `no-cache` on page
   `FileResponse`s and via a `StaticFiles` subclass for `/ui`;
   `test_pages.py` pins it (pages/assets `no-cache`, `/api` untouched).
3. **Cache-phase progress bar could never render** -- dashboard.js
   toggled `hidden` while the ported CSS expects the legacy `.active`
   class pair. Fixed to match the ported CSS (`.classList` toggles).
4. **Stale `_suppressClick`** (found while testing) -- a wire released
   off-canvas left the flag set, eating the next canvas click. Fixed:
   any new `pointerdown` clears it.
5. **"Continue from" select rendered blank** (found by the Playwright
   suite during the main-page redesign) -- the placeholder option was
   `disabled` without `selected`, leaving `selectedIndex` at `-1`, so
   Chromium displayed nothing. Present since the M6 port (visible in
   the original screenshots). Fixed: placeholder gets
   `disabled: true, selected: true`.
6. **Config widgets had no `id`** (found by the M8a smoke) -- labels
   pointed at `#f-<dotted id>` but `renderForm` never assigned
   `input.id`, so label association was broken and the smoke's
   `select_option("#f-tuning-method")` timed out. Fixed at the render
   site.
7. **`live` overlay written under dotted keys** (found by the M8a
   smoke) -- `onEdit` set `live["tuning.method"]` literally while
   `getDeep(live, "tuning.method")` reads `live.tuning.method`, so
   `visible_when` kept evaluating the stale loaded value (rank stayed
   visible after switching to `distillation`); the initial overlay was
   also a shallow copy that would have let edits mutate `values`. Fixed
   with `setDeep` writes + a JSON deep copy on load.

## Known deferred (not bugs)

* Previews, per-run `/events` history, `logs/clear`, `/files/{kind}`
  -- deferred per docs 03 section 3.1.
* Graph autosave to localStorage: the new frontend never writes
  `ng_graph_v1`; the key is read once as an import source (docs 03
  section 5).
* Training-loop behavioral tests stay deferred until the full server
  rework is done (user decision).
* Preset suggestion menu (wire-drop compatible-node hints): legacy
  used presets only for suggestions, not the payload -- deferred.
