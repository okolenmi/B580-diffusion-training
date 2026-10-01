# 06 -- Frontend visual smoke checklist (browser-enabled session)

Status: **executed 2026-10-01** from a desktop-app browser session
against a scratch-DB backend on 8766, **automated the same day** -- the
checklist below now runs as `backend/tests/visual_smoke.py` (186
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
`style.css`), the config editor (schema-driven form, visibility,
dirty tracking, E2E save against a throwaway copy), run detail views
(hand-off link, honest 404, completed + failed renders), and the
dataset manager against the real library (create guard, honest empty
states, preview bytes over the files route, filters, throwaway
cleanup), the M8f card previews + item context menu (thumb images
resolve on real cards, the one-option menu opens/disables/closes --
never clicked, so the real datasets stay read-only), the M8d shell
(icon rail inventory + active states +
hover tips, floating console minimize/FAB/restore/persistence,
help + settings pages), and the M8e dataset flows (add-data dialog
option sets + local validation, Browse/Edit modes (selection is an
edit-mode tool: browse renders no checkboxes or select-all and hides
multi-edit mid-selection, and a selection survives the round-trip
back to edit), the advanced item
editor's dirty/revert/walk cycle, the multi-edit panel -- all
read-only against the real curated datasets). Console is asserted
clean across every page.
Desktop-browser execution of the same checklist below stays as the
manual fallback.

Console geometry gestures (resize -> move -> resize, and item
checkbox measurements) are exercised by
`scripts/probe_ui_bugs.py` against the same live server:

```bash
~/.venvs/pw/bin/python scripts/probe_ui_bugs.py
```

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

* [x] Page renders: icon rail (shell.js, all pages), topbar + state
      hero (start form when idle, live run when active), the monitor
      hand-off strip, history table (6 column headers) + log pane.
      Layout is state-driven: idle never shows empty metric cards,
      running never shows a start form that would 409.
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
* [x] `/api/v1/events` SSE connects (floating console: "Connected to
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

## 5. Run detail -- `http://127.0.0.1:8766/run/{id}`

* [x] Hand-off from the main page: the log card's "open full ↗" link
      appears with a selection and follows it (`/run/12` after
      clicking run 12's row, scenario B).
* [x] Missing run renders an honest state: probing `/run/9999`
      against the real backend keeps the details hidden and shows a
      "not found" message; the browser's own resource-status line for
      this deliberate 404 is filtered noise (`is_expected_noise`).
* [x] Completed run (mocked): title, `status-completed` badge, all 16
      detail rows (steps, exit code 0, absolute + relative timestamps,
      empty error as em dash), log tail fills. Line structure is
      pinned via `white-space: pre-wrap` -- `.log-pane` moved to
      `style.css` (both pages share it); it was missing on this page
      until the screenshot showed run-together lines.
* [x] Failed run (mocked): error text renders with the `error` class,
      red (computed color differs from a normal row), exit code shown.
* [ ] Live active-run refresh (5s log tail + 1s duration ticker +
      terminal-event SSE reload) is exercised only through its code
      path; the smoke has no real running run on `/run/{id}`.

## 6. Dataset manager -- `http://127.0.0.1:8766/datasets`

Runs against the REAL backend and the repo's real datasets (opened
read-only; a throwaway `m8c-smoke-ds` is created and deleted inside
the scenario, its confirmation dialog captured).

* [x] Library list renders the real datasets; the icon rail links to
      `/datasets` (M8c hand-off).
* [x] Create guard: an empty name refuses inline, fires no request,
      adds no card; a real create appends the card.
* [x] Fresh-dataset detail: title follows the route, seven stat chips
      with real zeros (API-computed, never placeholders), honest empty
      states on all three tabs, the Add-data entry point renders on
      the Tasks tab (the dialog itself is scenario H).
* [x] Real curated dataset (`1024 aes` -- space in the name pins the
      URL-encoded route): stats show the true counts, all 201 item
      cards render, a preview image actually loads through
      `/datasets/{name}/files/{path}` (`naturalWidth > 0`).
* [x] Card previews (M8f): every dataset card renders a preview thumb
      and a real card's image resolves through the files route; the
      empty throwaway's card shows the honest `NO PREVIEW` placeholder
      (no guessed image).
* [x] Item `⋮` context menu (M8f): the half-transparent trigger
      renders on the thumb; the menu opens with exactly one option
      ("Set as dataset preview"); the item currently fronting the card
      is honestly **disabled**, another item's option renders enabled
      (**never clicked** -- the scenario watches requests and asserts
      zero `PUT /preview`, so the real dataset is never mutated);
      Escape and an outside click both close the menu; screenshot
      `datasets_menu.png`.
* [x] Membership filters: Pending is honestly empty on a fully
      curated dataset (its empty state shows), Used brings the items
      back; the training-sets tab lists the real set rows.
* [x] Delete asks for confirmation and removes the throwaway (the
      scenario asserts the card count returns to baseline; the repo's
      `datasets/` directory is checked clean afterwards).
* [ ] Item mutations (prompt edit, good/bad toggle, discard, bulk
      apply, commit-to-set, **set card preview**) are covered by
      `test_api_datasets.py` at the API level only -- the smoke edits
      nothing and applies nothing; scenario H exercises the
      editor/multi-edit UI in a read-only walk (dirty -> revert, rows
      rendered, never saved).
* [ ] Task start/stop is never fired from the smoke (it would spawn a
      real child process against user checkpoints); scenario H pins
      that the browser sends no task POST during its walk.

## 7. Shell -- rail + floating console + help/settings (M8d)

Exercises the shell mounted on every page (icon rail + floating
system console) plus the two new pages.

* [x] Icon rail: visible, exactly 6 items (Graph Editor, Dataset
      manager, Pre-built workflows, System tracker, Help, Settings);
      the workflows slot is honestly disabled (`aria-disabled`);
      `aria-current="page"` marks the active destination, on `/` and
      after rail navigation alike; the logo links home.
* [x] Rail hover tips carry the label (opacity actually transitions
      in -- `is_visible()` would pass on an unstyled tip, so the check
      waits for `opacity > 0.9`).
* [x] Floating console: mounted on every page (tracker and the
      monitor/graph regression pages both assert it), natively
      resizable (`resize: both`), minimizes to the bottom-right FAB,
      the minimized state survives a reload (localStorage), clicking
      the FAB restores the window.
* [x] Rail navigation: tracker -> datasets lands on `/datasets` with
      the destination item active; the console follows to the new
      page.
* [x] Help (`/help`): title, six stub sections, every stub honestly
      marked "To be written.", the factual where-things-live table
      lists six destinations.
* [x] Settings (`/settings`): Design theme first -- Dark selected
      (the theme that ships), Light honestly disabled and tagged
      "planned"; the config editor stays reachable from here.
* [ ] Dragging the console by its header and native corner-resize
      are exercised interactively (desktop session); the automated
      run pins persistence of whatever geometry it gets, not the
      gestures themselves -- and `scripts/probe_ui_bugs.py` now
      drives resize -> move -> resize automatically and asserts the
      window keeps its size (the 2026-10-01 shrinking-window bug).

## 8. Dataset add-data + edit modes -- scenario H (M8e)

Read-only walk of the M8e flows against the real library: the dialog
opens and validates but never posts, the editor dirties and reverts
but never saves, the multi-edit panel renders but never applies.

* [x] Card entry point: a dataset card's "Add data" opens the dialog
      bound to that dataset; Generate is the default tab; the form
      exposes its option set (prompt list, cfg/steps/t ranges, batch,
      conditions/samples, latent size, prediction type); the total
      preview computes `conditions x samples` and the latent size
      renders its pixel equivalent.
* [x] Local validation: an empty checkpoint refuses inline with the
      error box visible and the dialog still open; the scenario
      watches requests and asserts **zero** task POSTs left the
      browser.
* [x] Import tab: swaps panels, resize mode carries a written
      description, the max-aspect-ratio knob is hidden until the
      `fit` (split) mode makes it relevant; the import option set
      (dir, recursion, resize mode, latent size, prediction type,
      negative prompt, seed) renders.
* [x] Edit mode: the toolbar toggle marks the grid and shows the
      per-card edit affordance; clicking a card opens the advanced
      editor bound to that item with read-only metadata and its
      position in the walk.
* [x] Dirty tracking: editing the prompt enables Save, Revert
      restores the snapshot (Save disabled again), next moves to the
      following item, close works from both the walk and a
      prompt-click (Browse mode reaches the same editor).
* [x] Multi-edit: two checkboxes raise the panel with exactly four
      field segments; the CFG and Verdict value rows swap in on
      demand; counts (selected / apply-target) follow the selection;
      clearing selection hides the panel; still zero task POSTs and
      no confirmation dialog fired (the walk deleted nothing).
* [x] Selection is edit-mode-only: browse renders no item checkboxes
      and hides select-all; switching to browse mid-selection hides
      the panel without dropping the selection, and switching back
      restores it (2 selected).
* [x] Screenshot `datasets_edit.png` captured.

## 9. Record

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

## Findings from the 2026-10-01 post-M9 user reports (all fixed same day)

Reported from a live desktop session; reproduced first with
`scripts/probe_ui_bugs.py`, then pinned in the smoke.

8. **System console shrank while being dragged; resize seemed
   ignored** -- `apply()` writes border-box `width`/`height` (the global
   `box-sizing: border-box`), but the ResizeObserver synced from
   `clientWidth`/`clientHeight` (border excluded). Every move ->
   observer -> move round-trip wrote the border-less size back as
   border-box: **-2px per cycle**, so dragging a 566px window by ~12
   steps left it at 542px, and a fresh native resize was ratcheted down
   the next time the window moved. A second defect compounded it: the
   observer clamped stored `x` without moving the element, so stored
   state diverged from the visible position and the first drag snapped
   the window. Fixed: observer reads `offsetWidth`/`offsetHeight`
   (border-box, exactly what `apply()` writes) and never clamps; the
   drag starts from the element's live `offsetLeft`/`offsetTop`.
   Probe pins: move keeps size, resize-after-move sticks.
9. **Item selection checkboxes collapsed to a ~2px sliver** -- the
   design-system base rule `input[type="checkbox"]` (specificity
   0,1,1) outranked `.ds-item-check` (0,1,0), so the checkbox never got
   `position: absolute` and stayed a flex child of `.ds-thumb`, where
   the preview image's flex pressure crushed it (2 x 26px measured on
   all 40 cards). Fixed with a contextual selector `.ds-thumb
   .ds-item-check` (0,2,0): absolute top-left, fixed 18px hit target,
   same visual language as "select all". Probe pins 18x18 on every
   card.
10. **Browse mode exposed the selection tools** -- item checkboxes and
    select-all rendered in browse mode; clicking one raised the
    multi-edit panel while the toolbar still read "Browse". Fixed by
    contract: selection (checkboxes, select-all, bulk panel) renders
    only in edit mode; a live selection is preserved (not destroyed)
    across mode switches. Smoke pins browse rendering zero checkboxes,
    select-all hidden, the panel hidden mid-selection, and the
    2-item selection surviving the round-trip.

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
