# 06 -- Browser findings and coverage

The missing test layer. `backend/tests/run_all.py` and the full gate cover
the API, the SSE frame contracts, and that the pages and assets *serve*.
None of that runs JavaScript, and a page whose module wiring is broken
serves perfectly. A real browser is the only thing that catches it.

That layer is `backend/tests/visual_smoke.py`, whose invocation is below.
It prints its own check count on completion, so the number is never a
figure quoted in this document. It needs a live server plus the
Playwright venv, which is why it is not part of `scripts/full_gate.sh`.

What the browser layer found is recorded below; what it does not yet
check is listed under *Coverage not yet checked*.

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

## Coverage not yet checked

* **Graph executions**
  * [ ] The smoke test never starts an execution. It checks the panel is
        rendered (`#exec-list`), nothing more: no run is launched, so no
        node result, progress frame, terminal status or stop button is
        ever driven through the UI.
  * [ ] **This is not only a missing test -- the page makes it awkward.**
        `/graph` polls `/api/v1/graphs/executions` every 2.5 s while any
        execution is non-terminal, and holds an event stream open, so
        **Playwright's `networkidle` can never fire while a run is
        active**. Measured: idle, `networkidle` on `/graph` settles in
        0.6 s (3 of 3); with a 3200-node run going, it times out every
        time (3 of 3 at a 12 s limit), while
        `domcontentloaded` + `wait_for_selector('.rail')` takes 0.2 s.
        The nine `networkidle` waits in `visual_smoke.py` are therefore
        only safe because no run is active. Any section that starts a run
        must wait on a concrete element instead, and must stop the run
        before the next `networkidle`.
* **Monitor dashboard**
  * [ ] **needs a live producer**: points appear live, mid-stream reload
        replays history, `clear` empties the chart, CSV export + series
        toggles on real data (the M6 harness covered the frame contract;
        real-data rendering waits for a producer that emits real frames).
* **Config editor**
  * [ ] `PUT /config/raw` rejection (invalid TOML -> 422, file
        untouched) is API-tested (`test_config.py`); not driven in the
        browser.
* **Dataset manager**
  * [ ] Item mutations (prompt edit, good/bad toggle, discard, bulk
        apply, commit-to-set, **set card preview**) are covered by
        `test_api_datasets.py` at the API level only -- the smoke edits
        nothing and applies nothing; scenario H exercises the
        editor/multi-edit UI in a read-only walk (dirty -> revert, rows
        rendered, never saved).
  * [ ] Task start/stop is never fired from the smoke (it would spawn a
        real child process against user checkpoints); scenario H pins
        that the browser sends no task POST during its walk.
* **Shell**
  * [ ] Dragging the console by its header and native corner-resize
        are exercised interactively (desktop session); the automated
        run pins persistence of whatever geometry it gets, not the
        gestures themselves -- and `scripts/probe_ui_bugs.py` drives
        resize -> move -> resize automatically and asserts the window
        keeps its size (the shrinking-window bug, finding 8 below).

## Two rules for this checklist

Fix regressions in the milestone that owns the code -- do not adjust the
checklist to match broken behavior.

The desktop app's `screenshot` tool has served stale frames: byte counts
matched fresh captures while the images showed earlier states. DOM
inspection via `evaluate` and the console reader were authoritative. This
only matters for the desktop browser, which is for viewing; the Playwright
suite takes real frame-buffer screenshots from headless Chromium and is
the driver for everything below.

## Bugs this checklist found

Twelve, all fixed. Two routes in: the Playwright suite above catches them
on a run, and `scripts/probe_ui_bugs.py` reproduces the ones a user
reports from a live desktop session before they are pinned in the smoke.
Each entry records the root cause, because in every case the symptom was
in one layer and the cause was in another -- a stylesheet specificity
fight, a geometry round-trip, or a port that renamed something the other
half of the port still expected.

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
4. **Stale `_suppressClick`** -- a wire released
   off-canvas left the flag set, eating the next canvas click. Fixed:
   any new `pointerdown` clears it.
5. **"Continue from" select rendered blank** -- the placeholder option was
   `disabled` without `selected`, leaving `selectedIndex` at `-1`, so
   Chromium displayed nothing. Present since the M6 port (visible in
   the original screenshots). Fixed: placeholder gets
   `disabled: true, selected: true`.
6. **Config widgets had no `id`** -- labels
   pointed at `#f-<dotted id>` but `renderForm` never assigned
   `input.id`, so label association was broken and the smoke's
   `select_option("#f-tuning-method")` timed out. Fixed at the render
   site.
7. **`live` overlay written under dotted keys** -- `onEdit` set `live["tuning.method"]` literally while
   `getDeep(live, "tuning.method")` reads `live.tuning.method`, so
   `visible_when` kept evaluating the stale loaded value (rank stayed
   visible after switching to `distillation`); the initial overlay was
   also a shallow copy that would have let edits mutate `values`. Fixed
   with `setDeep` writes + a JSON deep copy on load.


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
11. **Node search box overflowed the palette rail** -- `.ed-search` carries `.cfg-input`, whose
    `width: 100%` was combined with horizontal margins, so the box was
    1.2rem wider than the 280px rail and stuck out over the canvas;
    the form-sized padding/type also looked oversized next to the
    palette rows. Fixed: `width: auto` + `align-self: stretch` (fills
    the rail minus its margins) with compact padding/font-size. Same
    pass removed the page-local Console section (notes now ride the
    floating console, the shell's documented contract), made both rails
    collapsible from the toolbar (grid tracks animate to 0 so the
    canvas truly grows; state persisted), and turned Executions into a
    drawer collapsed by default -- canvas went from ~66% to 91% of the
    viewport height.


12. **Consecutive palette drops stacked on one point, making wire
    starts ambiguous** -- with
    the view centered on the origin, `dropPosition` produced negative
    coordinates and `state.addNode` clamped them with `Math.max(0, ...)`,
    so every drop landed at exactly (0, 0); a wire dragged from that
    stack resolved to the *topmost* overlapping node and rejected itself
    as "same node". Even without the clamp the old 30px jitter kept
    230px-wide nodes ~87% overlapped. Fixed: the clamp is gone (the
    plane is infinite in all directions, matching the drag path and the
    free-form library layout), and drops cascade 260px apart in a
    3-column grid, always landing inside the viewport. Related fix in
    the same pass: pointer/click/dblclick handlers moved from
    `#canvas-inner` to the viewport -- once the plane is panned, strips
    of the viewport are no longer covered by inner's box and must still
    pan, deselect, and recenter like empty plane. Smoke pins: overflow
    hidden with 0px scrollbar chrome, the origin circle centered (0px
    off), wheel and background-drag pans changing the transform, the
    drop cascade + in-view landing, a wire forming across the
    transformed plane, and a negative `left` after dragging past the
    origin.

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
