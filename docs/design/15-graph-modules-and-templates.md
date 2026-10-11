# Graph modules and the Templates tab

*Built 2026-10-10. Answers the five open questions in
[`14-node-attached-monitor-widgets.md`](14-node-attached-monitor-widgets.md)
and goes one step past it: the same mechanism that lets a node offer a live
view also lets it offer an editable field, so one board engine serves the
Templates tab (pages for people who should not meet the graph) and monitor
windows.*

## What this is for

The node graph stays: it is where graphs are authored, and nothing about it
is replaced. The **Templates tab** is a constructor on top of it. A template
is a *page* — a board of blocks, each one pointing at something the graph
offers — plus a snapshot of the graph those blocks belong to. A beginner opens
a page, changes the few values it exposes, presses Run and watches the live
blocks. They never see a node.

Two modes, one document:

| | **use** (default) | **edit** |
|---|---|---|
| fields | editable | editable |
| live views | live | live |
| layout | fixed | drag, resize, collapse, rename, remove |
| tray | hidden | everything the graph offers and the page does not yet show |

Hiding a setting is just not placing it: it keeps the value the page's author
saved. A **monitor window** is the same document with `kind: "monitor"`: only
live-view blocks, no Run, optionally popped out (`?window=1`) onto a second
screen.

## The model: modules

A *module* is one self-describing thing a node (or the graph) can put on a
page.

* **Field modules** — one per settable input `Port`. Derived automatically; a
  node author writes nothing. `Port` gained presentation hints (`label`,
  `group`, `advanced`, `unit`, `minimum`, `maximum`, `step`) which refine how
  it is drawn and which the graph editor honours too.
* **Display modules** — declared in `Node.MODULES` (`nodes/ui.py`
  `DisplayModule` / `Series`): a read-only live view with a `kind` from a
  closed vocabulary (`chart`, `scalar`, `bars`, `text`, `progress`) naming the
  report keys it plots.
* **Graph-scope modules** (`backend/domain/modules.py`) — the graph's own
  settings (`memory.vram_max_mb`, `vram_min_mb`, `strict`, `ram_max_mb`) and
  views (`run` status, device-memory chart, memory scalar). They are data in
  the same shape as a node's, not nodes: a node would have to sit in the
  graph, take part in validation and be wired.

Adding a metric to a node is "add a number to the report and one `Series` to
`MODULES`". No frontend edit.

## Answers to the open questions in doc 14

1. **Descriptor or registered component?** Both, split at the seam: the
   backend sends a *descriptor* (what the data is: kind, series keys, scale,
   x axis, reference key, suggested size); the frontend owns *how each kind is
   drawn* (`frontend/js/board/displays.js`). The `kind` vocabulary is closed,
   so the backend stays out of pixels and an unknown kind degrades to a
   labelled placeholder. A new module of an existing kind needs no frontend
   change; a new kind is one function. Per-phase timings, whose keys the
   trainer builds at run time, are declared with a `pattern` (`*_ms`) and
   discovered from the stream.
2. **Identity.** Per *node instance* plus module name. A block's target is
   `{scope, node, kind, name}`; the stream is `node:<node_id>`; graph-scope
   modules use `graph:run`. "The loss chart of *this* trainer" is
   unambiguous when a graph has two. Renaming a module or a port breaks saved
   pages exactly as renaming a class breaks saved graphs — deliberately (see
   4).
3. **Sampling.** Per module: `cadence` (`step`/`run`) is declared; the page
   downsamples at draw time (`board/series.js` `decimate`) by keeping, per
   bucket, the rows holding each series' minimum and maximum plus the first
   and last. Original rows only, nothing averaged, so a one-step loss spike
   survives. Rows are capped per stream (50 000, oldest dropped, the count
   shown). A key absent from a report — or non-finite — is a gap, never zero.
4. **Missing targets.** They degrade visibly and are *kept*. `POST
   /templates/resolve` answers block by block: `ok`, `wired` (the graph now
   feeds that field, so it is read-only), `missing_node`, `unknown_class`,
   `missing_module`. A missing block renders as an orphan naming what is
   gone; the page still loads and still runs.
5. **Where does the composition live?** In a saved *template document* (SQLite
   table `templates`, migration 009), per name, with the graph snapshot inside
   — so a page is portable as a unit and a graph edited later cannot change
   what a saved page does. Blocks pointing at things the graph no longer has
   are question 4. Presets of the monitor kind are the same documents.

## Document

```
{format: 1, kind: "template"|"monitor", columns: 1..24, source_graph, graph,
 blocks: [{id, ref: {scope: "node"|"graph", node?, kind: "field"|"display", name},
           x, y, w, h, collapsed, title, options}]}
```

Validated by `backend/domain/template.py` (every problem reported, with a
path); stored verbatim so keys a newer frontend adds survive an older backend.

## Running a page

`POST /api/v1/templates/{name}/run` takes `values` — `{"nodes": {id: {port:
v}}, "memory": {key: v}}` — and applies them to the saved graph **only for
fields the saved page exposes** (`application/template_binding.py`). A value
for a hidden field, a wired field or a missing node is refused (422), not
applied, so a stale tab or a hand-built request cannot reach past the page.
It is then the *same* `StartGraphExecution` as the editor's Run (one shared
single-active lock), so validation, admission and error codes are identical.
A value typed on the page is used for the run without being saved; Save
stores layout and values.

The page disables Run for exactly one client-side reason the server would
also refuse: the page shows a field the saved page does not expose (save
first).

## API

| | |
|---|---|
| `POST /templates/modules` | what a graph offers a page (fields, live views, graph modules) |
| `POST /templates/resolve` | block-by-block binding status of a draft + graph validation issues |
| `GET /templates?kind=` · `PUT/GET/DELETE /templates/{name}` | the library |
| `POST /templates/{name}/run` | run with typed values |
| `GET /monitor/{stream}/stream` | existing SSE; now also `node:<id>` and `graph:run` |

Error codes `template_not_found`, `template_invalid`,
`templates_unavailable` are in `docs/design/backend/02-api-reference.md`.

## Frontend

Pure, unit-tested modules (`node --test frontend/tests/*.test.mjs`):
`board/layout.js` (grid with push-down and compaction, move/resize/collapse,
keyboard nudge), `board/model.js` (the tray, auto-fill from a graph, field
values in the draft graph, what Run will send, dirty detection),
`board/series.js` (frame handling, series discovery, decimation, progress and
ETA). DOM: `board/board.js` (grid, drag/resize/rename; blocks are re-placed,
never rebuilt, when the layout moves, so a chart keeps its data and a field
its focus), `board/displays.js`, `board/fields.js` (the editor's own
`buildWidget`, so a value looks the same on the page and on the canvas),
`board/live.js` (one `EventSource` per stream, shared by every block on it),
`templates.js` (the two routes). On a narrow screen the grid is one column in
reading order and drag is off.

A reconnecting stream is replayed in full by the server (state frames
included), so the page empties a stream on `connected`; otherwise a
reconnect would draw history twice. ETA/rate use arrival times and are
withheld when reports arrive in a burst (a page opened late is replayed the
whole history at once).

## Graph settings and the memory budget

Two problems the same work surfaced:

* **A graph had nowhere to say how much memory it may use** — the editor sent
  only nodes and edges. `GraphDoc.memory` now exists, is stored in the graph
  (library save/load, run, template snapshot) and is edited in a *Graph
  settings* section of the editor, drawn from the backend's
  `graph_fields` by the same controls the Templates tab uses.
* **Default runs were refused to start.** With `vram_max_mb: "auto"` and no
  remembered peak, the exploratory claim was *all* of capacity, and the
  child's physical check refused whenever the card had anything else on it
  (a desktop, a browser). *Adaptive* mode: when the demand is unknown and the
  budget is `auto`, the child shrinks the grant to `free − 256 MB` if that is
  at least `max(vram_min, 1 GB)` and only otherwise refuses, with the numbers.
  An explicit number is never shrunk. (`graph_task_worker.py`,
  `GraphMemory.shrink_to`.)

## What was verified, and how

Evidence classes, because they are not equal:

* **Real browser, fixture nodes, CPU** (headless Chromium against the real
  app with toy nodes and the real SSE bus): create a page from a graph, edit a
  value, save, reload (value persists), run with an *unsaved* value (the run
  used it; the saved one was untouched), live progress/ETA/chart, status
  block through to `finished`, timing bars discovered from a pattern, drag /
  resize / Escape-cancel / keyboard nudge / collapse / rename / remove / add
  from the tray / advanced toggle, layout persisting across save and reload,
  phone width (one column, no overflow), orphan / wired / unknown-class /
  missing-port blocks, a monitor window popped out and receiving a run started
  elsewhere, graph settings round-tripping through the library and Run.
* **Unit tests** — 118 frontend (`node --test`), `backend/tests/
  test_templates.py` (100 checks: descriptors, document validation, binding,
  resolve statuses, the HTTP API, a real template run).
* **Code read only** — why default runs were refused (the diagnosis above was
  read from the code, and the adaptive path is tested with a fake probe, not
  on a card).

Not verified: anything on real hardware or with the real trainer; memory use
during graph *initialisation* (loading the UNet/CLIPs) — the budget governs
what the ledger knows about, and the load peak was not measured; the
`graph:run` memory chart was drawn from synthetic frames in the browser test
(the real frames are covered by supervisor tests, not by a GPU run).

## Known limitations

* A dynamic node's fields (`resolve_inputs`) are described from the params at
  describe time; changing the param that reshapes them needs a re-resolve
  (the page does this after edits, debounced).
* No `advanced` hints are set on the real nodes yet, so the starter page for
  a real trainer graph leans on "fields the author set" and "required with
  no default" to stay short.
* No seeded starter template ships: a correct training graph cannot be
  authored blind. "New from a graph" builds one from any saved graph.
* One run is active at a time, so `graph:run` is one stream; per-run
  history of a stream is not kept after the next run starts.

## Related

- [`14-node-attached-monitor-widgets.md`](14-node-attached-monitor-widgets.md)
  — the proposal this implements.
- [`docs/design/backend/09-event-contract.md`](backend/09-event-contract.md).
