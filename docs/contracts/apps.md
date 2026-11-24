# Apps and boards

An **App** is a published, versioned page over a graph it owns: the way a workflow is used by someone who should not have to meet the node
canvas. A **board** is the grid of cards that both an App and a graph's *views* (`graph-json.md`) are made of. Code: `backend/domain/board.py`,
`backend/domain/app.py`, `backend/application/app_binding.py`, `backend/presentation/api/apps.py` and `boards.py`; the frontend engine is
`frontend/js/board/`, the page `frontend/js/apps.js` (entry) and `frontend/js/apps/` (document, selection, run tracker, tray, toolbar, page controller).

## Board

A flat board (a view inside a graph document) is `{mode: "page" | "monitor", columns, blocks}`. A **container** is
`{columns, tabs: [{id, title, blocks}]}`: each tab has its own grid. A card carries either a `ref` (it points at one module) or a `container`
(it is a titled section with one tab, or a tabbed card with several); containers nest at most 3 deep and are not allowed in a flat board.

A card is `{id, x, y, w, h, collapsed, title, style?, options, ref | container}`. `ref` is `{scope: "node" | "graph" | "note", kind: "field" | "display" | "action" |
"note", name, node?}`; `node` is the node uid and only for scope `node`. `style` is one of `plain`, `outlined`, `accent`, `bare`; unset, a multi-line text field is `bare` and everything else `plain`.

Limits: columns 1..24 (default 12); at most 200 cards in a document, containers included; at most 12 tabs in a container; ids `[A-Za-z0-9_-]{1,40}`,
unique across the whole document (tab ids unique within their container); a card must fit its container's grid (`x + w <= columns`); `options` at most
4 KB. Every problem is collected with a path, so one round trip tells the editor everything to fix. Unknown keys survive a round trip.

Card kinds the editor writes (all inside the shapes above, so the backend stores them as they are): a **note** is `ref` scope/kind `note` with
`options.text`; with `options.look: "heading"` it is a one-line heading (the editor adds it with `style: "bare"`, which hides the frame in use mode).
A **section** is a card with a `container` of one tab; a **node form** is a section holding one field card per free, non-advanced field of a node
(ordinary cards: binding and exposure are unchanged). Moving a card into another tab keeps its id, so the document stays valid. In the editor, Ctrl/Shift-click selects several cards of one tab and **Put them in one box** wraps them in a new section (one tab, cards as arranged) that takes their place (`groupCards`, model.js); a box then gets more tabs from its strip or the inspector. The page's own
tabs are `root.tabs`.

Every card is a box with a compact **title tab** on its top-left corner (folder-tab shape: rounded on top, square where it meets the box); `options.tone`
(`blue teal green amber red violet gray`, absent = neutral) colours the tab and outlines the box in the same colour. `style` changes the box (`plain`,
`outlined`, `accent`; `bare` has no frame and, in use mode, no tab).

A **field card** draws the control the graph editor would draw for that port (`frontend/js/graph/widgets.js`, one place for both): a model file is chosen
from the server's list, a prompt is a text area, a number has its bounds. The backend sends each field with the port's `ui` hints so the page can choose.
Its layout (`fieldLayout`, model.js): a one-line value **inside a box** is a *row*, "name: value" on one line without a tab (as in a form; a status
sentence a node reports is a row too); a one-line value **on the page** has the title tab and the value in the box below; a text area or list is always the
tab over a box, and the text area is the box itself. `options.layout` (`inline` | `stacked`) overrides this; `options.hideLabel` leaves the name out.

Editor-only state is not stored: the selected card, the active tab of each container, undo history (snapshots of `root`, so a value typed into a
field is not undone) and the phone preview.

## App document

```
{format: 1, graph, source_graph, style: {accent, density}, root}
```

* `graph` is the app's own graph, a graph document (`graph-json.md`) with its layout and views: what a run executes, and what the graph editor's
  *app mode* opens and saves (`PUT .../graph`). It starts empty or as a copy of a library graph; after that nothing links it to the library, and
  changing either never reaches the other. `source_graph` only remembers which library graph the copy was made from (or `null`).
* `style.accent` is one of `blue teal green amber red violet`; `style.density` is `comfortable` or `compact`.
* `root` is a container: the app's page. Cards point at modules of the snapshot.

A card whose target is not in the graph is kept and shown as an *orphan* (`missing_node`, `missing_module`, `unknown_class`); an app whose graph
changed must still load.

## Versions

`graph_hash` is the SHA-256 of what a run would do with the snapshot: each node's uid, class, params and addons (sorted by uid), the edges and event edges
in order, the settings and the graph addons. Layout, views and titles are not in it. Saving an app whose hash differs from the stored one makes the next
`version`; a layout, style or description change does not. A new app is version 1.

## Using an app

Running an app changes only the fields its cards expose: `values` is `{"nodes": {uid: {port: value}}, "memory": {key: value}}`, a value of `null` clears
the parameter, a field that is wired from another node or not exposed by a card is refused (`app_invalid`, every problem in `details`). The run is the
same `StartGraphExecution` as the editor's Run and records its **origin**: `{app, version, values}`, returned as `origin` on every execution summary
(`null` for runs not started from an app).

## Actions

An **action card** (`ref.kind: "action"`, scope `node`, name = the action) is a button for something a node declared in `ACTIONS` (`graph.ActionModule`): "Save now" on the
trainer. Ownership: the node that holds the state declares it; what it does to other nodes goes through ordinary events (the trainer answers with a
`CheckpointDue` that any Save Adapter in the graph writes). `boards/modules` lists each node's `actions` with `available` and, when false, a `reason`
(`needs` names the event type some other node must listen for; nothing listens, the card shows the reason and stays disabled). The button works only while
the run is `running`. Pressing it calls `POST /api/v1/graphs/executions/{id}/actions` with `{node, action}` and nothing else (202 `{command_id, status: "queued"}`;
404 `action_not_found`; 409 `graph_execution_not_active`, `action_unavailable`); the event is built inside the run from the declaration, never from the request.
The node's answers come back as frames `{kind: "action", id, action, status: queued | delivered | rejected | failed, detail}` on `node:<uid>`, which the card shows.
The draft builder puts every available action on a new page. How a command reaches the run: `events.md` (Commands).

## Status, results and runs

Two more display kinds read what a node reports, kept by the page's stream state (the newest value of each key, so a frame that lacks it does not blank it):

* `line` (`DisplayModule(kind="line", value_key=...)`): one live sentence. The trainer says what it is doing (`Preparing...`, `Training · step N of M · x s/step`,
  `Finished at step N of M`), Save Adapter the file it wrote last, Load Model what it split the checkpoint into.
* `result` (`DisplayModule(kind="result", series=(Series(key, label, unit), ...))`): the newest value of each named key as a small list, text as it came: where
  the trainer got to (step, total, last loss) and the newest adapter file with how many were written.

A node reports a sentence as a key of an ordinary frame on `node:<uid>` (`{step, status: "..."}`); a frame must carry `step` for the stream reader to keep it.

Two graph displays are about runs, not a node, and read persisted state the page polls (`GET /api/v1/graphs/executions`), not a stream:

* `history` (graph display): the runs this app started, newest first (`?app=<name>` limits the list to runs whose origin is that app). Each row has the status,
  time, duration and the first values it was given; **Run again** puts the run's recorded `origin.values` back on the page (only for cards the page still has;
  `null` clears, a key the run did not record is left alone). It does not start anything.
* `queue` (graph display): runs waiting for memory (`planned`), in the order they will start, each with **Cancel** (`POST /graphs/executions/{id}/stop`).

The page's **Add to queue** button runs with `mode: "queue"` (`POST /apps/{name}/run`), which waits behind whatever lives or waits instead of starting beside it.
When the run the page follows has ended and the next run of the app has started, the page follows that one.

The draft builder (`autoPage`) sorts a new app into tabs: **Run** (inputs, action buttons, run status, sentences, results, progress, images), **Charts** (charts,
bars, scalars, memory) and **Runs** (history, queue); a tab nothing belongs on is left out.

## REST

| Route | |
|---|---|
| `GET /api/v1/apps` | list: `{apps: [{name, description, version, card_count, node_count, created_at, updated_at}], count}` |
| `PUT /api/v1/apps/{name}` | upsert `{description, doc}`: 201 first save, 200 replace; 422 `app_invalid` |
| `GET /api/v1/apps/{name}` | one app with `doc`, `version`, `graph_hash`; 404 `app_not_found` |
| `DELETE /api/v1/apps/{name}` | `{deleted: true}`; 404 |
| `PUT /api/v1/apps/{name}/graph` | body `{graph}`: replaces the app's graph, page, style and description stay: `{app, orphaned_cards}` (the cards whose target the new graph no longer has); a changed run makes the next version, a moved node does not; 404 `app_not_found`, 422 `invalid_query` |
| `POST /api/v1/apps/{name}/run` | body `{values, memory_overrides?, mode?}` (`auto` default, `parallel`, `queue`): 201 execution summary; 422 `app_invalid` / `graph_invalid`, 409 `graph_execution_active` / `memory_unavailable` |
| `POST /api/v1/boards/modules` | body `{graph}`: per node its fields and live displays, and the graph's own: what a card can bind to |
| `POST /api/v1/boards/resolve` | body `{doc, graph?}` (an App document, or a flat board with the graph beside it): per card `ok`, `wired`, `missing_node`, `missing_module`, `unknown_class`, plus the graph's validation issues; 422 `board_invalid` |

| `GET /api/v1/artifacts/{path}` | a PNG a node left for the page (`Preview Image` and `Sampling Preview` write them under `artifacts/<node uid>/`, newest 48 kept per node); 404 `artifact_not_found` for anything else, a path leaving the folder included |

An `image` display module (`DisplayModule(kind="image", value_key=...)`) names the report key that holds `[{path, seed, width, height}]`; the page draws the pictures of the newest frame.

Names are trimmed, 1..120 characters. A server started without app support answers 503 `apps_unavailable`.
