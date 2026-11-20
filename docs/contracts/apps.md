# Apps and boards

An **App** is a published, versioned page over a frozen copy of a graph: the way a workflow is used by someone who should not have to meet the node
canvas. A **board** is the grid of cards that both an App and a graph's *views* (`graph-json.md`) are made of. Code: `backend/domain/board.py`,
`backend/domain/app.py`, `backend/application/app_binding.py`, `backend/presentation/api/apps.py` and `boards.py`; the frontend engine is
`frontend/js/board/`, the page `frontend/js/apps.js`.

## Board

A flat board (a view inside a graph document) is `{mode: "page" | "monitor", columns, blocks}`. A **container** is
`{columns, tabs: [{id, title, blocks}]}`: each tab has its own grid. A card carries either a `ref` (it points at one module) or a `container`
(it is a titled section with one tab, or a tabbed card with several); containers nest at most 3 deep and are not allowed in a flat board.

A card is `{id, x, y, w, h, collapsed, title, style?, options, ref | container}`. `ref` is `{scope: "node" | "graph" | "note", kind: "field" | "display" | "action" |
"note", name, node?}`; `node` is the node uid and only for scope `node`. `style` is one of `plain`, `outlined`, `accent`, `bare`.

Limits: columns 1..24 (default 12); at most 200 cards in a document, containers included; at most 12 tabs in a container; ids `[A-Za-z0-9_-]{1,40}`,
unique across the whole document (tab ids unique within their container); a card must fit its container's grid (`x + w <= columns`); `options` at most
4 KB. Every problem is collected with a path, so one round trip tells the editor everything to fix. Unknown keys survive a round trip.

Card kinds the editor writes (all inside the shapes above, so the backend stores them as they are): a **note** is `ref` scope/kind `note` with
`options.text`; with `options.look: "heading"` it is a one-line heading (the editor adds it with `style: "bare"`, which hides the frame in use mode).
A **section** is a card with a `container` of one tab; a **node form** is a section holding one field card per free, non-advanced field of a node
(ordinary cards: binding and exposure are unchanged). Moving a card into another tab keeps its id, so the document stays valid. The page's own
tabs are `root.tabs`.

Editor-only state is not stored: the selected card, the active tab of each container, undo history (snapshots of `root`, so a value typed into a
field is not undone) and the phone preview.

## App document

```
{format: 1, graph, source_graph, style: {accent, density}, root}
```

* `graph` is the snapshot a run executes (graph document, `graph-json.md`; layout and views are not needed). `source_graph` is the library graph it
  was taken from, or `null`.
* `style.accent` is one of `blue teal green amber red violet`; `style.density` is `comfortable` or `compact`.
* `root` is a container: the app's page. Cards point at modules of the snapshot.

A card whose target is not in the snapshot is kept and shown as an *orphan* (`missing_node`, `missing_module`, `unknown_class`); an app whose graph
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

## REST

| Route | |
|---|---|
| `GET /api/v1/apps` | list: `{apps: [{name, description, version, card_count, node_count, created_at, updated_at}], count}` |
| `PUT /api/v1/apps/{name}` | upsert `{description, doc}`: 201 first save, 200 replace; 422 `app_invalid` |
| `GET /api/v1/apps/{name}` | one app with `doc`, `version`, `graph_hash`; 404 `app_not_found` |
| `DELETE /api/v1/apps/{name}` | `{deleted: true}`; 404 |
| `POST /api/v1/apps/{name}/diff` | body `{graph}`: `{added, removed, changed, wiring_changed, settings_changed, orphaned_cards, new_version}`; a pure read, the review before "update from graph" |
| `POST /api/v1/apps/{name}/run` | body `{values, memory_overrides?}`: 201 execution summary; 422 `app_invalid` / `graph_invalid`, 409 `graph_execution_active` / `memory_unavailable` |
| `POST /api/v1/boards/modules` | body `{graph}`: per node its fields and live displays, and the graph's own: what a card can bind to |
| `POST /api/v1/boards/resolve` | body `{doc, graph?}` (an App document, or a flat board with the graph beside it): per card `ok`, `wired`, `missing_node`, `missing_module`, `unknown_class`, plus the graph's validation issues; 422 `board_invalid` |

| `GET /api/v1/artifacts/{path}` | a PNG a node left for the page (`Preview Image` writes them under `artifacts/<node uid>/`, newest 48 kept per node); 404 `artifact_not_found` for anything else, a path leaving the folder included |

An `image` display module (`DisplayModule(kind="image", value_key=...)`) names the report key that holds `[{path, seed, width, height}]`; the page draws the pictures of the newest frame.

Names are trimmed, 1..120 characters. A server started without app support answers 503 `apps_unavailable`.
