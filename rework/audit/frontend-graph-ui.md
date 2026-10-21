# Audit: frontend graph UI (read from code, origin/main a7cc71e)

Method: read `frontend/js/editor/*.js`, `lib/events.js`, `templates.js` header,
and the design docs they cite. Line references are to those files.

## 1. Module map

| File | Lines | Job |
|---|---|---|
| `editor.js` | 406 | page wiring: catalog load, toolbar, run, memory panel, graph settings section |
| `editor/canvas.js` | 670 | infinite plane, node DOM, sockets, wires (SVG), selection, drag, wire drag, execution badges, live diagnostics |
| `editor/widgets.js` | 322 | the ONLY place a port becomes a control (select / checkbox / number / JSON textarea / text / path picker with upload) |
| `editor/state.js` | 288 | `GraphDoc`: nodes, edges, `memory`, payload forms, load/save shapes |
| `editor/executions.js` | 231 | run lifecycle list, per-node badges from SSE, stop, wipe |
| `editor/inspector.js` | 227 | side panel: same widgets, id rename, delete, "Run diagnostics" |
| `editor/library.js`, `palette.js`, `assets.js`, `draft.js` | 156/98/72/70 | saved graphs, node palette, asset catalog, localStorage draft |
| `editor/settings.js` | 81 | "Graph settings" section; fields come from the backend |
| `lib/events.js` | 133 | the one SSE client (`/api/v1/events`), with replay and resync |
| `templates.js`, `board/*` | 556 + ~1700 | templates page and monitor windows (blocks bound to node modules) |

## 2. How a node is drawn

Generic, from catalog metadata (`cls.inputs`, `cls.outputs`: name, type,
`type_mro`, default, doc, required, `visible_when`, `choices`, `path_kind`,
`widget_only`, label/group/advanced/unit/min/max/step). There is **no
per-node-type code in the editor**. That is a strength to keep.

`canvas.js:79-117` builds a node as: header (title, badge), two columns (inputs
left, outputs right). Each input row is socket + label + (widget | wire hint
`<- node.port`). A primitive-typed port has both a socket and a widget; a wire
overrides the widget value by design (`state.js:25-27`, docs 05 section 4).

## 3. Grouping, collapsing, visibility

* `Port.group` and `Port.advanced` exist in the catalog and are used by the
  **templates page** and the graph-settings section, but **`canvas.js` ignores
  both**. A node always shows every port row.
* `visible_when` is honoured (`widgets.js:40-49`, `canvas.js:157`): a gated row
  gets class `gated` and no widget. Its socket is still drawn.
* **There is no collapse of any kind** on a node (no `collapsed` field in
  `GraphDoc` nodes, no UI). The owner's complaint ("minimized node shares one
  big point") is about a feature that does not exist in this editor; it
  probably refers to the legacy editor or to the templates board, where blocks
  collapse (`board/layout.js`). Either way the requirement stands: collapse is
  new work, and it must keep sockets alive.
* Consequence for the 31-port managed trainer: the node is one very tall card.

## 4. Live update: what exists and what is missing

Exists:

* **Server-computed content on a node**: `has_diagnostics` classes get a
  debounced `POST /graphs/nodes/{class}/diagnostics` after any re-render
  (`canvas.js:366-413`), result lines render under the input row. It is
  sequenced (`_diagSeq`) so a late answer cannot overwrite a newer one. This is
  the closest existing thing to "node shows server result", and it is the
  pattern the Load Model node should extend.
* **SSE with replay** (`lib/events.js`, design `docs/design/backend/09-event-contract.md`):
  lifecycle events are numbered and ring-buffered; reconnect replays or says
  `resync_required`.
* **Per-node execution badges**: `graph_execution_progressed` paints `ok`/`err`
  and duration when a node *finishes* (`executions.js:174`).

Missing:

1. **Re-render on every change.** `doc.changed` -> full `render()` removes and
   rebuilds every `.gnode` (`canvas.js:59-72`). Any node-level live behaviour
   (animations, a chart, a spinner, text being typed) has to survive a rebuild
   by being re-applied from maps (this is why badges and diagnostics live in
   maps). It does not scale to node-embedded widgets.
2. **No "running" or "processing" state.** Badges are set only on completion.
   There is no state for "waiting for the server", and no per-node progress.
3. **Dynamic ports are not drawn.** The backend has `Node.resolve_inputs /
   resolve_outputs(params)` (`nodes/core.py:431`) and the validator uses them,
   and `catalog.resolved_inputs` exists, but the editor draws `cls.inputs` /
   `cls.outputs` from the static catalog. Only the templates page calls
   resolve. There is no `resolve_outputs` endpoint at all
   (`catalog.py:42` has inputs only).
4. **Node-embedded widgets**: display modules (chart/scalar/bars/text/progress)
   are drawn by `board/displays.js` on the templates page only. The graph
   canvas has no slot for them.
5. **Multiple wires into one port: not supported.** `state.js:138` removes any
   existing edge into the same input ("one wire per input, newest wins").
   Answer to the open question in the component-map comment: **no**.
   Several output wires from one output port are fine.

## 5. Templates and monitors vs the graph

(`docs/design/15-graph-modules-and-templates.md`, verified against `templates.js`)

* A template page is a board of blocks. A block's target is
  `{scope: node|graph, node, kind: field|display, name}`. Fields are the same
  controls as on the canvas. Displays are read-only live views by closed
  `kind` vocabulary. Streams: `node:<node_id>` and `graph:run` on
  `/monitor/{stream}/stream`.
* The page **stores a snapshot of the graph inside the template document**
  (SQLite `templates`, migration 009). That is the weak spot the owner senses:
  a template is a copy, so a graph edited later does not change a saved page,
  and the two can drift; "new from a graph" is a one-way copy.
* The monitor is not a node already (`TrainingProgressMonitorNode` is the
  legacy producer-side handle; displays are declared on `Node.MODULES`).
  Monitors as windows exist (`?window=1`).

## 6. Graph-wide settings today

`GraphDoc.memory` (`state.js:67`), stored in the graph, edited in
`settings.js`, field list from `backend/domain/modules.py` `GRAPH_FIELDS`:
`memory.vram_max_mb`, `memory.vram_min_mb`, `memory.strict`,
`memory.ram_max_mb`. So the **graph settings menu already exists, for memory
only.** It is a section inside the editor, not a separate menu, and has no
slot for device, dtype, residency, prewarm, graph capture.
`VRAMBudgetControllerNode` still exists as a node (see backend audit).

## 7. Wire rules

`typesCompatible` (`state.js:46`) is UX sugar over `type_mro`; the server is
authoritative (`POST /graphs/validate`). No wire may start and end on the same
node. Edges and node params are flat; there are no subgraphs, groups of
nodes, or reroute points.

## 8. Serialisation

Library/run/draft payloads: `{format: 1, nodes:[{id,class_name,params}],
edges:[{from_node,from_port,to_node,to_port}], layout:{id:{x,y}}, memory,
description}`. Unknown classes are kept verbatim; validation happens at run
time. There is **no per-node-class version** and no migration table: renaming
a class or a port breaks saved graphs (docs 15 states this is deliberate).

## 9. UI gaps against the owner's wishes

| Wish | State today |
|---|---|
| Node field groups | catalog has `group`/`advanced`; canvas ignores them |
| Collapse without losing wires | no collapse at all |
| Dynamic outputs with "processing" state | resolve hooks exist server-side; editor does not call them; no pending state |
| Node live update | full re-render per change; badges/diagnostics survive via side maps |
| Events driving widgets | SSE and module streams exist; nodes cannot host widgets on the canvas |
| Settings menu | memory-only section exists |
| Several wires into a port | not supported |
