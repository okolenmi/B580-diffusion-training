# 60. Templates and monitors: from "a copy of a graph" to views over a live one

Status: **proposal.** The owner: monitor is not something to attach to the graph;
nodes own their UI parts; presets can be **subscribers of graph events** that
update widgets; the "graph -> templates" design "is still not good enough, think
more". Needs 10 (events), 20 (settings), 30 (node displays). Inputs:
`docs/design/14-node-attached-monitor-widgets.md`, `15-graph-modules-and-templates.md`,
`audit/frontend-graph-ui.md` section 5.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What exists (so we do not redo it)

* **Modules**: a node offers *field* modules (one per input port) and *display*
  modules (`Node.MODULES`: chart, scalar, bars, text, progress). The graph offers
  its own (`memory.*` fields; `run`, `memory`, `memory_now` displays).
* A **template** is a document `{format, kind: template|monitor, columns,
  source_graph, graph, blocks[]}`; each block targets `{scope, node, kind, name}`.
  Stored in the `templates` table. **The graph snapshot is inside it.**
* Streams: `node:<id>` and `graph:run` on `/monitor/{stream}/stream`
  (`monitor_bus.py`). The board engine, field and display renderers are shared
  with the canvas. Monitor windows pop out (`?window=1`).
* Legacy: trainers still have a `monitor: MonitorHandle` port and
  `TrainingProgressMonitorNode` exists, while displays are injected
  automatically (`_attach_module_stream`).

## 2. What is wrong with "graph -> templates"

1. **Two copies of the truth.** Editing the saved graph does not change a saved
   page; "new from a graph" is a one-way copy. A page can run a graph that no
   longer matches the one the person is editing.
2. **Brittle addressing.** Blocks target `(node id, port name)`. Node ids are
   user-editable (`renameNode`), and renaming does not update pages, so a rename
   orphans blocks. `resolve` then reports `missing_node` for a node that merely
   changed its label.
3. **Two jobs in one document.** A template is both "a way to start a graph"
   (a recipe) and "a page to operate and watch a graph" (a view). The code even
   says "no seeded starter template ships: a correct training graph cannot be
   authored blind."
4. **Monitors are a second document kind** with no link to a run; history of a
   stream is gone when the next run starts.
5. Pages cannot see **events** or **settings** beyond the memory fields, and
   cannot act (no button).

## 3. The split: Graph, View, Starter

| Concept | What it is | Persisted as |
|---|---|---|
| **Graph** | the executable document: nodes, data edges, event edges, settings, layout | library graph (exists) |
| **View** | a board of blocks bound to *this* graph, in a mode: `page` (fields + displays + Run) or `monitor` (displays only) | **inside the graph document**, key `views` (no copy) |
| **Starter** | a graph factory: a skeleton plus a few questions ("family", "dataset", "steps"); instantiating **creates a new graph** | files shipped in the repo, plus user-saved starters |

* One truth: the page edits the graph's own fields. Open the editor and the page
  at once and they show the same values.
* "A preset that only subscribes to events" is a **View**: a set of blocks that
  subscribe to display modules, event topics and settings of a graph.
* "A way to begin" is a **Starter**. Making a graph from a starter is an explicit
  copy the person sees ("New graph from starter"), which is the honest version
  of today's "new from a graph".
* An **App** is a graph that has a default page view, opened in *use* mode at
  `/apps/<name>`: what a beginner sees. It is not a separate document type.

## 4. Stable addressing

Nodes get a **`uid`** (short random id, assigned at creation, never edited)
separate from the editable display id. Blocks, event edges and layout refer to
`uid`. Rename changes the label only; the edge list can keep display ids or move
to uids too (decision V2). Every node has one from creation (no old graphs are loaded, plan 90). Resolve statuses (`ok`, `wired`,
`missing_node`, ...) stay.

## 5. Blocks: what a view can show or do

| Kind | Binds to | Status |
|---|---|---|
| `field` | a node field or a graph setting (the dialog's fields, plan 20) | exists, extended to settings |
| `display` | a node display module or a graph display | exists |
| `status` | `node_state` of one node, or the run state | new (events) |
| `event-feed` | a filtered stream of events (plan 10) | new |
| `note` | markdown text | new |
| `image` | a preview or probe image from a node | new |
| `button` | an action: Run, Stop, or **trigger an event-in** | new; the trigger needs a control channel into the child process (see 8) |

A block subscribes through the **same streams** as now; a new kind is one
renderer function in `board/displays.js` plus a descriptor, as today.

## 6. Monitors are views over runs, not nodes

* A **monitor view** is a view in `monitor` mode. It is attached to **a run**: the
  live one by default, or a past one. Opening it in a second window keeps working
  (`?window=1`).
* **Remove the monitor wiring from graphs**: the trainer's `monitor` port, `on_step`
  and `TrainingProgressMonitorNode` go away once the displays cover what the
  legacy monitor shows (task: list the report keys the legacy path consumes and
  confirm each has a `Series` in some node's `MODULES`; **not yet checked**).
* **Persist per-run metrics** (downsampled, to the run's folder, e.g. one JSON
  Lines file per stream). Then a monitor can open an old run and two runs can be
  overlaid. Today a stream's history ends when the next run starts
  (doc 15, known limitations). Optional but cheap, and it makes "compare this run
  to the last one" possible.

## 7. How the pieces connect

```
 Node (declares MODULES, EVENTS_OUT, fields)
        │ runs in child process
        ▼
 EventHub / report handles ──► MonitorBus streams  node:<uid>, graph:run
        │                              │
        │                              ▼
        └── lifecycle events ──► EventBus (numbered, replayed)
                                       │
            View blocks (browser) ◄────┘   subscribe by address, redraw on frame
```

Nothing new on the wire: the mirroring in plan 10 section 6 feeds the same
streams a block already reads.

## 8. Acting from a view (the one hard part)

A `button` that triggers an event-in (for example "Save now" wired to the saver's
`save`) needs a **server-to-child control channel**. Today the child only
receives the graph and a cancel flag. Options:

| Option | Cost |
|---|---|
| Extend the cancel mechanism to a small command queue the supervisor writes and the run polls at safe points | moderate; reuses the polling model |
| Only Run/Stop buttons now; triggers later | trivial; recommended for the first pass |

Decision V6: ship Run/Stop; design the command queue with plan 70 when the saver
and probe nodes become event subscribers.

## 9. Starters

* A starter is `{id, label, description, questions: [...], graph: {...}}`, where
  each question substitutes into the skeleton (a choice of **family** swaps the
  family-specific nodes: SDXL conditioner vs Anima conditioner, the right
  dataset latent cache, the right default loss). The skeleton is an ordinary graph
  with placeholder markers.
* Ship at least: **SDXL LoRA** and **Anima LoRA**. They also serve as the
  integration tests of the whole rework (a starter must validate and run on the
  synthetic fixtures).
* The component map's complaint "a correct training graph cannot be authored
  blind" is answered by building starters *from the verified graphs* the project
  already runs (the hardware validation script wires one), not from scratch.

## 10. Existing templates

Not converted. Templates and the `templates` table are not carried into the new
tree (plan 90): views live inside the graph from the start, node uids exist from the
first save, and the executions table is reused as is (it already snapshots the
graph per run). Starters (section 9) replace the way people begin.

## 11. API changes

| Today | After |
|---|---|
| `GET/PUT/DELETE /templates/{name}` | removed; `/graphs/{name}` carries `views` |
| `POST /templates/modules`, `/templates/resolve` | kept; moved under `/graphs/.../modules` and `/resolve`, with `settings`, `events` added |
| `POST /templates/{name}/run` with typed `values` | kept (run with overrides, nothing saved), applying only to fields the view exposes |
| new `GET /starters`, `POST /starters/{id}/instantiate` | creates a graph |
| new `GET /runs/{id}/metrics?stream=` | persisted per-run frames |

## 12. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| V1 | Where a view lives | own table with snapshot; inside the graph | Inside the graph (`views` key; the library already stores extras verbatim) |
| V2 | Node addressing | display id; stable `uid` | `uid` for views and event edges; data edges may follow later |
| V3 | Starters | templates doubling as starters; separate | Separate factories that create a graph |
| V4 | Monitor | node/handle; view over a run | View over a run; remove the node and the `monitor` port after the coverage check |
| V5 | Run history | live only; persisted metrics | Persist downsampled frames per run |
| V6 | Buttons | Run/Stop only; event triggers | Run/Stop now; triggers with the control channel |
| V7 | Naming in the UI | Templates; Apps / Starters / Views | Apps (use mode), Starters (new graph from), Views (inside the editor) |

## 13. Risks

* A view inside the graph document grows the graph JSON and executions snapshot.
  Mitigation: strip `views` from the run payload and the snapshot.

## 14. Open questions for the owner

1. Is "Apps" (a graph with a default page) the right idea for the beginner
   audience, or do you want that audience to see Starters only?
2. Do you want to compare runs (persisted per-run metrics, section 6) in this
   rework, or later?
3. Should a monitor window ever outlive its graph (watch a finished run after the
   graph was edited)? The design says yes, because runs snapshot their graph.
