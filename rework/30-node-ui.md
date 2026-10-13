# 30. Node UI: groups, collapse, live state, dynamic ports, embedded views

Status: **proposal.** The editor must change to support the other plans (dynamic
Load Model, events, settings, grouped fields). Inputs: `audit/frontend-graph-ui.md`
(read it first; section numbers below refer to it).

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* Nodes "should be more universal": better **live update**, **groups of fields**
  (the trainer has too many parameters and nothing can be minimized).
* A minimized node must not collapse into **one shared point that cannot receive
  new wires**.
* Server-computed state shown on the node ("processing... please wait").
* Nodes may carry their own UI parts (monitors, widgets), not only sockets.
* The editor "should be reworked as well to support all new features".

## 2. Findings that shape the design (audit 2 to 4)

* The editor is **generic**, no per-node-type code: keep that.
* `Port.group` and `Port.advanced` exist in the catalog and the templates page
  uses them, **`canvas.js` ignores both**. There is **no collapse** today (the
  complaint is about a feature that is not there; see audit section 3).
* Every change **rebuilds every node's DOM** (`canvas.js:59-72`). Badges and
  diagnostics survive only because they are copied back from side maps.
* Dynamic ports are possible in the backend (`resolve_*` hooks) but the editor
  never calls them; there is no `resolve_outputs` endpoint.
* **One wire per input** (`state.js:138`); outputs may fan out.
* Node status is set only when a node *finishes*.

## 3. Principles

1. **The catalog describes, the editor renders.** Everything below is data on
   the node class (groups, widgets, states), never per-node editor code.
2. **A node's UI state is not its params.** Collapsed groups, size, "show
   advanced" live in the graph's `layout`, so they never change run payloads or
   templates.
3. **Wires never depend on layout.** A port that exists can always be wired,
   whatever is collapsed.
4. **Incremental updates.** A change touches the DOM of what changed.
5. **Stay vanilla** (no build step, as now): a small keyed-render helper, not a
   framework (decision U1).

## 4. Field groups

Backend (`nodes/core.py`):

```python
@dataclass(frozen=True)
class Group:
    id: str; label: str
    collapsed: bool = False          # initial state
    advanced: bool = False           # hidden until "show advanced"
    visible_when: tuple | None = None   # same contract as Port.visible_when
    doc: str = ""
class Node:
    GROUPS: ClassVar[tuple[Group, ...]] = ()     # order = display order
```

* `Port.group` (exists) names a group id. Ports with no group fall into an implicit
  first group `main` (no header drawn when it is the only group).
* **Lint rule (test):** a node class with more than 8 widget ports must declare
  groups, and every `Port.group` must exist. Enforced in the node smoke tests the
  way `visible_when` typos are enforced today in `__init_subclass__`.
* `Port.ui` hints (new, optional): `widget` (`slider`, `multiselect`, `table`,
  `chips`, `textarea`, `code`, `path`), `rows`, `inline`, `placeholder`, `unit`.
  The widget registry (section 9) maps them; unknown hint falls back to type.

Example grouping the **current** managed trainer would get (31 ports), to show
the effect before the trainer itself is redesigned (plan 70):

| Group | Ports |
|---|---|
| Main | `batches`, `trainer`, `optimizer`, `lr_schedule`, `steps` |
| Loss | `loss_weighting`, `diffusion_process`, `bucket_balance`, `grad_accum`, `grad_clip_max_norm` |
| Conditioning | `conditioning_blend` |
| LoRA gate (advanced) | `gate_enabled`, `gate_train_low`, `gate_train_high`, `gate_width` |
| Residency and memory (advanced; to graph settings) | `resource_control`, `residency_safety_margin`, `calibration_steps`, `empty_cache_every_n_steps`, `use_xpu_graph`, `prewarm_text_encoder` |
| Saving and probes (advanced; to events) | `save_every_n_steps`, `save_prefix`, `probe_*` (4) |
| Monitoring | `monitor`, `profile`, `on_step`, `project_layout` |

## 5. Collapse without losing wires

A collapsed group is not a hidden group of ports; it is **the same ports drawn
smaller**.

* **Group header**: chevron, label, count badge ("5 fields, 1 wired"), and a **pin
  strip**: one small real socket for every socket-capable port in the group.
  Sockets keep their identity (`data-port`), so wires attach to the pin exactly
  as they would to the full row. Widget-only ports have no pin.
* **Too many pins** (more than 8): the first 8 plus a "+n" pin; dropping a wire on
  the header or the "+n" pin opens a small chooser of the compatible hidden ports.
* **Spring-loaded expand**: while dragging a wire, a collapsed group that contains a
  compatible port expands after ~400 ms of hover and collapses again when the drag
  ends (as folders do in file managers). Compatible sockets are highlighted as
  today.
* **Whole-node minimize**: body hidden, title row only; **every** socket becomes a
  pin along the left (inputs) or right (outputs) edge, vertically distributed.
  There is never a single shared connection point. Wired pins show a filled dot;
  hovering a pin shows port name and type.
* Wires to a pin on a collapsed group are drawn to that pin; the geometry reads
  `socketPos(port)` from the node view, which returns the pin or the row socket
  depending on state, so edge code does not know about collapse.
* Collapsed state per group, per node, in `layout[id].collapsed = {group: bool}`.

## 6. Node state and live update

### 6.1 State machine

`idle | resolving | stale | running | done | error | blocked`

| State | Meaning | Look |
|---|---|---|
| `resolving` | server computing the node's shape or choices ("please wait") | spinner + "processing... please wait", old ports dimmed, fields stay editable |
| `stale` | inputs changed since the last run result | muted badge |
| `running` | the node's `build()` is executing | spinner + optional progress bar (from `node_progress`) |
| `done` | finished | duration badge (today's "ok 120ms") |
| `error` | failed or resolve failed | red, message in a tooltip and the inspector |
| `blocked` | cannot run (a required input missing) | amber, names the input |

Driven by **events** (plan 10): the supervisor's progress, the node's own
`node_state` events, and the local resolve flow. Today only "done" and "error"
exist, and only after the fact.

### 6.2 Incremental rendering

Replace `Canvas.render()` (full rebuild) with:

* `GraphDoc` emits fine-grained changes: `node:add`, `node:remove`, `node:move`,
  `node:params`, `node:ports`, `node:state`, `node:ui` (collapse/size),
  `edge:add`, `edge:remove`.
* `NodeView` (one per node, owns its DOM): `setSpec`, `setParams`, `setState`,
  `setUi`, `patchPorts(patch)`, `socketPos(port)`. A keyed helper updates rows in
  place, so a half-typed text field keeps focus without the "store without
  notifying" workaround in `widgets.js`.
* `EdgeLayer` recomputes only paths whose endpoint node changed or moved
  (already how drag works); a `ResizeObserver` per node triggers a re-measure.

This is the part of `canvas.js` (670 lines) that is rewritten; `widgets.js`,
`inspector.js` and the board stay structurally as they are.

### 6.3 Dynamic shape

(Server contract is in plan 41 section 7.) On the client:

1. A class's `dynamic.depends_on` lists the params whose change must re-resolve.
2. `NodeView` debounces (default 300 ms), sets `resolving`, calls
   `POST /graphs/nodes/{class}/resolve`, tags the request with a sequence number,
   ignores superseded answers (as `_diagSeq` does for diagnostics).
3. On `ready`: `patchPorts` replaces ports; wires on surviving ports are kept;
   wires on vanished ports become **dangling** (dashed red, validator error)
   instead of being deleted. On `error`: keep the last good shape and show the
   message. On `needs_choice`: highlight the field that must be chosen.
4. The response may also carry **computed choices** (the adapter groups of plan 43)
   and an **info line**; both render in the node.

## 7. Event ports

Plan 10 adds a second kind of port. In the editor:

* An **event strip** along the top (inputs) and bottom (outputs) of the node, so
  the data layout on the sides is untouched. Event sockets are diamonds.
* Event wires are **dashed** and coloured by payload type; they accept many wires
  on both ends.
* A node flashes its event strip when it emits or receives; the last event name
  shows in the tooltip. A bottom **Events** panel lists a filtered trace.
* Collapsed nodes keep event pins on the top and bottom edges.

## 8. Multi-input ports

Needed for: several adapted parts into one trainer, several savers, a list of
datasets. New port flag `multi=True`:

* accepts any number of wires; the runtime passes a **list in wire order**;
* the editor draws each wire to the same socket, offset slightly; the order is
  editable in the inspector (drag to reorder);
* the validator checks every wire's type against the element type;
* data inputs without `multi` keep today's "one wire per input" rule.

This answers the open question left in the component map: **the editor does not
support several wires into one port today, and the design adds it explicitly
(`multi`) instead of removing the rule everywhere.**

## 9. Widgets

Keep the invariant "`widgets.js` is the only place a port becomes a control"; make
it a **registry**: `registerWidget(kind, builder)`. Built-ins today: select,
checkbox, number, JSON textarea, text, path picker. Planned additions, each a
small module: `slider` (number with bounds), `multiselect` (adapter groups),
`table` (region weights), `chips` (tag lists), `textarea` (prompt text),
`pathpicker` with an optional detection badge. The inspector, the node body, the
templates board and the settings dialog all call the same registry.

## 10. Embedded displays in nodes

Display modules (`Node.MODULES`: chart, scalar, bars, text, progress) already
exist and stream per node. Give the **canvas** a slot for them: a collapsible
**Live** group at the bottom of the node, rendered by the same
`board/displays.js` functions, fed by `LiveStreams` (one `EventSource` per
stream, shared). Result: the trainer node shows its loss curve and step bar, the
Load Model node shows its info line, a dataset node can show a thumbnail grid
(`image-grid`, a new display kind). Node resizing (drag a corner) is stored in
`layout[id].{w,h}`.

## 11. Smaller UI wins (cheap, high value)

| Item | Why |
|---|---|
| Drag a wire to empty canvas -> quick-add list of **nodes with a compatible port** | With dynamic ports this removes most palette hunting |
| Search palette (type to add) | 38 node classes and growing |
| **Frames**: a titled, coloured rectangle behind nodes ("Data", "Model", "Training"), moves its nodes | Mirrors the nine parts; layout only, no run meaning |
| Handle-type colours on sockets | Makes SDXL/Anima mix-ups visible |
| Undo / redo | **Does not exist** (no match in `editor.js` or `editor/*.js`); worth adding once changes are fine-grained events (6.2), because each event can be inverted |

## 12. Where UI and code design pull apart, and the compromise

| Topic | UI wants | Code wants | Compromise |
|---|---|---|---|
| Collapse | pins stay wirable | one render path | `socketPos()` abstraction; collapse is a presentation of the same ports |
| Many fields | groups, advanced hidden | flat port dict | `Group` metadata is optional; ports stay a flat dict in the class |
| Live data | widgets in the node | nodes know nothing about presentation | nodes only declare `MODULES` (already so); the editor owns drawing |
| Dynamic ports | change on the canvas | static `INPUTS`/`OUTPUTS` | `DYNAMIC` marker plus resolve hooks; static dicts stay for non-dynamic nodes |
| Responsiveness | update as you type | build() runs once | resolve is an **editing-time** call, separate from run time |
| Multi-wire | several wires | typed single values | opt-in `multi` port with element type |

## 13. Backend/catalog additions

`NodeInfo.groups`, `ports[].ui`, `ports[].multi`, `dynamic`, `events_in/out`,
`modules` (exists), `reads_settings`; a `catalog_format` integer so the editor can
tell an old server from a new one. Saved graphs gain per-node `layout` entries
`{x, y, w, h, collapsed, advanced}`; old graphs load unchanged.

## 14. Tests and verification

* Pure units (`node --test`): pin placement, collapse geometry, wire-order logic,
  dangling-wire retention, sequence handling, keyed patching.
* Headless Chromium (the project already runs browser checks): collapse and drag
  a wire onto a collapsed group, spring-loaded expand, resolving state, focus
  kept while typing, **200-node synthetic graph** drag and pan time before and
  after the rewrite.
* Backend: catalog snapshot tests for groups, dynamic, multi, events; lint test
  for the 8-port rule.

## 15. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| U1 | Rendering approach | keep full re-render; vanilla keyed views; adopt a library | Vanilla keyed `NodeView`/`EdgeLayer`; no build step |
| U2 | Collapse | one shared handle; per-port pins | Per-port pins on groups and on the minimized node |
| U3 | Spring-loaded expand | none; on drag hover | On hover while dragging a wire |
| U4 | Multi-wire | everywhere; opt-in port flag | Opt-in `multi` |
| U5 | UI state storage | in params; in layout | In layout |
| U6 | Frames | now; later | Early: cheap and it mirrors the parts |
| U7 | Displays in nodes | no; yes | Yes, via a Live group, reusing `board/displays.js` |
| U8 | Node size | auto; resizable | Resizable, saved in layout |

## 16. Risks

* Rewriting `canvas.js` can regress dragging, wiring, selection. Mitigation:
  write the pure geometry and doc-change code first with tests, then build the new
  renderer as a new module set (`frontend/js/graph2/`, plan 90) with the same
  public surface; the old editor stays untouched until the swap.
* Pin geometry on very small nodes. Mitigation: minimum node height from the
  larger of inputs and outputs; a "+n" overflow.
* Groups on 38 existing classes is mechanical but large. Mitigation: only the
  classes over the lint threshold first (managed trainer, dataset source, LoRA
  injector, resources controller).

## 17. Open questions for the owner

1. Is "pins on the group header, spring-loaded expand while dragging" the
   behaviour you want for wiring into collapsed groups, or would you rather
   always see every socket and only collapse widgets?
2. Do you want undo/redo on the canvas (it does not exist today) in this rework?
3. Frames and quick-add are extras I added; keep or drop?
