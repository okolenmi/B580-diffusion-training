# 31. Node addons: optional parts attached to a node instead of separate nodes

Status: **owner's idea, 2026-11-03.** The direction (items marked DECIDED) is the owner's; the mechanics below are a proposal
to refine. Builds on `30-node-ui.md` (groups, pins), `10-events.md` (event plane) and the current `next/graph` (ports, groups,
events). Nothing here is implemented; section 11 says what could be built before the editor exists.

## 1. What the owner asked for

* **DECIDED** Small single-purpose classes (a learning-rate warmup, a region ladder, a weight average, a self-perceptual loss)
  should not each be a node. A node is overloaded when its extras are ports; a graph is cluttered when its extras are nodes.
  They become **addons**: optional additions to a node that supports them through a specific interface.
* **DECIDED** A node's interface works like a settings window with tabs. The native part and the additional part are
  visibly separate, so the user sees what is native, what was added, and can disable an addon.
* **DECIDED** Every connection that is not visible on the canvas goes to **one circle on the node, "other connections"**,
  instead of many sockets.
* **DECIDED** A further tab for **optional inputs** the node supports natively (kept apart from addons and from the main fields).
* **DECIDED** (2026-11-03, second message) An addon is defined by an **interface**, not by one host: any node that implements
  the interface for a slot makes that addon available. Several node classes with similar mechanics share the same addons.
* **DECIDED** An addon's inputs (and a slot itself) can be filled by a **wire from another node**, not only by typing values.
  Wires belonging to a tab that is not open all point into the same hub dot (section 7).
* **DECIDED** No strict specification of "addon or node": what targets a specific internal process of a node should rather be an
  addon (section 4). It is a guideline, not a rule the code enforces.
* **DECIDED** (2026-11-03, later) Events are not tied to one scope: they bubble up (addon, host, frame, graph) and each element on the way
  may catch them, as in the HTML DOM (section 6c). Graph settings stay a separate window for technical things.
* **DECIDED** (third message) No copyable "preset" feature inside the node. Addons must be **easy to enable and configure from outside**
  (section 6b); an outside loader reads a specific JSON and applies a configuration, so presets are files, not UI state.
* **DECIDED** Event triggers such as Every N Steps become addons: things with a simple interface that only need an event to exist to
  act. Where they attach is answered by event bubbling (section 6c).
* Goal: workflows with fewer nodes and far fewer wires.

## 2. The problem in the present code

`next/nodes` has about 30 node classes. Several exist only to hand a collaborator to one other node:

| Micro node today | What it is | Wired to |
|---|---|---|
| Region Ladder | how many timesteps per image at each batch | Noiser |
| Warmup LR | wraps a learning-rate schedule | Trainer (between the schedule and the trainer) |
| Adapter EMA | averages of the adapter weights, subscribes to step events | Save Adapter |
| Blend Conditioning | wraps a conditioner | Trainer |
| Tag Swap, Auto Caption | stages in the prompt chain | Batcher / Captions |
| Every N Steps, Every Epoch | emit "checkpoint due" (they only need the trainer's step event) | Save Adapter (and EMA's saver) |
| Time Balance | weights and draws by timestep | Trainer **and** Noiser |

The other failure is the opposite one: options that are only ports. The trainer node carries 9 fields in its `loop` group
and needed a new group just to stay under the surface limit (`check_node_surface`); `perceptual_weight`, `policy_calibrate_every`,
`aligned_init_batches` are optional features with their own code. The old managed trainer had 31 ports.

The key observation: **a micro node's output is a collaborator object that a host input port receives** (a ladder object into
the Noiser's `ladder` port; an EMA object into the saver's `ema` port). An addon is the same object created *inside* the host,
with no wire. Conversion is therefore mechanical: the host's collaborator port becomes an addon slot of the same type, and the
micro node's `build` becomes the addon's.

## 3. Vocabulary

| Term | Meaning |
|---|---|
| **Native port** | A field or socket the node always has (today's `Port`, with groups). Main tab. |
| **Optional input** | A native port that is off the canvas until the user enables it (a rarely used input the node supports itself). Optional tab. |
| **Addon** | A small separate class that plugs into a node (its **host**) through an interface. Own parameters, own optional wires, own events. Addons tab. One instance sits on one host; the class is usable by every host class that offers its slot. |
| **Slot** | A place in the host where addons of one interface fit (`ladder`, `lr_modifier`, `step_observer`, `loss_term`). |
| **Interface** | The type a slot demands (a Python `Protocol` in `engine/` or `families/api`), the same kind of type ports use today. |
| **Hub** | The single "other connections" circle on a node. |

## 4. Guideline: node, addon or port

No formal definition (the owner cannot give a stricter one, and none is needed). The principle:

> When a piece targets a **specific internal process of a node**, implement it as an addon. When it is a stage of the data flow,
> or a thing in its own right, it is a node.

Examples, not rules:

| It is... | Usually |
|---|---|
| One number or switch that host code reads | a native port in a group |
| A rarely used input the host supports | an optional input |
| A change to one process inside a node (how the learning rate ramps, which timesteps a batch gets, an extra loss term, a weight average, a caption transform) | an **addon** |
| A stage in the data flow, or the thing loaded or trained | a node |
| One object that several nodes use at the same time (Time Balance) | a node wired into the slots of each host (section 5), or an addon of one of them; either works |

Because a slot can also be filled by a wire (section 5), a wrong guess is cheap: the same class works in both forms.

## 5. What the host and the addon declare

Python shape (names are proposals):

```python
class Addon:
    """Base of every addon: parameters and groups like a Node; fills one slot, in any host class that declares that slot."""
    DISPLAY_NAME: ClassVar[str]
    SLOT: ClassVar[str]                       # which slot of the host it fills
    INPUTS: ClassVar[dict[str, Port]] = {}    # its parameters; they may be wireable like a node's
    GROUPS: ClassVar[tuple[Group, ...]] = ()
    EVENTS_IN: ClassVar[dict[str, EventIn]] = {}
    REQUIRES: ClassVar[tuple[str, ...]] = ()  # other addon kinds that must be on the same host
    CONFLICTS: ClassVar[tuple[str, ...]] = ()
    def build(self, host: HostContext, **inputs) -> object:   # returns the collaborator for the slot
        ...

@dataclass(frozen=True)
class AddonSlot:
    name: str
    interface: type            # what build() must return
    many: bool = False         # one addon, or an ordered list (learning-rate modifiers)
    doc: str = ""

class NoiserNode(Node):
    ADDON_SLOTS = (AddonSlot("ladder", RegionSchedule, doc="How many timesteps per image at each batch."),)
    def build(self, **inputs):
        ladder = self.addons.one("ladder")        # None when no addon is attached
        ...
```

* The host never imports an addon class. It asks `self.addons.one(slot)` or `self.addons.all(slot)` and gets objects that
  implement the slot's interface; adding an addon means writing a class and registering it, never editing a host.
* Addons are discovered by the same registry as nodes (`NodeRegistry`), listed with their `SLOT` and which node classes have that
  slot. The Addons tab of a node offers exactly the addons whose slot the node declares, and (when the family matters)
  whose `FITS` capability the loaded family has.
* An addon with `many=True` slots has an order; the host applies them in that order (warmup, then decay). Order is part of
  the saved data.
* Addons subscribe to events like nodes (`EVENTS_IN`). The event hub addresses them as `host_uid.addon_id`, and they are
  delivered right after the host in rank order, so an addon never sees an event before its host's state is ready.
* Disabled addon: not built, not validated, its parameters kept in the file.
* **Reuse across host classes.** The interface is the contract: a second node (say another trainer, or the Anima family's) that
  declares `AddonSlot("ladder", RegionSchedule)` gets the Region Ladder addon in its Addons tab without any change to the addon.
  An addon may add a narrower requirement (`FITS`, a capability of the family or of the host) and is then offered only where it holds.
* **Slots can be filled by a wire.** Every slot also has a socket of its interface type in the Addons tab. A node wired there
  (the ladder node today) fills the slot like an addon would; a slot filled twice (addon and wire) is a validation error, except
  `many=True` slots, where wired objects and addons are applied in the saved order. This is how one object serves two hosts, and
  how every micro node that exists now keeps working as a node. An addon's own parameters are wireable ports as well
  (a number from an Int node, a path from a Text node).

## 6. Graph file

A node spec gains one list; edges can target an addon's port.

```json
{ "uid": "n_gggggg", "class": "NoiserNode", "params": {"regions": 3},
  "addons": [
    {"id": "a1", "kind": "RegionLadder", "enabled": true, "params": {"start": 9, "span": 120}}
  ] }
```

* `addons[*].id` is stable inside the node (not an index), so edges and layout survive reordering.
* An edge to an addon's wireable parameter names `(node uid, "a1.start")`; the validator treats it as a port of the node
  with the addon's prefix.
* UI state (active tab, collapsed addon sections, hub open) lives in `layout`, as decided for groups.
* Compatibility with graphs written before addons is not a goal (rework rule), so there is no automatic "wire to addon" migration;
  the starters are simply written with addons.

## 6b. Configuring addons from outside

Addons are data first: a node's `addons` list in the graph file is the whole configuration. So everything the editor does is also
possible without it:

* **API** (`graph` package): `enable(doc, node, kind, **params)`, `configure(doc, node, addon_id, **params)`, `disable(doc, node, addon_id)`,
  `remove(...)`, each returning a new `GraphDoc`, validated; `available(doc, node)` lists the addons the node's slots accept.
* **Addon set file**: plain JSON mapping nodes (by title or class) to addon entries, the same entries as in the graph file:

  ```json
  { "noiser": [ {"kind": "RegionLadder", "params": {"start": 9, "span": 120}} ],
    "trainer": [ {"kind": "Warmup", "params": {"steps": 50}} ],
    "graph":   [ {"kind": "EveryNSteps", "params": {"every": 250}} ] }
  ```

* An outside loader (the command line `--addons file.json`, a starter, a test, a script, later the editor's import) merges such a
  file into a graph. A configuration someone wants to reuse is such a file; no preset feature is needed in the node itself. The
  loader refuses an unknown kind, a slot the node lacks, or a double-filled slot with the validator's messages.
* The command-line flags of today (`--warmup`, `--regions-from`, `--ema`) are shorthand that produces the same entries.

## 6c. Event addons: events bubble like in the DOM

An event trigger needs nothing but the existence of an event to act, and it emits one back (`CheckpointDue`). The owner's model
(2026-11-03): events travel **up** a chain of scopes, as in the browser's DOM, and every element on the way may catch them.

```
addon  ->  host node  ->  frame (if the node sits in one; plan 30 frames)  ->  graph
```

* An event is emitted by its source (a node or an addon) and is offered first to the handlers at the source, then to each enclosing
  scope in turn, up to the graph. A handler may act and let it pass, or **stop** it (like `stopPropagation`).
* A trigger such as Every N Steps is an addon placed where its scope is meant: on the **trainer** (it sees the trainer's own
  `StepFinished`, and the `CheckpointDue` it emits rises through the trainer's frame to the graph), in a **frame** (it serves
  only the nodes inside that frame: two trainers or two savers in one graph can each get their own trigger), or on the **graph**
  (it serves everything). The question "trainer or graph" becomes "where do I put it", with no separate mechanism.
* A consumer (Save Adapter) declares the event types it accepts. It receives an emitted event when one of its enclosing scopes is
  an ancestor of the emitter's scope chain, or when a wire joins them explicitly: a wired event is an explicit source and target, as
  today. With one trainer and one saver in one graph, nothing has to be configured.
* Defaults for a consumer without a wire: it listens to events that reach its own scope chain (itself, its frame, the graph). The
  saver therefore hears a graph-scope trigger, and a trigger on a trainer reaches it only through a shared frame or the graph, which
  keeps "which saver is triggered by which trainer" a visible decision (a frame, or a wire), never an accident.
* Ordering follows the rank rule of the event plane (10-events): handlers deeper in the chain first (addon, host, frame, graph), so an
  addon sees the host's state when it is ready and the graph sees it last.
* Loops are refused as today (an event type may not re-enter a handler that is already running).

This needs the event hub to know scopes (a node's frame, the graph) and to deliver by event type along the chain, in addition to
today's wire edges, which remain.

**Graph settings stay separate** (plan 20): they are the technical things (VRAM budget, device, seed base). Graph-scope addons are
behaviour attached to the graph; where the editor shows them (a tab beside the settings window or a frame on the canvas) is a UI
question for the editor phase.

## 7. Interface on the canvas

```
┌ Noiser ───────────────────── +1 ┐   header: addon count badge
│ [Main] [Optional] [Addons]      │   tabs (a settings window in the node's inspector, the compact node shows Main)
│  regions        3               │
│  layout         strata          │
│  ...                          ◯ │   ◯ = the hub: "other connections" (count)
└─────────────────────────────────┘
```

* **Main**: native ports, grouped as in `30-node-ui.md` section 4. Native sockets stay visible on the canvas.
* **Optional**: native optional inputs (off by default). Enabling one adds it to the canvas as a socket.
* **Addons**: a list of attached addons, each its own section with an enable switch, its parameters in its own groups, and a
  remove button; a "+ Add addon" menu filtered by the host's slots. Addons are drawn apart from native fields, so what is native
  and what is added is never mixed, and a disabled addon is visibly off.
* **Hub**: every wire that ends on a port which is not shown at the moment (a parameter of an addon, an optional input not enabled,
  a port inside a collapsed group) is drawn to the one hub circle. The circle shows how many wires end there. Hover lists them
  (port, source node); click opens the tab that holds the port. Dropping a dragged wire on the hub opens a chooser of the
  compatible hidden ports of that node (the same chooser as the "+n" pin of `30-node-ui.md` section 5).
* This **replaces the per-port pin strips on collapsed groups** (`30-node-ui.md`, decision U2, still a recommendation there):
  one hub is simpler to draw, to route and to read than a strip of pins per group. A fully minimized node shows the title and
  the hub, plus its native input and output sockets only if wired.
* Wires out of an addon (event outputs) leave through the same hub.
* **Switching tabs moves wires.** With a tab open, the wires of its ports run to those ports; the wires of the tabs that are not open all
  point into the one hub dot. A wire dragged from another node can be dropped on any visible port, on a slot socket, or on the hub.

## 8. What changes for the starters and the trainer

`starters.sdxl_lora` builds the graph from `SdxlLoraOptions`. With addons the groups `_extras` add nodes and wires now
(`ema`, `ladder`, `balance`, `warmup`) become entries in the host's `addons` list and the wires disappear: the SDXL graph with
ladder, warmup and EMA loses three nodes and about eight edges. The CLI flags (`--regions-from`, `--warmup`, `--ema`) map to
the same addon entries, so the command line and the editor give the same file.

Candidates, host first (all are candidates until each is reviewed against the rule in section 4):

| Host | Addons |
|---|---|
| Noiser | Region Ladder |
| Trainer / schedule | Warmup; Self-perceptual loss; Calibrated one-pass; Aligned init (it has its own pass over batches) |
| Inject Adapter | Adapter EMA (it only reads the adapter), LoRA+ ratio (a number; stays a native port unless it grows) |
| Trainer, a frame or the graph (section 6c) | Every N Steps, Every Epoch (and other event-only things: stop after N steps, notify) |
| Conditioner | Blend, Persistent conditioning cache |
| Captions / prompt chain | Tag Swap, Auto Caption |
| Stays a node | The dataset and model nodes; Time Balance as long as two hosts share one instance (it can still be wired into their slots) |

One object serving two hosts (Time Balance: Trainer and Noiser) is either a node wired into the slot of each, or an addon on one host
that the other host does not need to know about; the slot sockets (section 5) make both forms equal. An addon never reaches into
another node on its own.

## 9. Why not "just groups"

Groups (plan 30) organise fields of one node but do not solve optional behaviour with its own code: the code still lives in
the host's `build` behind `if inputs.get("perceptual_weight")` branches (see `DiffusionTrainerNode._perceptual`, `_initializer`,
`_calibrate_every`). Addons move that code behind the slot interface, so the host's `build` shrinks and a new optional feature is a
new class, not an edit of a 350-line node.

## 10. Checks (when built)

* Every `AddonSlot.interface` is a Protocol; `Addon.build` is checked against it with a stand-in host.
* Every addon names a `SLOT` that at least one registered node declares.
* Validation: unknown kind, slot not on the host, `many=False` slot filled twice (by addon and wire, or by two addons), `REQUIRES` missing, `CONFLICTS` present.
* A second stand-in host class that declares the same slot offers the same addon without any change to the addon.
* Surface test (`check_node_surface`) counts only Main ports for the "fields visible" limit; addon parameters count per addon.
* Round trip: a graph with addons parses, validates, runs and serialises to the same JSON.
* A disabled addon leaves the run identical to a graph without it.

## 11. Build order

1. **Backend, no editor needed** (`next/graph`): **built** (`addons.py`, `addon_edit.py`, hub scopes in `events.py`; `check_addons`).
   What the code settled, so the proposals above read as decisions:
   * Names on the wire and in the file: an addon's parameter is the port `a1.start` of its host; a slot's socket is `slot:ladder`
     (`many` sockets are multi ports). Event wires may end at `uid.id` or `graph.id`.
   * Every host has the implicit `events` slot (addons that only act on events, no collaborator); the graph is a host that has only that slot.
   * The scope tree is graph > node > addon. An event is offered to the source's own addons, then to each enclosing scope: that scope's
     catchers (its addons, or the node's own handlers for events of its addons), then the consumers inside it that declared
     `EventIn(..., scoped=True)`. A handler may call `context.stop()`: the scopes above are skipped, the rest of that scope still hears it
     (pass by default, stop explicit). Wired delivery is unchanged and a target reached by a wire is not delivered the same event again.
   * Frames are not in the tree yet; a frame adds one level between node and graph when plan 30 builds them.
   * A scoped consumer must exist before its emitter runs, so the validator builds nodes with scoped inputs first among the nodes that are
     ready at the same time (data wires still decide).
   * Outside: `enable / configure / disable / remove / available / apply_set / load_set` (`graph/addon_edit.py`), and
     `--addons FILE.json` on the command line. Nothing is registered yet, so the flag has nothing to switch on until step 2.
2. Convert the first candidates one at a time, behaviour unchanged, each checked by the existing tests. **Done:** Region Ladder (on the
   Noiser, slot `ladder`), Batch Ramp (Batcher, slot `ramp`), Warmup (schedule nodes, slot `lr_modifier`, many), and the Every N Steps and
   Every Epoch triggers (event addons on the graph; Save Adapter's `checkpoint` input is `scoped`, so it hears them without a wire). The
   starter writes them as addons and `--param ladder.start=6` reaches an addon by its id. **Kept as nodes:** Adapter EMA (Save Adapter and
   Resume both use the object it makes), Time Balance (Noiser and Trainer share one). Also done: Self-perceptual loss and the gradient-aligned start
   (Trainer slots `perceptual` and `initializer`; their addons return a part that the trainer attaches to its model and batches, since both exist only when
   it is built). Also done: the gradient policy (Trainer slot `gradient_policy`; its four settings `gradient_policy`, `policy_normalize`, `policy_floor`, `policy_calibrate_every` are now the addon's parameters, and the command line has `--gradient-policy` and the `--policy-*` flags). Still to convert: the others. The starter and the command line follow.
3. Editor (phase 6, with the owner's go-ahead): tabs, Addons tab, hub circle, optional inputs; the catalog (`NodeInfo`) gains
   `addon_slots`, `addons` and `optional` fields so the editor needs no per-node code.

Until step 1 exists, new optional features continue as in `next/` now (a node or a native port), written so the conversion stays
mechanical: the collaborator is a class with a small interface, built from plain parameters.

## 12. Risks and open questions

* **A hidden feature is easy to forget.** The addon count badge on the header and the Addons tab being one click away are the answer;
  a disabled addon is shown struck through, not removed.
* **Slot interfaces freeze early.** Each slot's Protocol must be reviewed once when it is introduced; a slot with one addon is
  a sign that a native port was enough.
* **Order of addons.** Only `many=True` slots have one; if a second kind of ordering appears (between slots), `REQUIRES` is the tool,
  not hidden priorities.
* Answered by the owner: an addon is defined by its interface and may fit several host classes; a slot or an addon input may be filled by
  a wire; "addon or node" is a guideline, not a strict rule.
* Also answered: no preset feature in the node (outside loaders and JSON files instead); event triggers become addons.
* Also answered: events bubble through scopes (section 6c); graph settings stay separate from graph addons.
* Built with the proposal for (1) a handler passes by default and stops explicitly. Still open: (2) is a frame an event scope
  only (no other meaning), as proposed? (3) where the editor shows graph-scope addons (UI phase).
