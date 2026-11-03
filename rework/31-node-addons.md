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
| Every N Steps, Every Epoch | emit "checkpoint due" | Save Adapter |
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
| **Addon** | A small separate class attached to one node (its **host**). Own parameters, own optional wires, own events. Addons tab. |
| **Slot** | A place in the host where addons of one interface fit (`ladder`, `lr_modifier`, `step_observer`, `loss_term`). |
| **Interface** | The type a slot demands (a Python `Protocol` in `engine/` or `families/api`), the same kind of type ports use today. |
| **Hub** | The single "other connections" circle on a node. |

## 4. Rule of thumb: node, addon or port

| It is... | Use |
|---|---|
| One number or switch that host code reads | a native port in a group |
| A rarely used input the host supports | an optional input |
| Optional, has its own code and parameters, modifies or observes **one** host | an **addon** |
| Produces a value that **several** nodes consume, or is a stage in the data flow | a node (Time Balance stays one; so do the event triggers when several consumers listen) |
| Is the thing being trained or loaded | a node |

Test used when reviewing: if removing the wire and putting the object inside its only consumer changes nothing else in the
graph, it should have been an addon.

## 5. What the host and the addon declare

Python shape (names are proposals):

```python
class Addon:
    """Base of every addon: parameters and groups like a Node, one host, one slot."""
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
* **Hub**: every wire that ends on a port which is not on the canvas (a parameter of an addon, an optional input not enabled,
  a port inside a collapsed group) is drawn to the one hub circle. The circle shows how many wires end there. Hover lists them
  (port, source node); click opens the tab that holds the port. Dropping a dragged wire on the hub opens a chooser of the
  compatible hidden ports of that node (the same chooser as the "+n" pin of `30-node-ui.md` section 5).
* This **replaces the per-port pin strips on collapsed groups** (`30-node-ui.md`, decision U2, still a recommendation there):
  one hub is simpler to draw, to route and to read than a strip of pins per group. A fully minimized node shows the title and
  the hub, plus its native input and output sockets only if wired.
* Wires out of an addon (event outputs) leave through the same hub.

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
| Save Adapter | Save schedule (Every N Steps / Every Epoch) when only this saver listens |
| Conditioner | Blend, Persistent conditioning cache |
| Captions / prompt chain | Tag Swap, Auto Caption |
| Stays a node | Time Balance (Trainer and Noiser both consume it), the dataset and model nodes |

Cooperation between two hosts goes through a node (the Time Balance case), not through an addon reaching into another node.

## 9. Why not "just groups"

Groups (plan 30) organise fields of one node but do not solve optional behaviour with its own code: the code still lives in
the host's `build` behind `if inputs.get("perceptual_weight")` branches (see `DiffusionTrainerNode._perceptual`, `_initializer`,
`_calibrate_every`). Addons move that code behind the slot interface, so the host's `build` shrinks and a new optional feature is a
new class, not an edit of a 350-line node.

## 10. Checks (when built)

* Every `AddonSlot.interface` is a Protocol; `Addon.build` is checked against it with a stand-in host.
* Every addon names a `SLOT` that at least one registered node declares.
* Validation: unknown kind, slot not on the host, `many=False` slot filled twice, `REQUIRES` missing, `CONFLICTS` present.
* Surface test (`check_node_surface`) counts only Main ports for the "fields visible" limit; addon parameters count per addon.
* Round trip: a graph with addons parses, validates, runs and serialises to the same JSON.
* A disabled addon leaves the run identical to a graph without it.

## 11. Build order

1. **Backend, no editor needed** (`next/graph`): `Addon`, `AddonSlot`, registry listing, `addons` in `NodeSpec`/`GraphDoc`,
   validator rules, `host.addons` accessor, event delivery by `host.addon`. CPU-testable with the toy nodes.
2. Convert the first candidates one at a time, behaviour unchanged, each checked by the existing tests: Region Ladder (Noiser),
   Warmup, Adapter EMA, then Self-perceptual and the others. The starter and the command line follow.
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
* Open for the owner: (1) may an addon also be filled by a wire from a node, so a power user can still share one object between
  two hosts, or is "addon or node" strict? (2) Should a node's addon set be copyable as a preset ("my training extras")?
  (3) Are the event triggers (Every N Steps) addons of the saver, or do they stay nodes because the saver and the EMA both listen?
