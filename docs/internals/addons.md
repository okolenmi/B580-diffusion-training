# Node addons

Optional parts attached to a node (or to the graph) instead of separate nodes: a warm-up, a region ladder, an anchor loss, a trigger.
Code: `graph/addons.py` (classes), `graph/addon_edit.py` (change a graph's addons without the editor), host nodes declare `ADDON_SLOTS`;
addon classes live beside their hosts (`nodes/train/addons.py`, `nodes/dataset/addons.py`, `nodes/train/triggers.py`). Saved format:
`contracts/graph-json.md` (rule 10); catalog fields: `contracts/catalog.md`.

## Model

- A host declares `AddonSlot(name, interface, many)`. An addon is defined by the interface of the slot it fills, so any host that declares
  the slot offers it. The host never imports an addon: it asks `self.addons` and gets objects that satisfy the interface.
- A slot with `many=True` holds an ordered list; the order is saved. Every host also has the `events` slot for addons that only react to events.
- A slot has a socket (`slot:<name>`): a wire from another node fills the slot like an addon would. A slot filled twice (addon and wire)
  is refused unless it is a `many` slot.
- An addon is saved as `{id, kind, enabled, params}`; `id` is stable inside its host. A disabled addon is neither built nor checked; its
  parameters stay in the file. An addon parameter can be wired like a node port.
- The validator reports problems with `addon_*` codes (unknown kind, not offered by this host, duplicate or bad id, slot taken, requires or
  conflicts, wired). `graph.addon_edit` (enable, configure, disable, remove, load an addon-set file) raises `AddonError` with the same messages.

## Events travel up a chain of scopes

An event is offered to the handlers at its source, then to each enclosing scope up to the graph (`graph.events.GRAPH`), and a handler
may call `stop()` to keep it from rising further (handlers of its own scope still run). A trigger such as Every N Steps is placed where
its scope is meant: on the trainer, or on the graph. A consumer such as Save Adapter declares its `EventIn` as `scoped`: it then hears
events whose emitter's scope chain contains its own scope, with no wire. An explicit event wire still joins two nodes directly
(`contracts/events.md` for the event types and dispatch rules).

## Adding an addon

1. Pick the slot interface from a host that declares it (or add a slot to a host: one `AddonSlot` and a call to `self.addons`).
2. Write the `Addon` subclass: `DISPLAY_NAME`, `SLOT`, its ports and groups, and `build(**inputs)` returning the collaborator (`self.host` is the host context).
3. It is discovered with the nodes (`NodeRegistry`); the editor lists it in the host's Addons tab. Cover it in `tests/check_addons.py`.
