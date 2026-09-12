*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 3 -- Interactive node support (editor + core.py + introspection)

**Status: audit done, no implementation yet -- reporting back before
any design commitment, per this phase's own gate.**

Read `server/static/nodegraph.js` end to end (1327 lines). Real
findings, not estimates:

**The wire protocol to `/run` doesn't need to change.** `toRunPayload()`
already sends opaque `{id, class_name, params}` per node, and
`graph_executor.py` already resolves everything fresh from
`class_name` + `params` at Run time. Since a Phase-3 node settles into
a concrete shape before Run is pressed (this phase's whole premise),
nothing about what's actually transmitted needs to change.

**But `classInfo.inputs`/`classInfo.outputs` -- static, one shared
object per class, fetched once at page load -- are read directly in at
least 10 separate places across `GraphModel`/`GraphView`, not one:**
`GraphNode`'s constructor (default param values), node rendering (all
three display modes), `validate()`, `toRunPayload()` itself (misses
serializing a dynamically-resolved input's value entirely if not
fixed), connection drag/hit-testing (four separate spots), and
`suggestNodesForDroppedWire()` (the "drop a wire on empty space, see
compatible node suggestions" feature). Every one of these needs to read
a node's *resolved current* shape instead of the shared static one for
an interactive node to work correctly everywhere, not just in whichever
spot gets tested first.

**One genuinely hard case, not just a long list of mechanical
changes -- resolved.** `suggestNodesForDroppedWire()` iterates the
*entire registry's* static `classInfo.inputs/outputs` to find
compatible matches; a class whose shape depends on params can't be
cheaply enumerated that way in general. Resolution: for suggestion
purposes, **each preset of a dynamic node counts as its own separate
searchable entry**, contributing only its *required* inputs/outputs
(optional ports are "just helpers" and don't carry suggestion-worthy
signal either way). This turns "enumerate all possible shapes" (hard,
open-ended) into "enumerate `(class, preset)` pairs, each with its own
fixed required-port list" (exactly as cheap and enumerable as today's
one-shape-per-class search, just with one more dimension). Concretely:
`NodeInfo` (`server/nodegraph_introspect.py`) gains a `node_kind:
"static" | "dynamic"` field -- **real, requested metadata dividing
default nodes from dynamic-with-presets ones, built to scale past the
one dynamic node that exists today** -- and, for a dynamic node, a
`presets: [{name, required_inputs, required_outputs}]` list, each
entry pre-resolved (that preset's own default configuration, not
requiring a live params round-trip just to enumerate it). The
suggestion search in `nodegraph.js` then iterates static classes and
`(class, preset)` pairs uniformly, over required ports only.

This also gives `nodes/core.py` and Phase 4's `ResourcePreset`
something concrete to satisfy: a dynamic node class needs to be able to
enumerate its own presets and each preset's required-only shape
*without* being asked to fully resolve or construct anything -- a
lighter, separate query than "resolve my current shape given these
live params" (still needed for the interactive-editing case itself),
but related, and worth designing together rather than as two
disconnected asks on the same class.

**Nothing like "resolve shape given params" exists in the introspection
layer at all today.** `nodegraph_introspect.py`'s `introspect_node_class(cls)`
takes a bare class, no params -- confirmed while doing the section 11.5
and DoRA work earlier, not newly discovered here, but worth restating
as a concrete gap this phase has to fill, not something to extend.

**Real touch points, once a design is chosen (not committed to yet):**
`nodes/core.py` (some way for a class to declare "my shape is a
function of these params" instead of a fixed `INPUTS`/`OUTPUTS`
`ClassVar`, additive -- every existing static node keeps working
unchanged), `server/nodegraph_introspect.py` (a params-aware resolver),
`server/graph_executor.py`'s `_is_compatible()` (resolve shape from
`spec.params` before checking edge types -- server-side validation at
`/run` time needs this regardless of what the client already checked,
same "don't trust the client" posture `resolve_safe_model_path`
already takes elsewhere), and the ~10 call sites above in
`nodegraph.js`.

**One small, unrelated loose thread found while reading this file, not
blocking anything:** its own top-of-file comment cites
`docs/node_architecture_refactor_plan.md` as "the node-graph design the
project's OOP rule is about." That file doesn't exist anywhere in this
repo -- either stale or never written. Flagging, not fixing --
out of scope for this audit.

**Dependency:** Phase 2 (done). **Metadata infrastructure done:**
`nodes/core.py` gained `Node.NODE_KIND` ("static"/"dynamic", default
"static" -- purely additive, every existing node unchanged),
`NodePreset` (a preset's own name + required-only inputs/outputs,
deliberately narrower than the full Phase 4 `ResourcePreset`), and
`list_presets()`, enforced by `__init_subclass__` at class-definition
time (a `NODE_KIND == "dynamic"` class that doesn't override
`list_presets()`, or declares an invalid `NODE_KIND`, fails loudly the
moment it's defined, not the first time something calls it).
`NodePreset` itself rejects a `required=False` `Port` inside
`required_inputs`/`required_outputs` at construction -- self-
contradictory, since those dicts are specifically the required-only
subset. `server/nodegraph_introspect.py`'s `NodeInfo` gained matching
`node_kind`/`presets` fields, resolved in `introspect_node_class()` and
serialized in `node_info_to_dict()`.

Verified, `server/smoke_tests/smoke_test_node_presets.py`: both
`__init_subclass__` enforcement paths raise at definition time with a
clear message; the `required=False`-inside-`required_inputs` rejection;
a synthetic two-preset dynamic node introspects correctly end to end
(names, required inputs/outputs, per preset) including through
`node_info_to_dict()`; a genuine multi-level-inheritance edge case (an
abstract intermediate class provides `list_presets()`, a concrete
subclass doesn't re-override it) is correctly satisfied -- the override
check resolves through the real MRO via `__func__` identity, not a
naive `cls.__dict__` check that would wrongly flag this legitimate
case; and every one of the 36 real nodes already in the registry is
still `node_kind == "static"` with `presets is None` -- confirming this
is purely additive, nothing about the real, already-shipped nodes
changed. Adjacent server smoke tests (`smoke_test_nodegraph_introspect`,
`smoke_test_graph_executor`, `smoke_test_execution_registry`,
`smoke_test_asset_inspect`) and two `nodes/smoke_tests/` files that
exercise `Node` directly still pass.

**`nodegraph.js`'s suggestion-search: done.** `suggestNodesForDroppedWire()`
now searches every class's own common shape (unchanged for a static
class) *plus*, for a `node_kind === "dynamic"` class, one additional
search target per declared preset (required-only ports). Real bug
caught before landing, not just reasoned about: the first version
treated "search the common shape" and "search per-preset" as mutually
exclusive for a dynamic node -- silently dropping matches against its
own common ports (which stay present no matter which preset is chosen,
per `NODE_KIND`'s own docstring). Caught by a standalone check of the
exact algorithm against mock registry data before it landed (see
below), not assumed correct by inspection. The suggestion menu shows
`ClassName (preset_name)` for a preset match, distinct from a plain
class-name match. `spawnAndConnect()` handles the real, honest
limitation this exposes: a preset match's port doesn't exist on the
freshly-spawned node yet (it still renders its default/common shape --
sockets reshaping to match a chosen preset is the separate,
not-yet-built "actual interactive editing" piece below), and
`addConnection()` does no validation of its own -- so rather than
silently create a dangling connection, spawning stops and tells the
person what's left to do by hand.

**Verified without a browser, honestly scoped.** This project has no
JS test infrastructure (no `module.exports` anywhere in `nodegraph.js`,
never tested before this session) -- adding that is a real structural
decision, not made unilaterally here. Instead: the exact matching
algorithm (checked line-for-line against the committed source) was run
standalone against mock registry data shaped like a real
`node_info_to_dict()` response, covering a dynamic node's preset-only
match, its common-shape match, and a static node's match, each checked
for the right count and the right `preset`/port identity -- this is
what actually caught the bug above. Syntax and CSS brace-balance
checked directly on the real file. **Not independently confirmed in an
actual browser** -- worth a real look before calling this done, same
caveat as the earlier node-header CSS change.

**Still open, unaffected by this:** `graph_executor.py`'s
`_is_compatible()` resolving live shape from `spec.params` for the
actual interactive-editing case -- a separate, heavier operation than
the required-only preset enumeration this phase built, and what's
needed before a spawned node's sockets can actually reshape to match a
selected preset (closing the `spawnAndConnect()` limitation above).
