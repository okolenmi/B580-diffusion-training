*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 3 -- Interactive node support (editor + core.py + introspection)

**`classInfo.inputs`/`classInfo.outputs` -- static, one shared
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

**Real bug caught before landing, not just reasoned about:** the
first version treated "search the common shape" and "search
per-preset" as mutually exclusive for a dynamic node -- silently
dropping matches against its own common ports (which stay present no
matter which preset is chosen, per `NODE_KIND`'s own docstring).

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
checked directly on the real file.
