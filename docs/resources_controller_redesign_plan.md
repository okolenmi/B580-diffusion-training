# Resources Controller & precision redesign -- step-by-step plan

**Status: Phase 1 + Phase 2 done, Phase 3's suggestion-menu question
resolved (node_kind/presets metadata), a consolidation pass done
connecting this to `docs/training_pipeline_design.md`'s remaining open
items -- no new node/server/frontend code yet beyond Phases 1-2.**
This tracks a real,
multi-session redesign, not a single patch -- update it as phases land
or as open questions get resolved, the same way `PROGRESS.md` tracks
the rest of this project.

**Priority rule, made explicit rather than left implicit:** where this
plan and `docs/training_pipeline_design.md` conflict, this plan wins --
it reflects the more recently decided direction. Section 11's older
items get updated or superseded as needed (see "Consolidation" below
for the concrete cases found so far), not treated as equally
authoritative history that both documents have to be reconciled around
forever. `training_pipeline_design.md` stays the record of what
actually shipped and why for everything outside this redesign's scope;
it isn't frozen, just not the tie-breaker for anything this plan
touches.

## Why this exists

`docs/training_pipeline_design.md` section 11.3 identified three
separate, scattered precision decisions (frozen weight storage dtype,
optimizer state dtype, compute dtype) and deliberately recommended
*against* bundling them, matching `ResourcePolicy`'s own precedent of
keeping `adapter_strategy`/`frozen_weight_store` orthogonal rather than
folded into one object. That recommendation included: "no preset
bundle proposed... worth adding later if real, explicit demand shows
up, not speculatively now." That demand has now shown up, from a real,
concrete sketch of a "Resources Controller" node: one place that owns
loading, dtype detection/override, and validity for every resource a
training run needs, replacing today's fully eager, dtype-blind loader
nodes.

This is not a reversal of 11.3's reasoning so much as it finishing what
11.3 said would happen if real demand arrived -- but the actual shape
requested is bigger than a single node: an interactive node whose
inputs/outputs settle live as the user configures it (not a fixed
`INPUTS`/`OUTPUTS` dict resolved once at class-definition time, which
is what every node in this project has been so far), backed by a new
kind of server<->editor query channel, driven by a formal, composable
preset abstraction. Each of those is independently real engineering.

## Ground truth this plan is built against, not assumed

Checked directly before writing this, so the phases below are grounded
in what's actually there rather than guessed at:

- **Loading is currently eager and dtype-blind.**
  `SafetensorsCheckpointNode.build()` (`nodes/model/checkpoint_loader.py`)
  calls `safetensors.torch.load_file()` -- the entire checkpoint,
  materialized, in whatever dtype it was saved in -- right at that
  node's own `build()`. `ModelWeights` (`nodes/model/handle.py`) is
  plain, already-loaded `dict[str, Tensor]` data, not a reference. This
  is exactly the coupling that has to change for a Resources Controller
  to be able to detect dtype *before* committing to a load.
- **Header-only dtype inspection is a real capability, not a
  hypothetical.** safetensors files store a JSON header (tensor name ->
  dtype/shape/offset) before the actual weight bytes;
  `safetensors.safe_open()` can answer "what dtype is this tensor" for
  a multi-GB file without materializing any tensor data. This is what
  makes "query dtype the instant a resource is attached" cheap enough
  to actually do.
- **Every edge in the graph is type-checked before any node runs.**
  `server/graph_executor.py`'s `_is_compatible()` checks every edge
  against `from_cls.OUTPUTS`/`to_cls.INPUTS` -- static, class-level --
  before the topological run starts. An interactive node's shape has to
  be *settled* (concrete `params`, not still-open choices) by the time
  Run is pressed, or this check has nothing to check against. This is
  compatible with "the node is interactive in the editor" -- it is not
  compatible with "the node's shape depends on a value carried by a
  wire at Run time."
- **A real, close sibling for the new query endpoint already exists.**
  `server/asset_paths.py` (`_safe_resolve()`, `browse()`,
  `list_options()`) already sandboxes exactly this kind of untrusted,
  network-reachable, `kind`-scoped path access, wired in via
  `server/routes_nodegraph.py`'s existing `/assets/{kind}/...` family
  (`browse`, `upload`, `mkdir`). A dtype-inspection endpoint is a
  natural new sibling in that same family and should reuse
  `_safe_resolve()` directly rather than reinvent sandboxing.
- **Composition-over-inheritance already has a real precedent here for
  exactly this kind of two-axis combination.** `DoRALinear`
  (`nodes/model/dora_layer.py`) deliberately wraps
  `core.lora.LoRALinear` via composition rather than inheriting from
  it. The bugs fixed in the previous two patches
  (`split_into_new_generation`, `reenable_dora_requires_grad`,
  `dora_trainable_parameters`) were all downstream of code elsewhere
  assuming an attribute layout that composition doesn't give you for
  free -- a real, paid-for lesson about combining two independent
  concerns in this exact codebase, not a generic preference.

## Open design question blocking Phase 4 -- one item left

The Task axis ("LoRA Training", later "distillation", ...) and the
Architecture axis ("SDXL", later others) compose into one concrete
preset per (task, architecture) pair. The `SDXLArchitecture` side is
deliberately a fixed, concrete, non-generic implementation for now --
SDXL's real two-encoder text-encoder setup (CLIP-L + OpenCLIP-G) is
masked behind a single, simple "clip" surface at this layer, not
exposed as a pluggable N-encoder abstraction. Generalizing past SDXL
happens later, when a second real architecture actually needs it, not
speculatively now -- matches this project's own established preference
throughout `docs/training_pipeline_design.md` for building the one
real thing before extracting an abstraction from it.

`SDXLArchitecture`'s own job is purely mechanical -- checkpoint
splitting/parsing, the CLIP-masking above, adapter-injection targets --
and has **no relationship to dtype at all**. Dtype lives entirely in
the `ResourcePreset` interface layer described below, not in the
architecture layer.

**The `ResourcePreset` interface contract (settled):**

| Piece | Contract | Maps to (node-graph side) |
|---|---|---|
| list of inputs | Declared inputs, some required only conditionally (e.g. only if a checkbox is checked) | `INPUTS`, resolved dynamically per Phase 3 rather than fixed at class-definition time |
| validators | Per-input `validate(value) -> list[str]` -- human-readable diagnostic lines, not just pass/fail | Rendered under that socket, matching the sketch's "UNET dtype: bf16" / "[valid]" rows |
| parameter-value dictionary | Node-facing: `{resource: [available dtype choices]}` (multi-choice, for the dropdown). Processor-facing: `{resource: chosen dtype}` (single, resolved) | The dtype override table at the bottom of the sketch |
| processor method | Takes resolved inputs + the resolved (single-choice) dtype dict, returns the real object | `build()` |

The object `processor()`/`build()` returns is a plain,
undecorated domain object (e.g. `SDXL_LoraTrainer(unet=..., clip=...,
vae=..., lora=None)`) -- none of the interface/preset machinery that
built it rides along on the output. It should grow a
`footprint_bytes()` method matching the pattern already established by
`ComfyUNetTrainableModel.footprint_bytes()`/`FrozenWeightStore.footprint_bytes()`,
not invent a new one, and should be shaped so it can later plug into
`nodes/memory/manager.py`'s `MemoryManager`/`ResourceProfile` (built,
real, but not yet threaded through anything real per `PROGRESS.md`) --
a genuine, already-flagged use case for "extend this object with extra
memory control later," not a hypothetical one.

**Composition mechanism -- one line still open.** Both are workable:

- *Composition*: `SDXL_LoraTrainer` holds an `SDXLArchitecture`
  instance, delegates to it. No MRO to reason about. Matches the
  `DoRALinear`-over-`LoRALinear` precedent already in this codebase.
- *Multiple inheritance*, done correctly: `class
  SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton)` --
  concrete mixin listed *before* the abstract skeleton. Python resolves
  a method by the first match walking the MRO left to right, not by
  "is there a better one further down" -- listing the skeleton first
  instead silently makes the class fail to instantiate (its abstract
  stub wins attribute lookup over the real implementation sitting right
  there as a later base). Standard idiom for exactly this pattern once
  the base order is right.

Not yet decided which one this project uses.

## Phases

Sequenced so that every phase through Phase 2 is independently buildable
and smoke-testable with zero frontend changes, matching how every other
patch in this project has been verified -- the frontend/interactive-node
work (the single riskiest, least-scoped piece) comes after the backend
foundation it depends on exists and is proven, not before.

### Phase 1 -- Lazy resource references + header-only inspection

**Status: done.**

- `nodes/model/resource_inspection.py` (new): `classify_key()` (unet/
  vae/clip by real SDXL prefix -- `model.diffusion_model.`,
  `first_stage_model.`, everything else masked as one CLIP bucket) and
  `inspect_checkpoint_dtypes()` (header-only, via `safe_open()`,
  confirmed directly against a real file to touch no tensor data).
  Reports `ComponentDtype(dtype, key_count)` per component -- `dtype`
  is `None` for either "absent" (`key_count == 0`) or "genuinely mixed"
  (`key_count > 0`, disagreeing dtypes), kept distinguishable rather
  than collapsed into one sentinel.
- `ModelWeights` (`nodes/model/handle.py`): now constructed from a path,
  not materialized dicts. `unet_sd`/`non_unet_sd` are lazy, cached
  `@property`s -- real load only happens on first access, exactly once.
  `from_state_dicts()` classmethod keeps the old eager, no-file
  construction available (test fixtures, mainly). `inspect_dtypes()`
  delegates to `resource_inspection.py`, raises a clear error if called
  on a `from_state_dicts()` instance (nothing to peek at -- read dtype
  off the tensors directly instead).
- `SafetensorsCheckpointNode.build()` no longer calls `load_file()` --
  does a cheap header-only existence/format check (fails loudly at
  build() time on a bad file, not later when something first touches
  `.unet_sd`) and hands back a lazy `ModelWeights`.
- Zero call-site changes needed in `ComfyUNetLoRANode` or the text
  encoder node -- both still do plain `weights.unet_sd`/
  `weights.non_unet_sd` attribute access, unaware anything changed.

**Verified**, `nodes/smoke_tests/smoke_test_resource_inspection.py`:
`build()` itself never calls `load_file()`; first real access loads
exactly once, a second access doesn't reload; the eventually-
materialized data is identical (keys, values, dtypes) to what the old
eager path produced, not just "didn't crash"; `inspect_dtypes()`
matches a real load's actual dtypes and never triggers the lazy
full-load cache; the mixed/absent distinction; a bad checkpoint path
fails clearly at `build()` time. All 52 `nodes/smoke_tests` files pass
(51 before this phase + this one).

### Phase 2 -- Server query endpoint

**Status: done.**

- `GET /nodegraph/assets/{kind}/inspect?path=...` (`server/routes_nodegraph.py`),
  a thin wrapper -- same convention as `browse_assets`/`list_assets` --
  around the real logic in `server/asset_paths.py`'s new `inspect()`,
  built directly on the existing `_safe_resolve()` (same sandboxing
  `browse()`/`upload()` already use).
- Response contract, deliberately narrow: `{kind, path, components:
  {unet|clip|vae: {dtype, key_count}}}` -- nothing else. `dtype` is a
  plain string (`nodes/model/resource_inspection.py`'s new
  `dtype_to_str()`, shared rather than reimplemented -- also what a
  future validator's human-readable text line would use), not a raw
  `torch.dtype`.
- Only `kind="checkpoint"` is supported -- LoRA-file inspection needs
  its own function (different key format, different meaningful fields
  like rank) and is explicitly left as real, separate follow-up rather
  than silently folded in here.

**Verified**, `server/smoke_tests/smoke_test_asset_inspect.py` (targets
`asset_paths.inspect()` directly -- same "test the function the route
delegates to" convention as `smoke_test_execution_registry.py`, not the
HTTP layer): the real answer checked field-by-field against an explicit
allowlist; the response is provably narrow (no key beyond the
documented contract, top level and per-component); path traversal /
absolute / empty path rejected the same way `resolve_safe_model_path`
already rejects them elsewhere in this project; a nonexistent file, a
directory instead of a file, a corrupt/non-safetensors file, and an
unsupported `kind` all fail with a clear message rather than a crash or
a silent wrong answer. Router wiring confirmed (`/nodegraph/assets/
{kind}/inspect` present alongside the existing asset routes); the three
existing `server/smoke_tests/` files still pass.

**Dependency:** Phase 1 (done).

### Phase 3 -- Interactive node support (editor + core.py + introspection)

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

**Dependency:** Phase 2 (done). Not started -- next real step once
picked back up is designing the actual `node_kind`/`presets` shape in
`nodes/core.py` and `NodeInfo`, now that the suggestion-menu question
has a concrete answer rather than being an open blocker.

### Phase 4 -- `ResourcePreset` abstraction

**Goal:** formalize the Task x Architecture composition from the open
design question above, once it's actually settled. Concrete classes
not written here yet -- see that section.

**Dependency:** none technically (pure abstraction design, can proceed
in parallel with Phases 1-3), but Phase 5's concrete preset needs it
finished first.

### Phase 5 -- The Resources Controller node itself

**Goal:** the node from the sketch, built for real: preset selector,
per-resource dtype detection/override with validity indicators, using
Phases 1-4. One concrete preset at first -- LoRA training on SDXL --
matching "only one available for start" from the original description.

**Dependency:** Phases 1-4.

### Phase 6 -- Downstream integration (`TrainerNode` and friends)

**Goal:** resolve the still-open question from earlier in this same
conversation -- does this node's output replace `TrainerNode`'s
separate `model`/`optimizer`/`text_encoder` ports with one bundled
port, or does it feed into the existing `ComfyUNetLoRANode`-style
construction path instead, with those ports staying as they are today?
Explicitly deferred until Phase 5 exists and its actual output shape
is settled -- the original sketch's own annotation already flagged the
output as undecided ("not sure what should be output, so there is
nothing currently").

**Dependency:** Phase 5.

## Consolidation -- resolving low-synergy items instead of leaving them
to drift

Explicit instruction behind this section: don't let this redesign
become one more thing sitting next to the rest of the project's design
rather than actually reconciled with it. Went back through
`docs/training_pipeline_design.md`'s remaining open section-11 items and
the already-shipped `ComfyUNetLoRANode`/`LoRAPhaseSplitNode` against
this plan specifically looking for redundancy, not just letting them
coexist.

**Checked, genuinely independent, no action needed:** section 11.1
(optimizer node consolidation -- which legacy optimizer nodes are safe
to mark deprecated). Pure optimizer-construction concern, nothing to do
with resource loading/dtype. No overlap.

**Checked, genuinely independent, no action needed:** `LoRAPhaseSplitNode`
vs. the sketch's own "Continue training" checkbox. These looked like
they might be the same idea told twice -- they're not. Continue
training (Resources Controller) is "start this run from an existing
saved LoRA's weights." Phase-splitting is "freeze what's been trained
*during this run* and grow a new, independently-trainable generation on
top of it, mid-pipeline." Different points in time, different real
mechanics (`nodes/model/lora_phases.py`'s generation-chain machinery
has no equivalent in "load a checkpoint at the start"). Stays a
separate node.

**Real synergy found, recommend unifying rather than building twice:**
section 11.4 (`Port.choices` -- a generic closed-choice-dropdown
mechanism for *any* `Port`, not resource-specific) and the
`ResourcePreset` interface's own "parameter-value dictionary (multiple
choices in node, single choice for processor)" from the open design
question above are, underneath the different names, **the same
mechanism**: a `Port` that declares a closed set of valid choices,
rendered as a dropdown, resolved to one concrete value by build time.
11.4 was scoped as "a larger item... its own piece of work" back when
nothing concrete needed it yet; it now has a real, immediate consumer.
Building it generically at the `core.py`/`Port` level, once, serves
both 11.4's original standalone case (`strategy`, `device`, and similar
plain string ports elsewhere in the graph) *and* Phase 4/5's dtype
dropdowns -- rather than the Resources Controller inventing its own
bespoke choice-rendering path that 11.4 would later duplicate, or 11.4
shipping first in a shape Phase 4 then has to work around. Practical
consequence: **`Port.choices` (11.4) should likely be built as part of
this redesign's own Phase 3/4 work, not as a separate, later,
disconnected item.** Not started -- flagging the connection now so
whoever picks up either piece next doesn't build it twice.

**Real synergy found, recommend folding in:** section 11.3's item 2
(`state_dtype` on the `Composed*` optimizer nodes, "needs one shared
implementation") is a third precision axis that predates this whole
redesign's starting motivation -- and the redesign's entire premise is
*one place* for precision decisions, not scattered ports. Recommend
`state_dtype` become part of the Resources Controller's own
parameter-value dictionary once that exists (Phase 4/5), rather than a
separate port added directly to each optimizer node in isolation.
Concretely unresolved and worth a real decision when Phase 4/5 gets
designed in detail: does the Resources Controller's own scope extend to
optimizer-adjacent config at all, or does it stay model-resource-only
(UNet/CLIP/VAE/LoRA) with `state_dtype` reading from it via a separate,
smaller wire? Not decided -- but implementing `state_dtype` as an
isolated `Composed*`-node port *before* that's settled risks building
exactly the kind of thing this section exists to avoid.

**Real redundancy risk found, recommend a concrete action now, not just
noting it:** `ComfyUNetLoRANode`'s own `dtype`/`frozen_weight_store`/
`adapter_strategy` ports and its `build()` method are -- once Phase 5
ships -- doing a subset of exactly what the Resources Controller's
processor method needs to do internally (resolve dtype, construct the
injected model). Left alone, Phase 5 either duplicates that
construction logic (two copies to keep in sync, the exact failure mode
`dora_layer.py`'s own composition-over-inheritance choice and the
`_is_unet_key`/`get_lora_weights()` bugs from the last two DoRA patches
both trace back to -- two things secretly needing to stay in sync,
nothing enforcing that they do) or Phase 5 has nothing to build on and
reinvents it. **Recommend extracting `ComfyUNetLoRANode.build()`'s real
construction logic (the `adapter_strategy_scope` + `ComfyUNetWrapper` +
`reenable_dora_requires_grad` sequence) into a standalone, reusable
function now** -- independent of Phase 3/4/5's timeline, low-risk, and
exactly the kind of thing worth doing *before* Phase 5 needs it rather
than as part of Phase 5 under time pressure. `ComfyUNetLoRANode` itself
keeps working unchanged (thin wrapper around the extracted function);
once Phase 5 ships, it becomes the manual/advanced path for someone who
wants fine-grained control without a preset -- same "mark deprecated in
the docstring, point at the replacement, don't delete" pattern section
11.1 already established for the optimizer nodes, reused here rather
than inventing a second deprecation story. Not started.

---
Last synced against `docs/training_pipeline_design.md` at commit
`2c1f0ff` (2026-08-25).
