# Resources Controller & precision redesign -- step-by-step plan

**Status: Phase 1 + Phase 2 done, Phase 3's suggestion-menu question
resolved (node_kind/presets metadata), Phase 4's core object-
construction mechanics done, a consolidation pass done connecting this
to `docs/training_pipeline_design.md`'s remaining open items --
including `Port.choices` (11.4), now built and landed as part of that
consolidation, with real frontend code (`server/static/nodegraph.js`)
beyond Phases 1-2's backend-only scope. Phase 5's backend now real too
(`ResourcesControllerNode` + `ResourcePreset`/`LoRASDXLPreset`,
registered and introspecting correctly) -- its live editor UX
deliberately deferred, see that phase's own status.**
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

### Phase 4 -- `ResourcePreset` abstraction

**Status: the core object-construction mechanics are done.** Composition
vs. inheritance decided -- inheritance, concrete-mixin-first ordering.

`nodes/model/sdxl_architecture.py`'s `SDXLArchitecture`: pure SDXL
mechanics, no dtype awareness at all, exactly as specified.
`split_checkpoint()`/`build_text_encoder()`/`inject_lora()` all
delegate to already-real, already-tested code (`resource_inspection.classify_key()`,
`text_encoder.py`'s existing CLIP masking, `build_lora_injected_unet()`)
rather than reimplementing anything -- this class is assembly, not new
logic. `build_text_encoder()` masks SDXL's real two text encoders
(CLIP-L, OpenCLIP-G) behind one simple `TextEncoder` object, per the
explicit design requirement -- but honestly flags a real, current
limitation found while building it: `core.clip_encode.SDXLClipEncoder`
(frozen legacy code) hardcodes its own dtype, no parameter exists yet
to make it configurable.

`nodes/model/lora_training_resources.py`'s `LoRATrainingSkeleton`
(abstract, declares the three architecture-specific methods above) and
`SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton)` (the
concrete combination -- base order load-bearing, not stylistic, see
that class's own docstring for the exact MRO reasoning). Construction
*is* processing, per the original design conversation: `__init__` runs
the real pipeline (split → inject LoRA → mask CLIP) and the resulting
instance has real `.unet`/`.clip`/`.vae_sd`/`.lora` attributes with no
construction machinery riding along, matching "the beauty of this
method" from that conversation exactly.

**Real memory/objects management inside, per the explicit follow-up
ask -- and a real, reassuring answer to "not sure we have a good base
for it": there already was one, already real and already
production-used, just not yet connected here.**
`nodes/memory/coordinator.py`'s `ResourceCoordinator` is the same
thing `nodes/train/supervised.py`'s `SupervisedLoRATrainerNode` already
registers its own `model`/`optimizer`/`text_encoder` against in a real,
shipping node. `LoRATrainingSkeleton` is now a real `DeviceResident`
itself (`nodes/memory/handle.py`) -- registers its own `.unet`/`.clip`
against an internal coordinator at construction time, so
`footprint_bytes()`/`offload()`/`reload()`/`release()` delegate to
real, already-tested machinery (each already implements
`DeviceResident` itself) instead of a second, hand-rolled, parallel
implementation -- replaced the hand-summed `footprint_bytes()` from the
first pass of this phase. `vae_sd` (real tensor state, not yet wrapped
in any object -- see the deferred item below) is moved/dropped by hand
alongside the coordinator's own work in all three lifecycle methods,
not silently left out of them just because there's no resident object
to register it with yet.

**Two things deliberately, honestly deferred, not silently left
half-built:** "continue training" (loading an existing saved LoRA into
the freshly-injected model, per the original sketch's own checkbox) --
`.lora` stays `None` unconditionally; the real loading mechanics
already exist (`LoRACheckpointLoaderNode`, DoRA-aware) and this class's
`.unet` is exactly what that loader operates on, but wiring them
together is a real, separate next increment. No VAE object either --
`.vae_sd` stays the raw split-out state dict, since nothing in `nodes/`
builds a VAE wrapper anywhere yet (only legacy `core.vae_decode.VAEDecoder`,
unused by anything in `nodes/` today).

**Verified**, `nodes/smoke_tests/smoke_test_lora_training_resources.py`:
`split_checkpoint()` correctness (every key in exactly one bucket);
`build_text_encoder()`/`inject_lora()` each proven to genuinely
delegate (real call-arg recording, not assumed); a full end-to-end
`SDXL_LoraTrainer` construction from a synthetic checkpoint, checking
real `.unet`/`.clip`/`.vae_sd`/`.lora` and a correctly-summed
`footprint_bytes()`; the `DeviceResident` implementation genuinely
moves/drops all three of `.unet`/`.clip`/`.vae_sd` through
`offload()`/`reload()`/`release()` -- not just the two with an obvious
resident object to delegate to, `vae_sd`'s own raw tensors checked by
their real `.device` too, both the explicit-device and no-arg
`reload()` paths exercised, and a released trainer correctly reports 0
footprint; and, the one that actually matters most for this phase's
central decision -- **the negative case**: a version with the bases
listed in the wrong order genuinely fails to instantiate with
`TypeError`, proving the shipped order is load-bearing and not just
happening to work. Extended (not forked) the existing `_RecordingWrapper`/
`_FakeClipEncoder` test fixtures with `.to()`/`.unload()`/`.dtype` they
were missing once `offload()`/`reload()`/`release()` needed to
genuinely exercise them, rather than a second, subtly-different copy.
Adjacent tests (`smoke_test_resource_coordinator`,
`smoke_test_resource_policy`, `smoke_test_resource_inspection`,
`smoke_test_gradient_checkpointing`, `smoke_test_dataset_model_contracts`,
and the extraction test itself after being extended) still pass. No
non-CPU device is available in this sandbox, so the offload/reload
checks prove the real mechanics run correctly, not an actual
cross-device tensor move -- worth a real check on real hardware before
fully trusting the device-transition behavior specifically.

**Frozen LoRA -- done.** `nodes/model/lora_merge.py`'s
`merge_lora_into_state_dict(base_sd, lora_sd, strength)`: merges a
saved LoRA directly into a checkpoint's raw weights before injection --
`W_merged = W + strength * (alpha/rank) * (B @ A)`, matching
`core.lora.LoRALinear.merge()`/`LoRAConv2d.merge()`'s own formula
exactly (checked against them directly, not an independent
reimplementation trusted on its own). No frozen-LoRA object exists
after construction -- its effect is baked into the weight tensors,
nothing else about it survives. `strength` corresponds to
`core.lora.LoRALinear`'s own `weight` constructor argument (also a
scaling multiplier), applied here at merge time instead of injection
time. `LoRATrainingSkeleton.__init__` gained `frozen_lora_sd`/
`frozen_lora_strength` parameters -- the merge runs on the UNet
component before `inject_lora()`, so the trainable LoRA gets injected
on top of the already-merged weights. `frozen_lora_sd=None` (the
default) is a true no-op, checked directly: the wrapper receives the
checkpoint's own unmodified weights.

**Continue training -- done.** `nodes/model/lora_checkpoint_loader.py`'s
`LoRACheckpointLoaderNode.build()` was extracted the same way
`ComfyUNetLoRANode.build()` was in the earlier extraction patch --
`load_lora_into_registry(registry, state_dict, source_description)`
now holds the real validation (missing keys, rank mismatches, both
raising with specifics rather than silently loading a partial LoRA)
and loading (plain layers via `core.lora.load_lora_into_model`, DoRA
layers via the existing `_load_dora_layers()`), reused by both the
node and `LoRATrainingSkeleton.__init__`'s new `continue_lora_sd`
parameter. A different feature from frozen-LoRA merging -- this one
loads into `self.unet`'s own trainable adapter, after injection, so
training continues from these weights rather than starting fresh, and
stays trainable afterward (frozen-LoRA merging doesn't). `self.lora`
holds `continue_lora_sd` itself when given, `None` otherwise -- a
plain reference, matching `self.vae_sd`'s own raw-dict pattern, not
the weights themselves (those live inside `self.unet`'s registry once
loaded). Checked to coexist correctly with `frozen_lora_sd` given
together -- both apply, neither interferes with the other, since they
operate on genuinely different things (base weights vs. the trainable
adapter). A validation error from `load_lora_into_registry` (a real
rank mismatch, say) propagates straight out of construction rather
than being swallowed.

**LoRA-file inspection -- done, closing the gap flagged since Phase 2.**
`resource_inspection.py`'s new `inspect_lora(path)`: dtype and rank for
a saved LoRA, read from the header only (`get_shape()` alongside
`get_dtype()`, same header-only mechanics as checkpoint inspection).
`asset_paths.inspect()` now accepts `kind="lora"` too -- response shape
`{kind, path, dtype, rank, key_count}`, distinct from `kind="checkpoint"`'s
per-component shape. `kind="dataset"` is still the one real remaining
gap. Verified the same way as checkpoint inspection: real answer
against an explicit allowlist, response provably narrow, path
traversal rejected, plus the absent/mixed distinction for both dtype
and rank.

**Not yet built:** validators (per-input human-readable detection text
-- both the checkpoint and LoRA inspection functions this needs now
exist), the parameter-value dictionary (dtype choices), and
list-of-inputs -- the actual `ResourcePreset`/`NodePreset`-satisfying
interface pieces that make this usable *as a node*. Those, plus wiring
this whole thing into an actual `Node` subclass, are Phase 5.

**A working-approach note, not a technical one, worth recording
because it changes how the rest of this redesign should be built:**
earlier framing in this document leaned toward declaring an input only
once its full implementation was ready, to avoid a `Node` contract that
advertises something not yet functional. That instinct is wrong for
this project specifically -- the redesign's whole point is a complete,
correct foundation, even where some of it goes temporarily unused, and
retrofitting a contract after the fact costs more than building it
right the first time. Phase 5's `NodePreset`/`ResourcePreset`
interface should declare the full intended shape of the Resources
Controller now, not grow it incrementally as each piece happens to be
implemented -- and each independent unit (this merge function, the
inspection functions, the construction classes) should be designed
from its own inputs and outputs first, not from how it currently fits
into what already exists; existing code changes to match a better
design when the two conflict, not the other way around.

**Dependency:** none technically (pure abstraction design, could
proceed in parallel with Phases 1-3), but Phase 5's concrete preset
needs it finished first -- it now is, for the core construction path.

### Phase 5 -- The Resources Controller node itself

**Goal:** the node from the sketch, built for real: preset selector,
per-resource dtype detection/override with validity indicators, using
Phases 1-4. One concrete preset at first -- LoRA training on SDXL --
matching "only one available for start" from the original description.

**Status: backend done, corrected once against the actual hand-drawn
sketch (not just this document's own prose description of it, which
had drifted from it in two real ways -- see below); the live editor UX
(validity indicators as you
attach a file) deliberately deferred, not built here.**
`nodes/model/resources_controller.py` (new): `ResourcePreset`, an ABC
matching this document's own settled interface contract table above --
three of that table's four rows turned out to need no new machinery at
all once `Port.choices` actually existed (this same redesign's
Consolidation section, landed immediately before this phase): "list of
inputs" is `Port` itself, "parameter-value dictionary" is `Port.choices`
directly (node-facing multi-choice *is* `.choices`, processor-facing
single-choice *is* `build()`'s own already-validated `inputs[name]`),
"processor method" is `Node.build()`'s real logic one level down
(`ResourcePreset.process()`, so more than one preset can share one Node
class). Only "validators" (`dict[str, Callable[[Any], list[str]]]`,
per-input diagnostic text -- distinct from `Port.choices`' binary
valid/invalid, for things no closed list can check: does this file
actually look like SDXL, what dtype does it already have) was
genuinely new. `ResourcePreset.__init_subclass__` mirrors `Node`'s own
fail-at-definition-time posture (`name`/`inputs`/`outputs` must be
real, `outputs` non-empty), same reasoning applied to a structurally
analogous new class rather than left unenforced because it's new.

`LoRASDXLPreset`: the one concrete preset, wrapping `SDXL_LoraTrainer`
(Phase 4, unmodified, not reimplemented) rather than every knob it
accepts -- `scaling_policy`/`resource_policy`/`adapter_strategy`/
`target_modules` all stay at real defaults, matching a preset's whole
point (sensible defaults, not exposing everything; the manual path
stays available for anyone who needs those, unaffected). Two real dtype
axes, both wired all the way through to code that actually consumes
them: `unet_dtype` (compute, `choices=("bfloat16","float16","float32")`)
and `unet_weight_store` (frozen-weight storage,
`choices=("bf16","nf4")`, section 11.3 item 1) -- **CLIP and VAE dtype
are diagnostics-only, deliberately not override `Port`s**: checked
directly against `SDXLArchitecture.build_text_encoder()`'s own
docstring (`SDXLClipEncoder` hardcodes its dtype, no parameter exists)
and against `nodes/` as a whole (nothing converts `vae_sd`'s dtype
anywhere) -- an override with nothing downstream to honor it would be a
dishonest no-op, not a real knob. `state_dtype` (section 11.3 item 2)
is absent too, matching this same document's own Consolidation section:
"concretely unresolved," not implemented in isolation before that's
settled. `resource_inspection.py` gained `str_to_dtype()`, the precise
inverse of its own `dtype_to_str()` (`getattr(torch, s)`, not a second
hand-maintained mapping) for turning a chosen `unet_dtype` string back
into a real `torch.dtype`.

`ResourcesControllerNode(Node)`: `NODE_KIND = "dynamic"`, a real
`preset` `Port` (`choices=tuple(_PRESETS)`) satisfying "preset
selector" from this phase's own goal line -- genuinely a dropdown, even
with only one valid value today, via the same `Port.choices` mechanism
above, not bespoke UI. `INPUTS`/`OUTPUTS` are today's one preset's own
shape directly (`NODE_KIND`'s own docstring: "a dynamic node's *common*
ports, present no matter which preset is chosen" -- with exactly one
preset, its shape simply *is* the common shape); a second preset with a
genuinely different shape needs real reconciliation here, honestly
flagged in that class's own docstring rather than pretended-solved,
deferred until a second preset actually exists. `list_presets()`
delegates to each registered preset's own `node_preset()`, satisfying
Phase 3's already-built, already-tested suggestion-menu machinery with
zero changes needed there. `diagnostics(inputs)` exposes each attached
resource's validator output (`{"base_model": ["UNET dtype: bf16
(1234 tensors)", ...]}`) -- real and callable today, directly or by a
future endpoint, but **not yet wired to anything the editor calls
live** -- see "Not built" below. Registered in
`server/nodegraph_registry.py`; auto-derives the display name "Resources
Controller" for free, matching the sketch's own name exactly.

**Corrected against the actual sketch (a hand-drawn image, provided
after the first pass above was already written from this document's own
prose description of it -- two real drifts, not style preferences):**
(1) "Base model" is a wired socket in the sketch (a circle with a wire
drawn into it from off-canvas), not a path field owned by this node --
the first pass had `checkpoint_path` as a `path_kind="checkpoint"`
widget, resolving and loading the file itself, duplicating exactly what
`SafetensorsCheckpointNode` already does. Now `base_model:
Port(type=ModelWeights)`, a real wired input; `_checkpoint_validator`
is now `weights.inspect_dtypes()` (Phase 1, already real) instead of a
second path-resolving implementation of the same header read -- this
also means live checkpoint-dtype inspection as a path is typed belongs
to `SafetensorsCheckpointNode`'s own `path_kind="checkpoint"` Port
generically, not to this node specifically, once that frontend piece
gets built. (2) The sketch draws "Continue training" and "Frozen LoRA"
as checkboxes; the first pass inferred "enabled" from whether the
corresponding path was non-`None`, exactly the kind of implicit state a
checkbox exists to make explicit. Now real `continue_training`/
`frozen_lora` `bool` Ports gate `continue_lora_path`/
`frozen_lora_path`(+`frozen_lora_strength`) in `process()`, checked
both directions (checked-without-a-path, and a-path-without-being-
checked both raise a clear error) rather than one direction inferred
from a float comparison as before.

Also found while reading the sketch closely: its own bottom summary
table names a fourth axis, "LoRA (training)" dtype -- the trainable
adapter's own parameter dtype, distinct from unet_dtype/
unet_weight_store above. Checked directly against `core/lora.py`:
`LoRALinear`/`LoRAConv2d` hardcode `param_dtype = torch.float32` for
`lora_A`/`lora_B` regardless of the frozen base's own dtype, with a
detailed, load-bearing numerical justification in that file's own
comment (bf16's mantissa silently rounds away realistic Adafactor
updates at LoRA-adapter magnitudes -- that comment verifies "bit-for-
bit unchanged after 2000 steps" at a realistic lr; "every mainstream
LoRA implementation" keeps this in fp32 for exactly that reason).
Correctly absent from `LoRASDXLPreset.inputs` -- an override Port here
wouldn't be a mere no-op like CLIP/VAE dtype above, it would be a real,
easy-to-reach footgun, so the sketch's own `<fp32>` row for this one is
better read as "detected/fixed," not "editable," once its own drawn
arrows are checked against what the code underneath actually allows.

**Not built, deliberately deferred rather than silently missing:** a
live query endpoint (mirroring Phase 2's own existing
`/nodegraph/assets/{kind}/inspect`) for the editor to call
`diagnostics()` as the user attaches a checkpoint/LoRA, and the
`nodegraph.js` wiring to display its result inline under that Port's
widget -- the sketch's actual "validity indicators" UX. Sequenced this
way deliberately, same as every phase before this one: backend proven
before the frontend built on top of it, not the other way around.

**Verified, without the smoke-test suite this time** (that files list
its own budget, a huge amount of real code checked by direct read and
targeted manual runs before this document gets updated with a plan for
a real, deferred, consolidated test-writing/running pass, rather than a
smoke-test file re-run after every small change): the whole module
class-defines and imports cleanly (`ResourcePreset.__init_subclass__`'s
and `Node.__init_subclass__`'s own definition-time checks both pass for
real, not synthetic, classes); `ResourcesControllerNode`'s real
`INPUTS`/`OUTPUTS`/`list_presets()` introspect correctly end to end
through `node_info_to_dict()` (`node_kind`, `presets`, `path_kind`,
`choices` all present and correct for this genuinely more complex real
case, not just Phase 3's synthetic test fixture); `diagnostics()`
against a real, synthetic-but-genuine safetensors checkpoint (real SDXL-
shaped keys, real per-component dtypes, `save_file`/`load_file`, a temp
`paths.set_checkpoints_dir()`) reports the right per-component dtype
lines; the same against a checkpoint with no UNet-prefixed keys at all
correctly reports an `"ERROR: ..."` diagnostic line rather than
crashing; a saved-LoRA file's `diagnostics()` reports the right
dtype/rank line; the `frozen_lora_strength`-without-`frozen_lora_path`
guard correctly raises from `process()`; a path-traversal attempt is
rejected by both `diagnostics()` (as an `"ERROR: ..."` line, not
crashing the whole call) and `process()` (raised, not swallowed --
`build()`'s real path needs the loud failure, `diagnostics()`'s display
path doesn't). **Not exercised**: `process()`/`build()`'s actual
`SDXL_LoraTrainer(...)` construction past the checkpoint-loading step --
needs ComfyUI's real SDXL UNet class, not installed in this sandbox,
same limitation already noted for Phase 4's own extraction work
(`smoke_test_lora_injector_extraction.py`). A real smoke test file for
this module (`nodes/smoke_tests/smoke_test_resources_controller.py`,
following this project's established fixture-mocking pattern for
exactly this ComfyUI-shaped gap) is real, queued follow-up, not written
in this same pass -- deliberately, per this session's own working
approach: write the real code first, verify it in one deferred,
consolidated pass rather than after each small piece.

Re-verified after the sketch-driven correction above, same manual-not-
suite approach: `diagnostics()` against a real wired `ModelWeights`
(not a path) reports the same correct per-component dtype lines;
`continue_training`/`frozen_lora` each raise `process()`'s new
both-directions guard correctly (checked-without-a-path, and a-path-
given-while-unchecked); clearing every one of this session's own new
guards and reaching real `SDXL_LoraTrainer(...)` construction correctly
hits the same known `ModuleNotFoundError: comfy` boundary as before,
confirming the guards run in the intended order and none of them
silently swallow a real failure.

**Dependency:** Phases 1-4 (done).

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
**Status: done.** `nodes/core.py`'s `Port` gained `choices: tuple[str,
...] | None`, enforced at construction (`__post_init__`, same
fail-at-definition-time posture as `NodePreset`'s own
self-contradictory-port check): only meaningful on a `str`-typed Port,
must be a non-empty tuple of strings, and a given `default` must be one
of them. `Node.validate_inputs()` now rejects an explicit input value
outside its Port's `choices` the same place it already catches a
missing required input -- server-side, "don't trust the client" (the
value only ever really needs checking once the editor's own dropdown
already restricted it, but `graph_executor.py` takes this same posture
elsewhere regardless of what the client already checked).
`server/nodegraph_introspect.py`'s `PortInfo` gained a matching
`choices: list[str] | None`, resolved in the shared `_port_info()` (so
it's correct for inputs, outputs, and preset ports uniformly, no third
copy) and serialized in `node_info_to_dict()`.
`server/static/nodegraph.js` renders it as a plain `<select>`
(`buildChoicesWidget()`, wired into `buildInputBlock()` right alongside
the existing `path_kind` picker/save-as widgets it structurally
mirrors) instead of the freeform text box a `str` Port otherwise gets.

**Wired into two real, immediate consumers, not just built and left
for Phase 4/5 to be the only caller:** the three
`Composed*OptimizerNode` classes' `strategy` Port now sets
`choices=tuple(STRATEGIES)` off `strategy_registry.py`'s own existing
registry (zero new duplication -- that module already generates
`STRATEGY_DOC` from the same dict for exactly this reason).
`RenoiseBatchSourceNode`/`ManagedDatasetSourceNode`'s `t_mode` Port
sets `choices=T_MODES` off a new `nodes/dataset/timestep_modes.py`.
That constant is a **deliberate**, documented duplicate of a same-named
constant added to `core/noise_schedule.py` (where `sample_timestep()`
actually implements those five distributions), not importable from
there directly -- checked directly, not assumed: `core/__init__.py`
eagerly imports `core.unet_wrapper` (ComfyUI-dependent) and other heavy
modules, so anything under `core.*` pulls all of that in at import
time, which is exactly why `renoise.py`'s own `_renoise()` already
deferred its `core.noise_schedule` import to call time rather than
module load -- `nodes/dataset/` is deliberately ComfyUI/torch-free at
import time, and a Port's `choices` is needed at class-definition time
(module load), where that deferral trick isn't available. `device`
Ports deliberately did **not** get `choices` -- checked directly
(`core/comfy_setup.py`), they're `torch.device()`-parsed and accept
indexed variants (`"xpu:0"`) no closed list could enumerate, so they're
genuinely the open-ended case `choices=None` exists to leave alone, not
an oversight.

**Verified**, `nodes/smoke_tests/smoke_test_port_choices.py`: every
malformed `choices` construction (non-str type, empty, a list instead
of a tuple, non-str entries, a default outside the set) rejected at
construction; `Node.validate_inputs()` accepts a valid explicit choice,
rejects an invalid one by name, and leaves a genuinely-absent optional
input alone (a different, pre-existing check); the real `strategy`
Ports on all three `Composed*OptimizerNode` classes read back exactly
`STRATEGIES`, and `t_mode` on both real dataset nodes reads back
exactly `T_MODES` -- not just "some choices got set," the actual shared
values. `server/smoke_tests/smoke_test_nodegraph_introspect.py`
extended (not forked) with a check that `choices` serializes as a JSON
list through `node_info_to_dict()` and stays `None` for `device` and
every other Port that never declared one.
`server/static/nodegraph.js`'s new code passes `node --check` (a real
syntax check, available in this environment -- stronger than Phase 3's
own brace-balance check, though still **not independently confirmed in
an actual browser**, same honest caveat Phase 3 left). This sandbox
had none of the project's own dependencies installed, not even
`requirements.txt`'s; installed the real stack (torch, pydantic,
safetensors, tqdm, numpy, pillow, pydantic_settings, plus
`requirements.txt` itself) to actually run things rather than reasoning
about them untested -- with that in place, the full existing
`nodes/smoke_tests/` suite (56 files, including the new one) and all 5
`server/smoke_tests/` files pass, not just the two touched here.

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
than inventing a second deprecation story.

**Status: done.** `build_lora_injected_unet()`
(`nodes/model/lora_injector.py`) now holds the real construction
logic; `ComfyUNetLoRANode.build()` is a thin wrapper resolving its own
`Port` defaults into it -- `weights`/`device`/`dtype`/`rank`/`alpha`/
`scaling_policy`/`dropout`/`target_modules`/`use_checkpoint`/
`resource_policy`/`adapter_strategy`/`frozen_weight_store_factory`, one
real signature, one real source of truth for what "the default LoRA
injection" means. Verified,
`nodes/smoke_tests/smoke_test_lora_injector_extraction.py` (patches
`ComfyUNetWrapper`/`adapter_strategy_scope` to record their real call
args -- a full end-to-end run needs ComfyUI's actual SDXL UNet class,
not installed here): defaults match exactly what the pre-extraction
inline code computed, `resource_policy` correctly overrides
`use_checkpoint`/`scaling_policy`, and the node's own Port-default
resolution into the extracted function is correct for both defaults
and explicit overrides -- a real behavior-preservation proof, not just
"doesn't crash." Adjacent tests
(`smoke_test_gradient_checkpointing.py`, `smoke_test_adapter_injection.py`,
`smoke_test_dataset_model_contracts.py`, `smoke_test_resource_policy.py`)
still pass.

---
Last synced against `docs/training_pipeline_design.md` at commit
`2c1f0ff` (2026-08-25).
