# Resources Controller & precision redesign -- step-by-step plan

**Status: Phase 1 + Phase 2 done, Phase 3's suggestion-menu question
resolved (node_kind/presets metadata), Phase 4's core object-
construction mechanics done, a consolidation pass done connecting this
to `docs/training_pipeline_design.md`'s remaining open items --
including `Port.choices` (11.4), now built and landed as part of that
consolidation, with real frontend code (`server/static/nodegraph.js`)
beyond Phases 1-2's backend-only scope. Phase 5 done: the Resources
Controller node produces a verified, NOT-yet-LoRA-injected resource
pack (`LoRATrainingResources`) -- see that phase's own status for how
its scope got corrected from an earlier, wider version that did
injection itself, and for what's still browser-unverified. Phase 5
also landed real, generic editor mechanics (`Port.visible_when`,
`Port.widget_only`, a live `Node.diagnostics()` endpoint) any future
node can use, not just this one. Phase 6's own `LoRATrainingConfigNode`
done too -- takes that resource pack and actually injects LoRA
(rank/alpha/frozen-weight-storage), including locking rank when
continuing training from an existing LoRA; downstream `TrainerNode`
integration itself is still open.**
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
| list of inputs | Declared inputs, some required only conditionally (e.g. only if a checkbox is checked) | `INPUTS`, always present structurally; `Port.visible_when` (built, real -- see Phase 5) hides a conditional one's own row in the editor rather than actually removing it from `INPUTS` per preset choice -- a UI hint, not the dynamic-shape-resolution Phase 3 originally left open (still open) |
| validators | Per-input `validate(value) -> list[str]` -- human-readable diagnostic lines, not just pass/fail | Rendered under that socket, matching the sketch's "UNET dtype: bf16" / "[valid]" rows |
| parameter-value dictionary | Node-facing: `{resource: [available dtype choices]}` (multi-choice, for the dropdown). Processor-facing: `{resource: chosen dtype}` (single, resolved) | The dtype override table at the bottom of the sketch |
| processor method | Takes resolved inputs + the resolved (single-choice) dtype dict, returns the real object | `build()` |

The object `processor()`/`build()` returns is a plain,
undecorated domain object (e.g. `LoRATrainingResources(unet_sd=...,
clip=..., vae_sd=..., continue_lora_sd=None)` -- Phase 5's actual
return shape, once its own scope got corrected to stop short of LoRA
injection; see that phase's own section for why) -- none of the
interface/preset machinery that built it rides along on the output. It
should grow a `footprint_bytes()` method matching the pattern already
established by
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

**Goal, as clarified directly (this section previously described a
wider, wrong scope -- see "How this section's scope got corrected"
at the end):** a node interface over the basic functions needed to
turn a checkpoint (plus optional frozen/continue-training LoRAs) into
a ready-to-use, **verified** pack of resources for LoRA training.
Inputs and outputs are meant to be the same shape for any future
(task, architecture) preset, so adding one later is additive, not a
rewrite. For LoRA training specifically, a verified pack is exactly
four things: base unet, clip, vae, and an optional continue-training
LoRA. Frozen LoRA is not a fifth field -- it merges directly into the
base unet at construction time and has no separate identity
afterward. This node does **not** do LoRA injection (no rank, no
alpha, no frozen-weight-storage choice) -- that's a separate, later
node's job (Phase 6 below), which can then do its own real work, like
sizing a continuing LoRA's adapter to that LoRA's own actual rank --
a decision this node has no business making.

**Status: done, including real editor mechanics.**
`nodes/model/resources_controller.py`: `ResourcePreset`, an ABC
matching this document's own settled interface contract table above --
three of that table's four rows need no new machinery at all now that
`Port.choices` exists (this document's own Consolidation section):
"list of inputs" is `Port` itself, "parameter-value dictionary" is
`Port.choices` directly, "processor method" is `Node.build()`'s real
logic one level down (`ResourcePreset.process()`, so more than one
preset can share one Node class). "Validators" (per-input diagnostic
text, distinct from `Port.choices`' binary valid/invalid) was the one
genuinely new piece. `LoRASDXLPreset`: the one concrete preset,
producing `LoRATrainingResources` -- renamed from this class's first
name, `VerifiedResourcePack`, on direct feedback: a standardized,
LoRA-training-specific output type here (rather than a generic
"verified resource pack" name) is what lets Phase 6's own node type
its input against this specific name, the same way `SDXL_LoraTrainer`/
`LoRATrainingSkeleton`'s own naming already anchors the *post*-injection
stage -- "...Resources" (this one, verified/uninjected) vs.
"...Skeleton" (that one, injected/trainable) is the actual
standardization, not just a cosmetic rename
(`nodes/model/lora_training_resources.py`, new) -- `unet_sd`, `clip`
(a real, already-loaded `SDXLTextEncoder`, since
`build_text_encoder()` does real work and isn't LoRA-specific at
all), `vae_sd`, `continue_lora_sd` (`None` unless `continue_training`
is checked). `SDXL_LoRATrainingResources(SDXLArchitecture,
LoRATrainingResources)` mirrors `SDXL_LoraTrainer`'s own
multiple-inheritance shape exactly, reusing
`SDXLArchitecture.split_checkpoint()`/`build_text_encoder()` rather
than reimplementing either -- deliberately does **not** need
`inject_lora()` at all, so the type system itself reflects the scope
boundary above. `LoRATrainingResources` is a real `DeviceResident`
(`footprint_bytes()`/`offload()`/`reload()`/`release()`) and has its
own `describe()` (dtype/footprint per component, `rank` for
`continue_lora` if present) -- the same "universal interface other
nodes may use later" ask this phase's goal already covers, answered
for the pre-injection stage the same way `LoRATrainingSkeleton`
(Phase 4, unmodified, now Phase 6's own tool) already answers it for
the post-injection one.

`ResourcesControllerNode(Node)`: `NODE_KIND = "dynamic"`, a real
`preset` Port (`choices=tuple(_PRESETS)`) -- a dropdown even with only
one valid value today, via `Port.choices`, not bespoke UI.
`checkpoint_path` is a `path_kind="checkpoint"` Port this node
resolves and loads itself -- self-contained, not a wire (a wire
doesn't remove the string/path from the picture either way, it just
moves it to a different node for no real gain here). `unet_dtype`
includes a real `"inherited"` choice (the default), resolving to the
attached checkpoint's own detected dtype at build time via a shared
`_inspect_checkpoint()` helper (also used for the "doesn't look like
SDXL" check) -- neither `process()` nor `_checkpoint_validator()`'s
own display text parses the other's output for structured data.
`continue_training`/`frozen_lora` are real `bool` Ports (checkboxes)
gating `continue_lora_path`/`frozen_lora_path`(+`frozen_lora_strength`)
both structurally (`Port.visible_when`, below) and semantically
(`process()`'s own both-directions check: checked-without-a-path and
a-path-without-being-checked both raise a clear error).

Three mechanisms landed here that are real, generic (not
Resources-Controller-specific) additions to `nodes/core.py`/the
server/editor, not one-off plumbing: **`Port.visible_when`**
(`nodes/core.py`) -- `(other_port_name, value)`; the graph editor
hides a Port's own row unless the named sibling currently holds that
value. Checked at class-definition time (`Node.__init_subclass__`):
the referenced name has to actually be in that class's own `INPUTS`,
so a typo fails loudly there. A UI hint only -- `Node`/`Port` never
read it, so `process()` still enforces the real invariant.
**`Node.diagnostics()`** (`nodes/core.py`) -- a real, generic,
`{}`-default method any node can override; `NodeInfo.has_diagnostics`
(`server/nodegraph_introspect.py`) reports whether a class actually
did, the same is-this-actually-overridden check `NODE_KIND ==
"dynamic"` already needs for `list_presets()`. New endpoint,
`POST /nodegraph/node/{class_name}/diagnostics`
(`server/routes_nodegraph.py`) -- same registry lookup `/run` already
uses, a fresh instance per call, `{params: {...}}` in (exactly `/run`'s
own shape), the node's own `diagnostics()` result out; a
bad/incomplete params dict is an ordinary 400, not a 500.
`server/static/nodegraph.js`: `buildInputBlock()` tags each row with
`data-visible-when-*`/`data-port-name`; `updatePortDotState()`
(already the one real per-change choke point every widget handler
calls) also calls `updateFieldVisibility()` (re-scans/re-hides/shows
every tagged row live) and `scheduleDiagnostics()` (debounced 400ms,
calls the new endpoint for any `has_diagnostics` node, renders the
per-input result as read-only text under that Port's own row via
`fetchDiagnostics()`, new `.ng-diagnostics`/`.ng-diagnostic-line`
styles in `nodegraph.html`). Best-effort throughout: a failed
diagnostics call is silently ignored, Run still goes through the real
`validate_inputs()`/`build()` contract regardless.

A fourth, smaller mechanism, same generic-not-specific posture:
**`Port.widget_only`** (`nodes/core.py`) -- a Port with this set never
gets a wire socket at all, only its own widget; direct feedback that a
checkbox with a redundant wire-point row above it, plus a widget
literally labeled "true", was confusing rather than a real extra
capability. `continue_training`/`frozen_lora` set it now; a
`humanizePortName()` in `nodegraph.js` gives the checkbox's own label
a readable "Continue training" instead of the value it already
represents. Building this exposed a real bug before it shipped:
`updatePortDotState()` (the same per-change choke point
`updateFieldVisibility()`/`scheduleDiagnostics()` above ride on)
returned early whenever a port had no dot to update -- true for every
ordinary port with a connection problem worth flagging, but now also
true for every `widget_only` port, silently breaking both of those for
exactly the two checkboxes this was meant to fix. Fixed: the dot
update and the visibility/diagnostics refresh are independent now: the
one only runs if a dot exists, the other always does.

**Verified without the smoke-test suite** (a deferred, consolidated
testing pass is still the plan -- see below): every method above
checked by direct read plus targeted manual runs against real
synthetic safetensors files -- `diagnostics()`/`process()` against a
real checkpoint and a real saved LoRA (correct per-component dtype
lines, correct rank/dtype detection); both checkbox-guard directions
raise correctly; `unet_dtype="inherited"` resolves correctly and
`process()` reaches the real (ComfyUI-gated) construction boundary
with no `rank`/`alpha` involved at all, confirming the injection code
path is genuinely gone, not just hidden; `LoRATrainingResources`
checked in isolation against a minimal concrete fake subclass (a real
`SDXL_LoRATrainingResources` needs ComfyUI to construct at all) --
`describe()`, `footprint_bytes()`, `offload()`/`reload()`/`release()`,
dtype conversion, and the frozen-LoRA-merge code path all run
correctly; the new `/diagnostics` endpoint called directly (success,
404, path-traversal-as-embedded-error, `{}` for a node with no
overridden `diagnostics()`). `nodegraph.js`: `node --check` only (a
real syntax check, not a functional one) -- **not run in an actual
browser**. The person did hand-test the `Port.choices` dropdown and
the `visible_when` hide/show behavior directly and confirmed both
work; the live diagnostics fetch/render specifically has not been
separately confirmed in a browser yet.

**Not built, honestly deferred:** a second preset (needs the Task x
Architecture matrix to actually grow past one entry, and real
reconciliation of `ResourcesControllerNode.INPUTS`/`OUTPUTS` once a
second preset's shape genuinely differs from the first); a
genuinely dynamic dropdown gaining a *new* choice from a live server
response after a node's already spawned (the still-open, harder
version of what `"inherited"` sidesteps by being a static choice
instead).

**How this section's scope got corrected:** the first two working
passes had this node calling LoRA injection directly (`rank`/`alpha`/
frozen-weight-storage as its own inputs, constructing
`SDXL_LoraTrainer` in `process()`) -- reasonable given
`docs/training_pipeline_design.md`'s own Phase 4 work already built
that exact pipeline, but wrong: rank/alpha/frozen-weight-storage are
properties of a LoRA *injection*, not of a verified *resource*, and
conflating the two put a decision that belongs on the training node
(Phase 6) onto this one instead. Corrected directly, not discovered
independently -- `LoRATrainingResources` and this node's current,
narrower shape are the result. Earlier drafts of this section walked
through that correction and two earlier, smaller ones (a wired-socket
detour for `checkpoint_path`, correcting checkbox inference to real
`bool` Ports) in full blow-by-blow; removed from here on the same
direct feedback that the accumulated correction history had itself
become the confusing part of this document. The reasoning for each
individual decision above (why a wire was rejected, why `"inherited"`
is static, why `LoRATrainingResources` doesn't do injection) still
lives in the code's own docstrings, not just here.

**Dependency:** Phases 1-4 (done).


### Phase 6 -- `LoRATrainingConfigNode`, and downstream integration (`TrainerNode` and friends)

**Goal:** a node that takes Phase 5's `LoRATrainingResources` --
`unet_sd`/`clip`/`vae_sd`/`continue_lora_sd`, not yet LoRA-injected --
and actually creates the trainable adapter: decides `rank`/`alpha`/
frozen-weight-storage, calls `inject_lora()` (Phase 4, unmodified --
`LoRATrainingSkeleton`/`SDXL_LoraTrainer`
(`nodes/model/lora_training_resources.py`) already implement exactly
this, not rebuilt here), and produces something `TrainerNode` can use.

**Status: `LoRATrainingConfigNode` done. Downstream integration into
`TrainerNode` itself still open (see below).**
`nodes/model/lora_training_config.py` (new): `resources` is a wired
`LoRATrainingResources` input (Phase 5's own output -- nothing left to
load, everything real and already in memory by the time this node
runs). Dispatches on `resources`'s own concrete type
(`_TRAINER_FOR_RESOURCES`, one entry today:
`SDXL_LoRATrainingResources -> SDXL_LoraTrainer`) rather than a
user-facing preset selector -- there's nothing to choose, the
architecture was already decided by whichever
`ResourcesControllerNode` preset produced this specific `resources`
object; grows the same way `_PRESETS` does in
`resources_controller.py`, one dict entry per architecture. Real,
concrete job that belongs here specifically because it doesn't belong
on Phase 5: `rank` is ignored entirely, not merely defaulted, whenever
`resources.continue_lora_sd` is set -- its own shape
(`lora_down.weight`'s own first dimension, shared `_lora_rank()`
helper) is used instead, matching direct feedback that this should be
"impossible to override." Honestly **not** attempted: showing this as
a visually locked/disabled `rank` widget in the editor --
`Port.visible_when` only compares against a sibling Port's own widget
value, evaluated client-side before the graph runs, but whether
`continue_lora_sd` is set isn't a Port's own value at all, it's a
property of whatever object is actually wired into `resources`, which
doesn't exist until the graph executes that far. Same reason this node
has no `diagnostics()` override: Phase 5's live-diagnostics endpoint
sends plain JSON widget values, and a wired object is exactly what it
can't carry.

Required a real refactor of `LoRATrainingSkeleton`
(`nodes/model/lora_training_resources.py`), not just a new caller: its
`__init__` did split -> merge frozen LoRA -> inject -> build text
encoder all in one call, which would have meant
`LoRATrainingConfigNode` either re-deriving `unet_sd`/`clip` from a raw
checkpoint a second time (`resources` no longer even exposes one) or
duplicating the inject/continue-load/coordinator-setup logic itself.
Extracted a shared `_inject()` (inject, load an optional continuing
LoRA into the fresh adapter, set up the coordinator) that both
`__init__` (the from-a-raw-checkpoint path) and a new
`from_resources()` classmethod (the from-Phase-5's-own-output path)
call -- one real implementation of "inject and finalize," not two.

Caught two real bugs while doing this refactor, neither shipped:
`LoRATrainingResources.reload()`'s device fallback was hardcoded
`"xpu"` regardless of what device the object was actually built for
(unlike `LoRATrainingSkeleton.reload()`, which correctly remembers its
own construction device) -- `reload(None)` after `offload()` on
anything built for `"cpu"` would have silently moved everything to a
device never actually asked for. Fixed by storing `self._device` in
`LoRATrainingResources.__init__`, same as `LoRATrainingSkeleton`
already does. Same method also called `self.clip.reload(device)` with
the *unresolved* argument while `unet_sd`/`vae_sd` used the resolved
fallback -- `clip` and the raw tensors could have ended up on two
different devices from one `reload(None)` call. Both fixed together.

**Verified manually, real dispatch path (not the smoke-test suite):**
a minimal, real (not faked) `SDXL_LoRATrainingResources` instance
routed through `LoRATrainingConfigNode.build()` reaches the same real,
expected `ModuleNotFoundError: comfy` boundary inside `inject_lora()`
-- confirms the dispatch table and `from_resources()` wiring are
correct end to end, not just at the mock level. Mock-level checks
(a fake `inject_lora()`, since a real one needs ComfyUI) confirm:
`from_resources()` reuses `resources.clip`/`.vae_sd` by identity
(never rebuilds them) and never calls `split_checkpoint()`/
`build_text_encoder()`; rank is honored when given and no
`continue_lora_sd` exists; rank is silently overridden to the
detected value when `continue_lora_sd` does exist, even when an
explicit, different rank was also given; `unet_weight_store="nf4"`
resolves to the real `NF4WeightStore` class; an unregistered resources
type raises a clear, actionable error naming
`_TRAINER_FOR_RESOURCES`; the two `LoRATrainingResources` bug fixes
above (device fallback, `clip`/tensor device consistency) both
verified against a minimal fake `DeviceResident`-shaped object.
Registered in `server/nodegraph_registry.py`; auto-derives the display
name "LoRA Training Config".

**Still open, unaffected by any of the above:** whether `TrainerNode`
consumes `LoRATrainingConfigNode`'s own `trainer` output as one
bundled port, replacing its separate `model`/`optimizer`/
`text_encoder` ports, or feeds the existing `ComfyUNetLoRANode`-style
construction path instead -- the original sketch's own annotation
already flagged this as undecided, and nothing built in Phase 5 or 6
has settled it yet.

**Dependency:** Phase 5 (done).

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
