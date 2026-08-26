# Resources Controller & precision redesign -- step-by-step plan

**Status: Phase 1 + Phase 2 done, Phase 3 audited (no implementation
yet -- see that section for what's actually needed and the one open
question blocking it).** This tracks a real,
multi-session redesign, not a single patch -- update it as phases land
or as open questions get resolved, the same way `PROGRESS.md` tracks
the rest of this project.

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
changes:** `suggestNodesForDroppedWire()` iterates the *entire
registry's* static `classInfo.inputs/outputs` to find compatible
matches. A class whose shape depends on params can't be cheaply
enumerated that way -- there's no single "the inputs" to check against
for the whole class. Realistic options: exclude interactive nodes from
this suggestion feature, or suggest based on their default/initial
configuration only and accept that it may be incomplete. Not decided.

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

**Dependency:** Phase 2 (done). Not started pending direction on the
suggestion-menu question and the overall shape decision above.

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

---
Last synced against `docs/training_pipeline_design.md` at commit
`2c1f0ff` (2026-08-25).
