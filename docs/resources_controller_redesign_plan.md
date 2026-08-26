# Resources Controller & precision redesign -- step-by-step plan

**Status: Phase 1 done, Phase 2 next.** This tracks a real,
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

**Goal:** expose Phase 1's inspection capability over HTTP, scoped and
provably unable to leak anything beyond the specific fields requested.

- New endpoint in the existing `/assets/{kind}/...` family in
  `server/routes_nodegraph.py`, e.g. `GET /assets/{kind}/inspect?path=...`,
  built directly on `asset_paths._safe_resolve()` -- same sandboxing
  `browse()`/`upload()` already use, not reinvented.
- Response is exactly the detected fields (dtype per component,
  validity), nothing else -- no raw header dump, no full key list
  unless that's specifically one of the requested fields. This is what
  "protected" means in practice: the response shape is a fixed,
  narrow contract, not a passthrough of whatever safetensors' header
  happens to contain.

**Definition of done:** a new `server/smoke_tests/` file (matching
`smoke_test_graph_executor.py`'s style) covering the real answer *and*
the adversarial cases explicitly: a path-traversal attempt is rejected
the same way `resolve_safe_model_path` already rejects one elsewhere in
this project, a request for a kind/path combination that doesn't exist
fails cleanly, and the response body is checked field-by-field against
an allowlist rather than just checked for "does it look about right."

**Dependency:** Phase 1.

### Phase 3 -- Interactive node support (editor + core.py + introspection)

**Goal:** the actual mechanism letting one node's displayed inputs/
outputs change live as the user configures it in the editor, settling
into a concrete, static shape by the time Run is pressed -- not a
value carried by a wire at Run time (see "ground truth" above for why
that distinction matters to `graph_executor.py`).

**This phase starts with an audit, not a design.**
`server/static/nodegraph.js` has not been read in full in this
conversation, and no implementation approach should be committed to
before it has been -- today's editor model (drag a class from the
palette, get its fixed socket set) may or may not tolerate a node
whose sockets change after placement without a more invasive rewrite
than the rest of this plan assumes. First real step of this phase is
reading that file end to end and reporting back what it would actually
take, before writing any of it.

Known real touchpoints once that audit is done: `nodes/core.py` (some
way for a node class to report "resolve my shape from these params"
rather than a fixed `INPUTS`/`OUTPUTS`, additive -- every existing node
keeps working unchanged), `server/nodegraph_introspect.py` (extend
introspection to resolve shape given partial params, not just a bare
class), `server/graph_executor.py`'s `_is_compatible()` (resolve the
concrete shape from `spec.params` before checking edge types, for
nodes opting into this -- backward compatible with every static node
already in the registry).

**Dependency:** Phase 2 (needs something real to query as the user
configures the node).

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
