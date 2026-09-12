*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

# Context and ground truth

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

**Resolved -- see the "Composition mechanism" sub-section below and
Phase 4's own section further down: multiple inheritance,
concrete-mixin-first, was chosen.** The reasoning that follows is kept
as the historical record of the question itself, not as a still-open
item.

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

**Decided (see the resolution note at the top of this section and
Phase 4 below): multiple inheritance, concrete-mixin-first.** (This
sentence originally read "Not yet decided which one this project
uses" -- corrected in place during a later docs pass once Phase 4 had
already settled it, so this section stops contradicting the rest of
the document.)

## Phases

Sequenced so that every phase through Phase 2 is independently buildable
and smoke-testable with zero frontend changes, matching how every other
patch in this project has been verified -- the frontend/interactive-node
work (the single riskiest, least-scoped piece) comes after the backend
foundation it depends on exists and is proven, not before.
