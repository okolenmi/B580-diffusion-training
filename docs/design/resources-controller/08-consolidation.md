*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

# Consolidation -- resolving low-synergy items instead of leaving them
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

**Real synergy found when this was written; resolved since, by how
Phase 5 itself actually turned out -- kept here as the historical
record, not still an open question:** section 11.3's item 2
(`state_dtype` on the `Composed*` optimizer nodes, "needs one shared
implementation") looked at the time like a third precision axis that
might belong folded into the Resources Controller's own eventual
parameter-value dictionary rather than living as an isolated port on
each optimizer node. It doesn't, as it turned out: Phase 5's actual
scope correction settled `ResourcesControllerNode` as strictly
model-resource-only (`unet_sd`/`clip`/`vae_sd`/`continue_lora_sd`,
nothing optimizer-adjacent at all -- see that phase's own section) --
optimizer construction happens entirely downstream of it, on the
`Composed*` optimizer nodes, which is exactly where `state_dtype`
(shipped as `state_precision`, block-wise 8-bit quantization rather
than a plain dtype cast -- see `docs/training_pipeline_design.md`
section 11.3 item 2 for what actually shipped) ended up living, right
alongside `strategy`/`device` on those same nodes, via the exact
`STRATEGIES`/`resolve_strategy()` shape `strategy_registry.py` already
used for a different Port there. Not the redundant, isolated-before-
the-fact addition this section originally worried about: the "one
place for precision decisions" this whole redesign cares about turned
out to be "the node that actually owns the thing being configured," not
literally one single node for every precision decision regardless of
which resource it concerns.

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
Last synced against `docs/design/` (formerly the single file
`docs/training_pipeline_design.md`) at commit `2c1f0ff` (2026-08-25).
