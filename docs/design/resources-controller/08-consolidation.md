*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

# Consolidation -- resolving low-synergy items instead of leaving them
to drift

Explicit instruction behind this section: don't let this redesign
become one more thing sitting next to the rest of the project's design
rather than actually reconciled with it. Went back through
`docs/design/10-node-surface-and-precision-control.md`'s remaining open
section-11 items and
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
shipping first in a shape Phase 4 then has to work around.

`choices=T_MODES` off a new `nodes/dataset/timestep_modes.py`.
That constant is a **deliberate**, documented duplicate of a same-named
constant added to `core/noise_schedule.py` (where `sample_timestep()`
actually implements those five distributions), not importable from
there directly -- checked directly, not assumed: `core/__init__.py`
eagerly imported `core.unet_wrapper` (ComfyUI-dependent) and other heavy
modules, so anything under `core.*` pulls all of that in at import
time, which is exactly why the original deferral example
(`renoise.py`'s `_renoise()`, since retired with the baked-grid format
it corrected -- `managed.py`'s build()-local `from manager.loader
import ...` is the live one) deferred its `core.noise_schedule` import
to call time rather than module load -- `nodes/dataset/` is deliberately
ComfyUI/torch-free at
import time, and a Port's `choices` is needed at class-definition time
(module load), where that deferral trick isn't available. `device`
Ports deliberately did **not** get `choices` -- checked directly
(`core/comfy_setup.py`), they're `torch.device()`-parsed and accept
indexed variants (`"xpu:0"`) no closed list could enumerate, so they're
genuinely the open-ended case `choices=None` exists to leave alone, not
an oversight.
