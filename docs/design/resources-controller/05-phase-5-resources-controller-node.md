*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 5 -- The Resources Controller node itself

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
this redesign's own
[Phase 4](04-phase-4-resource-preset-abstraction.md) work already built
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
