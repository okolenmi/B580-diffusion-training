*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 6 -- `LoRATrainingConfigNode`, and downstream integration (`TrainerNode` and friends)

**Goal:** a node that takes Phase 5's `LoRATrainingResources` --
`unet_sd`/`clip`/`vae_sd`/`continue_lora_sd`, not yet LoRA-injected --
and actually creates the trainable adapter: decides `rank`/`alpha`/
frozen-weight-storage, calls `inject_lora()` (Phase 4, unmodified --
`LoRATrainingSkeleton`/`SDXL_LoraTrainer`
(`nodes/model/lora_training_resources.py`) already implement exactly
this, not rebuilt here), and produces something `TrainerNode` can use.

**Status: `LoRATrainingConfigNode` done. Downstream integration into
`TrainerNode` also done -- see
`docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`
for how (a separate unpack node, not the mechanism originally sketched
below).**
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

**Superseded by `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`:**
this section originally read "`TrainerNode` consumes `LoRATrainingConfigNode`'s
own `trainer` output as one bundled port, replacing its separate
`model`/`optimizer`/`text_encoder` ports" -- a direct answer to the
question the original sketch's own annotation had left undecided
("not sure what should be output"), but wrong once actually checked
against the rest of the graph: `ComfyUNetLoRANode`/`SDXLTextEncoderNode`
(the existing, working main route into `TrainerNode`) never produce a
`LoRATrainingSkeleton` at all, only a bare `TrainableModel`/`TextEncoder`
each -- replacing `TrainerNode`'s own ports outright would have broken
that route to unblock this one, not an acceptable trade for a route
meant to run *alongside* the main one for comparison, not replace it
sight unseen. What shipped instead in 09: a small, separate
`TrainerResourcesUnpackNode` that adapts `trainer` into the plain
`model`/`text_encoder` ports every existing node already understands,
so nothing about `TrainerNode`, `ModelParametersNode`, or
`CachingTextEncoderNode` needed to change at all. `LoRATrainingSkeleton`/
`LoRATrainingResources` staying real objects with their own working
methods, outliving and outnumbering any one node's own Ports, is still
exactly the design posture this paragraph originally argued for --
only the specific "replace TrainerNode's ports" mechanism was wrong.

**Dependency:** Phase 5 (done).
