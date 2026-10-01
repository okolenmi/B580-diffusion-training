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
for how.**
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
sight unseen. What shipped instead in 09: not an adapter into
`TrainerNode`'s own ports at all -- a genuinely independent trainer
node (`ManagedLoRATrainerNode`) that takes `trainer` directly, with its
own step loop built around deterministic residency rather than the
main route's reactive-only offloading (an intermediate version that
*did* adapt into `TrainerNode`'s ports and shared its loop was tried
first and reverted -- see 09's own "First attempt, reverted"). Nothing
about `TrainerNode`, `SupervisedLoRATrainerNode`, `ModelParametersNode`,
or `CachingTextEncoderNode` needed to change either way.
`LoRATrainingSkeleton`/
`LoRATrainingResources` staying real objects with their own working
methods, outliving and outnumbering any one node's own Ports, is still
exactly the design posture this paragraph originally argued for --
only the specific "replace TrainerNode's ports" mechanism was wrong.
