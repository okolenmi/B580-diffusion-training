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
become the confusing part of this document.
