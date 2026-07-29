# VRAM optimization + LoRA phase-split ("warm-up"), 2026-07-28

Session goals, from the user directly: (1) reduce VRAM usage for the
existing working LoRA training graph, and (2) a specific feature -- train
a fraction of steps as a "warm-up," then have a checkpoint whose own
weights contain none of the warm-up phase's changes, because early
training's coarse changes were observed to damage fine object structure
("one object can morph into another... less knowledge about how to
properly draw things"). Strict OOP, no copying old code's shape into
`nodes/`, comments short (detail lives here).

## Part 1: VRAM

### Gradient checkpointing, fixed

**Continues prior work, not a fresh discovery.** This exact bug was
already diagnosed on 2026-07-26 (commit `5096a94`, "Fix training crash
(gradient checkpointing vs LoRA)") by reading the same upstream source.
That session's fix was to add the `use_checkpoint` port at all (it didn't
previously exist, so the wrapper's own default of `True` was silently
firing on every graph run) and default it to `False` -- i.e., stop the
crash by disabling the feature, not by fixing it. This session re-fetched
the same file directly (independent confirmation, not assumed from the
earlier session's notes) and actually fixes the underlying call instead.

**The bug, precisely.** Fetched `comfy/ldm/modules/diffusionmodules/util.py`
directly from `github.com/comfyanonymous/ComfyUI` during this session
(not assumed from training data) to confirm the exact mechanism rather
than pattern-match the error message. `CheckpointFunction.backward()`
calls:

```python
torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params,
                     output_grads, allow_unused=True)
```

`ctx.input_params` is a checkpointed block's *entire* `parameters()` list,
unfiltered by whether any given one actually requires grad.
`torch.autograd.grad()` requires every tensor in its `inputs=` list to
have `requires_grad=True`, full stop -- `allow_unused=True` only excuses
a tensor that's *unused in this particular graph*, not one that
structurally can never require grad. In a full fine-tune every parameter
requires grad, so this never surfaces; the moment a block has even one
frozen parameter (any norm weight, any bias, any Linear/Conv2d LoRA
didn't target) sitting next to a trainable LoRA adapter, this raises
`RuntimeError: One of the differentiated Tensors does not require grad`
on the very first backward pass.

**The fix** (`nodes/model/gradient_checkpointing.py`): filter
`ctx.input_params` down to the ones with `requires_grad=True` before the
`torch.autograd.grad()` call, then reconstruct a full-length gradient
tuple afterward with `None` at the frozen positions -- `torch.autograd.
Function.backward()` must return one gradient slot per original `forward()`
argument regardless of whether that argument required grad, and PyTorch
accepts `None` for a position that didn't. Forward, the shallow-copy
re-run under `torch.enable_grad()`, and the autocast context are all
copied unchanged from ComfyUI's own implementation -- this is a filter on
proven logic, not a reimplementation of it. Applied via a runtime
monkeypatch (`comfy.ldm.modules.diffusionmodules.util.CheckpointFunction`
reassigned), idempotent, only triggered when `ComfyUNetLoRANode` is built
with `use_checkpoint=True` -- zero footprint when unused.

`ComfyUNetLoRANode.INPUTS["use_checkpoint"]`'s default flipped `False` ->
`True`. This wasn't a judgment call about whether the tradeoff is worth
it in the abstract -- it was categorically broken before (crashed
immediately), so there was no real prior default to weigh against; now
that it works, it's the single biggest lever available for this specific
ask (activations dominate LoRA training's VRAM profile, not the frozen
base weights sitting in memory or the tiny adapters), so defaulting it on
directly serves the stated priority. It's still a real recompute-time
tradeoff (roughly 20-30% slower steps is the typical range for this kind
of checkpointing, not measured on this specific model) -- `use_checkpoint=False`
is one input away if that's the wrong tradeoff for a given run.

**Verification level, precisely.** ComfyUI itself isn't installed in this
sandbox (no `comfy` package, no `COMFY_DIR` -- same constraint the rest of
this project's `nodes/` work has run into repeatedly). What's actually
verified (`smoke_test_gradient_checkpointing.py`, real torch): a faithful,
verbatim reproduction of the real `CheckpointFunction`/`checkpoint()`
(the exact source fetched above) is registered as a stand-in module at
the same import path the patch targets; against that stand-in, (a) the
*unpatched* version really does raise the exact documented error on a toy
block with one frozen and one trainable parameter -- the diagnosis is
demonstrated, not assumed -- and (b) the *patched* version produces
gradients that exactly match an independent non-checkpointed reference
computation on the same block, and the frozen parameter's `.grad` stays
`None` throughout (no fabricated gradient). This verifies the patch
function's actual logic correctly. It does not verify that ComfyUI's
current source still has `CheckpointFunction`/`checkpoint()` at that exact
path/shape -- that needs confirming once on a machine with ComfyUI
installed, ideally alongside the very first real LoRA training run with
`use_checkpoint=True`.

### CachingTextEncoderNode

`SupervisedLoRATrainerNode._run_step` calls `ctx.text_encoder.encode(...)`
fresh, every step, for every batch's prompt -- for a managed dataset whose
captions repeat (typical: a handful of style/character tags reused across
many images), that's a full CLIP forward pass, with its own real
activation memory on top of the UNet's, recomputing something already
computed. `nodes/model/text_encoder_cache.py`'s `CachingTextEncoder`
wraps any `TextEncoder`, LRU-caching by `(prompt, batch_size, height,
width)` on CPU (bounded, default 512 entries, so an open-ended/randomized-
prompt dataset can't grow it without limit). Every call site in this
codebase already does `.to(device=...)` on `encode()`'s return value, so
handing back CPU tensors on a hit needs no device-placement
special-casing.

### Considered, not implemented

- **int8/quantized base weights.** Real VRAM win (frozen base is the
  single biggest static allocation), but a genuinely large, separate
  effort (quantization-aware forward math, dequant-on-the-fly or
  quantized matmul kernels) with real risk of numerical drift in a
  frozen "ground truth" the LoRA adapter is delta-ing against. Out of
  scope for this session; flagging as the next real lever if activation
  checkpointing alone isn't enough.
- **CPU offload of the frozen base between steps.** Would need the base
  UNet moved device<->host every step (SDXL UNet is large; the PCIe
  round-trip cost per step would likely dominate), and this project's own
  device abstraction doesn't currently have a per-block streaming-offload
  primitive to build on. Not attempted -- would need real hardware to
  even evaluate whether the transfer cost is tolerable.
- **Gradient accumulation.** Doesn't reduce peak VRAM by itself (it
  reduces effective batch size at fixed VRAM, not the other way around,
  unless paired with also shrinking the microbatch) and is explicitly
  out of `TrainerNode` v1's scope already (see
  `docs/nodes_package_design.md`'s "Explicit v1 scope reduction" list).

## Part 2: LoRA phase-split ("warm-up")

### The ask, restated precisely

Train the first *X%* of steps as a "warm-up." After that boundary, two
things should exist: (1) a LoRA containing the warm-up phase's own
changes, and (2) a *separate* LoRA containing only the changes made
*after* the boundary -- none of the warm-up phase's modifications in it.
Motivation: early training's coarsest, most destructive changes are
usually not the ones a human would recognize as "the right first thing to
learn," and were observed to produce structurally broken output (object
morphing, loss of drawing competence) when they stay baked into the final
adapter.

### Two candidate designs, and why one was rejected

**Rejected: merge the current adapter into the frozen base weights, then
reinitialize it in place, and keep training.** `core.lora.LoRALinear.merge()`
already does exactly "fold `B @ A * scaling` into `base_weight`, zero
`lora_A`/`lora_B`" -- reusing it looked like the smallest possible change.
Two real problems, not stylistic ones:

1. **Gradient-dead adapter.** `merge()` zeros *both* `lora_A` and
   `lora_B`. LoRA's forward is `(x @ A.T) @ (B.T * scaling)` -- bilinear
   in `A` and `B`. If `B` is zero, `d(loss)/dA` is proportional to `B`
   (zero) via the chain rule, so it's *also* zero. Both parameters at
   zero is a fixed point with zero gradient in every direction -- the
   fresh "generation" would never leave initialization. (The reason the
   *original* single-generation init works, `A` random/nonzero and only
   `B` zero, is exactly to avoid this: nonzero `A` keeps `dL/dB`
   nonzero even though the adapter's initial *output* is zero.) Reusing
   `merge()` as-is for this purpose would need a follow-up reinit of `A`
   back to a fresh `kaiming_uniform_` draw -- doable, but already one
   more moving part than the alternative below needed.
2. **Optimizer-state entanglement.** Folding into base and reinitializing
   `lora_A`/`lora_B` *in place* means the exact same `nn.Parameter`
   *objects* an `OptimizerNode`'s handle has already accumulated
   momentum/second-moment state for are the ones now holding brand-new
   values. That state was computed from an entirely different parameter
   trajectory (the pre-fold one) -- applying it to the freshly
   reinitialized values is not a "continue training" operation, it's
   applying momentum computed for one function to a different one. Since
   LoRA-only training means *every* trainable parameter is a LoRA
   adapter, this isn't a partial-state issue, it's the whole optimizer's
   state. Fixing it correctly means manually resetting the optimizer's
   per-parameter state at exactly the fold boundary -- another moving
   part, and one that has to reach into `OptimizerHandle`-specific
   internals to do.

There's also a smaller, second-order cost: `base_weight` is bf16 (see
`core/lora.py`'s "CRITICAL: lora_A/lora_B are always fp32" comment for
why that distinction matters at all), so folding into it is a real
(if one-time, not iterative) rounding event.

**Chosen: stack a fresh, independent adapter on top of the frozen one.**
`nodes/model/lora_phases.py`'s `LoRAGeneration`: `forward(x) = inner(x) +
this_generation's_own_delta(x)`, where `inner` is either the original
`core.lora.LoRALinear`/`LoRAConv2d` or an earlier `LoRAGeneration`.
`inner`'s parameters get `requires_grad_(False)` the moment a new
generation is stacked on top. This sidesteps both problems by
construction: the new generation's `lora_A`/`lora_B` are genuinely new
tensors, initialized the normal way (`A` random, `B` zero) -- no
gradient-dead fixed point, and no optimizer has ever seen them, so a
*fresh* `OptimizerNode` for just the new generation starts with clean
state, no manual reset needed anywhere. No base-weight mutation at all,
so no bf16 rounding event either. The base's frozen values stay exactly
as loaded, for as many generations as get stacked.

Cost of this choice: the live model's module tree grows one wrapper deep
per split (bounded -- nothing about this project's use case calls for
more than a handful of phases), and combining N generations back into one
portable LoRA file for external tools needs a real (if small) derivation,
below, rather than being free.

### The combination math

Each generation's own forward contributes `delta_i = scaling_i * B_i @ A_i`
(this is what `core.lora.LoRALinear.forward` computes, and
`LoRAGeneration.forward` reuses the identical shape). The live model's
total contribution after N generations is `sum_i(delta_i)`. For **Linear**,
concatenating along the rank axis is exact:

```
A_cat = cat([A_0, A_1, ..., A_N], dim=0)               # (sum(r_i), in_features)
B_cat = cat([B_0*s_0, B_1*s_1, ..., B_N*s_N], dim=1)   # (out_features, sum(r_i))
B_cat @ A_cat == sum_i(scaling_i * B_i @ A_i)
```

because matrix multiplication distributes over a block-concatenated
contraction dimension -- this is the standard "rank-stacking" trick for
composing multiple LoRA adapters, not something specific to this project.
`alpha` is saved as `sum(r_i)` (the combined rank) rather than any
particular generation's own alpha, so that `scaling = alpha/rank == 1` on
reload -- each generation's real scaling is already baked into its `B`
half above, so the combined pair needs an effective scaling of exactly 1.
This is what makes the file loadable by anything that infers rank from
tensor shape and reads `alpha` at face value (this project's own loader
included, and standard external tools), not just something only this
codebase's own loading path understands.

**Conv2d needs more care, and the first version of this got it wrong.**
`LoRAConv2d.forward` does two convolutions: `F.conv2d(x, A, groups=g)`
then `F.conv2d(that, B*scaling)` (the second always `groups=1`, a plain
1x1 mixing across the rank channels, independent of the original layer's
own `groups`). For `groups == 1` the same concatenation trick as Linear
works unchanged. For `groups > 1`, naive `cat(dim=0)` is **wrong**:
PyTorch splits a grouped conv weight's output channels into `groups`
*contiguous, equal-sized* blocks, one per input-channel group. Each
generation's own `A_i` already has its own `rank_i` channels correctly
partitioned into `groups` blocks that respect *that generation's own*
input-channel grouping -- but concatenating generation 0's whole block of
`rank_0` channels followed by generation 1's whole block of `rank_1`
channels, then asking PyTorch to re-split the total into `groups` new
*equal* chunks, draws the boundaries in the wrong place the moment
`rank_0 != rank_1` (and even when they're equal, nothing guarantees the
new boundary lines up with either generation's own).

`smoke_test_lora_phase_split.py`'s Conv2d check with `groups=2` and
mismatched ranks (4 then 2) caught this directly -- the round-trip through
`core.lora`'s own loader produced a completely different forward output,
not a close-but-off one, which is exactly what a misaligned-groups bug
should look like. Fix (`_combine_conv_generations`): reshape each
generation's own `A_i`/`B_i` into `(groups, rank_i/groups, ...)` --
i.e. make its *own* group-blocks explicit -- before concatenating along
the `rank_per_group` axis and flattening back down. That places the
boundaries PyTorch's own automatic group-splitting will draw on the
concatenated tensor exactly on every generation's real group boundaries,
by construction, and reduces to the plain Linear-style concatenation when
`groups == 1`. Verified the same way as the Linear case: round-tripped
through `core.lora.load_lora_into_model` (untouched) into a fresh
`LoRAConv2d`, forward output compared against the live stacked model,
with `stride=2, padding=1, groups=2` (i.e. not the trivial all-defaults
case). This project's actual LoRA targets (`to_q`/`to_k`/`to_v`/`to_out.0`)
are all `nn.Linear`, and SDXL's UNet convs aren't grouped, so `groups > 1`
is unlikely to be exercised by this project's real usage -- fixed anyway,
since a silently-wrong result for a case the code doesn't prevent isn't
acceptable just because it's unlikely to be hit.

### Why this needed generalizing `LoRAWeightsExportable` into `TrainedWeightsExportable`

First pass: a narrow `LoRAWeightsExportable` ABC (just `TrainableModel`'s
saveable subset), with `LoRACheckpointSaverNode.INPUTS["model"]` retyped
to it. This looked right until checking it against the actual mechanism
the graph editor uses to accept or reject an edge
(`server/graph_executor.py._is_compatible`): a real `issubclass()` check
against each port's *declared* type, not the runtime object's actual
class. `SupervisedLoRATrainerNode.OUTPUTS["model"]` is declared
`TrainableModel` (the ABC) -- for that to remain wireable into a
`LoRACheckpointSaverNode` input declared `LoRAWeightsExportable`,
`TrainableModel` itself would need to be a subclass of it, which it
wasn't, even though the one *concrete* class involved
(`ComfyUNetTrainableModel`) happened to implement both. That would have
silently broken the main, already-working save-at-the-end wiring the
instant the port got retyped -- caught by writing
`smoke_test_lora_phase_split.py`'s `check_contracts` against the real
`_is_compatible` function (not a reimplementation of its logic) before
assuming the design was fine, which is the whole reason to check a
contract against the real mechanism instead of reasoning about it in the
abstract.

Fix: renamed to `TrainedWeightsExportable` and made `TrainableModel`
extend it (`trained_state_dict()` -> `trainable_parameters()` etc. all in
one ABC now). This is a *more* honest contract, not a workaround: "a
model that's ready to be trained" reasonably ought to always be able to
say how to export what it learned -- a hypothetical future non-LoRA
`TrainableModel` would just export its own full weight diff through the
same method, nothing LoRA-specific about the requirement itself, just the
LoRA-specific *implementation* `ComfyUNetTrainableModel` happens to give
it. With that hierarchy, `LoRACheckpointSaverNode.INPUTS["model"]` can
stay typed `TrainedWeightsExportable` and both real wires -- a normal
`TrainerNode` output, and a `LoRAPhaseSplitNode.completed_generation`
snapshot -- satisfy it through a real, structural `issubclass()`
relationship, no runtime special-casing needed in the saver node at all.

### What `LoRACheckpointSaverNode` doesn't do (unchanged gap, not a regression)

`core.save.save_lora_checkpoint` (the legacy function
`LoRACheckpointSaverNode` used to call) temporarily removes a
`FusedXPUAdafactor`'s backward hooks during the weight read, to avoid a
hook firing mid-save on a tensor mid-detach. That race needs a *live,
still-training* model and a save happening *while training is paused
mid-loop* -- not reachable today, since nothing in `nodes/` calls
`LoRACheckpointSaverNode` from inside a `TrainerNode`'s step loop (no
periodic-checkpoint orchestration node exists yet, per
`docs/nodes_package_design.md`'s scope-reduction list). The node's
previous version also never passed an optimizer through to get that
protection either, so removing the (unused) legacy call path isn't a
regression -- worth building the equivalent protection once an
orchestration node that could actually trigger the race exists, not
before.

## Files touched

- `nodes/model/gradient_checkpointing.py` (new)
- `nodes/model/lora_phases.py` (new)
- `nodes/model/text_encoder_cache.py` (new)
- `nodes/model/lora_injector.py` (`use_checkpoint` default, patch wiring,
  `trained_state_dict()`)
- `nodes/model/lora_saver.py` (rewritten against `TrainedWeightsExportable`,
  no longer calls `core.save.save_lora_checkpoint`)
- `nodes/model/handle.py` (`TrainedWeightsExportable`, `TrainableModel`
  extends it)
- `server/nodegraph_registry.py` (registered `LoRAPhaseSplitNode`,
  `CachingTextEncoderNode`)
- `nodes/smoke_tests/smoke_test_gradient_checkpointing.py`,
  `smoke_test_lora_phase_split.py`, `smoke_test_text_encoder_cache.py` (new)
