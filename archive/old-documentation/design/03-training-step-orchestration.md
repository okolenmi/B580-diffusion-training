*[← docs/design index](README.md)*

# 2. Orchestrating a training step

## 2.1 The step loop as a pipeline of phases, not one method

A training step is a fixed sequence of concerns -- fetch a batch, encode
conditioning, run the forward pass, compute loss, backward, apply the
optimizer update, report progress -- each with a genuinely different
reason to change (a new conditioning scheme, a new loss weighting, a new
optimizer family), but it used to live as sequential code inside one
~90-line method, which is exactly what made each "explicit v1 scope
reduction" item (CFG dual-pass, gradient accumulation, resume cadence)
mean editing that same method rather than adding something next to it.

More granular than the design's own illustrative 7-phase list
(`nodes/train/step_pipeline.py`): `update_lr()`/`zero_grad()`/
`begin_step()` and diffusion-input prep (`x_t`/`target`/`t`/`sigma`/`xc`)
each got their own phase rather than folding into the forward and
conditioning phases, since each is genuinely its own reason to change --
the same test this design was built around. The profiling output's
*shape* genuinely changed as a result, checked against every real
consumer before shipping per `step_pipeline.py`'s own docstring -- not a
silent behavior change.

**Still not built, and this is the concrete, real payoff the refactor was
for:** `SupervisedLoRATrainerNode`'s own scope note still lists no CFG
cond/uncond dual pass, no gradient accumulation, no cyclic/teacher-rollout
caching, no DAgger, no adversarial pre-conditioning, and no resume/
checkpoint cadence beyond `on_step`. None of these are designed further
here -- each is now genuinely additive ("construct one more phase, insert
it in the list") rather than monolith surgery, which was the actual goal;
designing any one of them in detail is real, separate future work, sized
independently once actually needed.

## 2.2 Resource budget as a first-class value, resource policy as a Strategy

**Implemented, then later removed -- the record of both is worth
keeping, since the reasoning that shaped it is still the reasoning
behind the Ports that replaced it.** `ResourceBudget`/`ResourcePolicy`/
`ManualResourcePolicy` covered exactly three choices --
`checkpointing_strategy()`, `lora_scaling_policy()`, and
`parameter_group_policy()` -- not the seven this section originally
sketched. `adapter_strategy()` was cut, and stayed cut even once
`AdapterStrategy` became reachable from `ComfyUNetLoRANode`'s real
construction path (3.1): it's wired as its own standalone
`adapter_strategy` port instead, the same way `checkpointing_strategy`
was a real choice made independently of (and overridden by)
`resource_policy` rather than routed only through it -- a `ResourcePolicy`
method for this would have duplicated a choice that already had a real,
working home, not filled a gap. `frozen_weight_store()` was cut because a
chosen `FrozenWeightStore` still wasn't reachable from
`ComfyUNetLoRANode`'s real construction path (3.3) at the time -- a
policy method for a choice nothing could act on would have been
scaffolding, not a feature. `optimizer_execution_strategy()` was cut
because `ExecutionStrategy` selection is already a real Port (`strategy`)
on each `Composed*OptimizerNode`, but which *Algorithm* to use (CAME/
Adafactor/AdamW) is a choice of *Node class*, not a Port value -- there's
no single generic "the" optimizer node a `ResourcePolicy` method could
have handed this to. `enable_text_encoder_cache()` was cut because
whether caching happens is decided by *which Node* is wired into the
graph (`CachingTextEncoderNode` vs. a plain `TextEncoderNode`), not a
flag any single Node's `build()` could read and act on after the fact --
graph topology, not a constructor-time choice.

**A real architectural correction, found by building this rather than
foreseen in the design:** a `ResourcePolicy` returning
`ActivationCheckpointingStrategy`/`LoRAScalingPolicy` (both `model/`) and
`ParameterGroupPolicy` (`optimizer/`) needed those types in its method
signatures -- but a module living outside any one domain package can't
`import` from two domain packages at once without violating the Acyclic
Domain Dependency Rule (5.7). Resolved the same way `DiffusionProcess`
(1.4) already resolves an analogous problem: every method used a
forward-reference string type hint only (structural, via `from __future__
import annotations`, not a per-method style choice), so the module
needed zero real cross-domain imports. That pattern is still real and
still in use -- `nodes/resource_budget.py`'s own docstring and
`nodes/memory/profile.py`'s `DeviceContext` example both point back to
it.

**Wiring, at the time, and a second deliberate deviation from the
original illustration:** the three `Composed*OptimizerNode` classes got a
direct `group_policy` port instead of a `ResourcePolicy` one --
deliberately: routing `parameter_group_policy()` selection through a full
`ResourcePolicy` there would have forced an optimizer node to also supply
a `checkpointing_strategy`/`lora_scaling_policy` it had no use for, just
to pick a parameter-group policy, which would have been worse ergonomics
than the scattered-flags problem this item existed to fix. So this was
never "one `ResourcePolicy` object, four identical consumers" -- it was
`ResourcePolicy` where two of its three choices naturally co-located
(`ComfyUNetLoRANode`), and a simpler, direct port where the third one
didn't. `group_policy` was never routed through `ResourcePolicy` at
all, in fact -- it's `ParameterGroupPolicy` on its own, a fully separate
mechanism from the start; an earlier version of this doc said otherwise,
which was simply wrong, not something that changed later. A real,
previously-undocumented gap was closed as a side effect regardless:
`LoRAPlusGroups` (3.4) had existed since the `ParameterGroupPolicy` fix
landed, but no Node ever exposed a way to actually select it from the
graph until `group_policy` existed.

**Why it was removed.** No `Node` in the registry ever produced a
`ResourcePolicy` value -- `ManualResourcePolicy` was constructible only
by hand, in Python, and the only thing that ever did was its own smoke
test's contract check. The `resource_policy` port that accepted one was
therefore real, tested, and permanently unreachable from the actual
graph editor: nothing a person building a graph in the browser could
ever wire into it. The two concerns that were ever exercised in
practice (checkpointing, LoRA scaling) already had their own real
ports; `parameter_group_policy()` was never called by
anything outside that same smoke test either, `group_policy` being
separate as described above. Removed along with its smoke test.

`ResourceBudget` survived them and is now **live**: the VRAM budget
controller node constructs one and the trainer's budgeted resource
control handle consumes it (a `ResourceBudget.strict`
mode was added later -- offload-then-continue versus refuse outright).

## 2.3 Activation checkpointing: strategy and placement

What was missing from the underlying fix was that it used to be exposed
only as a global, process-wide monkeypatch triggered by a bare `bool`
port -- correct, but not itself an object another piece of code could
compose with or substitute.

`FrozenParamSafeCheckpointing` takes no `placement` parameter yet --
deliberately: adding one with nothing real to pass it would be
scaffolding, not a feature.

All-or-nothing checkpointing (every block, or none) is correct and
maximizes VRAM savings at maximum recompute cost. A more
principled middle ground has real published grounding: Chen et al.,
"Training Deep Nets with Sublinear Memory Cost" (arXiv:1604.06174, 2016)
show that checkpointing roughly every `sqrt(N)` layers achieves
near-optimal memory/recompute tradeoff for a *uniform*-cost network;
Korthikanti et al., "Reducing Activation Recomputation in Large
Transformer Models" (NVIDIA, 2022) generalize this to *selective*
recomputation -- ranking candidate checkpoint points by their actual
memory-saved-per-recompute-cost ratio, which is the more directly
applicable idea here since a UNet's blocks aren't uniform cost (attention
blocks vs. plain conv/resnet blocks differ in both activation size and
recompute time).

**Two real findings, confirmed against ComfyUI's actual source
(cloned directly, not guessed) while building this:**
- In `comfy/ldm/modules/diffusionmodules/openaimodel.py`,
  `ResBlock.forward()` calls `checkpoint(self._forward, ...)` -- a bound
  method, so `ctx.run_function.__self__` is the real block instance.
  That's what makes real per-block labels (`type(instance).__name__` +
  a first-seen ordinal, e.g. `"ResBlock#4"`) possible without needing
  the block's dotted path in the UNet.
- `comfy/ldm/modules/attention.py`'s `BasicTransformerBlock.forward()`
  does **not** call `checkpoint()` at all in this pinned ComfyUI
  version -- its `checkpoint=True` constructor parameter is unused dead
  wiring. In practice, only `ResBlock` instances ever reached this
  profiler or got placed by `GreedyRatioPlacement` -- a real, grounded
  constraint on what this item could decide over, not a gap in the
  implementation. **Closed** (see `docs/known-issues/resolved.md` --
  confirmed on real hardware 2026-09-28: without the patch the same
  run OOMs on its first forward pass, with it the run peaks at
  8592 MB): the attention-block checkpointing patch routes
  `BasicTransformerBlock.forward()` through the same `checkpoint()`/
  `CheckpointFunction` seam `ResBlock` already used, so
  `use_checkpoint=True` now actually reaches SDXL's dominant
  activation cost (attention blocks, not just the two convolutions in
  each `ResBlock`), and this profiler now sees both block types. The
  checkpoint patch itself is confirmed on real hardware; the profiler
  *instrumentation* (`ProfilingCheckpointing`'s timing output) still
  hasn't been exercised in a real hardware run.

**Fraction knob (added 2026-09-28, hardware-measured):**
`enable_attention_block_checkpointing(fraction=1.0)` can checkpoint a
*subset* of blocks; `fraction=0.0` skips the patch entirely. Exposed as
`scripts/hw_validate.py`'s `--attn-ckpt-fraction` for the VRAM-vs-
recompute sweep. Measured at 1024x1024/batch 2 on the B580: skipping 25%
of attention recompute is a net *loss*: recomputing these blocks is
cheaper than carrying their activations. The knob is real and honored,
but at this operating point the practical recommendation is the default
(density 1.0). Full numbers and the measured floor-lever costs are in
`docs/known-issues/open.md`.

## 2.5 Dataset prefetching, kept honest about what it does and doesn't save

`SupervisedLoRATrainerNode`'s own `profile=True` output already reported
`data_wait_ms` (now `fetch_batch_ms`, see 2.1) -- so whether data loading
is a real bottleneck was already measurable, not a guess, before this was
built.

`PrefetchingBatchSource` (`nodes/dataset/prefetch.py`) -- a decorator
over any `TrainingBatchSource`, the wrap-don't-reimplement pattern this
domain's batch sources established. One real deviation worth noting: a
fresh worker thread per `__iter__()` call rather than one shared for the
object's whole lifetime, since `TrainingBatchSource.__iter__()` is
expected to be restartable (a fresh pass each call) and `FetchBatchPhase`
relies on exactly that to wrap to a new epoch. Explicitly not built:
pinning the host-side buffers (page-locked memory) -- a real
platform-specific wrinkle (pinning support/benefit isn't identical across
CUDA and XPU), left for its own follow-up once this is in real use. Not
wired in by default -- still an opt-in node, per the same
"demand-driven, not speculative" reasoning as before.

---
