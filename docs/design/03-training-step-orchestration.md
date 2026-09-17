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

**Implemented**, and more granular than the design's own illustrative
7-phase list: `StepState`/`StepPhase`/`TrainingStepPipeline` plus nine
concrete phases (`FetchBatchPhase`, `PrepareDiffusionInputsPhase`,
`EncodeConditioningPhase`, `OptimizerBeginStepPhase`, `ForwardPhase`,
`LossPhase`, `BackwardPhase`, `OptimizerStepPhase`, `MonitoringPhase`) and
`TimedPhase` (the generic profiling decorator, replacing `profile: bool`
manually wrapping five points with `xpu_synchronize()` + `perf_counter()`)
-- all in `nodes/train/step_pipeline.py`, driving
`SupervisedLoRATrainerNode` in `nodes/train/supervised.py`.
`update_lr()`/`zero_grad()`/`begin_step()` and diffusion-input prep
(`x_t`/`target`/`t`/`sigma`/`xc`) each got their own phase rather than
folding into `ForwardPhase`/`EncodeConditioningPhase`, since each is
genuinely its own reason to change -- the same test this design was built
around. The profiling output's *shape* genuinely changed as a result
(checked against every real consumer before shipping, per
`step_pipeline.py`'s own docstring) -- not a silent behavior change.

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
`ManualResourcePolicy` (originally `nodes/resource_policy.py`; only
`ResourceBudget` survives, in `nodes/resource_budget.py`) covered
exactly three choices -- `checkpointing_strategy()`, `lora_scaling_policy()`,
and `parameter_group_policy()` -- not the seven this section originally
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
it. The concrete cost of the fix at the time: `ManualResourcePolicy`
didn't compute its own sensible defaults the way this section's
original illustration had it do (`adapter_strategy or PlainLoRAAdapter()`-
style fallbacks) -- doing that would have needed exactly the imports
being avoided. Instead `ManualResourcePolicy` was a pure carrier (every
field required, nothing defaulted internally), and each consuming Node
built its own default instance from classes it already imported -- the
same pattern `diffusion_process` (1.4) still uses for its `None ->
locally-constructed default` ports.

**Wiring, at the time, and a second deliberate deviation from the
original illustration:** `ComfyUNetLoRANode` had an optional
`resource_policy` port that, when given, fully replaced its
`use_checkpoint`/`scaling_policy` ports (both of which still worked
unchanged when `resource_policy` was `None` -- the default). The three
`Composed*OptimizerNode` classes got a direct `group_policy` port
instead of a `resource_policy` one -- deliberately: routing
`parameter_group_policy()` selection through a full `ResourcePolicy`
there would have forced an optimizer node to also supply a
`checkpointing_strategy`/`lora_scaling_policy` it had no use for, just to
pick a parameter-group policy, which would have been worse ergonomics
than the scattered-flags problem this item existed to fix. So this was
never "one `ResourcePolicy` object, four identical consumers" -- it was
`ResourcePolicy` where two of its three choices naturally co-located
(`ComfyUNetLoRANode`), and a simpler, direct port where the third one
didn't. `group_policy` was never routed through `ResourcePolicy` at
all, in fact -- it's `ParameterGroupPolicy` on its own, a fully separate
mechanism from the start; an earlier version of this doc (and, copying
it, `docs/status/progress.md`) said otherwise, which was simply wrong,
not something that changed later. A real, previously-undocumented gap
was closed as a side effect regardless: `LoRAPlusGroups` (3.4) had
existed since the `ParameterGroupPolicy` fix landed, but no Node ever
exposed a way to actually select it from the graph until `group_policy`
existed.

**Why it was removed.** No `Node` in the registry ever produced a
`ResourcePolicy` value -- `ManualResourcePolicy` was constructible only
by hand, in Python, and the only thing that ever did was its own smoke
test's contract check. `ComfyUNetLoRANode`'s `resource_policy` port was
therefore real, tested, and permanently unreachable from the actual
graph editor: nothing a person building a graph in the browser could
ever wire into it. `use_checkpoint`/`scaling_policy` already covered the
two concerns that were ever exercised in practice (`checkpointing_strategy`,
`lora_scaling_policy`); `parameter_group_policy()` was never called by
anything outside that same smoke test either, `group_policy` being
separate as described above. Removed along with its smoke test.

`ResourceBudget` itself is implemented but still inert -- nothing
constructs or consumes one in a live path yet. `CheckpointPlacementPolicy`
(2.3) is its first designed consumer, still blocked on the per-block
profiling instrumentation described there, not on `ResourceBudget`
itself.

## 2.3 Activation checkpointing: strategy and placement

The underlying fix (`nodes/model/gradient_checkpointing.py`) was, and
remains, *correct*: filter `ctx.input_params` to `requires_grad=True`
entries before `torch.autograd.grad()`, reconstruct the full gradient
tuple with `None` at frozen positions. What was missing was that it used
to be exposed only as a global, process-wide monkeypatch triggered by a
bare `bool` port -- correct, but not itself an object another piece of
code could compose with or substitute.

**Implemented**, unchanged from the design: `ActivationCheckpointingStrategy`/
`NoCheckpointing`/`FrozenParamSafeCheckpointing` (same mechanism, now with
an `apply()` method; `NoCheckpointing` is the explicit "did nothing" case
replacing an implicit "the if just wasn't taken") -- in
`nodes/model/gradient_checkpointing.py`.
`ComfyUNetLoRANode`'s existing `use_checkpoint: bool` port stayed wired to
this internally, so nothing that already used it broke.
`FrozenParamSafeCheckpointing` takes no `placement` parameter yet --
deliberately: adding one with nothing real to pass it would be
scaffolding, not a feature. That's the policy below.

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

**Implemented -- both halves, backlog item 1.** `BlockCost`/
`CheckpointPlacementPolicy`/`EveryBlockPlacement`/`GreedyRatioPlacement`,
essentially as sketched here originally (one real correction:
`GreedyRatioPlacement` fits against `vram_budget_mb - vram_reserve_mb`,
not the raw ceiling -- `vram_reserve_mb` postdates this sketch), in
`nodes/model/checkpoint_placement.py`. The actual blocker, the per-block
profiling instrumentation, is `nodes/model/block_profiler.py`'s
`BlockProfileCollector`/`ProfilingCheckpointing` -- a third
`ActivationCheckpointingStrategy`, composed with the existing correctness
fix via a new optional `recompute_wrapper` parameter on
`enable_frozen_param_safe_checkpointing()` rather than a second copy of
that delicate autograd code. It measures each checkpointed block's real
recompute (the existing backward-time re-run of the block's forward,
already paid for by checkpointing itself -- no separate profiling-only
pass needed): wall time directly, and activation memory via
`DeviceContext.memory_stats()`'s `allocated_mb` delta around that same
call.

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
  wiring. In practice, only `ResBlock` instances ever reach this
  profiler or get placed by `GreedyRatioPlacement` -- a real, grounded
  constraint on what this item can decide over today, not a gap in the
  implementation.

**Not wired into `ComfyUNetLoRANode`'s real construction path** (unlike
`AdapterStrategy`'s seam, now live-wired -- see 3.1): real, tested,
reachable by any caller that constructs a `ProfilingCheckpointing`
directly, but `use_checkpoint` doesn't yet have a way to select it (its
former `resource_policy` sibling port, which could have, was removed --
see 2.2), and `GreedyRatioPlacement` isn't wired in as a real choice
either -- `EveryBlockPlacement`'s unconditional behavior stays what
`use_checkpoint=True` actually does. Real, separate follow-up, once a
first real profiled run's `BlockCost` numbers exist to validate a real
placement against -- see section 10.

## 2.4 Text encoder cache becomes visible to resource accounting

`CachingTextEncoder` (bounded LRU, default 512 entries, CPU-resident) was
already real, working, and self-contained -- the gap was that its memory
usage was invisible to anything outside itself.

**Implemented**: `TextEncoder` (`nodes/model/text_encoder.py`) extends
`DeviceResident` directly, so `CachingTextEncoder` gets `footprint_bytes()`/
`offload()`/`release()` -- landed as part of backlog item 12, the last
conformance gap left open once `ResourceCoordinator`/`OffloadOrchestrator`
(5.1, 5.2) actually needed a second real `DeviceResident` besides the
model to coordinate anything meaningful. The aggregate "where did my
memory go" report this enables (5.5, `ResourceProfile`) is itself still
not built -- see that section.

## 2.5 Dataset prefetching, kept honest about what it does and doesn't save

`SupervisedLoRATrainerNode`'s own `profile=True` output already reported
`data_wait_ms` (now `fetch_batch_ms`, see 2.1) -- so whether data loading
is a real bottleneck was already measurable, not a guess, before this was
built.

**Implemented**, unchanged from the design: `PrefetchingBatchSource`
(`nodes/dataset/prefetch.py`) -- a decorator over any
`TrainingBatchSource`, same pattern `nodes/dataset/renoise.py`'s
`RenoiseBatchSource` already established for this domain. One real
deviation worth noting: a fresh worker thread per `__iter__()` call
rather than one shared for the object's whole lifetime, since
`TrainingBatchSource.__iter__()` is expected to be restartable (a fresh
pass each call) and `FetchBatchPhase` relies on exactly that to wrap to a
new epoch. Explicitly not built: pinning the host-side buffers
(page-locked memory) -- a real platform-specific wrinkle (pinning
support/benefit isn't identical across CUDA and XPU), left for its own
follow-up once this is in real use. Not wired in by default -- still an
opt-in node, per the same "demand-driven, not speculative" reasoning as
before.

## 2.6 `MemoryManager`'s reach widens; its interface doesn't

Every new device-memory consumer identified in this design
(activation-checkpoint recompute scratch, if a future custom block needs
it; a text-encoder cache's tensors, if it's moved to device rather than
kept CPU-resident; a `PrefetchingBatchSource`'s pinned host buffers)
should acquire memory through the existing `MemoryManager.get_buffer()`/
`release()`/`free()` vocabulary, under its own tag, exactly the way
`ChunkedScratchBufferStrategy` already does for optimizer scratch. No new
method is being proposed on `MemoryManager` itself -- the design problem
it solves (tagged, lazily-grown, reuse-vs-drop-tracked buffers) is
domain-independent already; the gap is adoption, not capability.

---
