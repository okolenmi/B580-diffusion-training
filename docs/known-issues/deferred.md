*[← docs/known-issues index](README.md)*

# Deferred (not urgent, revisit later)

- **[2026-07] New finding, informational only:
  `ChunkedXPUAdafactor`'s momentum handling corrupts `exp_avg` in place
  when a parameter's dtype is float32.** Found while building and
  verifying `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm`
  against this class directly. In `step()`: `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`, where
  `g` is `self.exp_avg[i]` a few lines above (aliased, not copied). When
  `p.dtype == torch.float32` (same as the internal state dtype),
  `.to(dtype=p.dtype)` is a documented no-op returning the *same tensor
  object* -- confirmed directly (`t.to(dtype=t.dtype) is t` -> `True`) --
  so the following `.mul_(alpha_t)` permanently shrinks the momentum
  buffer itself by `alpha_t` (~`lr`) every step, rather than only scaling
  a throwaway copy for the parameter update. **Does not affect real
  training**: this codebase trains in bf16, and `.to(dtype=bf16)` from a
  float32 buffer always allocates a fresh tensor, so the aliasing (and
  therefore the corruption) never happens in practice -- confirmed by
  re-running the same comparison under bf16 and seeing the divergence
  collapse to ordinary quantization noise, no larger than the
  no-momentum case's own bf16 noise. Left here as a record, not fixed --
  `nodes/` doesn't touch `core/optimizers.py`, and there's no evidence
  this has ever caused a real-training problem to chase.

- **[2026-07] Note for future sessions: `nodes/memory/manager.py`'s new
  `MemoryManager` structurally prevents the reset-vs-free asymmetry bug
  class behind the "CAME optimizer VRAM near-ceiling hang" entry above,
  for anything built through `nodes/` going forward.** This does **not**
  fix or touch `core/optimizers.py`'s legacy classes -- per `nodes/`'s
  existing rule, that file hasn't been modified. Left here as a pointer,
  not a claim of resolution: once the node-graph optimizer path replaces
  the legacy one, this whole class of VRAM-lifecycle bug should stop
  being something to watch for by construction, rather than something to
  keep re-auditing by hand.

- **CAME's tiny-param batching fast path.** `ChunkedXPUAdafactor` has a
  vectorized/batched fast path for many small parameters (relevant for LoRA's
  many small A/B matrices) that `ChunkedXPUCAME` doesn't replicate --
  deliberately deferred to keep the initial port reviewable. CAME has more
  per-parameter state than Adafactor (two factored row/col pairs instead of
  one, plus the momentum buffer), making the batching trick a real port, not
  a copy-paste. Only worth doing if CAME's per-step Python-loop overhead is
  actually a measured bottleneck for typical LoRA parameter counts.

- **CAME momentum buffer in bf16.** `exp_avg` is CAME's one genuinely new
  full-size buffer vs. Adafactor. Storing it in bf16 instead of fp32 would
  roughly halve that buffer's footprint at a small, untested precision cost
  on a smoothed EMA quantity. Shelved because the buffer-reuse fix above
  already resolved the near-ceiling hang on its own -- revisit only if VRAM
  is tight again after that fix.

- **`lora.py` legacy 2816->3072 padding path.** Fragile, hardcoded special
  case for loading old-format LoRA checkpoints. Now at least logs a warning
  when it fires (visibility fix already shipped). Generalizing it or removing
  it once no one has 2816-dim checkpoints left to load is lower priority.

- **[2026-08] No shared `MemoryManager` reachable from
  `SupervisedLoRATrainerNode.build()`.** Found while wiring
  `ResourceProfile` (`nodes/memory/profile.py`, design doc section 5.5):
  `nodes/optimizer/strategies/chunked.py`'s `ChunkedScratchBufferStrategy`
  constructs its own private `MemoryManager` when none is injected, and
  nothing between it and the trainer node passes one in, so there's no
  single instance to hand `ResourceProfile.capture()` --
  `memory_manager_stats` is `None` in every real run today. Harmless
  right now (each strategy's private manager is internally consistent on
  its own), not fixed here -- see
  `docs/design/09-prioritized-backlog.md` section 10's note on this for
  when it'd actually start to matter.

- **[2026-08] Only `ResBlock` instances ever route through this
  project's checkpoint patch -- attention blocks don't, in this pinned
  ComfyUI version.** Found while building `nodes/model/block_profiler.py`
  (design doc section 2.3, backlog item 1): cloned
  comfyanonymous/ComfyUI directly to confirm what `ctx.run_function`
  actually is. `comfy/ldm/modules/diffusionmodules/openaimodel.py`'s
  `ResBlock.forward()` calls `checkpoint(self._forward, ...)` for real;
  `comfy/ldm/modules/attention.py`'s `BasicTransformerBlock.forward()`
  does not call `checkpoint()` anywhere in its body, despite taking a
  `checkpoint=True` constructor parameter that looks like it should.
  Not a bug in this project -- `enable_frozen_param_safe_checkpointing()`
  only ever patches what ComfyUI itself routes through it -- but it does
  mean `GreedyRatioPlacement` (once wired in) can only ever place
  ResBlocks, and `use_checkpoint=True`'s real VRAM savings today only
  ever come from ResBlocks, never attention blocks, regardless of what
  the constructor parameter's name suggests.

- **`config_model.py` doesn't yet warn about grad_accum's real-update math
  anywhere in the UI/docs.** The step-counting refactor fixed the mechanism,
  but nothing explains "steps now means real updates, cache/compute cost
  scales with steps*grad_accum" to a new user reading the config file cold.
