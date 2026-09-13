*[← docs/known-issues index](README.md)*

# Deferred (not urgent, revisit later)

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
