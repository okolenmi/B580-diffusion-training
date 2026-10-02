*[← docs/known-issues index](README.md)*

# Deferred (not urgent, revisit later)

- **[2026-09-28] `torch.xpu`'s `pin_memory()` doesn't actually pin on
  this build -- `core/cache_utils.py`'s parallel pin thread-pool is a
  no-op optimization here.** Measured, not assumed: a 4 GB
  `pin_memory()`-ed tensor showed `Locked: 0 kB` in the allocating
  process's `/proc/<pid>/smaps_rollup`, and holding it (same-process or
  a second process) moved `torch.xpu.mem_get_info()` by single-digit MB.
  So the batches the cache pipeline "pins" before H2D transfer are
  ordinary pageable host memory, and `_PIN_POOL`'s comment ("pin_memory()
  is a kernel syscall (mlock) -- it releases the GIL") describes this
  build's *intended* behavior, not its actual one. Harmless (correctness
  unaffected; H2D transfers just don't get pinned-memory speed), left
  alone for now because removing the pinning would also be wrong -- if a
  future torch build fixes `pin_memory`, it starts helping again for
  free. Worth re-testing after any torch upgrade (the check above takes
  30 seconds); found incidentally during the 2026-09-28 VRAM-report
  investigation documented in [`resolved.md`](resolved.md).

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

- **`config_model.py` doesn't yet warn about grad_accum's real-update math
  anywhere in the UI/docs.** The step-counting refactor fixed the mechanism,
  but nothing explains "steps now means real updates, cache/compute cost
  scales with steps*grad_accum" to a new user reading the config file cold.
