*[← docs/known-issues index](README.md)*

# Pending user testing

- **[2026-08] VRAM ratchet on non-square datasets -- root cause found,
  fix implemented, not yet confirmed.** Original hypothesis here
  (caching-allocator fragmentation from varying tensor shapes) tested
  directly and found wrong: `num_alloc_retries` stayed `0` across a
  200-step run, `vram_reserved` stayed flat (delta in single-digit MB)
  -- no fragmentation, no growth, on a uniform-shape dataset. On an
  actual non-square dataset, real per-phase VRAM capture (built this
  investigation specifically because a single end-of-step snapshot
  couldn't answer "which phase caused this") caught the real mechanism
  live: the entire VRAM jump (+560MB in one step) happened at exactly
  one phase boundary, `forward`, nowhere else in the step moved at all.
  Root cause: `resize_mode="fit"` preserves aspect ratio with no cap on
  the long side -- a sufficiently tall/wide image forces genuinely
  larger tensors through `forward`, the allocator grabs a bigger
  reserved block and keeps it permanently (nothing in this pipeline
  calls `empty_cache()` on its own). Fix: `manager/builder.py`'s
  `run_lora_ingestion_task` gets a new `max_aspect_ratio` parameter
  (default `2.0`) -- images whose "fit"-resized long side would exceed
  it get split into multiple same-caption crops instead of one
  oversized sample. Wired through the UI
  (`server/static/dataset_manager.html`/`.js`). Crop-box arithmetic
  verified against the actual committed function. **Not run** -- needs
  the person to re-ingest a non-square dataset with the new cap and
  confirm `vram_reserved` stays bounded.

- **[2026-09] `use_checkpoint=True` only ever actually checkpointed
  `ResBlock`, never SDXL's attention blocks -- real fix landed, not yet
  confirmed on real hardware.** Follow-up on the `docs/known-issues/
  deferred.md` entry with the same title/date-2026-08: two separate real
  OOM reports (`docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`'s
  third addendum) showed activation memory as roughly half of real
  reserved VRAM even with `use_checkpoint=True` on for both trainer
  routes (checked directly: it defaults `True` all the way through
  `build_lora_injected_unet()`, and nothing overrides it for either
  route) -- consistent with checkpointing only ever reaching `ResBlock`,
  a minority of SDXL's UNet next to its `BasicTransformerBlock` stacks
  (`transformer_depth` up to 10 per level). Re-confirmed the root cause
  directly against a fresh comfyanonymous/ComfyUI `master` checkout
  (this project doesn't pin/vendor a ComfyUI version -- `docs/setup.md`):
  `BasicTransformerBlock.__init__` never assigns its own `checkpoint`
  argument to `self`, and `SpatialTransformer.forward()`'s
  `transformer_blocks` loop calls each block directly, no `checkpoint()`
  anywhere in the path. Fix: `nodes/model/attention_checkpointing.py`'s
  `enable_attention_block_checkpointing()` monkeypatches
  `BasicTransformerBlock.forward()` to route through the same
  `checkpoint()`/`CheckpointFunction` seam `ResBlock` already used
  (reusing, not reimplementing, whichever `CheckpointFunction` variant
  this project's own `gradient_checkpointing.py` has installed -- the
  frozen-param-safe fix applies here too, since LoRA freezes most of a
  `BasicTransformerBlock`'s own parameters the same way it does
  `ResBlock`'s), composed into both `FrozenParamSafeCheckpointing.apply()`
  and `ProfilingCheckpointing.apply()` so both the real training path and
  `block_profiler.py`'s own instrumentation pick it up automatically.
  Verified with a new mocked-comfy smoke test
  (`nodes/smoke_tests/smoke_test_attention_checkpointing.py`, same
  technique as `smoke_test_gradient_checkpointing.py`: a faithful stub of
  `BasicTransformerBlock` built from the real, freshly-fetched
  comfyanonymous/ComfyUI source, real gradients checked, not just
  "doesn't crash") -- confirms the patch logic and the frozen-parameter
  fix both hold for this block too. **Not run** -- needs a real
  before/after VRAM comparison on real hardware (ideally against one of
  the two reports that motivated this) to confirm the actual activation-
  memory reduction this was built to deliver, the same gap
  `gradient_checkpointing.py`'s own ResBlock-only test coverage already
  disclosed (verifies the patch's logic, not that it moves real peak
  VRAM by the expected amount).

- **[2026-09-20] `BudgetedResourceControlHandle` (`nodes/memory/control_handle.py`)
  gained an explicit `synchronize()` around every offload/reload/release
  transition, a `strict` mode that raises instead of continuing once
  usage is still over budget after offloading everything it can, and a
  new `release()` method (deterministic, unconditional offload of a
  named resident, used by `ManagedLoRATrainerNode`,
  `nodes/train/managed.py`, every step) -- not a fix for a diagnosed
  bug, defensive hardening motivated by this file's own "Device lost"
  entry above.** No real XPU hardware in the
  environment this was built in -- `synchronize()`'s call sites and
  `strict`'s raise/no-raise logic are verified against a scripted fake
  `DeviceContext` (`nodes/smoke_tests/smoke_test_resource_control_strict.py`,
  the actual thing under test in that file's own docstring), not a real
  VRAM-pressure event on real hardware. Whether this actually helps with,
  or is even related to, the open "Device lost" report above is
  unconfirmed and stays unconfirmed here -- that report is about
  `core/trainer.py` (the legacy pipeline), this change is in `nodes/`
  (the rewrite), and the "Device lost" entry's own root-cause note (a
  missing/incomplete synchronize on an XPU offload path, cited from a
  different project's report on the same hardware) is a plausible
  *shape*, not a confirmed diagnosis, in either codebase. **Not run** --
  needs a real training run on real XPU hardware, under real VRAM
  pressure (e.g. a `vram_budget_mb` set below what the run would
  otherwise use), to say anything about whether this actually changes
  observed stability. See
  `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`
  for the full reasoning.

- **[2026-09-20] `AdaptiveResidencyController` (`nodes/train/managed.py`)
  and the `SDXLTextEncoder.offload()`/`unload()` fix (`nodes/model/text_encoder.py`)
  -- both real, motivated by an actual reported run (~4.7x slower than
  the main route, offloading never once necessary at 9.0GB peak against
  a 12500MB budget), neither one run again after the fix on the hardware
  that produced that report.** See
  `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`'s
  own addendum for the full investigation and every change made. **Not
  run** -- needs the same real run repeated post-fix to confirm the
  actual steps/sec improves, and to check whether anything in the
  still-open list there (no text-encoder caching on this route, no
  pinned host memory anywhere) is now the next dominant cost.

- **[2026-09-20] `AdaptiveResidencyController`'s ongoing escalation +
  `residency_safety_margin` (`nodes/train/managed.py`), fixing a real
  OOM report on a variable-resolution dataset (calibration sampled only
  smaller images, "stay resident" was locked in before the true worst
  case was ever measured) -- not run again on the hardware that
  produced that report.** See
  `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`'s
  second and third addenda for the full investigation, including a
  second, separate finding (not fixed, disclosed as a real limit): more
  than half of reserved VRAM in that report was activation memory,
  which this controller and NF4/Int8 weight quantization both leave
  completely unmanaged -- gradient checkpointing
  (`nodes/model/gradient_checkpointing.py`, real, unwired) is the actual
  lever for that, still not attempted. **Not run** -- needs the same
  variable-resolution dataset repeated post-fix to confirm the OOM is
  actually gone, and, separately, a real test of whether escalation's
  fundamental "next occurrence, not this one" limit is acceptable in
  practice or whether the dataset needs bucketing/resolution-aware
  calibration instead.
