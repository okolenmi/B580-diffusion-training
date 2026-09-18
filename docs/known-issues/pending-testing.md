*[← docs/known-issues index](README.md)*

# Pending user testing

- **[2026-09] `core.optimizers.ChunkedXPUAdafactor`/`FusedXPUAdafactor`
  silently corrupted their own momentum buffer for float32 parameters
  with `beta1` (momentum) set -- root cause found, fix implemented, not
  yet confirmed.** `g = self.exp_avg[i]` aliased the momentum buffer (no
  copy); the following `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`
  called `.to(dtype=p.dtype)`, which for a float32 parameter (state is
  already float32) returned the *same object*, not a copy -- so the
  subsequent `.mul_(alpha_t)` mutated the momentum buffer in place. Net
  effect: every step, right after using the momentum buffer to compute
  that step's update, the buffer got permanently shrunk by `alpha_t`
  (~lr) as an unintended side effect. Confirmed directly, not theorized
  -- see `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`'s
  `check_legacy_float32_momentum_no_longer_corrupted()` for the
  `FusedXPUAdafactor` case; `ChunkedXPUAdafactor`'s copy of the same
  pattern (`core/optimizers.py`, main-parameter path) confirmed the
  identical way by reading it directly. bf16 parameters were never
  affected (`.to(dtype=p.dtype)` performs a real cast there, producing a
  genuine copy).
  **Scope differed between the two, checked precisely rather than
  assumed:** `FusedXPUAdafactor`'s buggy line was shared by both its
  tiny-parameter (< 10,000 element) and main-parameter code paths, so it
  affected every parameter size. `ChunkedXPUAdafactor`'s sat only in its
  main-parameter path (guarded by `p.numel() < 10_000` routing tiny
  parameters elsewhere instead) -- its own tiny-parameter path applies
  updates via a different mechanism (`ws[s:e].add_(..., alpha=-alpha_t)`,
  no aliasing problem) and was never affected. `ForeachXPUAdafactor`
  never had this bug at all -- both its factored and unfactored paths
  use non-in-place `.mul(alpha_t)`, which always returns a new tensor
  regardless of dtype, never aliasing the momentum buffer.
  `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm` never
  had this bug either way. Fix: forced a real copy
  (`.to(dtype=p.dtype, copy=True)`) in both `core/optimizers.py`
  locations, regardless of whether the dtype conversion alone would
  already produce one -- `core/` is untouched by the `nodes/` rewrite's
  own restructuring (see `docs/architecture.md`), but bugs found in it
  get fixed in place, which this is. Since the fix makes legacy's
  momentum math structurally identical to `AdafactorAlgorithm`'s (both:
  blend into the buffer, copy, scale, cast), the float32+momentum
  combination that `smoke_test_fused_adafactor_equivalence.py` used to
  deliberately exclude from its equivalence grid (because the bug made
  it meaningless to compare against) is now included, expected to agree
  within the same `1e-4` tolerance every other float32 case does.
  **Not run** -- needs `smoke_test_fused_adafactor_equivalence.py` run
  against real torch to confirm both the fix itself
  (`check_legacy_float32_momentum_no_longer_corrupted()`) and the
  newly-included float32+momentum equivalence checks.

- **[2026-09] `DeviceResident.footprint_bytes()` didn't check actual
  device placement -- root cause found, fix implemented across all
  four real implementations, not yet confirmed.** `footprint_bytes()`
  is documented as "current device-memory usage," but every
  implementation just summed `numel() * element_size()` over whatever
  tensors it held -- a number that doesn't change when a tensor moves
  from GPU to CPU, since shape/dtype don't change either. Practical
  consequence: after `offload()`, `footprint_bytes()` kept reporting
  the same pre-offload total, as if nothing had moved. This directly
  undermines a real, designed monitoring feature --
  `nodes/train/step_pipeline.py`'s own comment describes comparing
  `ResourceCoordinator.total_footprint_bytes()`'s sum against the real
  device-driver-reported VRAM stats as "a useful sanity signal," with "a
  growing gap between them worth investigating" -- offloading anything
  would produce exactly that gap, not because of a real accounting
  problem but because `footprint_bytes()` itself couldn't tell CPU-
  resident from device-resident. Confirmed by reading all four concrete
  `DeviceResident` implementations directly, not assumed to generalize
  from one: `SDXLTextEncoder` (`nodes/model/text_encoder.py`),
  `ComfyUNetTrainableModel` (`nodes/model/lora_injector.py`),
  `ComposedOptimizerHandle` (`nodes/optimizer/composed.py`, also covers
  `ComposedFusedOptimizerHandle` via inheritance), and
  `AdafactorOptimizerHandle` (`nodes/optimizer/adafactor.py`, the one
  remaining legacy-wrapping optimizer) -- all four had the identical
  gap. `nodes/model/text_encoder_cache.py`'s caching wrapper delegates
  to its inner resident's `footprint_bytes()`, so needed no separate
  fix. Fix: each implementation now tracks whether it's currently
  offloaded (`SDXLTextEncoder`/`ComfyUNetTrainableModel` already tracked
  `self._device_before_offload` for `reload()`'s own use, just never
  checked it here; `ComposedOptimizerHandle`/`AdafactorOptimizerHandle`
  needed a new `self._offloaded` flag) and return `0` from
  `footprint_bytes()` while it's set -- correct, not a compromise: `0`
  bytes of *device* memory is exactly right for something living in
  host RAM. Second, related bug caught while implementing this:
  `reload()` in both `SDXLTextEncoder` and `ComfyUNetTrainableModel`
  read `_device_before_offload` but never reset it afterward -- harmless
  before (nothing else read it), but would have made `footprint_bytes()`
  incorrectly report `0` forever after the *first* offload, even once
  reloaded, without also fixing this. **Not run** -- verification
  script written (`nodes/smoke_tests/smoke_test_footprint_bytes_after_offload.py`,
  covers all four implementations, the two model/text-encoder ones
  against minimal fakes rather than full legacy ComfyUI objects), not
  yet executed against real torch.

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
