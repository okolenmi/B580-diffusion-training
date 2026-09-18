*[← docs/known-issues index](README.md)*

# Pending user testing

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
