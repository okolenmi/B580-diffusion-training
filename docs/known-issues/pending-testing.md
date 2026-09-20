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
