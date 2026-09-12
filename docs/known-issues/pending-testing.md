*[← docs/known-issues index](README.md)*

# Pending user testing

- **[2026-08] LoRA timestep gate now wired -- candidate fix for a real
  deformation/quality report.** User-reported after a real training run
  (rank 48, alpha 1, dropout 0, weight_decay 0, **t range [150, 999]**,
  LR 1e-5, clip_threshold 1, 18000 steps): at LoRA power 2.0 a visible,
  useful change came with significant anatomy/content deformation; at
  power 1.0 the useful change was much less visible but some deformation
  was still present.

  **Strong candidate, not confirmed**: `t range [150, 999]` excludes the
  low end of the timestep range (t<150). Separately confirmed:
  `core/lora.py`'s `set_lora_gate()`/`compute_lora_gate()` -- which
  exists specifically to keep the LoRA's contribution close to the
  frozen base outside a dataset's own trained t-range -- had zero wiring
  anywhere in `nodes/` (confirmed via exhaustive grep). A LoRA trained on
  this t range, run through `nodes/`, would apply its full, ungated delta
  at every timestep during generation, including t<150 -- never
  supervised at all. That's a plausible, mechanistic match for "useful
  change and deformation showing up together, worse at higher power" (an
  un-gated delta doesn't get *weaker* at unsupervised timesteps, it gets
  applied at exactly the same strength as everywhere else).

  The gate is now called from `PrepareDiffusionInputsPhase`
  (`nodes/train/step_pipeline.py`) at the same point the legacy pipeline
  calls it. New `gate_enabled` (default `False`, matching the legacy
  default)/`gate_train_low`/`gate_train_high`/`gate_width` ports on
  `SupervisedLoRATrainerNode`. Verified: real `step_pipeline.py` file
  loaded directly (not a copy) with `torch`/`core.lora` mocked to record
  calls -- both the enabled and disabled branches call the right
  functions with the right arguments, and the gate correctly resets to
  `None` when disabled after having been enabled. **Not run** -- needs
  the person to set a restricted `t_low`/`t_high` on a dataset (their own
  earlier report used `[150, 999]` -- a good real case to retest),
  matching `gate_train_low`/`gate_train_high`, and compare LoRA quality
  with `gate_enabled` `True` vs. `False`.

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

- **[2026-08] CAME `optimizer_step` ~7x AdamW's -- confirmed structural,
  not the old "Chunked vs. Foreach host-sync" hypothesis, fix
  implemented, not yet confirmed.** Original hypothesis here (wrong
  optimizer node in use, "Chunked" instead of "Foreach") tested directly
  and found wrong: `ForeachCAMEOptimizerNode` and
  `ComposedCAMEOptimizerNode` (any strategy) showed the same ~1041ms
  `optimizer_step`, and running the legacy `CAMEOptimizerNode` through
  the current, properly-synchronized profiler gave the same number too
  -- ruling out both "wrong node" and "the old pre-nodes/ pipeline was
  genuinely faster" (its own timer, `core/timer.py`'s `StepTimer`, has
  zero `synchronize()` calls anywhere in it, so it never measured real
  GPU execution time to begin with). Real cause, confirmed by direct
  A/B: CAME vs. AdamW, same everything else, `1041ms` vs. `148ms` -- CAME's
  math runs as an un-batched per-parameter Python loop in every current
  implementation (confirmed by reading all four: `ChunkedXPUCAME`,
  `ForeachXPUCAME`, and every `ComposedCAMEOptimizerNode` strategy).
  Fix: new `ShapeGroupedBatchStrategy`
  (`nodes/optimizer/strategies/shape_grouped.py`), groups parameters by
  exact shape/dtype/device/lr and runs each group's entire update as one
  batched computation. Equivalence-verified via a numpy transcription of
  the exact math (~7e-7 max relative difference across group sizes
  1/2/5/20 and multiple shapes) and a shipped smoke test
  (`nodes/smoke_tests/smoke_test_shape_grouped_equivalence.py`). **Not
  run** -- needs the person to run that smoke test on real hardware,
  then a `profile=True` comparison against the `1041ms` baseline with
  `strategy="shape_grouped"`.

- **[2026-08] Same root cause confirmed for Adafactor too, and now fixed
  for both -- real user report, not just the CAME finding above.** User
  reported real, felt slowness on real hardware: AdamW 3-4x faster than
  Adafactor/CAME, specifically worse on a weak CPU, and "even legacy
  code converted into nodes (adafactor/CAME)" feeling slower. Two things
  checked directly, not assumed:

  1. **The node wrapper itself is not the cause.**
     `AdafactorOptimizerHandle.step()`/`CAMEOptimizerHandle.step()`
     (`nodes/optimizer/adafactor.py`/`came.py`) are one-line pass-throughs
     to the exact same legacy `core.optimizers` classes `core/trainer.py`
     calls directly -- confirmed by reading both. Negligible Python call
     overhead either way; wrapping in a node isn't where a real slowdown
     could come from.
  2. **`AdafactorAlgorithm` had the identical gap `CAMEAlgorithm` had
     before the fix above -- confirmed by reading it, not assumed from
     the CAME finding alone.** No `compute_update_batched()` override at
     all, so `ShapeGroupedBatchStrategy` silently fell back to
     `Algorithm`'s own default (loop + stack -- see algorithms/base.py),
     giving zero real batching benefit for Adafactor even when
     `strategy="shape_grouped"` was already selected. `ForeachApplyStrategy`
     doesn't help either for either optimizer -- confirmed by reading
     it: it only batches the final `decay`/`delta` *apply* step via
     `torch._foreach_*`, not the per-parameter algorithm math itself
     (its own module docstring says so directly). So `strategy="chunked"`
     and `strategy="foreach"` both still pay a real per-parameter Python
     loop for Adafactor/CAME's actual update computation -- on a weak
     CPU, where kernel-dispatch/interpreter overhead dominates over the
     (tiny, per-LoRA-matrix) actual compute time, this is exactly the
     kind of cost that shows up as "the optimizer section takes an
     enormous amount of time," and disproportionately worse than a
     faster CPU would show.

  Fix: `AdafactorAlgorithm.compute_update_batched()`
  (`nodes/optimizer/algorithms/adafactor.py`), same pattern as CAME's
  existing override -- covers the common, already-recommended
  `scale_parameter=False` case. Real, documented scope boundary, not
  papered over: `scale_parameter=True` still falls back to the slow
  per-member path, because `alpha_t` (and therefore `decay`) would
  genuinely vary per group member in that mode, breaking
  `compute_update_batched()`'s shared-decay contract -- extending that
  contract is real, separate work with no urgent need yet, since
  `scale_parameter=True` already has its own documented pathology for
  LoRA's zero-initialized B matrix (see `adafactor.py`'s own module
  docstring) and isn't the recommended setting regardless.
  Equivalence-tested (bit-exact, not just within tolerance, across every
  case checked) against the per-member reference --
  `nodes/smoke_tests/smoke_test_adafactor_shape_grouped_equivalence.py`.
  **Not run on real hardware** -- same caveat as CAME's own entry above:
  needs the person to run `strategy="shape_grouped"` on both Adafactor
  and CAME and compare against their own real `1041ms`-class baseline.

  One more real, honest caveat on the "GPU AdamW is 3-4x faster" part of
  the report: this is *expected*, not itself a bug. `SimpleAdamWOptimizerNode`
  wraps `torch.optim.AdamW(foreach=True)` -- PyTorch's own first-party
  batched multi-tensor kernels, real batching Adafactor/CAME's own math
  has never had until the fix above. Even with `shape_grouped`, some gap
  between AdamW and Adafactor/CAME is genuinely expected (more reduction
  operations per parameter, not just an execution-overhead difference) --
  the open question `shape_grouped` is meant to answer is whether the gap
  shrinks to something like that inherent-complexity difference, not
  whether it disappears entirely.


  Reported as compounding slowly (not just a one-time jump), first appeared
  sometime after an earlier preview-VRAM fix (exact point unknown). Ruled
  out two candidates by reading the code: CAME's own memory pool (only
  entered during `optimizer.step()`, which preview never calls) and
  `PreviewGenerator`'s cached conditioning (set once at construction, never
  mutated). Couldn't reproduce or narrow further without XPU hardware, so
  shipped `TRAIN_VRAM_DEBUG=1` env-var-gated diagnostics instead of guessing
  further: 8 checkpoints (`vram_snapshot()` in `comfy_setup.py`) across
  `preview_sampler.py`'s `generate()` (entry, after denoising, after VAE
  load, after decode loop, after `vae.free()`) and `trainer.py`'s
  `_generate_preview()` (entry/before offload, after offload, after
  `generate()` returns, exit/after reload) plus a baseline every 250
  micro-steps during ordinary training. Zero overhead when the env var is
  unset. Waiting on the user to run this and report which checkpoint's
  reading doesn't drop back down across 2-3 consecutive previews.

- **[2026-07] Corrected an overstated claim about unified-teacher LoRA's VRAM
  benefit.** Originally claimed removing the separate teacher model would
  meaningfully reduce steady-state training VRAM. Wrong -- traced the
  existing code and found `self.teacher` was already being moved to CPU
  (`self.teacher.to("cpu")`) right after cache generation, before the main
  training loop starts, in the *original* code too. So the old code's
  resident-during-training VRAM was already just one model, not two; the
  unified-teacher change's real benefit is reducing peak VRAM during the
  (shorter) cache-generation phase specifically, not steady-state training.
  Matches user's report of no measurable change in their monitored training
  VRAM after applying that patch.
