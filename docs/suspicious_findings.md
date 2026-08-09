# Suspicious findings / deferred work

Running list of things noticed during review that aren't confirmed bugs (or
are confirmed but not urgent) so they don't get lost. Newest first.

## Open

- **[2026-08] 8 real, working Node classes exist but aren't selectable in
  the graph editor -- `server/nodegraph_registry.py`'s list is stale.**
  Confirmed directly, not a hypothesis: walked every concrete `Node`
  subclass under `nodes/` and diffed against
  `server.nodegraph_registry.get_registry()`'s actual returned set.
  Missing:

  - `P2LossWeightingNode` (`nodes/train/loss.py`).
  - `PrefetchingBatchSourceNode` (`nodes/dataset/prefetch.py`).
  - `ComposedAdamWOptimizerNode`, `ComposedAdafactorOptimizerNode`,
    `ComposedCAMEOptimizerNode`, `ComposedFusedAdamWOptimizerNode`,
    `ComposedFusedAdafactorOptimizerNode`,
    `ComposedFusedCAMEOptimizerNode` (`nodes/optimizer/composed_*.py`) --
    predate the 2026-08 nodes/ session entirely (from the earlier
    Algorithm/ExecutionStrategy optimizer rewrite); not introduced by
    that session, a pre-existing gap its own registry check happened to
    surface.

  All 8 are real, already-tested classes with nowhere to be selected
  from. The fix is mechanical (add each to `server/nodegraph_registry.py`'s
  import list and `classes` list, same as every other entry there),
  low-risk, no new logic needed. **Applied** -- `_load()` now imports and
  registers all 8. Verified statically (no torch/ComfyUI in the sandbox,
  same limitation as always): an `ast`-based walk of every concrete `Node`
  subclass under `nodes/` against both `_load()`'s import list and its
  `classes` list shows an exact match in both directions (nothing missing,
  nothing registered that doesn't exist) -- the same check the prior
  session did by hand with a live import, redone here without needing a
  real import. Not yet confirmed against the actual running palette --
  waiting on the person to apply the patch and check `/nodegraph`.

- **[2026-08] Training produces unstable/deforming results -- useful
  change and anatomy/content deformation appear at the same LoRA power
  level.** User-reported after a real training run (rank 48, alpha 1,
  dropout 0, weight_decay 0, t range [150, 999], LR 1e-5, clip_threshold
  1, 18000 steps). At LoRA power 2.0 a visible, useful change is there
  but comes with significant anatomy/content deformation; at power 1.0
  the useful change is much less visible but negative effects (some
  deformation) are still present. Not investigated. See
  `docs/session_handoff.md` for candidate hypotheses and suggested
  cheap-first experiments -- none diagnosed as *the* cause, genuine
  experimentation needed, not "apply hypothesis #1 and declare it
  fixed."

- **[2026-08] VRAM usage grows (not just runs high) with a dataset of
  mixed image dimensions, eventually crashing.** User-reported: more
  than 2x the expected batch-size footprint, and growing over time
  rather than stabilizing, specifically correlated with varying image
  dimensions across the dataset. Not investigated. Leading hypothesis
  (general PyTorch allocator behavior, not traced in this codebase):
  caching-allocator fragmentation from constantly-varying tensor shapes,
  not a genuine leak -- see `docs/session_handoff.md` for the reasoning
  and the one real data point (`vram_allocated=5539MB` vs
  `vram_reserved=9802MB`) that's mildly consistent with it, plus a
  separate, real concern surfaced while looking at this:
  `DeviceResident.footprint_bytes()` (items 3/9/12 of the 2026-08
  session) doesn't check actual device placement anywhere, which could
  explain `tracked_footprint_mb` coming out *higher* than
  `vram_allocated_mb` in that same log line -- backwards from what
  static-weight-only accounting vs. real allocated-including-activations
  should show.

- **[2026-08] Training is much slower than the pre-nodes/ implementation
  -- `optimizer_step` dominates step time.** User-reported, with one real
  `profile=True` log line (batch size 2, 512px): `forward=245ms
  backward=287ms optimizer_step=1041ms total=1575ms` --
  `optimizer_step` alone is ~66% of total step time, disproportionate for
  a LoRA's small, fixed parameter set. Fairly confident leading
  hypothesis given `clip_threshold=1` in the reported settings (an
  Adafactor/CAME-family parameter): the optimizer node in use is a
  "Chunked" variant (`AdafactorOptimizerNode`/`CAMEOptimizerNode`, real
  per-parameter device syncs, built for VRAM-bounded large-parameter-count
  scenarios) rather than the matching "Foreach" variant
  (`ForeachAdafactorOptimizerNode`/`ForeachCAMEOptimizerNode`, vectorized,
  explicitly documented in `nodes/optimizer/foreach_came.py` as "the
  right default for LoRA's small, fixed parameter set"). See
  `docs/session_handoff.md` for the full reasoning and fallback steps if
  a Foreach node is already in use.

- **[2026-07] "Device lost" errors and silent training hangs after
  VRAM-pressure events, reported from real ComfyUI use (not this
  project's `nodes/` work).** User-reported, not yet investigated here.
  Symptom: not a normal OOM -- either a device-lost error or a silent
  hang, most reliably reproduced by a VRAM-heavy sequence (merging three
  6GB models, generating with an intermediate merge state, then
  generating again with a different base model) and, separately, seen in
  this project itself after a VRAM spike during preview generation's VAE
  decode step -- loss of a few hundred MB stabilizes (frees back down)
  but training hangs a few steps later *despite* free VRAM being
  available afterward. User's own read, worth taking seriously: something
  gets offloaded under memory pressure but isn't correctly loaded back,
  even though there's room for it. Likely related to the "Persistent
  ~500MB VRAM growth after preview generation" entry below (same VAE
  decode trigger point) but the *symptom* here (hang/device-lost, not
  just VRAM not dropping back down) is a distinct, arguably more serious
  report -- not confirmed to be the same root cause, not assumed to be
  either. A `kohya-ss/musubi-tuner` discussion training Wan2.2 on the
  same B580 hardware describes a matching hang-after-offload symptom,
  traced there to a `synchronize_device()` call missing its `device`
  argument on the non-CUDA path -- a plausible root-cause *shape* (async/
  non-blocking transfer without a matching explicit synchronize on the
  XPU path) worth checking `core/trainer.py`'s own offload code for, not
  a confirmed diagnosis here. Not investigated further this session --
  out of scope for `nodes/`-only work, and needs `core/trainer.py`,
  which `nodes/` doesn't touch.

- **[2026-07] Composition destruction on LoRA training -- DAgger chain-mixing
  now wired for LoRA end-to-end; the actual experiment is still unrun.**
  Long-standing issue and leading hypothesis unchanged from before (exposure
  bias / compounding distribution shift from teacher-trajectory-only
  training -- see prior entry text preserved below).

  This session's remaining piece is done: chain-mixing's "student" steps
  (`student_mix_frac`/`student_anchor_steps`/`student_chain_len`/
  `student_chain_noise`) now use the *live, currently-training*
  `self.student` in unified-LoRA mode (`trainer.py`'s `_get_cache()` passes
  `student_model=self.student`), not a separate stale checkpoint. Verified
  before shipping, not just assumed:
    - `_current_gate` defaults to `None` (full, ungated LoRA) at module load
      and nothing sets it before cache-build in LoRA mode (single-cycle,
      cache always built before any training step runs) -- so student-chain
      steps get correct full-strength LoRA behavior with no extra plumbing.
    - The target-computation pass (the actual supervised label for every
      collected state) unconditionally uses `teacher.forward()` inside
      `_teacher_ctx()` regardless of whether that state was reached via a
      teacher-only or student-chain step during rollout -- confirmed this
      matches the real DAgger recipe (aggregate states visited under the
      current policy, but always label with the expert's true output), not
      just "sometimes mix in different data."
    - `self.student` staying in `.train()` mode (vs. `.eval()`, which the
      existing code path never explicitly sets when `student_model` is
      passed) has no behavioral effect here: SDXL_CONFIG has no dropout key
      (ComfyUI defaults to 0), this architecture uses GroupNorm which is
      train/eval-mode-independent (unlike BatchNorm), and the entire rollout
      already runs under `torch.no_grad()` regardless.

  **Still unrun**: the actual experiment -- does DAgger-mixed training
  measurably reduce composition destruction vs. pure teacher-trajectory
  training for LoRA. Suggested test: short run, `student_mix` around
  0.3-0.5, compare previews against a `student_mix=0` baseline on the same
  seed/dataset.

  Distillation's own DAgger (student_mix + cyclic) is untouched by any of
  this, exactly as requested.

- **[2026-07] Distillation's existing DAgger chain-mixing always rolls out
  from the *initial* pre-training weights, never anything reflecting actual
  training progress -- across any cycle, not just "one cycle behind."**
  Discovered while wiring the LoRA version above; confirmed by tracing every
  assignment to `self.student_unet_sd` in trainer.py -- it's set exactly
  once, inside `load_models()`, and never touched again. Since distillation's
  chain-mixing always goes through `student_unet_sd=self.student_unet_sd`
  (a fixed dict reference) rather than a live model object
  (`student_model=None` is hardcoded at that call site), every cycle's
  chain-mixing constructs a fresh `ComfyUNetWrapper` from those same
  never-updated original weights. This matches the user's own independent
  assessment of distillation's DAgger as "just an imitation." Real, but
  explicitly out of scope this session -- the user asked to keep
  distillation's path untouched while LoRA gets a real (live-model) version.
  If distillation's DAgger is revisited later, the fix is straightforward
  given the LoRA work already done: refresh `self.student_unet_sd` (or
  better, pass `student_model=self.student` directly, mirroring the LoRA
  change) from the actual trained weights between cycles.

## Pending user testing

- **[2026-07] Persistent ~500MB VRAM growth after preview generation.**
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

## Resolved

- **[2026-07] CAME optimizer VRAM near-ceiling hang after ~60 steps.** Root
  cause: `res` and `update` in `ChunkedXPUCAME.step()` each allocated a fresh
  full-parameter-sized tensor per step (on top of Adafactor's baseline
  scratch-buffer usage), slowly fragmenting VRAM near the ceiling. Fixed by
  reusing the existing scratch buffer in place for both. Confirmed fixed by
  user.

- **[2026-07] Default `snr_weighting: "snr"` used the v-prediction Min-SNR
  formula (`snr/(snr+1)`) unconditionally, including for the default
  `student_type: "eps"`.** For eps-prediction the correct uncapped form is
  trivially 1.0 (uniform); the old default gave ~99% weight to easy/low-noise
  steps and ~1% to high-noise/structural steps -- close to the opposite of
  what's wanted. Fixed by branching `snr`/`min_snr_5`/`decay_snr` on
  `student_type`. Recommended switching configs to `min_snr_5` explicitly
  (the correctly-implemented standard choice for eps) rather than relying on
  `snr` reducing to a uniform no-op.

- **[2026-07] `grad_accum` inflated "steps" to mean micro-batches, not real
  optimizer updates.** `steps: 5000, grad_accum: 32` only did `5000/32 = 156`
  real weight updates; LR schedule, save/preview cadence, and the dashboard
  all silently used the wrong count. No warning, and the shipped example
  config (`convert-cfg.toml`) already had `grad_accum: 32`. Refactored so
  `steps` means real optimizer updates everywhere (dashboard, saves,
  previews, LR schedule); cache size and micro-batch loop scale internally by
  `grad_accum` instead. Confirmed working by user (correct step count,
  expected per-step timing).

## Deferred (not urgent, revisit later)

- **[2026-07] New finding, informational only:
  `ChunkedXPUAdafactor`'s momentum handling corrupts `exp_avg` in place
  when a parameter's dtype is float32.** Found while building and
  verifying `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm`
  against this class directly (see `docs/nodes_package_design.md`'s
  "Third data point: AdafactorAlgorithm" section for the full writeup).
  In `step()`: `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`, where
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
  for anything built through `nodes/` going forward** (see
  `docs/nodes_package_design.md`'s "Centralized memory management"
  section for the design). This does **not** fix or touch
  `core/optimizers.py`'s legacy classes -- per `nodes/`'s existing rule,
  that file hasn't been modified. Left here as a pointer, not a claim of
  resolution: once the node-graph optimizer path replaces the legacy one
  (see `nodes_package_design.md`'s "Concrete next step" list), this whole
  class of VRAM-lifecycle bug should stop being something to watch for by
  construction, rather than something to keep re-auditing by hand.

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

- **`config_model.py` doesn't yet warn about grad_accum's real-update math
  anywhere in the UI/docs.** The step-counting refactor fixed the mechanism,
  but nothing explains "steps now means real updates, cache/compute cost
  scales with steps*grad_accum" to a new user reading the config file cold.
