# Suspicious findings / deferred work

Running list of things noticed during review that aren't confirmed bugs (or
are confirmed but not urgent) so they don't get lost. Newest first.

## Open

- **[2026-07] Composition destruction on LoRA training -- infrastructure for
  testing the leading hypothesis now shipped, hypothesis itself still
  unconfirmed.** Long-standing issue: LoRA training reliably produces a
  "mess of badly-formed objects" rather than a clean quality/style shift,
  regardless of dataset/settings (sooner at high LR, later at low LR). Loss
  looks fine throughout -- the failure doesn't show up there.

  Leading theory: exposure bias / compounding distribution shift. Training
  samples come from the *teacher's own* deterministic denoising trajectory;
  the student is only ever supervised on states exactly on that trajectory.
  During real generation the student denoises from its *own* prior output,
  so the moment its prediction deviates even slightly from the teacher, every
  later step operates on a latent state never seen in training -- and the
  deviation compounds. Early (high-noise) steps stay close to trajectories
  the student trained on; by mid/low-noise steps it's improvising in
  unfamiliar territory, and "don't discard, keep pasting in plausible
  objects" is a plausible improvisation failure mode. This also explains why
  loss (measured only on in-distribution teacher states) doesn't catch it.

  This is a named, well-studied problem in imitation learning, and the
  standard fix -- DAgger (let the learner generate part of its own training
  trajectory so training data includes the drift states it actually hits) --
  already existed in this codebase (`student_mix` config), but only for full
  distillation, and only across *cyclic* boundaries (needed a second full
  model in VRAM, so was limited to loading a stale checkpoint from a
  previous cycle rather than the live model).

  Realized during this session: for LoRA (student = base weights + LoRA,
  same object as any "teacher" would be), DAgger doesn't need a second model
  at all -- the LoRA gate can be forced to 0 for a forward pass to get exact
  base-model behavior from the *same* live model object, with no separate
  VRAM cost and no staleness (always the actual current student, not a
  checkpoint from N steps ago). Implemented and verified this session:
  `lora_gate_override()` context manager (lora.py, nestable, restores prior
  state even on exception -- verified the scalar-gate broadcast zeroes the
  LoRA delta correctly for both 2D and 3D layer output shapes numerically),
  wired through every teacher.forward() call site in both cache_trajectory.py
  and cache_random.py (purely additive -- a no-op via contextlib.nullcontext()
  when the new `teacher_lora_gate` param isn't passed, so zero risk to
  existing distillation training), and trainer.py now skips building a
  separate teacher model entirely in LoRA mode, passing the live
  `self.student` (gate forced to 0) as the teacher instead. Fixed one real
  bug caught during this work: the cache-loop's `elif self.teacher:` gate
  became unreachable once self.teacher was deliberately left None in unified
  mode, which would have raised a RuntimeError on every LoRA run --
  corrected to `elif self.teacher or self._lora_unified_teacher:`.

  **NOT done yet**: this only removes the VRAM barrier and gives correct
  gate-based teacher substitution for the plain trajectory-rollout and
  target-computation passes. The actual DAgger *chain-mixing* logic
  (student_mix_frac, student_anchor_steps, student_chain_len,
  student_chain_noise) in cache_trajectory.py still expects a separate
  `student_model`/`student_unet_sd` (a *stale, previously saved* checkpoint)
  and is still wired for the cyclic-distillation case only -- it does not
  yet use the live `self.student` (gate=1) to mix in fresh, up-to-the-second
  student steps within a single-cycle LoRA run. That's the next piece:
  wire chain-mixing to use `self.student` directly (gate=1) instead of a
  loaded `student_cache`, and enable `student_mix_frac > 0` for LoRA's
  (always single-cycle) training loop. Once that's in, the actual
  experiment -- does DAgger-mixed training reduce composition destruction
  compared to pure teacher-trajectory training -- is still unrun.

  Distillation's own DAgger (student_mix + cyclic, loading a stale
  checkpoint) is untouched by any of this and continues to work exactly as
  before -- this was kept as a genuinely separate code path per explicit
  request, not merged/risked.

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
