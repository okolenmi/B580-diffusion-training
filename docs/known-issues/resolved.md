*[← docs/known-issues index](README.md)*

# Resolved

Cases that were real bugs, were fixed, and are worth remembering — either
because the failure mode is easy to reintroduce or because the measurement
is the only record of it. See the [index](README.md) for the rule that the
hardware numbers here are not to be trimmed: `runs/` and `datasets/` are
gitignored, so VRAM figures, steps-per-second and OOM thresholds exist
nowhere else.

Entries carry a **Confirmed** line when they were run on hardware rather
than reasoned about. Those runs were on an Intel Arc B580, 12 GB,
torch 2.12.1+xpu, through `scripts/hw_validate.py` and
`scripts/hw_validation_batch.sh`; the ones without it were found and fixed
by inspection.

- **[2026-10-04, confirmed on hardware] `keep_incomplete_batches` and the training diagnostics: the counts were right, and the probe turns out to cost wall time and no memory at all.**

- **[2026-09-30] Loader silently drops images whose (caption, size) group
  is smaller than `batch_size` (shuffle on).** `ManagedDatasetLoader` now
  prints a one-time warning with exact counts, and
  `ManagedDatasetSourceNode` has a new `keep_incomplete_batches` Port
  (default False = old behavior) that keeps those samples as smaller
  batches. Covered by `manager/smoke_tests/smoke_test_loader_incomplete_
  batches.py` (real sqlite + shard).

  **Both open questions answered on hardware, 2026-10-04.**

  *The warning's counts are correct on every real dataset.* Verified by
  recomputing the grouping independently from the loader's own buckets --
  `never` (samples in groups smaller than a batch) and `partial` (the
  remainder `len(group) % batch_size`) -- and comparing all four printed
  numbers (`never`, `total`, `partial`, and the derived "only N of M are
  used") plus the number of samples actually yielded:

  | dataset | bs | samples | never | partial | usable | yielded | match |
  |---|---|---|---|---|---|---|---|
  | `1024 aes` | 2 | 201 | 0 | 1 | 200 | 200 | yes |
  | `1024 aes` | 4 | 201 | 0 | 1 | 200 | 200 | yes |
  | `non-square` | 2 | 273 | 19 | 12 | 242 | 242 | yes |
  | `non-square` | 3 | 273 | 49 | 20 | 204 | 204 | yes |
  | `non-square` | 4 | 273 | 70 | 19 | 184 | 184 | yes |

  `1024 aes` and `test2` are a single `(prompt, size)` group each, so only
  the remainder is ever lost. **`non-square` is the one that matters**: 63
  groups, and at batch 4 **89 of its 273 samples (33%) are never trained
  on** without the flag. With `keep_incomplete_batches=True` all 273 are
  used, at both batch sizes.

  *But it costs throughput, and the warning does not say so.* Measured with
  `scripts/hw_validate.py main --dataset "non-square"`, 30 steps:

  | bs | shapes off | shapes on | steps/sec off | steps/sec on | change |
  |---|---|---|---|---|---|
  | 2 | 44 | 75 | 0.412 | 0.467 | +13% |
  | 4 | 22 | 73 | 0.412 | 0.288 | **-30%** |

  So the flag trades throughput for coverage, and at batch 4 on a
  many-shaped dataset it is a 30% throughput cost to train on 33% more
  images -- worth taking deliberately, not by default. `scripts/hw_validate.py`
  grew `--keep-incomplete-batches` for these measurements.

- **[2026-09-30] Fixed-probe / gradient-alignment diagnostics**
  (`probe_every_n_steps` etc. on `ManagedLoRATrainerNode`; see
  [`../training-diagnostics.md`](../training-diagnostics.md)). **Run on a
  real SDXL UNet and the B580, 2026-10-04.** `scripts/hw_validate.py` grew
  `--probe-every-n-steps`, `--probe-items`, `--probe-points-per-bucket` and
  `--probe-grad-alignment` for the measurements.

  *Step 1 reads as documented.* A LoRA's `B` is zero at step 1, so the
  adapter is the identity and `rel` must be 1 and `drift` 0:

      [probe step 1]  t_low: rel=0.999 drift=0.0000  t_mid: rel=0.999 drift=0.0000
                      t_high: rel=1.000 drift=0.0000  worst_rel=1.000

  Same under `--weight-store nf4` (rel 0.999 / 0.999 / 1.000, drift 0.0000,
  peak reserved 9,054 MB against 9,234 MB for bf16).

  *Cost, measured per step rather than per run.* Ordinary step 1.30 s.

  | | probe step | vs ordinary | peak allocated |
  |---|---|---|---|
  | forward-only | 4.22-4.24 s | **+226%** | 8,820 MB — **unchanged** |
  | `--probe-grad-alignment` | 18.68 s | **+1337%** | 8,820 MB — **unchanged** |

  The forward-only probe costs **no VRAM at all**, because it runs under
  `no_grad` and so retains no activations. Its cost is entirely wall time on
  the steps it fires, so the amortised price is set by `probe_every_n_steps`
  and nothing else.

  *`probe_grad_alignment`'s backward fits, and does not even raise the
  peak.* Byte-identical 8,820 MB against a plain training step, at batch 2,
  1024, on a run whose total peak reserved was 9,234 MB. It costs 14x wall on
  the step it fires on and nothing in memory. The numbers it produces are
  usable, and the doc's noise-floor warning is borne out at the defaults
  (`probe_items=2`, `probe_points_per_bucket=2`):

      align_t_low=0.970  align_t_mid=0.597  align_t_high=0.171
      self_t_low=0.021   self_t_mid=-0.072  self_t_high=0.248

  `self_t_high=0.248` is a split-half self-cosine of a quarter, i.e. the
  high-t bucket's own gradient estimate is mostly noise at the default sample
  count — which is exactly what `training-diagnostics.md` says to check
  before reading the cross-bucket numbers against it.

  **DoRA cannot be covered, and that is a wiring gap rather than a
  measurement.** The probe ports are on `ManagedLoRATrainerNode`, and
  `nodes/train/managed.py` has **no `adapter_strategy` input at all** — so
  the managed route cannot train DoRA at all. DoRA is reachable only from
  the main route, through `ComfyUNetLoRANode`'s `adapter_strategy` port
  (`nodes/model/lora_injector.py:297`), which the probe does not attach to.
  Covering it means threading `adapter_strategy` through
  `LoRATrainingConfigNode` into `ManagedLoRATrainerNode`. Small, and left
  for the same reason DoRA's real-run validation is still open in
  `docs/design/09-prioritized-backlog.md`: it changes what the managed route
  can do, which is a project's call.

- **[2026-10-04, confirmed on hardware] NF4 weight storage costs 0.10% of
  loss and 16% of throughput, and saves 180 MB — a bad trade at batch 2 /
  1024.** Measured as a *matched pair* on the B580: managed route, batch 2,
  dataset `1024 aes`, 40 steps, seed 1234, rank 64, identical in every
  respect except `--weight-store`.

  | | bf16 | nf4 | change |
  |---|---|---|---|
  | loss at step 0 | 0.176599 | 0.176780 | **+0.102%** |
  | mean relative loss difference, 40 steps | — | — | **0.147%** (max 0.747%) |
  | loss change over the run | -39.9% | -39.8% | same |
  | throughput | 0.707 steps/sec | 0.593 steps/sec | **-16%** |
  | peak reserved | 9,228 MB | 9,048 MB | **-180 MB** (-2%) |

  **The quality question is answered, and it is not close to a problem.**
  `NF4WeightStore`'s documented ~9% relative RMSE is in *weight* space; at
  step 0, before any update, it moves the loss by 0.10%, and the two loss
  curves stay within 0.147% on average for 40 steps while descending by the
  same 39.9%. So the quantisation error is three orders of magnitude smaller
  in function space than in weight space, and LoRA training on this UNet is
  unaffected. That is the diffusion-specific question
  `docs/design/09-prioritized-backlog.md` item 1 asked to be measured rather
  than assumed from QLoRA's LLM benchmarks.

  **The trade is the surprise.** 16% of throughput for 180 MB — 2% of a
  9.2 GB peak — when the floor leaves roughly 2.8 GB of headroom. It is a
  poor deal *at this operating point*, and it is a good deal only where
  memory is the binding constraint. Note this is a **smaller** saving than
  the 1,361 MB `docs/known-issues/open.md` records for the same lever, which
  was measured at a different operating point; the two should be read
  together rather than as one contradicting the other.

  **The limit of this measurement, stated plainly:** 40 steps at batch 2 on a
  201-image dataset is about 0.4 epochs. This shows NF4 does not *break*
  training over a short run. It does not show output quality over a full
  fine-tune, and reading it that way would be reading more than it says.

  Two parts of that backlog item remain open and are untouched by this:
  `NF4WeightStore`'s `MemoryManager`-backed scratch buffer for the
  dequantised tensor, and DoRA not honouring `NF4WeightStore` (QDoRA).


- **[2026-09-29] A cold `CachingTextEncoder.encode()` called
  `ensure_loaded()` twice (once per cache half), and the test that
  caught it was edited to expect two instead of fixing the encoder.**
  The split-key redesign gave each half (`encode_prompt_only` /
  `resolution_embedding`) its own `ensure_loaded()`, so a both-cold
  encode fired it once per half; `c703aa6` changed
  `check_resource_control_called_only_on_miss`'s assertion from one
  call to two, justified by "the second call is a no-op in real
  ResourceControlHandle (already resident)" -- not true:
  `BudgetedResourceControlHandle.ensure_loaded()` skips the reload when
  resident but unconditionally runs `_make_room()`, whose first act is a
  `memory_stats()` read -- a real device query on every call. Every
  both-halves-cold encode paid that query twice for the rest of the run
  with caching on. Fixed in the encoder this time:
  `CachingTextEncoder.encode()` checks both cache keys before either
  half loads and calls `ensure_loaded()` once, suppressing the halves'
  own per-miss calls only for the duration of that call (direct half
  calls keep theirs); the test asserts one again, plus a new check that
  direct half calls still ensure. The docstrings now state the real
  cost too: `ensure_loaded()`'s no longer calls the resident-path
  check just "cheap" -- it says the reload branch is skipped yet
  `_make_room()`'s `memory_stats()` read (a real device query) still
  runs every call.

- **[2026-09-28] User-reported wrong VRAM figure in this project's docs
  ("reserved flat at ~6034 MB" for the TOML-route health check) -- the
  report was right: real consumption was ~11.4/12.2 GB, and the
  misleading number was a snapshot-ordering artifact, now fixed.**
  Investigation (`core.cli`, `runs/hw_validation/legacy_check.toml`,
  instrumented phase snapshots + tensor census + a cross-process
  `torch.xpu.mem_get_info` monitor): steady state during training is
  **reserved 9942 MB, live tensors ("allocated" and an independent
  `gc`-based census, matching to 0.6 MB) 5624 MB, external driver total
  ~11406 MB** -- which reconciles exactly (allocator free-block pool
  ~4.3 GB + ~900 MB desktop baseline + context overhead), and matches
  both the user's own monitoring (11.4/11.9 GB) and the cross-process
  query. The ~6034 MB in the docs came from `core/train_step.py`'s
  maintenance block running `xpu_empty_cache()` *before* the
  `[vram]` snapshot every 250 micro-steps: `empty_cache()` collapses
  reserved to roughly allocated, so every snapshot the docs quoted was
  a post-free reading ~3.9 GB below steady state. Fix: the block now
  snapshots before the maintenance (labelled `pre-maintenance`) and
  again after (labelled `post-empty_cache`), so the reported reserved
  is the real device-relevant number and the drop is still observable.
  Ruled out along the way (each measured, not assumed): torch's
  allocator stats and the census agree everywhere tested (no
  under-counting by torch); the external reading is genuine device
  memory (host-RAM and pinned-memory injections don't move it);
  the earlier "5.6 GB hidden outside torch" reading was entirely the
  post-`empty_cache` artifact. Also cross-checked that the numbers in
  this file's `pending-testing` entries were *not* affected: those are
  `peak_reserved` values, and an independent driver-level rerun of
  `A_after` (`A_after_mon`) peaked at 10140 MB external ≈ 8592 MB
  reserved + ~950 MB desktop + overhead -- consistent. Side finding,
  tracked in [`deferred.md`](deferred.md): `torch.xpu`'s
  `pin_memory()` does not actually lock pages on this build.

- **[2026-08, confirmed 2026-09-28] VRAM ratchet on non-square
  datasets -- `max_aspect_ratio` cap confirmed bounded on real
  hardware.** Original diagnosis (unchanged, at the time of the move
  this was the newest of the five and the only one whose fix predated
  the run): the hypothesis that this was allocator fragmentation tested
  wrong (`num_alloc_retries` 0, flat on uniform datasets), and real
  per-phase capture found the actual mechanism -- `resize_mode="fit"`
  with no cap on the long side pushed genuinely larger tensors through
  exactly one phase boundary (`forward`), the allocator grabbed a
  bigger reserved block, and kept it permanently (+560 MB in one step).
  Fix: `manager/builder.py`'s `run_lora_ingestion_task`'s
  `max_aspect_ratio` parameter (default `2.0`) splits over-long
  "fit"-resized images into multiple same-caption crops, wired through
  the ingestion UI. **Confirmed:** `datasets/non-square` was
  re-ingested with the cap (its `sources.config` records
  `"max_aspect_ratio": 1.5`), then a 60-step main-route run on it
  (label `B_ratchet`, genuinely variable resolutions -- latents from
  48x64 to 96x64, both orientations, 273 samples) came back bounded:
  per-step reserved first 8104 MB, last 8168 MB, total drift 64 MB,
  **max single-step jump +40 MB** (vs the pre-fix +560 MB permanent
  jump). No ratchet.

- **[2026-09, confirmed 2026-09-28] attention-block checkpointing
  (`nodes/model/attention_checkpointing.py`) -- before/after on real
  hardware, and the "before" is an outright OOM.** The fix makes
  `use_checkpoint=True` actually reach SDXL's
  `BasicTransformerBlock` stacks (ComfyUI's
  `BasicTransformerBlock.__init__` never assigns its `checkpoint`
  argument, and `BasicTransformerBlock.forward()` never calls
  `checkpoint()` anywhere in its body despite taking that parameter --
  so only `ResBlock` was ever checkpointed). **Confirmed:** two
  identical main-route runs on
  the uniform 1024-dataset (40 steps, rank 64, budget 11500), differing
  only in whether `enable_attention_block_checkpointing()` was
  active: with the patch (`A_after`) -- 40/40 steps, per-step peak
  reserved **8592 MB**, 0.943 steps/sec; with it neutered to
  reproduce the pre-fix ResBlock-only behavior (`A_before`) -- hard
  `torch.OutOfMemoryError` on the **first forward pass** (10.76 GiB
  allocated of 11.93 GiB total, zero steps completed). So this wasn't
  a marginal activation reduction: 1024² at batch 1 is simply not
  trainable on this 12 GB card without it, which is the same
  activation-dominance shape as the two OOM reports that motivated
  the fix.

- **[2026-09-20, confirmed 2026-09-28] `BudgetedResourceControlHandle`'s
  `synchronize()` / `strict` / `release()` hardening -- exercised under
  real VRAM pressure, both outcomes as designed.** (Original entry:
  defensive hardening motivated by the open "Device lost" report,
  previously verified only against a scripted fake `DeviceContext` in
  `smoke_test_resource_control_strict.py`.) **Confirmed:** two
  identical main-route runs at `vram_budget_mb=2500` against actual
  ~8114 MB usage on the `1image` dataset: non-strict (`C_pressure`)
  -- 30/30 steps, no hang, no device-lost, the offload path taken
  every step (allocated dropped 7886 → 6325 MB after the first offload
  and held there -- text encoder offloaded, allocator kept its
  reservation, exactly the accounted-for behavior); strict
  (`C_strict`) -- raised precisely as designed: *"7538MB reserved
  still exceeds the 1988MB usable budget (2500MB minus 512MB reserve)
  after offloading every resident registered as offloadable"*. The
  same run also exercised `SDXLTextEncoder.offload()`/`reload()`
  under real pressure every step (via the caching wrapper), covering
  that companion fix from the same session. **Scope caveat kept from
  the original entry:** this exercises the `nodes/` rewrite's offload
  path, not the `core/trainer.py` path the open "Device lost"
  report is about -- that relationship stays unconfirmed (see
  [`open.md`](open.md)).

- **[2026-09-20, confirmed 2026-09-28] `AdaptiveResidencyController`
  perf regression (managed route ~4.7x slower than main) -- fixed on
  the hardware that produced the report.** Original report: 0.36 vs
  ~1.7 steps/sec (managed vs main), against a stated 12500 MB budget
  where offloading was never necessary. **Confirmed:** post-fix, two
  identical runs (uniform 1024-dataset, 40 steps, rank 64, budget
  11500, AdamW, attention checkpointing on) differing only in trainer
  node: managed (`D_managed`) **0.897 steps/sec** vs main
  (`A_after`) **0.943 steps/sec** -- managed at 95% of main-route
  speed (was ~21%), identical per-step peak reserved (8592 MB both).
  The residency controller calibrated (measured peak 8592 MB over 3
  calibration steps, usable 9889 MB) and correctly stayed fully
  resident. The still-open costs listed in
  `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`
  (no text-encoder caching on this route, no pinned host memory) are
  now the measured ~5% gap's plausible homes, not a 4.7x mystery.

- **[2026-09-20, confirmed 2026-09-28] `AdaptiveResidencyController`'s
  ongoing escalation + `residency_safety_margin` -- the OOM on the
  variable-resolution dataset is gone.** Original report: calibration
  sampled only smaller images, "stay resident" locked in before the
  worst case was measured, and a run then OOM'd at 10218 MB reserved
  on `datasets/non-square`. **Confirmed:** post-fix run on that same
  dataset (`E_managed_nonsq`, 60 steps, budget 8000 MB -- i.e. usable
  6739 MB after reserve + 10% margin, deliberately below what the run
  would use): calibration measured 8036 MB > usable, controller
  escalated ("releasing optimizer, text_encoder"), and the run
  finished **60/60 steps with no OOM**, per-step peak max 8036 MB.
  Escalation's "next occurrence, not this one" limit was acceptable
  in practice across 60 genuinely variable-resolution steps. The run
  also put a number on the *disclosed, still-open* limit: releasing
  residents could not bring reserved anywhere near the 6739 MB budget
  (model 4897 MB + activations dominate; reserved settled ~8000 MB),
  confirming that release-based escalation cannot budget-fit an
  activation-dominant run -- the lever for that remains activation
  checkpointing (the attention-block entry above is now wired and
  confirmed; full activation management is still the backlog item).

- **[2026-09] `core.optimizers.ChunkedXPUAdafactor`/`FusedXPUAdafactor`
  silently corrupted their own momentum buffer for float32 parameters
  with `beta1` (momentum) set.** `g = self.exp_avg[i]` aliased the
  momentum buffer (no copy), and the following
  `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))` -- `.to(dtype=p.dtype)`
  returns the *same object*, not a copy, for a float32 parameter (state is
  already float32) -- so the subsequent `.mul_(alpha_t)` permanently shrank
  the momentum buffer by `alpha_t` (~lr) every step. bf16 was never affected
  because there the cast is real. The blast radius differed between the
  two optimizers: `FusedXPUAdafactor`'s buggy line was shared by both its
  tiny-parameter (< 10,000 element) and main-parameter paths, while
  `ChunkedXPUAdafactor`'s sat only in its main-parameter path -- there
  `p.numel() < 10_000` routes tiny parameters to a path that updates via
  `ws[s:e].add_(..., alpha=-alpha_t)`, which doesn't alias, and was never
  affected; `ForeachXPUAdafactor` never had it at all. Fix: forced
  a real copy (`.to(dtype=p.dtype, copy=True)`) in both `core/optimizers.py`
  locations; covered by
  `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py` (which also
  now includes the float32+momentum case the bug had made incomparable),
  confirmed fixed by user.

- **[2026-09] `DeviceResident.footprint_bytes()` didn't check actual
  device placement.** Documented as "current device-memory usage," but
  every implementation just summed `numel() * element_size()` over whatever
  tensors it held -- unchanged by a GPU-to-CPU move, since shape/dtype don't
  change either. After `offload()` it therefore kept reporting the
  pre-offload total, as if nothing had moved, which fed the monitoring
  feature in `nodes/train/step_pipeline.py` (comparing
  `total_footprint_bytes()` against real driver VRAM as "a useful sanity
  signal") a fake "growing gap worth investigating" whenever anything
  offloaded. All four concrete implementations had the identical gap. Fix:
  each now tracks whether it's currently offloaded and returns `0` from
  `footprint_bytes()` while set (`0` device bytes is right for something
  living in host RAM), and `reload()` in the two that already tracked
  `_device_before_offload` now resets it -- without that, `0` would stick
  forever after the first offload. Covered by
  `nodes/smoke_tests/smoke_test_footprint_bytes_after_offload.py`;
  confirmed fixed by user.

- **[2026-09] `AdafactorOptimizerNode`'s legacy-wrapping siblings
  (`ForeachAdafactorOptimizerNode`, `FusedAdafactorOptimizerNode`) had an
  unreplicated small-parameter (< 10,000 element) code path relative to
  their `Composed*` equivalents -- closed on real torch, not just read from
  source.** `ChunkedXPUAdafactor`/`FusedXPUAdafactor` both had a
  tiny-parameter fast path (an elementwise second-moment EMA instead of the
  factored approximation) that `AdafactorAlgorithm` didn't cover, and the
  two didn't even agree on the mechanism -- `FusedXPUAdafactor`'s is
  per-parameter, `ChunkedXPUAdafactor`'s batches all tiny parameters into
  one shared clip and EMA state; `ForeachXPUAdafactor` had no such path at
  all. Closed via
  `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`: Foreach
  matched `ComposedAdafactorOptimizerNode(strategy="foreach")` at
  floating-point-noise magnitude, so both Foreach files were deleted with no
  algorithm change; Fused's gap was real (1e-3 to 1e-2 magnitude, not
  noise) and closed by an opt-in `tiny_parameter_threshold` on
  `AdafactorAlgorithm` -- used only by
  `ComposedFusedAdafactorOptimizerNode`, since extending it to the other
  strategies would have broken the just-confirmed Foreach match -- within
  this project's established tolerances for this pair (`1e-4` float32,
  `1e-2` bf16), and both Fused files were deleted too. `ChunkedXPUAdafactor`'s
  cross-parameter-batching version was a different, bigger problem, and
  `AdafactorOptimizerNode` stayed registered for it as a capability gap
  rather than a broken thing -- until 2026-10-02, when the decision was
  made to retire the node instead of building the machinery
  (`nodes/optimizer/adafactor.py` deleted). The reasoning, and the one
  unmeasured performance consequence, are recorded in
  `docs/known-issues/open.md`.

- **[2026-08] 8 real, working Node classes existed but weren't selectable
  in the graph editor -- the nodegraph registry's list was stale.** Found
  by diffing every concrete `Node` subclass under `nodes/` against
  `get_registry()`: `P2LossWeightingNode`, `PrefetchingBatchSourceNode`,
  and six `Composed*OptimizerNode` classes were all missing. Fix: added
  each to the registry's import and `classes` lists
  (`archive/server/nodegraph_registry.py`; the registry moved to `archive/`
  at M9). Confirmed fixed by user ("I can now see and use new nodes").

- **[2026-07] CAME optimizer VRAM near-ceiling hang after ~60 steps.**
  `res` and `update` in `ChunkedXPUCAME.step()` each allocated a fresh
  full-parameter-sized tensor per step (on top of Adafactor's baseline
  scratch-buffer usage), slowly fragmenting VRAM near the ceiling. Fixed by
  reusing the existing scratch buffer in place for both. Confirmed fixed by
  user.

- **[2026-07] Default `snr_weighting: "snr"` used the v-prediction Min-SNR
  formula (`snr/(snr+1)`) unconditionally, including for the default
  `student_type: "eps"`.** For eps-prediction the correct uncapped form is
  trivially 1.0 (uniform); the old default gave ~99% weight to
  easy/low-noise steps and ~1% to high-noise/structural steps -- close to
  the opposite of what's wanted. Fixed by branching `snr`/`min_snr_5`/
  `decay_snr` on `student_type`. Recommended switching configs to
  `min_snr_5` explicitly (the correctly-implemented standard choice for
  eps) rather than relying on `snr` reducing to a uniform no-op.

- **[2026-07] `grad_accum` inflated "steps" to mean micro-batches, not real
  optimizer updates.** `steps: 5000, grad_accum: 32` only did `5000/32 = 156`
  real weight updates; LR schedule, save/preview cadence, and the dashboard
  all silently used the wrong count, with no warning, and the shipped
  example config already had `grad_accum: 32`. Refactored so `steps` means
  real optimizer updates everywhere (dashboard, saves, previews, LR
  schedule); cache size and micro-batch loop scale internally by
  `grad_accum` instead. Confirmed working by user (correct step count,
  expected per-step timing).

- **[2026-07] "Device lost" errors and silent training hangs after VRAM
  pressure.** Resolved 2026-10-02 by diagnosis from the person who
  reported it: **a driver problem, not an offload bug in this codebase.**
  Two independent things follow from that, and they are recorded
  separately because only the first is a diagnosis.

  *Why the code was not at fault.* The reported shape -- something
  offloaded under pressure and not correctly loaded back, with free VRAM
  available afterwards -- had already been checked here against
  `archive/core/trainer.py`: every offload/reload transition called
  `xpu_synchronize()` explicitly, and the observed VRAM spike was at
  exactly the reload-after-preview transition that the sync there
  addresses. A matching hang-after-offload report on the same B580
  hardware was traced elsewhere to a missing `device` argument on
  `synchronize_device()`'s non-CUDA path -- i.e. to a driver/oneapi
  layer, which is where this now points.

  *Why the route that produced it is gone.* The trigger needs
  training-in-process to preview-sample. Preview sampling only existed on
  the `core/` route; `nodes/` has none, so a graph execution cannot reach
  it. `core/` moved to `archive/core/` on 2026-10-02 and the backend no
  longer spawns it, so the repro is unreachable regardless.

  *Why the route that replaced it is not exposed to it.* The live route
  has a ceiling rather than a hope. `VRAMBudgetControllerNode` takes
  `vram_budget_mb`, the trainer polls the resulting `ResourceBudget` at
  its own step boundaries, and `strict=True` **raises** instead of
  continuing over budget (`nodes/memory/control_handle.py`) -- so a
  ceiling set below the card's real capacity turns "drive the driver into
  a device-lost fault" into "stop and say so". The budget is measured
  against the allocator's `memory_reserved`, with a `vram_reserve_mb`
  margin left deliberately unused.

  Left worth remembering, not worth chasing: nothing here was diagnosed
  by measurement on this machine. The `nodes/` route had already been
  run under sustained pressure with no hang (`hw_validate.py` label
  `C_pressure`), so what closed this is the reporter's read of the
  cause plus the removal of the route -- not a reproduction that stopped
  happening.

- **[2026-10] Adafactor `foreach` as the replacement for the retired
  batched-tiny-parameter path: measured, and the answer was "no, but a
  different strategy already had".** Measured 2026-10-02 on the B580, so
  this no longer belongs in "Open" -- the measurement is recorded here
  because the *conclusion* still has a consequence for a config choice,
  and because the hypothesis it tested turned out to be wrong in an
  instructive way.

  **What was being asked.** `nodes/optimizer/adafactor.py` is deleted, so
  `ComposedAdafactorOptimizerNode` is the only Adafactor node. It was
  retired rather than reimplemented because `ChunkedXPUAdafactor`'s
  tiny-parameter path concatenates every parameter under 10,000 elements
  into one shared clip/EMA state, and
  `smoke_test_adafactor_tiny_parameter_gap.py` Part C measured that this
  *contaminates* results -- a parameter's update depends on unrelated
  parameters' gradients sharing its batch. Correctness was never lost.
  What was lost was performance, and the entry's own hypothesis was that
  `strategy="foreach"` recovered it through `torch._foreach_*`.

  **The measurement.** `scripts/hw_validate.py main`, `1024 aes`,
  100 steps, batch 1, rank 64, seed 1234, same process-per-run harness
  throughout. 1.70 s/step ≈ 1.0 s/step of that is optimizer work, so
  this is not a rounding difference:

  | optimizer | strategy | n | steps/s | sec/step | peak reserved |
  |---|---|---|---|---|---|
  | adafactor | `shape_grouped_foreach` | 2 | **1.007** | 0.993 | 7984 MB |
  | adafactor | `shape_grouped` | 2 | 0.991 | 1.009 | 7984 MB |
  | adamw | (default `simple`) | 1 | 0.919 | 1.088 | 8592 MB |
  | adafactor | `simple` | 3 | 0.590 | 1.694 | 7884 MB |
  | adafactor | `foreach` | 1 | 0.574 | 1.742 | 8234 MB |
  | adafactor | `chunked` | 1 | 0.567 | 1.764 | 7886 MB |
  | came | (default `simple`) | 1 | 0.471 | 2.123 | 8248 MB |

  Losses agree across every Adafactor strategy to 1.9e-5 (first step) and
  3.5e-5 (last) -- the same numbers, as the equivalence smoke tests
  require. Not bit-identical, which fp32 on a GPU is not.

  **The hypothesis was wrong.** `foreach` is *slower* than `simple` here
  (0.97x) and reserves 350 MB more. `torch._foreach_*` is not the win on
  this hardware.

  **The win was the shape grouping, and it already existed.**
  `shape_grouped_foreach` is **1.71x** `simple` and 10% faster than AdamW
  -- while computing the same values. Isolating the two halves:
  `shape_grouped` alone is 0.991, so the grouping does essentially all
  of it and the `_foreach` suffix adds ~1.6% on top. The lesson for the
  next time something looks like it needs a new batching strategy: ask
  what shape the work is, not how many launches it takes.

  **Consequence: the default changed.** `DEFAULT_STRATEGY` in
  `nodes/optimizer/strategy_registry.py` is now
  `shape_grouped_foreach`, and all three Composed*OptimizerNode
  `strategy` Ports default to it. No `TinyBatchedStrategy` was written
  -- the grouping it would have implemented is what `shape_grouped`
  already does.

  The other two nodes were measured before their defaults were touched
  too, against the same `simple` they were replacing: adamw 0.919 ->
  1.027 (1.12x) and came 0.471 -> 0.972 (2.06x). Not the full matrix,
  and deliberately not: three defaults had to be decided, each was
  measured against the one thing it replaced, and a cross-product
  would not have changed a decision.

  The default lives in the registry rather than in each node, for the
  reason `STRATEGY_DOC` does: a value three files repeat is three
  things that can disagree, and a disagreement about a *default* is
  invisible -- nothing fails, two nodes quietly stop behaving like
  their documented twin.

  **Caveats, because they bound the claim.** One dataset (`1024 aes`),
  batch 1, rank 64, one card, fp32 state. The shape_grouped arms are n=2;
  foreach, chunked and the two non-adafactor `simple` arms are n=1. The
  result is about *this* parameter population: LoRA rank 64 on SDXL,
  which is the case the retired code was written for. A different rank or
  target-module set moves the shape distribution and could move the
  ranking.

- **The backend test suite leaked its scratch directories.** Every file
  under `backend/tests/` creates scratch with `tempfile.mkdtemp`, which
  returns a name and hands back no handle, so nothing removed them. One
  full suite run left ~40 directories behind; over many runs that reached
  **4,834 directories and 2.1 GB**. `/tmp` here is a 20 GB tmpfs, so it
  was not disk that filled but RAM -- which makes the failure mode worse
  than slow: a test can fail because it could not create its scratch
  directory, and the failure gets attributed to a test with nothing wrong
  with it. One such failure was seen (`test_graph_execution.py`, once, not
  reproducible).

  Fixed in `backend/tests/run_all.py`: each file gets its own `TMPDIR`,
  which `tempfile` honours, and it is removed afterwards. A full suite run
  now leaves nothing behind -- re-measured, the count of `/tmp/tmp*`
  directories is unchanged across a run. `run_all.py` is the gate's entry
  point, so this covers every gate run; running a single test file
  directly still leaves what that file makes, which is the honest
  boundary.
