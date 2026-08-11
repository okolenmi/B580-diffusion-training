# Optimizer execution redesign — plan, 2026-08-09
(Updated 2026-08-11 — see "Status update" section below for what changed.)

Written after a real profiling report (`optimizer_step=967ms` vs.
`forward=250ms`+`backward=308ms`, CAME) plus follow-up from the person
that changes the diagnosis: `ForeachCAMEOptimizerNode` and
`ComposedCAMEOptimizerNode` show "almost the same performance" as each
other. That rules out the host-device-sync explanation (`ForeachXPUCAME`
in `core/optimizers.py`, read directly, has none) as the dominant cost,
and points at something structural shared by every current CAME path.
Separately: the VRAM reserved/allocated gap (9802MB vs 5539MB) was
measured on a **uniform 512x512 dataset** — no shape variation at all —
which rules out the mixed-image-dimension hypothesis as the explanation
for *that specific* gap (still worth testing on its own merits, just not
provenly the cause here).

Person's instruction for this round: most effective fix, not cheapest.
Long refactor chains are fine. This doc is the plan; no code changes yet
— implementation happens in follow-up patches once this is reviewed.

## Status update, 2026-08-11 — real diagnostics ran, plan revised

Phase 0/1 (below) are done — the person ran every experiment, real
hardware, real data. What actually came back changed several conclusions
in this original doc materially. Read this section before the rest —
some of what follows is superseded.

**Speed — confirmed, not just reasoned:**
- Strategy choice genuinely doesn't matter for CAME:
  `SimpleLoopStrategy` vs. `ChunkedScratchBufferStrategy`, `1042ms` vs.
  `1041ms`, identical within noise.
- CAME vs. AdamW, same everything else: `1041ms` vs. `148ms` — a clean
  **7.04x** ratio. Close enough to a from-first-principles prediction
  (CAME does roughly 7x AdamW's elementwise ops per parameter; if wall
  clock is dispatch-overhead-bound rather than compute-bound, which is
  plausible for LoRA-sized tensors, op-count ratio ≈ time ratio) that
  this looks like a real, structural, per-parameter-loop-overhead
  ceiling, not a fixable inefficiency in any one implementation.
- **The old ~1.5x number was a measurement artifact, not a real
  regression.** Traced `core/timer.py`'s `StepTimer` directly: zero
  `synchronize()` calls anywhere in it. On an async backend, unsynced
  `time.perf_counter()` measures CPU dispatch time, not GPU execution
  time — the old number never measured what it claimed to. Confirmed
  directly: ran the legacy `CAMEOptimizerNode` (same `ChunkedXPUCAME`
  class the old trainer used) through the current, properly-synchronized
  pipeline — same slow number as `Composed`. Same implementation, same
  cost, honest measurement this time. **There is no "get back to the old
  speed" — there never was a faster speed, only an unmeasured one.**
- This *increases* confidence in Phase 2 (shape-grouped batching, below)
  rather than reducing it — the problem is real, structural, and
  properly characterized now instead of guessed at.
- **Phase 2 is now implemented** (equivalence-verified, not yet
  real-hardware validated) — see its own section below for exactly what
  shipped and what's still open.

**VRAM — the original fragmentation hypothesis is dead. The real
mechanism is different and now directly measured, not inferred:**
- `alloc_retries=0` across 200 steps, both strategies, both algorithms.
  `delta_from_baseline` never moved past single-digit MB. No
  fragmentation, no retries, no growth on a uniform-shape dataset.
- On a **non-square** dataset: a real, reproducible, discrete jump —
  `forward` time 1452ms→2177ms, `reserved` 2438MB→2998MB delta, at the
  identical step boundary, `allocated`/`active` flat across both steps.
  New per-phase VRAM instrumentation (built this round, see Phase 0)
  caught a second such transition live: the entire +560MB movement
  happened at exactly one phase boundary — `forward` — nowhere else in
  the step moved at all.
- Mechanism, now measured rather than guessed: `resize_mode="fit"`
  preserves aspect ratio with **no cap on the long side** — only the
  short side is pinned to the target resolution. A sufficiently
  tall/wide image forces genuinely larger tensors through `forward`;
  the allocator grabs a bigger reserved block to serve it and — since
  nothing in this pipeline calls `empty_cache()` on its own — keeps that
  reservation permanently, even after the oversized tensors are freed.
  Each new, larger shape encountered for the first time ratchets the
  reserved watermark up one notch; it never comes back down on its own.
- Separately, and independently confirmed: verified directly against
  this project's own real, instantiated SDXL UNet config
  (`core/unet_wrapper.py`'s `transformer_depth: [0, 0, 2, 2, 10, 10]`,
  `transformer_depth_middle: 10`) that self-attention — which scales
  quadratically with spatial token count — lives at every resolution
  stage past the first. Aspect-ratio stretch preserves the *area* ratio
  at every downsampling stage, so a 2.34x-larger-area image (the actual
  512x1200-vs-512x512 case) predicts ~(2.34)² ≈ 5.5x more attention
  cost — closely matching the measured ~5-8x forward/backward slowdown.
  Not the same mechanism as the VRAM ratchet (that's about the
  allocator's own behavior, this is about why the tensors are
  genuinely bigger in the first place), but the same root cause
  (`resize_mode="fit"`'s uncapped long side) drives both.
- Ruled out as an explanation, directly: VAE encoding is not on this
  path at all for real per-step cost — confirmed it happens once,
  offline, at dataset-generation time (`manager/builder.py`'s
  `run_lora_ingestion_task`), not per training step. Tiling it wouldn't
  touch this problem.
- **Fix implemented this round** (see Phase 4 below): cap the long side
  at ingestion time, splitting an over-cap image into multiple
  same-caption crops instead of letting one oversized sample through.

**Speed AND quality — a separate, real gap found while tracing why LoRA
results were poor, not a speed/VRAM item, but tracked here since it came
out of the same investigation and matters more than either:**
- `core/lora.py`'s `set_lora_gate()`/`compute_lora_gate()` — a real,
  designed mechanism to keep the LoRA's contribution close to the frozen
  base at timesteps outside the dataset's own trained `t_low`/`t_high`
  range — is used throughout the legacy `core/` pipeline and has **zero
  references anywhere under `nodes/`**. If a nodes/-trained LoRA used
  anything narrower than the full `t_low=1`/`t_high=999` range, the
  learned delta applies at full, ungated strength at every timestep
  during actual generation, including ones it was never supervised on.
  High timesteps control coarse structure/composition — this is a
  plausible, concrete, previously-undiscovered explanation for "ruins
  backgrounds/composition regardless of LR." **Not yet fixed** — needs
  design work (how the dataset's own `t_low`/`t_high` reaches the step
  pipeline) before it can be. Tracked here, not started this round.
- No caption/conditioning dropout exists anywhere in this codebase (both
  pipelines, confirmed via exhaustive search) — the standard technique
  (kohya-ss's `--caption_dropout_rate`, diffusers' `--caption_dropout_prob`,
  same idea in OneTrainer/EveryDream) every major SDXL LoRA trainer uses
  to keep CFG's cond/uncond gap calibrated during fine-tuning. Without
  it, a LoRA only ever learns the conditional branch, and CFG's
  extrapolation (`uncond + scale*(cond-uncond)`) stops being meaningful
  — a real, standard explanation for "my LoRA changes how CFG behaves."
  **Explicitly deferred** — person wants this tracked, not built yet.

Both belong in `manager/loader.py`'s per-batch materialization (or,
for the gate, wherever the dataset's `t_low`/`t_high` can reach the step
pipeline) when picked up.

## Confirmed by reading the code directly (not assumed)

1. **No CAME implementation in this codebase does real multi-tensor
   batching of its actual math**, for any of the four variants
   (`ChunkedXPUCAME`, `ForeachXPUCAME` — both `core/optimizers.py` — and
   `ComposedCAMEOptimizerNode`'s `simple`/`chunked`/`foreach` strategies).
   `CAMEAlgorithm.compute_update()` (`nodes/optimizer/algorithms/came.py`)
   runs strictly one parameter at a time, by its own docstring.
   `ForeachApplyStrategy` (`nodes/optimizer/strategies/foreach.py`) only
   batches the *final* apply (`param -= delta`, one or two
   `torch._foreach_*` calls per (device, dtype) group) — everything
   before that (row/col normalize, clip, momentum, confidence residual,
   ~15-20 elementwise ops) still runs per-parameter, in a plain Python
   loop, in every strategy including "foreach".

2. **`ForeachXPUAdafactor`'s factored path (`_step_factored`,
   `core/optimizers.py`) is *also* a plain per-parameter loop** — zero
   `torch._foreach_*` calls. Only its unfactored (1D) path gets real
   batching. Every LoRA adapter weight is 2D, so **nothing in this
   codebase currently gets true multi-tensor batching for the shape LoRA
   training actually uses.** This was already known, precisely, in
   `nodes/optimizer/strategies/foreach.py`'s own docstring: *"the legacy
   ForeachXPUAdafactor's own 'factored' path already falls back to a
   plain per-parameter loop for exactly this reason."*

3. **This gap was already named as a real limitation, not discovered
   fresh here.** `algorithms/base.py`'s module docstring states it
   directly: `core/optimizers.py`'s four hand-written classes are "2
   algorithms x up to 3 memory strategies, hand-crossed, with CAME only
   getting 1 of the 3 possible strategies because writing each
   combination by hand is expensive" — and that the
   Algorithm/ExecutionStrategy split exists specifically to stop needing
   to hand-write that grid. The batched strategy this plan proposes is
   the follow-up work that same docstring already flagged as not yet
   done ("the memory-optimized strategies are real, valuable, separate
   follow-up work once the split itself is validated").

4. **`ForeachXPUCAME`'s per-step memory churn is already fairly
   controlled** — state buffers (`vr`/`vc`/`vs`/`exp_avg`/`res_r`/`res_c`)
   are allocated once and reused in-place across steps; the working
   tensor `g` is mutated in-place for normalize/clip/momentum/residual.
   Real per-step allocations are small: one `.float()` cast (only if the
   gradient isn't already fp32) and one `g2` temp, per parameter, per
   step. By contrast, `ComposedCAMEOptimizerNode`'s default `strategy`
   is `"simple"` (`SimpleLoopStrategy`), which never passes `scratch`,
   so `CAMEAlgorithm._compute_update_safe()` runs — genuinely
   out-of-place, allocating **four-plus fresh full-parameter-sized
   tensors per parameter, every single step, forever.** If VRAM was
   profiled while wired to `ComposedCAMEOptimizerNode` without
   explicitly setting `strategy="chunked"`, this is a real, distinct
   contributor, independent of dataset shape.

## Open unknowns that gate parts of this plan

- Which exact optimizer class produced the `967ms`/VRAM numbers —
  `ForeachCAMEOptimizerNode`, or `ComposedCAMEOptimizerNode` at which
  `strategy`? The print-on-construction lines
  (`core/optimizers.py`) or an explicit log of `strategy=...` answer
  this directly; not yet confirmed which.
- Whether the VRAM reserved/allocated gap is *growing* over a run or
  stable-but-high. Every number seen so far is a single snapshot.
- Whether periodic preview generation runs during the profiled window
  (a plausible independent VRAM contributor — different resolution/batch
  shape than training, unrelated to dataset uniformity).
- Whether `PYTORCH_ALLOC_CONF=expandable_segments:True` (or
  `max_split_size_mb`) does anything measurable on this XPU build.

None of these block starting Phase 0/1 (below) — they're exactly what
Phase 0/1 is for.

## Phase 0 — instrumentation (DONE)

Shipped: richer `DeviceContext.memory_stats()` (peak/active/requested/
segments/alloc_retries, not just allocated/reserved), `describe_optimizer()`
closing the "which CAME variant is this" ambiguity, a shape histogram
printed at training start, per-step baseline deltas, and — added
mid-investigation once a single end-of-step snapshot proved insufficient
to answer "which phase caused this" — per-phase VRAM capture
(`profile_memory_per_phase`). This last one is what actually caught the
`forward`-phase VRAM jump directly, not by inference.

## Phase 1 — diagnostics (DONE)

Every experiment in the original version of this section was run for
real, on real hardware. Results are folded into the "Status update"
section at the top of this doc rather than repeated here.

## Phase 2 — the core redesign: shape-grouped batched execution (IMPLEMENTED, equivalence-verified, NOT yet real-hardware validated)

The actual "most effective, not cheapest" fix for speed, now built:
`Algorithm.compute_update_batched()` (new, default falls back to a
per-member loop for any algorithm that hasn't overridden it — see
`algorithms/base.py`), `CAMEAlgorithm.compute_update_batched()` (the real
vectorized override), and `ShapeGroupedBatchStrategy`
(`nodes/optimizer/strategies/shape_grouped.py`), wired into
`ComposedCAMEOptimizerNode` as `strategy="shape_grouped"` — opt-in, not
the default.

**Design, as built:**

- Parameters grouped by exact `(shape, dtype, device, lr)` — **lr
  included deliberately**, not just shape/dtype/device:
  `ComposedOptimizerHandle` already supports per-parameter lr via
  `ParameterGroupPolicy` (`composed.py`'s `LoRAPlusGroups`, real,
  implemented, not wired to anything yet) — two same-shaped parameters
  at genuinely different effective lr must never share one batched
  scalar lr, or one of them silently gets the wrong step size. Grouping
  computed once, lazily, on the strategy's first `step()` call.
- Each step: stack the group's gradients into one `(k, rows, cols)`
  tensor (one allocation per group, not per parameter — the real,
  accepted cost this design always named). Run CAME's entire update —
  row/col means, clip, momentum, confidence residual, second
  normalize — as batched ops across the leading axis. State
  (`r`/`c`/`ea`/`rr`/`rc`) is stacked fresh from each member's existing
  per-parameter dict, computed on, then scattered back via `.copy_()` —
  `ComposedOptimizerHandle`'s state lifecycle (offload/reload/decay/
  reset, all written once generically over a flat list of per-parameter
  dicts) needed zero changes.
- One real algebra change from a literal transcription, done
  deliberately: `clip_div` uses `torch.clamp(rms/threshold, min=1.0)`
  (stays a device tensor, per-group-member) instead of
  `max(float(rms/threshold), 1.0)` (forces a host sync, and can't be a
  Python `if`-branch once it's a per-member tensor anyway). Numerically
  identical either way — confirmed, not assumed (see below).
- Exact-shape grouping only, no padding of near-matching shapes —
  deliberately simpler and lower-risk than padding-based batching, and
  real UNet target modules repeat exact widths often enough (every
  q/k/v/out projection at a given attention width, across many blocks)
  that this should already capture most of the available win.

**Verification actually done, precisely, and what's still open:**

1. Numpy transcription of both the batched math and the exact
   per-parameter reference (`_compute_update_safe`'s factored branch),
   float32, group sizes 1/2/5/20, multiple shapes including a transposed
   one, 20-40 steps per trial (state evolution checked across steps, not
   just one output). Max relative difference ~7e-7 — float32
   reduction-order noise, same order of magnitude this codebase's own
   existing verification already tolerates (this package's own earlier
   ~4e-6 CAMEAlgorithm-vs-ChunkedXPUCAME check).
2. The grouping key itself (`_build_groups`) exercised directly with
   synthetic fake parameters (no torch needed for this part — it only
   reads `.shape`/`.dtype`/`.device`): confirmed same-shape-different-lr
   params correctly land in separate groups, same for different dtype
   and different device.
3. A real smoke test shipped
   (`nodes/smoke_tests/smoke_test_shape_grouped_equivalence.py`,
   `torch.allclose()` not `torch.equal()` — this restructuring isn't
   claimed bit-exact, unlike `ForeachApplyStrategy`'s) covering basic
   correctness, the size-1 fallback path, and — its own dedicated case —
   the lr-aware grouping key using `ComposedOptimizerHandle`'s real
   `LoRAPlusGroups` policy, not a mock: two identical-shape,
   identical-init, identical-gradient parameters at 1x/16x lr must
   diverge, proving they weren't silently merged into one group.
4. **Not done, and this is the real gap before this is trustworthy**:
   this smoke test has not been *run* (no torch in the sandbox this
   session used) — needs the person to run it on real hardware. Real
   speed measurement (the actual payoff this exists for) also hasn't
   happened yet — needs a `profile=True` run with
   `strategy="shape_grouped"` compared directly against the `1041ms`/
   `1042ms` baseline. Do both before trusting or promoting this.

## Phase 3 — rollout for CAME

1. Add `strategy="shape_grouped"` as a new option on
   `ComposedCAMEOptimizerNode` — done, default stays `"simple"` until
   validated.
2. Bit-exact... — see "Verification actually done" above for the real
   status: equivalence-verified via numpy plus a shipped-but-not-yet-run
   smoke test, not yet run against real torch tensors.
3. Real-hardware run: same profiling setup as the original report
   (`profile=True`, same rank/resolution/batch), compare
   `optimizer_step` and reserved/allocated directly against the
   `1041ms`/`1042ms` baseline (the honest, properly-synchronized
   baseline, not the original `967ms` guess). **Still needed** — this is
   the number that actually answers whether the plan worked.
4. If validated: flip `ComposedCAMEOptimizerNode`'s default to
   `"shape_grouped"`, update `docs/suspicious_findings.md`'s optimizer
   entries accordingly.

## Phase 4 — VRAM fixes (IMPLEMENTED: ingestion-time cap; empty_cache lever still open)

Original version of this section assumed fragmentation and offered
three fragmentation-conditioned candidates. None of those apply — see
"Status update" at the top of this doc. The real mechanism (an uncapped
`resize_mode="fit"` long side, confirmed via direct per-phase
measurement) has a direct fix, now shipped:

**Implemented**: `manager/builder.py`'s `run_lora_ingestion_task` gets a
new `max_aspect_ratio` parameter (default `2.0`), wired through
`server/routes_datasets.py` and a new form field in
`server/static/dataset_manager.html`/`.js`. Only affects
`resize_mode="fit"` — every other mode already produces a fixed square
by construction, confirmed no behavior change for them. When a "fit"
image's long side would exceed `max_aspect_ratio * target_px`, it's
split into multiple crops along the long axis instead of letting one
oversized sample through — each crop becomes its own independent
dataset sample (own `x0`, own shard entry), full resolution preserved
per crop (no downsampling quality loss, unlike capping via a smaller
target resolution). Deliberately no overlap between crops and the
source image's caption reused unchanged for every crop — both explicit,
discussed tradeoffs (overlap trades redundant compute/storage for
smoother coverage; real per-crop captioning is separate future work if
it turns out to matter), not oversights. Crop-box arithmetic verified
directly against the actual committed function (extracted and executed
its real source, not a hand-copied approximation) — in-bounds, /8-aligned,
within-cap, full-coverage, across the real reported aspect ratio
(500x1180) and both tall/wide orientations.

**Not done, still open**:
- `max_aspect_ratio=2.0`'s default is a starting point, not tuned
  against a real dataset/hardware combination yet — the person should
  adjust based on what their actual VRAM headroom allows once they can
  measure the new ceiling on a real non-square run.
- `empty_cache_every_n_steps` (already existed, built earlier this
  investigation) is still a manual, opt-in lever, not wired to fire
  specifically on a shape transition (an event-driven trigger tied to
  the loader's own clump/shape-boundary changes would be more precise
  than a blind N-step cadence, and was proposed but not built this
  round — real, scoped follow-up work if the ingestion-time cap alone
  doesn't leave enough headroom).
- `PYTORCH_ALLOC_CONF` — tried by the person this investigation,
  `alloc_retries` was already 0 without it, so it had nothing to fix;
  not revisited given the real mechanism turned out to be unrelated to
  allocator fragmentation.

## Phase 5 — extend batched execution to Adafactor and AdamW

Once `ShapeGroupedBatchStrategy` is validated on CAME (the heaviest
math, so the hardest case), give `AdafactorAlgorithm` and
`AdamWAlgorithm` the same batched interface. Smaller expected win per
algorithm (their per-parameter math is already shorter than CAME's
double row/col normalization), but removes the *same* structural
ceiling from the two other real optimizer paths, not just CAME's.

## Phase 6 — retire redundant legacy paths

Once the new strategy is real-hardware-validated and promoted to
default for all three algorithms: retire `ChunkedXPUCAME`,
`ForeachXPUCAME`, `ChunkedXPUAdafactor`, `ForeachXPUAdafactor`
(`core/optimizers.py`), and `SimpleLoopStrategy`/`ChunkedScratchBufferStrategy`/
`ForeachApplyStrategy` as anything other than an explicit, documented
fallback. This directly follows through on what `composed_came.py`'s
own docstring already said was the plan ("once ... real-hardware
validated ... the legacy path can eventually be retired -- a
deliberate, separate later step, not this one" — this is that step).
Real payoff: right now "CAME optimizer" doesn't map to one obvious node
in this codebase — it's four separate implementations spread across two
files with different validation status each. Collapsing that to one
validated path per algorithm is a real reduction in the kind of
confusion this exact conversation started from ("It was Foreach or
Composed" — two different answers to what should be one question).

## Phase 7 — docs cleanup

Update `docs/nodes_package_design.md`'s Algorithm/ExecutionStrategy
section and `docs/suspicious_findings.md`'s optimizer-speed and
VRAM-near-ceiling entries to reflect whatever Phase 3/4 actually
measured — real numbers, not the hypotheses in this doc, once they
exist.

## Phase 8 — training quality (found this investigation; LoRA gate now implemented, caption dropout still tracked/deferred)

Two real, confirmed gaps, neither a speed/VRAM issue, both found while
tracing why produced LoRAs were low quality ("ruins backgrounds/overall
quality more than brings useful changes, at any LR"):

1. **Missing LoRA gate wiring — IMPLEMENTED.** `core/lora.py`'s
   `set_lora_gate()`/`compute_lora_gate()` keeps the LoRA delta close to
   the frozen base at timesteps outside the dataset's own trained
   `t_low`/`t_high` range. Used throughout the legacy `core/` pipeline;
   confirmed absent from `nodes/` via exhaustive grep before this round.
   Wired now: `PrepareDiffusionInputsPhase`
   (`nodes/train/step_pipeline.py`) calls `compute_lora_gate`/
   `set_lora_gate` at the same point the legacy pipeline does — right
   after the batch's `t` tensor is available, before the forward pass —
   using the same functions, not a reimplementation. New
   `gate_enabled`/`gate_train_low`/`gate_train_high`/`gate_width` ports
   on `SupervisedLoRATrainerNode`, `gate_enabled` defaulting to `False`
   to match `core/config_model.py`'s own documented legacy default
   exactly ("LoRA applies uniformly across all timesteps") — a real
   behavior change only for someone who explicitly turns it on.

   **Known limitation, not fixed this round**: `gate_train_low`/
   `gate_train_high` must be set manually to match whatever `t_low`/
   `t_high` the dataset source node (`ManagedDatasetSourceNode`) was
   actually configured with — no automatic sync yet. A real fix would
   have the dataset's batch carry its own `t_low`/`t_high` through to
   the step pipeline directly (touching `manager/loader.py`'s
   collation), avoiding the two-places-to-keep-in-sync risk this shares
   with the legacy pipeline's own design — deliberately not attempted
   this round to keep the change scoped and low-risk (this is
   correctness-critical code; a first cut matching a proven pattern
   beats a more ambitious redesign with less scrutiny behind it).

   Verified: loaded the real `step_pipeline.py` file (package-aware
   import so its actual relative imports resolve, not a hand-copied
   approximation) with `torch`/`core.lora` mocked to record calls —
   confirmed `gate_enabled=False` calls `set_lora_gate(None)` and never
   calls `compute_lora_gate`; `gate_enabled=True` calls
   `compute_lora_gate(t, train_low, train_high, width)` then
   `set_lora_gate()` with its exact result; and that disabling the gate
   again after it was enabled still resets to `None` unconditionally
   (matters because `_current_gate` is a module-level global, not
   scoped to one build — inherited from `core/lora.py`'s own existing
   design, not introduced here). Also statically confirmed the port
   declarations, the `PrepareDiffusionInputsPhase(...)` call site's
   keyword arguments, and the constructor's real parameter names all
   agree exactly, and that `core/lora.py`'s real `compute_lora_gate`
   signature matches the positional call. **Not done**: a real run —
   this needs the person to actually set a restricted `t_low`/`t_high`
   on their dataset, matching `gate_train_low`/`gate_train_high`, and
   compare generation quality with `gate_enabled` True vs. False.
2. **No caption/conditioning dropout.** Confirmed absent from this
   entire codebase, both pipelines, via exhaustive search. Standard
   technique (kohya-ss `--caption_dropout_rate`, diffusers
   `--caption_dropout_prob`, same idea in OneTrainer/EveryDream) every
   major SDXL LoRA trainer uses to keep CFG's cond/uncond gap calibrated
   during fine-tuning — without it, a LoRA only ever learns the
   conditional branch, and CFG's `uncond + scale*(cond-uncond)`
   extrapolation stops being meaningful, a real explanation for
   "my LoRA changes how CFG behaves." Belongs in `manager/loader.py`'s
   per-batch materialization (same place noise/timestep already get
   resampled fresh) — with probability `p` (typically 5-20% for LoRA),
   swap that sample's prompt to `""` for that step. Explicitly deferred
   by the person — tracked here, not built.

Neither is blocked on Phase 2-6 or vice versa — independent work.

## Sequencing summary

```
Phase 0 (instrumentation, DONE) ─┬─> Phase 1 (diagnostics, DONE) ─┬─> Phase 4 (VRAM cap, IMPLEMENTED)
                                   │                                │
                                   └─> Phase 2 (batched design,      └─> (empty_cache-on-shape-transition,
                                        CAME, IMPLEMENTED,                still open)
                                        equivalence-verified)
                                                │
                                                v
                                   Phase 3 (rollout, CAME --
                                   smoke test shipped, NOT yet
                                   run; real-hardware speed
                                   measurement NOT yet done)
                                                │
                                                v
                                   Phase 5 (extend: Adafactor, AdamW)
                                                │
                                                v
                                   Phase 6 (retire legacy paths)
                                                │
                                                v
                                       Phase 7 (docs)

Phase 8 (training quality: LoRA gate wiring IMPLEMENTED, caption
dropout still deferred) -- independent of everything above.
```

The concrete next actions, in order: (1) run
`smoke_test_shape_grouped_equivalence.py` on real hardware, then a real
`profile=True` run with `strategy="shape_grouped"` against the `1041ms`
baseline — everything from Phase 3 onward is gated on those two numbers.
(2) Separately, independently: set a real restricted `t_low`/`t_high` on
a dataset, matching `gate_train_low`/`gate_train_high`, and compare a
LoRA trained with `gate_enabled=True` against one with it `False` —
this is the check that tells us whether the missing-gate hypothesis was
actually the (or a) real cause of the reported quality problem, not
just a real, confirmed gap that may or may not have been relevant to
this specific person's actual settings.
