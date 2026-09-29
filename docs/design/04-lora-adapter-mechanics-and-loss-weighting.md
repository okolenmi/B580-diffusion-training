*[← docs/design index](README.md)*

# LoRA/adapter mechanics, and loss weighting

## 3. LoRA and adapter mechanics

### 3.1 `AdapterStrategy`: how a trainable delta composes with a frozen weight

Plain LoRA -- a low-rank pair of matrices added to a frozen weight -- is
one way to parameterize a trainable delta, not the only one. Today's
`core.lora`-wrapping code hardcodes it as the only option; this section
makes it an explicit choice.

**Implemented**: `AdapterStrategy`/`PlainLoRAAdapter` (both in
`nodes/model/adapter_strategy.py`) -- `PlainLoRAAdapter` wraps
`core.lora.LoRALinear`/`LoRAConv2d`'s math unchanged, per the existing
rule that genuinely-correct legacy math gets wrapped, not re-derived.
**Two real signature gaps had to be closed to make the illustrative
`wrap(frozen, rank, scaling_policy)` actually callable**, worth recording
here since they're genuine interface corrections, not just
implementation detail: an `alpha` parameter (`scaling_policy.scaling(alpha,
rank)` structurally needs one, and the original signature omitted it),
and an `original` parameter alongside `frozen` (`LoRALinear`/`LoRAConv2d`
need the whole original `nn.Linear`/`nn.Conv2d` -- bias, in/out features,
conv stride/padding/dilation/groups -- not just a weight tensor).
`PlainLoRAAdapter` only actually honors `BF16WeightStore` today and
checks that at `wrap()` time rather than silently ignoring `frozen` --
see `nodes/model/adapter_strategy.py`'s own docstring for exactly why
that's correct for `BF16WeightStore` and would not be for a real
`NF4WeightStore` (3.3).

**Implemented**: `DoRAAdapter` -- Liu et al., "DoRA: Weight-Decomposed
Low-Rank Adaptation" (arXiv:2402.09353, ICML 2024 Oral). `nodes/model/dora_layer.py`'s
`DoRALinear`/`DoRAConv2d`, `nodes/model/adapter_strategy.py`'s
`DoRAAdapter(AdapterStrategy)`. Decomposes each frozen weight matrix
into a magnitude component (one learnable scalar per output channel)
and a direction component (the weight-normalized matrix), applying LoRA
only to the direction while training the magnitude directly.

**Grounded directly in HuggingFace PEFT's real implementation**
(`peft/src/peft/tuners/lora/dora.py`, fetched and read directly), not
the paper's own notation -- which is genuinely ambiguous about which
axis "column-wise" norm means relative to `nn.Linear`'s
`[out_features, in_features]` layout. Cross-checked against Meta's
torchtune, which computes the same thing independently: both take
`torch.linalg.norm(weight, dim=1)`, one magnitude scalar per *output*
channel, matching ordinary weight-normalization intuition (Salimans &
Kingma, 2016). The efficient forward formulation (not "merge the full
weight, then run one linear/conv", which would cost real VRAM against
this project's own design goal) is PEFT's, re-derived here by algebraic
expansion rather than copied verbatim, restructured so bias is never
itself magnitude-scaled (bias isn't part of the decomposed weight at
all -- a real, easy-to-get-wrong detail working from the paper's
weight-only equations directly) and `base_result` is reused rather than
recomputed. `||base_weight + scaling*BA||`'s gradient is detached, per
the paper's own section 4.3 (quoted directly in PEFT's source).

**A real, deliberate extension beyond PEFT**: the LoRA timestep gate
(`core.lora.py`'s `set_lora_gate()`/`compute_lora_gate()`) applies to
the entire DoRA delta, not just the raw LoRA term inside it, matching
`LoRALinear`'s own gate semantics exactly -- `gate=0` produces exactly
the frozen base output. PEFT has no equivalent concept (no LLM
fine-tuning analogue to "only some timesteps were in the training
data").

**Built via composition over a real `core.lora.LoRALinear`/`LoRAConv2d`**,
not a second implementation of parameter setup -- only the forward math
is genuinely new. Same real limit `PlainLoRAAdapter` has today: only
`BF16WeightStore` honored (`NF4WeightStore`, 3.3, doesn't exist yet).

**Checkpoint save/load: now a real round-trip for the common case, one
honestly-scoped gap left for phase-splitting.** `DoRALinear.load_lora_weights()`
still loads the directional component and recomputes `magnitude` fresh from
it (useful for starting DoRA training from an existing plain-LoRA
checkpoint's direction, but not a full round-trip); `load_dora_weights()`
is the real round-trip, and `restore_alpha()` keeps alpha/scaling
consistent with a checkpoint-restored value. `nodes/model/lora_saver.py`
(via `nodes/model/lora_phases.py`'s `extract_combined_weights`/
`extract_own_generation_weights`) and `LoRACheckpointLoaderNode` (via its
own `_load_dora_layers()`) both know about a `.dora_scale` key now --
name matches ComfyUI's own `comfy/lora.py` convention (`{key}.dora_scale`,
read alongside `.alpha`), not invented here. `DoRAAdapter` is trainable in
a real run today (live-wired via 3.1's `adapter_strategy_scope`, same as
`PlainLoRAAdapter`); saving and loading that training's real result
(direction + magnitude + alpha) correctly is now wired for an unsplit
DoRA layer, the overwhelmingly common case. See 9.1 for the one real edge
case still open by design (a phase-split DoRA layer's magnitude can't be
folded into a combined checkpoint) and for a second, deeper bug this
landing found and closed along the way: phase-splitting a DoRA layer had
never actually worked at all, independent of the checkpoint question.

The seam this needed (`AdapterStrategy` existing at all, with a real
second conformance checked against it) exists, **and is now live-wired
into `ComfyUNetLoRANode`'s real construction path** --
`nodes/model/adapter_injection.py`'s `adapter_strategy_scope`, a new
`adapter_strategy` port (default `None` -> `PlainLoRAAdapter()`, so
nothing wired to this Node today changes). Neither modifying
`core/lora.py` (against this project's standing rule) nor re-deriving
`_inject_lora`'s tree-walk inside `nodes/` turned out to be necessary:
`_inject_lora` constructs its target layers by calling
`LoRALinear(...)`/`LoRAConv2d(...)` as plain module-level names, which
Python resolves from `core.lora`'s own namespace at call time -- a real,
exploitable seam. `adapter_strategy_scope` temporarily replaces what
those two names point to for the duration of one `ComfyUNetWrapper(...)`
construction call, restored on every exit (exception or not), so
`_inject_lora`'s real, unmodified, already-correct targeting logic keeps
running exactly as before -- only what happens at each target it finds
changes. `PlainLoRAAdapter` selected (the default) installs no patch at
all, since it *is* `core.lora`'s own behavior by definition -- there's
nothing to intercept.

**A real recursion hazard, found by hitting it, not by predicting it in
advance**, and now fixed generally rather than routed around: any
`AdapterStrategy` whose `wrap()` internally constructs real
`LoRALinear`/`LoRAConv2d` (`PlainLoRAAdapter` does; a future
`DoRAAdapter` reusing `PlainLoRAAdapter`'s base construction naturally
would too) would, if it re-imported those classes live from `core.lora`
while a patch is active, resolve to the patch itself and recurse
forever. Fixed with a small cache
(`adapter_strategy.py`'s `_real_lora_classes()`/`_real_lora_classes_cache`)
that `adapter_strategy_scope` populates with the real classes at the one
moment they're still guaranteed real -- immediately before patching --
so `PlainLoRAAdapter.wrap()` gets the real ones regardless of what's
currently patched or what's calling it.

Also fixed while landing this: `ComfyUNetLoRANode.build()` already
resolves `scaling_policy` into a single effective alpha *before*
`core.lora` ever runs (3.2's seam) -- so the `alpha` `_inject_lora` hands
to each target is already final. `adapter_strategy_scope`'s patched
construction always passes `ClassicLoRAScaling()` (a proven identity on
an already-effective alpha) as `wrap()`'s `scaling_policy` argument,
regardless of what the person actually chose -- passing the real one
would apply it a second time. Equivalence-tested directly: a `RankStabilizedScaling`-produced
effective alpha, run through both the real, unpatched path and the
patched path, land on byte-identical `layer.alpha`/`layer.scaling`.

`DoRAAdapter` **is now implemented** too -- see above. Once
`AdapterStrategy` was live-wired, building it was the only remaining
piece, and it's trainable in a real run immediately, no further
live-wiring needed.

### 3.2 `LoRAScalingPolicy`

Standard LoRA scales its output by `alpha/r`. Kalajdzievski, "A Rank
Stabilization Scaling Factor for Fine-Tuning with LoRA" (arXiv:2312.03732,
2023) proves this causes the adapter's output and gradient magnitude to
collapse as rank `r` grows -- which is why LoRA in practice is usually
kept at low rank, since higher ranks "should" add capacity but
empirically don't help, because the scaling itself suppresses them. The
fix is a one-line change: scale by `alpha/sqrt(r)` instead. Proven, not
just observed, and costs nothing extra at inference or training time.

**Implemented**, unchanged from the design: `LoRAScalingPolicy`/
`ClassicLoRAScaling`/`RankStabilizedScaling` (`nodes/model/lora_injector.py`),
wired as an opt-in `scaling_policy` port on `ComfyUNetLoRANode` -- default
`None` resolves to `ClassicLoRAScaling`, reproducing today's `alpha/rank`
exactly, so nothing changes for an existing run. Adopted per the original
calibration verdict (zero VRAM cost, zero inference cost, the closest
thing in this document to a strict improvement with no tradeoff) -- its
actual value still depends on training at higher rank than this project's
current default (`rank: 64`) to have anything to stabilize; that
higher-rank run itself hasn't happened, so the improvement is
implemented and available, not yet observed in practice here.

### 3.3 `FrozenWeightStore`

The frozen base is this project's own single biggest static VRAM
allocation. Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient
Finetuning of Quantized LLMs" (arXiv:2305.14314, NeurIPS 2023) is the
concrete, published, extensively-benchmarked answer: **NF4** (4-bit
NormalFloat), a quantile-based 4-bit type shaped for the near-Gaussian
distribution of pretrained weights, plus **double quantization** of the
per-block scale factors themselves (another ~0.37 bits/parameter saved on
average). The frozen base stays 4-bit in storage; every forward/backward
dequantizes on the fly to bf16 for the actual matmul, so numerical
compute happens at full working precision -- only storage shrinks (4x vs.
bf16, before double quantization's further saving). The paper reports NF4
+ double quantization *fully recovering* 16-bit LoRA's benchmark accuracy
on models up to 65B parameters.

**The honest caveat generic QLoRA writeups mostly don't mention:** QLoRA
was developed and benchmarked on LLM linear layers with roughly-Gaussian
weight distributions -- this project's target is an SDXL UNet, a
genuinely different architecture (convolutions, GroupNorm,
cross-attention), and specifically a *diffusion* model whose weight-usage
pattern varies by timestep rather than being uniform across a single
forward pass. Ryu, Lim, Shim, "Memory-Efficient Fine-Tuning for Quantized
Diffusion Model" (TuneQDM, arXiv:2401.04339, KAIST) studied this exact
question and found that a naive quantized-diffusion-model finetuning
baseline "neglects the distinct patterns in model weights and the
different roles throughout timesteps," trading prompt fidelity against
subject fidelity rather than achieving both -- i.e. generic QLoRA applied
unmodified to a diffusion UNet has a documented, real quality gap versus
its LLM results.

**Implemented**: `FrozenWeightStore`/`BF16WeightStore`
(`nodes/model/frozen_weight_store.py`) -- the frozen base kept exactly as
loaded, no change to any existing forward path. This closed the
`TrainableModel.footprint_bytes()` gap (1.2) it existed for.

`NF4WeightStore` (`nodes/model/nf4_weight_store.py`) is **implemented
too** -- real blockwise NF4 quantization plus double quantization of the
per-block scale factors, grounded directly in bitsandbytes' real,
current source (`bitsandbytes/functional.py`, fetched and read directly,
not recalled or derived from the paper's equations alone) rather than
guessed: the 16-value codebook is reproduced in pure PyTorch
(`torch.special.ndtri`, the inverse standard-normal CDF, in place of
`scipy.stats.norm.ppf`, avoiding a new dependency) and verified against
bitsandbytes' own published codebook constants directly, matching to
float32-rounding precision. Real numbers, checked at a realistic weight
size (1280x1280, matching an actual SDXL cross-attention projection):
3.875x compression vs. bf16, and double quantization's own savings
(0.371 bits/parameter measured) landing almost exactly on the QLoRA
paper's own reported ~0.37 bits/parameter figure -- a real, independent
confirmation, not tuned to match.

**One real, deliberate simplification, not a byte-exact port**: double
quantization's second level (compressing the per-block absmax values
themselves) uses plain linear min-max 8-bit quantization here, not
bitsandbytes' own general-purpose "dynamic" 8-bit map
(`create_dynamic_map()`) -- a separate, more involved piece of machinery
whose own exact reproduction would add real complexity for a small share
of this class's total value (the ~0.37 bits/parameter figure above is
itself already close to bitsandbytes' own reported number, suggesting
the choice of second-level scheme matters less than getting the primary
4-bit NF4 quantization right).

**Not yet wired into a real forward pass.** `materialize()` exists
specifically so an `AdapterStrategy` could call it each forward for a
fresh dequantized tensor, but `PlainLoRAAdapter`/`DoRAAdapter` both still
only honor `BF16WeightStore` and read `core.lora.LoRALinear`/
`LoRAConv2d`'s own `base_weight` buffer directly -- `materialize()` is
never actually called from a real forward path yet. That, plus the
diffusion-specific quality caveat above (which needs an actual training
run on this project's own UNet to check, not assumed from the LLM
literature), are both real, separate follow-up work -- see the backlog,
section 10.

### 3.4 Per-parameter-group learning rates

The real gap, checked against the actual code before this landed: even
though `nodes/optimizer/composed.py`'s `ComposedOptimizerHandle` already
stored `param_lr` as a list (one entry per parameter), `update_lr()`
(called by the LR schedule every step) unconditionally overwrote every
entry with the same value -- anything that set a per-group ratio at
construction would have had it silently erased on the very next step.

**Implemented**: `ParameterGroupPolicy`/`UniformGroups`
(`nodes/optimizer/composed.py`), and the `ComposedOptimizerHandle` fix --
`update_lr()` now recomputes `param_lr` from `[new_lr * r for r in
self._group_ratios]` instead of overwriting uniformly. Behavior-preserving
for every existing caller (`UniformGroups` produces exactly the old
`[lr] * len(params)`).

This unlocked Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of
Large Models" (arXiv:2402.12354, ICML 2024): standard LoRA trains both
adapter matrices (`A`, random-initialized; `B`, zero-initialized) at the
same rate, which an infinite-width scaling argument proves is inefficient
for large-width models. Using a fixed ratio `lr_B = lambda * lr_A` with
`lambda > 1` restores efficient feature learning; the paper reports up to
~2x finetuning speedup and 1-2% task-performance improvement at identical
computational cost. **`LoRAPlusGroups` is also implemented**
(`nodes/optimizer/composed.py`, `ratio=16.0` default -- a commonly-used
starting point in public implementations like Hugging Face PEFT's
`LoraPlusModel`, not independently verified as optimal for SDXL LoRA
here) -- genuinely free once the fix above exists (same parameter count,
same forward/backward cost), but **it isn't wired to anything by
default, and hasn't actually been run: opting in is a one-line
`parameter_group_policy=LoRAPlusGroups(...)` change once a caller wires
it, but whether it actually helps *this* project's SDXL LoRA training,
at what ratio, is untested.** This is validation work, not construction
-- see the backlog, section 10.

---

## 4. Loss weighting

`nodes/train/loss.py`'s `LossWeighting` ABC was already a clean
Strategy-pattern interface needing no change -- confirmed by adding a
second implementation to it and finding zero friction.

**Implemented, both pieces**: `MinSNRLossWeighting`'s v-prediction branch
(`min(SNR, gamma) / (SNR + 1)`, selected via the `Parameterization` it's
given rather than a second, redundant eps/v-pred flag -- cross-checked
against a public reference implementation that had this formula wrong in
an earlier version, `huggingface/diffusers#5654`) and `P2LossWeighting`
(Choi et al., "Perception Prioritized Training of Diffusion Models", CVPR
2022, weighting by `1 / (k + SNR)^gamma`) -- both in `nodes/train/loss.py`.

**Application site fixed 2026-09-29**: the interface was never the
problem -- the call site was. Both trainers' `LossPhase` used to feed it
the batch's *mean* sigma and multiply the resulting scalar into the mean
loss (`w(mean sigma) * mean(loss)`). Since `weight()` is nonlinear in
sigma for Min-SNR and P2, that scalar is not the same number as the
intended `mean(w(sigma_i) * l_i)` whenever a batch's t-samples differ,
which for a batch drawn uniformly over t is essentially always. Both
`ManagedLoRATrainerNode`'s and `step_pipeline.py`'s `LossPhase` now
compute the weight per sample (shared-sigma schedules keep the
scalar path; uniform weighting is bit-identical either way), and stash
the detached raw per-sample MSE in `extras["per_sample_loss"]` -- which
is also what feeds the per-t bucket diagnostics
(`t_bucket_losses()`, design doc 09's fifth addendum). The
`LossWeighting` ABC itself needed no change, as the first line above
predicted.

---

## 5. Per-t-bucket rebalancing (optional, `nodes/train/bucket_balance.py`)

### 5.1 The problem it exists for

The optimizer minimizes one scalar -- the batch mean of per-sample
loss -- and a sum freely trades one t region's progress against
another's: `loss_t_low` descending can pay for `loss_t_high` rising,
and nothing objects. The per-t bucket numbers are diagnostics only
(section 4's `t_bucket_losses()`): a bucket that stalls or regresses
never changes what the gradient does. The only region levers are
static -- `t_mode` picks a sampling distribution once at config time,
Min-SNR/P2 are fixed functions of sigma -- so neither can react to
what is actually happening per region mid-run. The dashboard's
per-series trend lines make the symptom directly visible: three
movement numbers disagreeing (one down, one flat, one up) *is* the
tradeoff, quantified.

### 5.2 What it is: one shared object, two independently optional sides

`BucketBalance` tracks each bucket's window means (`observe()` folds
in `{loss_t_*: mean raw MSE}`) and exposes:

- **Gradient side** -- `weight_for_t(t)`: a per-sample multiplier
  applied in both routes' `LossPhase`, composing with the existing
  sigma weighting as `w(sigma) * w_bucket(t)`. Modes:
  - **`off` (default)**: tracking only. `weight_for_t()` returns
    `None`, and both `LossPhase`s keep their original code path
    expression-for-expression -- wiring a balance in this mode is a
    guaranteed bit-identical no-op (tested), so the tracking can exist
    for the sampler side alone.
  - **`normalize`**: after warmup, `w ∝ 1/baseline`, renormalized to
    mean 1. Equalizes contribution *magnitude* -- a bucket whose raw
    loss lives at 0.2 can't outshout one at 0.02 just by being 10x
    the number. Static after warmup, no controller.
  - **`speed`**: training-rate matching (the GradNorm idea, minus its
    per-layer gradient norms -- here the rate is measured off the
    reported bucket losses directly): each bucket's rate is
    `fast_ema / slow_ema` (< 1 = descending), compared to the mean
    rate across buckets; weights step by `(relative)^eta`,
    renormalized, clamped. The control target is literally "all
    losses should go down at the same speed" -- laggards gain weight,
    fast descendents lose it.
  - **`dro`**: worst-bucket emphasis (Group-DRO flavored):
    `w ∝ exp(dro_lambda * current/baseline)` over buckets,
    renormalized -- the bucket furthest above its own baseline
    dominates, so one broken region pulls the run's capacity instead
    of being averaged away.

  Every mode renormalizes to mean 1 over *eligible* buckets and then
  clamps to `[clip_min, clip_max]` (defaults 0.25 / 4.0): the floor is
  the "no bucket gets ignored" guarantee, the ceiling keeps one noisy
  window from flinging the loss scale. Renormalization happens before
  the clamp, so in saturation a winner may sit inside the ceiling
  rather than on it -- the guarantee is bounded weights, not a
  specific saturation point.

- **Data side** -- `sample_t(rng, t_low, t_high)`: adaptive t
  sampling. Picks a bucket with probability ∝
  `(current/baseline)^sample_bias` over whichever buckets
  `[t_low, t_high]` actually intersects, then draws uniformly inside
  that intersection. `sample_bias=0` keeps sampling uniform whatever
  the gradient side is doing, so either side can be tested alone.
  Pre-warmup it is plain uniform over the coverage.

Wiring: `BucketBalanceNode` (registered in the graph) produces the
instance; it goes to a trainer's `bucket_balance` port (both routes,
in `TrainerNode.COMMON_INPUTS`) and/or to `ManagedDatasetSourceNode`'s
`bucket_balance` port with `t_mode="adaptive"` -- `adaptive` (and the
`exact` mode in 5.5) is carried as `T_MODES_TRAIN_TIME` in
`nodes/dataset/timestep_modes.py`, deliberately *not* appended to
`T_MODES` itself: `core.noise_schedule.sample_timestep` would silently
degrade an unknown mode to uniform, and a silently-uniform mode would
be a lie. One instance, shared: the trainer's `observe()` feeds what
the sampler reads. Either side optional, both optional -- the point is
that all four mechanisms are independently wireable for A/B testing.

`adaptive` without a wired balance is a build-time `ValueError` on the
node *and* on `ManagedDatasetLoader`'s own constructor (both route
through `manager/t_sampling.py`'s `TrainTimeSampler`, which interprets
every t_mode), raised before any filesystem/DB access -- a config
error, not a mid-iteration crash. `manager/` stays duck-typed (it
must never import `nodes/`), documented by contract in `t_sampling.py`.

### 5.3 Cadence, honesty, reports

`observe()` runs once per optimizer step, at the *reported window*:
the managed route folds in the window-accumulated means at the
grad_accum boundary, the main route its per-report means -- batch-2
per-step numbers are too noisy to steer by. Both routes'
`MonitoringPhase` observe even when no monitor is wired and profiling
is off: the balance drives training, so tracking must not depend on
reporting. A window that sampled no bucket X leaves X's state
untouched (a gap is data, never zero-filled); a bucket with fewer
than `warmup_reports` observations has no baseline, gets a neutral
1.0 multiplier, and reports no key -- nothing is fabricated for a
region the run hasn't measured.

Reports gain `weight_t_low/mid/high` (mode `!= off`, only buckets
past their own warmup, values as applied *after* this step's update)
and, only once an adaptive sampler has actually stated its range,
`prob_t_low/mid/high` (the real current sampling distribution over
the covered buckets -- before that there is no distribution to
report). Absent keys, not zeros -- the monitor chart's gap rule.

### 5.4 Knobs (all Port-configurable on `BucketBalanceNode`)

| Port | Default | Meaning |
|---|---|---|
| `mode` | `off` | Which gradient-side mechanism (table above) |
| `warmup_reports` | 10 | Observations a bucket needs before it gets a baseline / any weight |
| `ema_alpha` | 0.05 | Slow EMA of bucket loss (tracking + difficulty) |
| `fast_alpha` | 0.25 | `speed` only: fast EMA; `fast/slow` is the descent rate |
| `eta` | 0.5 | `speed` only: step gain (0.5 = square-root correction) |
| `clip_min` / `clip_max` | 0.25 / 4.0 | Multiplier clamp after mean-1 renormalization |
| `dro_lambda` | 1.0 | `dro` only: worst-bucket sharpness (0 = flat) |
| `sample_bias` | 1.0 | Data side: difficulty exponent (0 = uniform sampling) |

Honest limits, stated as such: `speed` equalizes *speed*, not level --
a hopelessly broken bucket still has high absolute loss, just
descending at the same relative rate as everything else; the only
option that provably prevents one region's update from harming
another is gradient surgery (per-bucket backward passes), which was
deliberately not done here (×2-3 backwards on a 12 GB B580).
Verified by `nodes/smoke_tests/smoke_test_bucket_balance.py` (mode
directions against hand-computed values, mean-1/clip invariants,
bit-identical `off` path, window-cadence observe on both routes,
coverage- and bias-correct sampling, and the config-error checks
above); no GPU involved.

### 5.5 One latent per image: the single-latent consolidation and `exact` t targeting

Everything in section 5 -- adaptive sampling above, exact targeting
below -- rides on one format property: **the dataset is one clean latent
(x0) per image, and noise + t are injected at draw time.** That was
already the live reality when this section was written (verified on
disk: every trajectory of all six real datasets is `format=lora_raw`,
`sample_count=1` -- the standard kohya/diffusers/OneTrainer model, one
VAE encode per image). What changed is that the *old* concept stopped
existing around it:

- **Retired:** `manager/builder.py`'s `run_ingestion_task` (the legacy
  real-image path that baked a fixed ~20-value t grid per image into the
  shard -- the "sampled wasn't sampled" bug), the loader branch that read
  those shards (including its dual-pass target blending;
  `use_dataset_cfg` survives on the node as a documented no-op so old
  graphs still load), the `real` ("Real (VAE Encoding)") option in the
  dataset generator UI plus its route branch, and
  `RenoiseBatchSourceNode` (`nodes/dataset/renoise.py` + its smoke test)
  -- a workaround node whose entire reason for existing was undoing that
  baked grid.
- **Loader:** `ManagedDatasetLoader` now reads `format=lora_raw`
  trajectories only; anything else (teacher/compressed sequences, old
  baked shards) is *skipped* -- count and formats printed, never
  misread as a clean latent.
- **Kept:** `run_teacher_task` (distillation trajectories are genuinely
  sequential multi-t data -- a different format for a different purpose)
  and the shard-reader functions the dataset UI uses.

**New: `t_mode="exact"` + a `t_values` Port.** `t_values="500"` pins
every sample to one precise timestep; `t_values="200,500,800"` cycles
the list in draw order, one value per sample drawn, so each listed t
gets an equal long-run share regardless of shuffling. Every value must
be an integer inside `[t_low, t_high]` (and inside 1..999); anything
else is a build-time `ValueError`. `t_low`/`t_high` narrows the range,
`exact` removes it -- they compose as bounds, not competitors.

`manager/t_sampling.py`'s `TrainTimeSampler` is now the single
interpreter of all three t_mode families (the static five, delegated to
`core.noise_schedule.sample_timestep`; `adaptive`; `exact`) and the
single place they are validated -- in the node's `build()` before path
resolution, in the loader's ctor before any DB access, never silently:
unknown modes used to degrade to uniform through
`alpha_beta.get(mode, ...)`, and a silently-uniform mode is a lie. The
node-side choices constant is `T_MODES_TRAIN_TIME` (still a deliberate
copy -- a Port's `choices` is needed at class-definition time, when
neither `core.*` nor `manager.*` is importable there).
`nodes/smoke_tests/smoke_test_t_sampling.py` checks the copy against
core's list and t_sampling's accepted set, the exact cycle as an exact
sequence, and the adaptive-bias end-to-end that used to run through
`RenoiseBatchSource._renoise()`; `manager/smoke_tests/
smoke_test_lora_raw_dataset.py` covers the pinned-cycle and
skip-non-single-latent behavior against a real temp dataset.
