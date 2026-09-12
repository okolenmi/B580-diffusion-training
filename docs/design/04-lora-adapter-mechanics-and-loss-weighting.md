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

---
