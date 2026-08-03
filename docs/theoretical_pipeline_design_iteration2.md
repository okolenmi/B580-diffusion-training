# Iteration 2: reviewing the complete design, modern methods for the training-science pieces

Continuation of `docs/theoretical_pipeline_design.md`. Direct feedback on
that document, and the framing for this one: the three "iterations" in it
were actually one design built up in three additive layers (foundation,
then orchestration, then registry/observability) -- not three independent
passes over a complete design. Per direction, that whole document is now
being treated as **iteration 1**: one complete, working design. This
document is the real second pass over it -- re-reading it as a finished
whole, not a partial foundation, and specifically hunting for places where
a better, *published* training-science technique exists for something the
design currently does the ordinary way. Software-architecture polish that
doesn't block a new capability is explicitly left for iteration 3 (listed,
not resolved, at the end) -- that's what "iteration 3 should be just
fixes" means, taken literally.

**A sequencing correction, stated plainly rather than quietly fixed:**
`docs/theoretical_pipeline_design.md`'s section 4 (the gap analysis against
`nodes/`) was written immediately after what I'd internally called
"iteration 3" -- i.e., after the whole thing, under the *old* (wrong)
understanding of what "iteration 3" meant. Per the original instruction,
that comparison belongs after all three real iterations are done, not
after one. Nothing in that section is factually wrong -- it accurately
describes iteration 1 as it stood -- but it's provisional: it needs a
second pass once the real iteration 3 lands, not before. Not redoing it
here on purpose; doing it twice (now, provisionally, and then again
properly) would be wasted work in exactly the way "don't overcomplicate"
warns against.

## Method for this pass

Two kinds of change below:

1. **Revisions (Part A)** -- things iteration 1 got structurally
   not-quite-right, where leaving them unfixed would block a real
   iteration-2 addition. Anything that's pure polish, without blocking a
   new capability, is left for iteration 3 instead (section list at the
   end) -- fixed here only when a Part B addition genuinely can't be
   designed cleanly without it.
2. **Additions (Part B)** -- new design elements, each motivated by an
   actual published technique, not a hunch. Verified this session (arXiv
   IDs given, not cited from memory) rather than trusted from training
   data alone, given how much these interfaces would end up shaping if
   adopted uncritically. Each item states what it is, why it matters for
   *this* project's actual constraints (single consumer GPU, SDXL UNet,
   LoRA-only training today, VRAM-first), and its honest calibration:
   adopt now, adopt with real caveats, or seam-only (room left, not
   built). Getting the calibration right matters more than the length of
   the list -- a design document that recommends every paper it read
   isn't more useful than one that reads none of them.

---

## Part A: revisions to iteration 1 (structural, blocking)

### A.1 `NoiseSchedule` narrowed to discrete-time on purpose; `Parameterization` gets a third member

Iteration 1's `NoiseSchedule.alpha_sigma(t)` silently assumed `t` is
always a discrete integer index into a precomputed table -- correct for
what it was modeling (DDPM-style variance-preserving diffusion, which is
what SDXL actually is), but it would have to be broken to add a
continuous-time process later (B.8), and a broken-then-patched interface
is worse than one scoped honestly from the start.

Revision: `NoiseSchedule` stays exactly as iteration 1 defined it, but is
now documented as deliberately scoped to discrete-time,
variance-preserving diffusion -- not a general "any noising process" base
class. A continuous-time process (flow matching) would implement a
separate, smaller `Interpolant` contract instead of `NoiseSchedule`,
as a sibling, not a subtype forced to fake having a discrete alpha/sigma
table. `DiscreteLinearNoiseSchedule` (iteration 1) is unchanged.

`Parameterization` (`EpsParameterization`, `VPredParameterization`) gains
no interface change either, but a note the iteration-1 doc didn't have:
a v-prediction-only `VelocityParameterization` variant is what a
continuous-time process would need (B.8) -- the existing `convert_to()`
signature already generalizes to a third implementer without any change,
which is worth confirming explicitly rather than assuming, the same way
iteration 1 confirmed `Node`/`Port` matched `Builder`/`Port`
independently (see the original doc's section 4.1).

### A.2 `ResourcePolicy` gains real decision axes

Iteration 1's `ResourcePolicy` had three methods:
`checkpointing_strategy()`, `optimizer_execution_strategy()`,
`enable_text_encoder_cache()`. Every Part B addition below that's a real
per-run choice (which `AdapterStrategy`, which `LoRAScalingPolicy`, which
`FrozenWeightStore`, which `CheckpointPlacementPolicy`) needs a home on
this same object, or the whole point of having one `ResourcePolicy`
type -- "every VRAM/quality-affecting choice implements the same
shape" -- stops being true the moment a second such choice exists outside
it.

```python
class ResourcePolicy(ABC):
    @abstractmethod
    def checkpointing_strategy(self) -> "ActivationCheckpointingStrategy":
        ...
    @abstractmethod
    def optimizer_execution_strategy(self) -> type:
        ...
    @abstractmethod
    def enable_text_encoder_cache(self) -> bool:
        ...
    # New this iteration:
    @abstractmethod
    def adapter_strategy(self) -> "AdapterStrategy":
        ...
    @abstractmethod
    def lora_scaling_policy(self) -> "LoRAScalingPolicy":
        ...
    @abstractmethod
    def frozen_weight_store(self) -> type["FrozenWeightStore"]:
        ...
    @abstractmethod
    def parameter_group_policy(self) -> "ParameterGroupPolicy":
        ...
```

`ManualResourcePolicy` (iteration 1's only implementation) gets one more
constructor argument per new method, each defaulting to today's actual
behavior (`PlainLoRAAdapter()`, `ClassicLoRAScaling()`,
`BF16WeightStore`, `UniformGroups()`) -- so nothing wired to it changes
unless a run explicitly opts into a Part B technique.

### A.3 Per-parameter-group learning rates -- a real, precise gap, not a guess

Checked against the actual current code (`nodes/optimizer/composed.py`),
not assumed: `ComposedOptimizerHandle` already stores `param_lr` as a
*list*, one entry per parameter -- `self.param_lr = [lr] * len(self.params)`
at construction. Per-parameter rates are almost already representable.
The real gap is two places that currently force every entry to be
identical: there's no builder-level input to set them differently in the
first place, and, more importantly:

```python
def update_lr(self, new_lr: float) -> None:
    self._lr = new_lr
    self.param_lr = [new_lr] * len(self.params)
```

-- called by the LR schedule *every step* -- unconditionally overwrites
every entry with the same value. Anything that set a per-group ratio at
construction would have it silently erased on the very next step. This
is exactly the kind of thing that would ship as a subtle, hard-to-notice
bug if LoRA+ (B.3) were bolted on without noticing this first -- worth
fixing as infrastructure before the technique that needs it, not as part
of it.

```python
class ParameterGroupPolicy(ABC):
    """One multiplier per parameter (aligned with `params`' order),
    applied to whatever base rate the LRSchedule produces this step."""
    @abstractmethod
    def group_ratios(self, params) -> list[float]:
        ...


class UniformGroups(ParameterGroupPolicy):
    def group_ratios(self, params) -> list[float]:
        return [1.0] * len(params)
```

Revised handle (base lr and per-group ratio kept separate, so
`update_lr()` can no longer clobber the ratio):

```python
class ComposedOptimizerHandle(OptimizerHandle):
    def __init__(self, algorithm, strategy, params, lr, device,
                 group_policy: ParameterGroupPolicy | None = None):
        ...
        self._group_ratios = (group_policy or UniformGroups()).group_ratios(self.params)
        self.update_lr(lr)  # now the single place param_lr gets computed

    def update_lr(self, new_lr: float) -> None:
        self._lr = new_lr
        self.param_lr = [new_lr * r for r in self._group_ratios]
```

Behavior-preserving for every existing caller (`UniformGroups` produces
exactly today's `[lr] * len(params)`), and now genuinely supports B.3
without a hidden trap.

### A.4 `TrainableModel.footprint_bytes()` -- iteration 1's honest gap, now closeable

Iteration 1's gap analysis (original doc, section 4.2) flagged this
directly: "the wrapper doesn't expose frozen-base size today." That was
correctly left unresolved in iteration 1 because there was nothing to
compute it *from* -- the frozen base was an opaque quantity inside
`core.unet_wrapper.ComfyUNetWrapper`. Once `FrozenWeightStore` (B.5)
exists as an explicit object with its own `footprint_bytes()`, this stops
being a gap: `TrainableModel.footprint_bytes()` becomes `sum(p.numel() *
p.element_size() for p in self.trainable_parameters()) +
self._frozen_store.footprint_bytes()`. Noted here as resolved-by-B.5, not
independently designed -- it wasn't a gap in `DeviceResident`, it was a
gap in what `TrainableModel` had access to.

---

## Part B: modern-technique additions

Each item: what it is, why (or whether) it matters for this project
specifically, its cost/risk, the design element, and a verdict. Sources
verified via web search this session (arXiv IDs given); none of this is
recalled-and-trusted from training data alone.

### B.1 Zero terminal SNR schedule correction

**What.** Lin et al., "Common Diffusion Noise Schedules and Sample Steps
are Flawed" (arXiv:2305.08891, WACV 2024). The standard linear beta
schedule -- exactly what `DiscreteLinearNoiseSchedule` (iteration 1)
replicates from ComfyUI -- never reaches SNR=0 at the final training
timestep. At t=T the model is trained on an input that still contains a
small amount of real signal (Lin et al. compute
`x_T = 0.068265*x_0 + 0.997667*eps` for Stable Diffusion's actual
schedule, not near-pure noise), but at inference the sampler starts from
literal pure Gaussian noise. The mismatch is a documented, real cause of
generated images clustering around medium brightness and an inability to
generate very dark or very bright images -- the model has implicitly
learned to expect a bit of real low-frequency signal (the leaked
channel means) even at the noisiest step. The fix: rescale the beta
schedule so `sqrt(alphas_cumprod[-1]) == 0` exactly.

**Real, load-bearing caveat, not optional:** the same paper states that
enforcing zero terminal SNR requires switching to v-prediction. Epsilon
prediction's own math (`x0 = x_t - sigma*eps`) becomes numerically
degenerate as `sigma -> infinity` at the true zero-SNR limit (verified
directly against the paper's own Section 3.1, not assumed) -- v-prediction
stays well-defined there. This project's current default is
`student_type: eps` (`docs/suspicious_findings.md`), and its own
`MinSNRLossWeighting` already documents, as a known gap, that it "only
implements... the epsilon-parameterization form." Adopting B.1 fully
means finishing that already-flagged v-prediction gap too, not just
adding a schedule rescale in isolation.

**Design.**

```python
class RescaledZeroTerminalSNRSchedule(DiscreteLinearNoiseSchedule):
    """Same construction as DiscreteLinearNoiseSchedule, but rescales the
    computed alphas_cumprod sequence (Lin et al. 2023, Algorithm/Sec 3.1)
    so the final value is exactly zero, before deriving alpha_t/sigma_t.
    Overrides only the one method that determines the table -- everything
    else (per-device caching) is inherited unchanged."""

    @staticmethod
    def _compute(n, beta_start, beta_end):
        alpha_t, sigma_t = DiscreteLinearNoiseSchedule._compute(n, beta_start, beta_end)
        sqrt_ac = alpha_t.clone()
        sqrt_ac_T, sqrt_ac_0 = sqrt_ac[-1].clone(), sqrt_ac[0].clone()
        sqrt_ac -= sqrt_ac_T
        sqrt_ac *= sqrt_ac_0 / (sqrt_ac_0 - sqrt_ac_T)
        alphas_cumprod = sqrt_ac ** 2
        return alphas_cumprod.sqrt(), ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
```

A `DiffusionProcess` (iteration 1) built from
`RescaledZeroTerminalSNRSchedule` and `EpsParameterization` together is a
silently-wrong combination per the paper's own finding above -- worth a
constructor-time check rather than a documentation-only warning:

```python
@dataclass(frozen=True)
class DiffusionProcess:
    schedule: NoiseSchedule
    parameterization: Parameterization
    input_transform: "ModelInputTransform"

    def __post_init__(self):
        if isinstance(self.schedule, RescaledZeroTerminalSNRSchedule) \
                and isinstance(self.parameterization, EpsParameterization):
            raise ValueError(
                "Zero-terminal-SNR schedules are numerically unsound with "
                "epsilon prediction at t=T (Lin et al. 2023, Sec 3.1) -- "
                "use VPredParameterization.")
```

**Verdict.** Real, well-established, essentially free once v-prediction
training is finished end to end (already a partially-started, separately
tracked gap in the current codebase, not new scope this adds). Worth
adopting -- but the actual unlock is finishing v-prediction support, not
the schedule rescale by itself, and that's real work this document
doesn't do for you.

### B.2 Rank-Stabilized LoRA (rsLoRA) scaling

**What.** Kalajdzievski, "A Rank Stabilization Scaling Factor for
Fine-Tuning with LoRA" (arXiv:2312.03732, 2023). Standard LoRA scales its
output by `alpha/r`. The paper proves this causes the adapter's output
and gradient magnitude to collapse as rank `r` grows, which is why LoRA
in practice is usually kept at low rank -- higher ranks "should" add
capacity but empirically don't help, because the scaling itself
suppresses them. The fix is a one-line change: scale by `alpha/sqrt(r)`
instead. Verified directly against the paper (not a secondhand summary):
this is proven, not just observed, and costs nothing extra at inference
or training time -- pure scaling-formula change.

**Design.**

```python
class LoRAScalingPolicy(ABC):
    @abstractmethod
    def scaling(self, alpha: float, rank: int) -> float:
        ...

class ClassicLoRAScaling(LoRAScalingPolicy):
    """Today's actual behavior -- core.lora's existing alpha/rank formula,
    unchanged."""
    def scaling(self, alpha, rank) -> float:
        return alpha / rank

class RankStabilizedScaling(LoRAScalingPolicy):
    def scaling(self, alpha, rank) -> float:
        return alpha / (rank ** 0.5)
```

**Verdict.** Adopt. Zero VRAM cost, zero inference cost, a genuine
correction to a proven instability in the exact scaling formula this
project's LoRA math already uses -- the closest thing in this whole
document to a strict improvement with no tradeoff. Its actual value
depends on also training at higher rank than this project's current
default (`rank: 64` in the example config) to have anything to stabilize
-- worth pairing with a rank increase, not treated as independently
useful at the current default rank.

### B.3 LoRA+ : differential learning rates for A and B

**What.** Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of
Large Models" (arXiv:2402.12354, ICML 2024). Standard LoRA trains both
adapter matrices (`A`, the random-initialized projection down, and `B`,
the zero-initialized projection back up) at the same learning rate. Using
an infinite-width neural-network scaling argument, the paper proves this
is provably inefficient for large-width models, and that using a fixed
ratio `lr_B = lambda * lr_A` with `lambda > 1` (tuned once, not
theoretically pinned to an exact value by the proof itself -- confirmed
directly, the theorem gives an asymptotic `Theta(n)` relationship, not a
constant) restores efficient feature learning. Reported result: up to
~2x finetuning speedup and 1-2% task-performance improvement, at
identical computational cost to plain LoRA.

**Why relevant here.** Zero extra VRAM (same parameter count, same
forward/backward cost) -- purely a training-dynamics improvement, and one
this codebase's real optimizer plumbing already almost supports (see A.3
above, which this technique is what motivated fixing).

**Design.**

```python
class LoRAPlusGroups(ParameterGroupPolicy):
    """B matrices at `ratio`x the base rate, everything else (A matrices,
    any other trainable parameter) at 1x. is_b_matrix is a predicate over
    a parameter, decoupled from any one LoRA implementation's own naming
    -- this project's core.lora would supply it via each layer's own
    lora_B reference, not a name-string match."""

    def __init__(self, is_b_matrix, ratio: float = 16.0):
        self._is_b_matrix = is_b_matrix
        self._ratio = ratio

    def group_ratios(self, params) -> list[float]:
        return [self._ratio if self._is_b_matrix(p) else 1.0 for p in params]
```

The `ratio=16.0` default here is a commonly-used starting point in public
implementations (e.g. Hugging Face PEFT's `LoraPlusModel`), not a value
this document independently verified as optimal for SDXL LoRA -- stated
as a reasonable default to tune from, not a proven-correct constant, per
the paper's own point that the exact ratio isn't theoretically pinned
down.

**Verdict.** Adopt, once A.3 lands -- genuinely free, well-supported by
theory and reported empirical results, real synergy with the fix A.3
already made necessary.

### B.4 DoRA: weight-decomposed adapters

**What.** Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation"
(arXiv:2402.09353, ICML 2024 Oral). Decomposes each frozen weight matrix
into a magnitude component (one learnable scalar per output channel) and
a direction component (the weight-normalized matrix), applying LoRA only
to the direction while training the magnitude vector directly. Motivated
by a weight-decomposition analysis showing full fine-tuning updates
magnitude and direction in ways plain LoRA's single multiplicative update
can't represent well. Reported to consistently outperform plain LoRA
across LLaMA/LLaVA/VL-BART benchmarks, with no added inference cost (the
decomposition is foldable back into a single weight matrix after
training, same as plain LoRA). Confirmed directly: the extra trainable
parameter count is one scalar per output channel -- for a typical SDXL
UNet linear layer, negligible next to the LoRA matrices themselves, let
alone the frozen base.

**Compatibility, checked directly.** DoRA and quantized-base training
(B.5) already compose in published work -- "QDoRA" (referenced directly
in the DoRA paper's own repo, and in a public Answer.AI writeup on
FSDP+QDoRA) combines both. Relevant here because it means adopting B.4
doesn't foreclose B.5 later; they're genuinely orthogonal axes
(`AdapterStrategy` and `FrozenWeightStore`), which is a real, if modest,
validation that keeping them as two separate design elements (rather
than one combined "quality mode" flag) was the right call.

**Design.**

```python
class AdapterStrategy(ABC):
    """How a trainable delta composes with a frozen weight. Iteration 1
    assumed this implicitly (LoRA was the only option available); this
    is the seam that makes it a real choice."""
    @abstractmethod
    def wrap(self, frozen: "FrozenWeightStore", rank: int,
              scaling_policy: LoRAScalingPolicy) -> "AdaptedLayer":
        ...

class PlainLoRAAdapter(AdapterStrategy):
    """Today's core.lora.LoRALinear/LoRAConv2d math, wrapped -- unchanged,
    per the existing rule: genuinely-correct legacy math gets wrapped,
    not re-derived."""
    ...

class DoRAAdapter(AdapterStrategy):
    """Adds one trainable magnitude vector (out_features,) per wrapped
    layer; direction component still goes through `scaling_policy` and a
    LoRA pair exactly as PlainLoRAAdapter does. See B.4 for citations and
    the QDoRA compatibility note."""
    ...
```

**Verdict.** Real, credible quality improvement at near-zero extra VRAM
cost -- but a genuine new forward-pass code path (weight normalization +
magnitude scaling), not a formula tweak like B.1/B.2. Worth building and
equivalence-testing as a second `AdapterStrategy` once `AdapterStrategy`
itself exists (this is additive to `PlainLoRAAdapter`, not a replacement
for it) -- not free to add, but a clearly bounded, well-precedented piece
of work, not a research bet.

### B.5 QLoRA-style quantized frozen base weights (NF4)

**What.** Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient
Finetuning of Quantized LLMs" (arXiv:2305.14314, NeurIPS 2023). Three
components, confirmed directly against the paper: (1) **NF4** (4-bit
NormalFloat), a quantile-based 4-bit data type specifically shaped for
the near-Gaussian distribution of pretrained weights, shown to
outperform 4-bit integer and 4-bit float formats at the same bit width;
(2) **double quantization**, which also quantizes the per-block scale
factors themselves, saving roughly another 0.37 bits/parameter on
average; (3) **paged optimizers**, using CUDA unified memory to absorb
transient memory spikes during checkpointed backward passes rather than
OOMing. The frozen base stays 4-bit in storage; every forward/backward
dequantizes on the fly to bf16 for the actual matmul, so numerical
compute happens at full working precision -- only the *storage* of the
frozen weight shrinks (4x vs. bf16, before double quantization's further
saving). The paper reports NF4 + double quantization *fully recovering*
16-bit LoRA's benchmark accuracy on models up to 65B parameters --
i.e., not merely "close," but statistically matching in their own
reported results.

**Why this reverses iteration 1's stance, specifically.** The original
design (iteration 1, "deliberately deferred" list) declined to add a
`FrozenWeightStore` seam at all, citing "real numerical-drift risk
against a frozen ground truth" as the reason quantized base weights were
out of scope -- based on a vague, unresearched "maybe int8" idea, not a
specific method. QLoRA is exactly the concrete, published, extensively
benchmarked answer to that specific worry: NF4 was engineered
specifically to minimize the distributional mismatch that makes naive
quantization risky, and its accuracy-recovery claim is not this
document's assumption, it's the paper's own headline reported result.

**The honest caveat this project's own docs would want stated, and that
generic QLoRA writeups mostly don't mention:** QLoRA was developed and
benchmarked on LLM linear layers with roughly-Gaussian weight
distributions. This project's target is an SDXL UNet, not a transformer
LLM -- a genuinely different architecture (convolutions, GroupNorm,
cross-attention) and, more specifically, a *diffusion* model whose
weight-usage pattern varies by timestep rather than being uniform across
a single forward pass. Checked directly rather than assumed: Ryu, Lim,
Shim, "Memory-Efficient Fine-Tuning for Quantized Diffusion Model" (a.k.a.
TuneQDM, arXiv:2401.04339, KAIST) studied this exact question and found
that a naive quantized-diffusion-model finetuning baseline "neglects the
distinct patterns in model weights and the different roles throughout
timesteps," causing the naive approach to trade off prompt fidelity
against subject fidelity rather than achieving both -- i.e., generic QLoRA
applied unmodified to a diffusion UNet has a documented, real quality
gap versus its LLM results, motivating diffusion-specific adaptations in
that paper. Also confirmed: informal community reports (a public blog
walkthrough) of SDXL fine-tuned with QLoRA-quantized UNet + LoRA exist
and describe it as workable, but this is not peer-reviewed evidence and
isn't treated as one here.

**Design (seam only -- see verdict).**

```python
class FrozenWeightStore(ABC):
    @abstractmethod
    def footprint_bytes(self) -> int:
        ...
    @abstractmethod
    def materialize(self) -> "Tensor":
        """bf16 view for one forward pass."""


class BF16WeightStore(FrozenWeightStore):
    """Today's actual, only behavior -- the frozen base kept exactly as
    loaded. No change to any existing forward path."""
    def __init__(self, weight: "Tensor"):
        self._weight = weight
    def footprint_bytes(self) -> int:
        return self._weight.numel() * self._weight.element_size()
    def materialize(self) -> "Tensor":
        return self._weight


class NF4WeightStore(FrozenWeightStore):
    """QLoRA-style blockwise NF4 + double quantization. Deliberately not
    designed in more detail here -- see verdict below. materialize()
    would dequantize to bf16 each call; a real implementation needs a
    genuine decision about caching that dequantized tensor per step vs.
    re-dequantizing per use (a real VRAM/speed tradeoff this design
    doesn't resolve for you)."""
    ...
```

**Verdict.** This is the single most valuable *future* item in this
whole document -- the frozen base is this project's own documented
single biggest static VRAM allocation (`docs/vram_and_lora_phase_split.md`'s
"Considered, not implemented" section already said so), and NF4 is a
real, mainstream, extensively-validated answer to exactly that, not a
speculative one. **Not designed in full here, on purpose**: dequantizing
NF4 on the fly for every forward pass needs either a custom fused
dequant-matmul kernel or an explicit dequantize-then-matmul path with its
own scratch-buffer story (this design's `MemoryManager`, tagged
appropriately) -- real, substantial systems work, plus the diffusion-
specific quality caveat above genuinely needs checking against this
project's own real UNet, not assumed to transfer from the LLM literature.
Recommend: build `FrozenWeightStore`/`BF16WeightStore` now (small,
behavior-preserving, closes the A.4 gap), design and validate
`NF4WeightStore` as its own dedicated follow-up effort -- scoped like a
future `nodes/components/` migration with its own equivalence-testing
pass, not squeezed into this document's backlog as a minor item.

### B.6 Selective activation-checkpoint placement

**What.** Iteration 1 (section 2.3 of the original doc) left activation
checkpointing granularity as a vague "future coarser-grained variant,"
without a real method behind it. Two established results fill that in:
Chen et al., "Training Deep Nets with Sublinear Memory Cost"
(arXiv:1604.06174, 2016) show that checkpointing roughly every
`sqrt(N)` layers (for an N-layer network) achieves near-optimal
memory/recompute tradeoff for a *uniform*-cost network; Korthikanti et
al., "Reducing Activation Recomputation in Large Transformer Models"
(NVIDIA, 2022) generalize this to *selective* recomputation -- ranking
candidate checkpoint points by their actual memory-saved-per-recompute-
cost ratio (not assuming uniform cost per block) and greedily selecting
under a stated memory budget, which is the more directly applicable idea
here since a UNet's blocks are not uniform cost (attention blocks vs.
plain conv/resnet blocks differ in both activation size and recompute
time).

**Design.**

```python
@dataclass(frozen=True)
class BlockCost:
    """Per-block estimates a placement policy needs. Producing these
    accurately (profiling real activation sizes and real recompute time
    per block, on real hardware) is itself nontrivial work, explicitly
    out of scope here -- see verdict."""
    activation_bytes: int
    recompute_ms: float


class CheckpointPlacementPolicy(ABC):
    @abstractmethod
    def select(self, blocks: list[BlockCost], budget: "ResourceBudget") -> list[bool]:
        """One bool per block: True = checkpoint it (recompute during
        backward); False = keep its activations resident."""


class EveryBlockPlacement(CheckpointPlacementPolicy):
    """Today's actual, only behavior -- checkpoint everything. Correct;
    maximal VRAM savings; maximal recompute cost (the ~20-30% measured in
    docs/vram_and_lora_phase_split.md)."""
    def select(self, blocks, budget):
        return [True] * len(blocks)


class GreedyRatioPlacement(CheckpointPlacementPolicy):
    """Ranks blocks by activation_bytes/recompute_ms (memory saved per
    unit recompute cost) and checkpoints the best-ratio blocks first
    until the *remaining* uncompressed blocks' activation memory fits
    the budget. A direct, simplified reading of Korthikanti et al.'s
    cost-ratio ranking idea -- not their full method, which also
    reasons about which *operations within* a block to recompute, not
    just whole-block on/off. That finer granularity is a real further
    step this class doesn't take."""
    def select(self, blocks, budget):
        order = sorted(range(len(blocks)),
                        key=lambda i: blocks[i].activation_bytes / max(blocks[i].recompute_ms, 1e-6),
                        reverse=True)
        checkpoint = [False] * len(blocks)
        remaining = sum(b.activation_bytes for b in blocks)
        for i in order:
            if remaining <= budget.vram_budget_mb * 2**20:
                break
            checkpoint[i] = True
            remaining -= blocks[i].activation_bytes
        return checkpoint
```

`ActivationCheckpointingStrategy` (iteration 1) gains an optional
`placement: CheckpointPlacementPolicy` it consults per block instead of
applying uniformly -- `FrozenParamSafeCheckpointing`'s actual patch logic
(the verified `torch.autograd.grad()` frozen-parameter fix) is unchanged
either way; this only changes *which* blocks it gets applied to.

**Verdict.** The idea is sound and the ranking logic above is small and
buildable -- but it's only as good as `BlockCost`'s numbers, and
producing real, trustworthy per-block activation/recompute estimates for
this specific UNet needs actual profiling on real hardware (extending
the existing `profile=True` machinery, which already measures whole-step
phases, down to per-block granularity -- a real, separate piece of
instrumentation work). Recommend building `CheckpointPlacementPolicy` as
an interface now (so `EveryBlockPlacement` -- i.e., no behavior change --
is a real, typed default), but treat `GreedyRatioPlacement` as
unvalidated until real `BlockCost` numbers exist to feed it; shipping it
with guessed costs would be worse than not having it, since a bad
placement decision can checkpoint the wrong (low-value) blocks and get
neither the VRAM savings nor the untouched speed of the two extremes.

### B.7 A broader loss-weighting family

**What.** The current codebase's own `MinSNRLossWeighting` docstring
already states its known limitation plainly: "Only correct for an
eps-predicting student... the v-prediction form, not yet implemented
here." That specific completion is small, already-scoped, and doesn't
need new design -- it belongs on the iteration-3 fix list (below), not
here. What *is* a genuine iteration-2-shaped addition: a second weighting
scheme with a different shape, worth having as an option rather than
assuming Min-SNR is the only reasonable choice. Choi et al.,
"Perception Prioritized Training of Diffusion Models" (P2 weighting,
CVPR 2022) weight by `1 / (k + SNR)^gamma` -- a smoother, more
aggressive de-emphasis of the highest-SNR (near-clean-image, most
imperceptible-detail) steps than Min-SNR's hard clamp, on the argument
that those steps teach the least perceptually important information.

**Design.** Same `LossWeighting` ABC from iteration 1 -- literally a new
class, zero interface change, which is itself worth noting: the existing
`LossWeighting` design already generalizes cleanly to this, no revision
needed (confirms iteration 1 got this one right, the same kind of check
section 4.1 of the original doc did for `Node`/`Port`).

```python
class P2LossWeighting(LossWeighting):
    def __init__(self, k: float = 1.0, gamma: float = 1.0):
        self.k, self.gamma = k, gamma

    def weight(self, sigma: float) -> float:
        snr = 1.0 / (sigma ** 2 + 1e-8)
        return 1.0 / ((self.k + snr) ** self.gamma)
```

**Verdict.** Adopt as an available option -- cheap, well-precedented,
genuinely orthogonal to the eps/v-pred completion work, which stays on
the iteration-3 list since it's a fix to existing documented scope, not
a new capability.

### B.8 Flagged, not designed in detail

Each of these was seriously considered; each has a concrete, stated
reason it isn't being designed further in this document.

- **Continuous-time / flow matching** (Lipman et al., arXiv:2210.02747;
  Liu et al.'s rectified flow, arXiv:2209.03003 -- the formulation
  Stable Diffusion 3 and Flux actually train with). Genuinely modern,
  and A.1 above already leaves the seam open (`Interpolant` as a sibling
  to `NoiseSchedule`, `VelocityParameterization` as a third
  `Parameterization`). Not designed further because it's not a drop-in
  swap for this project's actual model: SDXL is a pretrained
  epsilon/v-prediction model, and converting an already-trained
  diffusion model's *sampling trajectory* into a flow-matching one is
  itself an active, nontrivial research question -- confirmed directly:
  Schusterbauer et al.'s "Diff2Flow" (CVPR 2025) exists specifically to
  do this alignment, which wouldn't be a real research topic if it were
  simple. Adopting flow matching here would mean either training a new
  model natively under it (out of scope -- this project fine-tunes an
  existing SDXL checkpoint) or adopting a nontrivial conversion procedure
  this document isn't in a position to design responsibly. Seam left
  open; technique not adopted.
- **GaLore** (Zhao et al., arXiv:2403.03507, ICML 2024) -- projects
  gradients into a low-rank subspace (via periodic SVD) so a full-
  parameter optimizer's momentum/variance state costs close to what
  LoRA's optimizer state already costs, without restricting the actual
  weight updates to a low-rank subspace the way LoRA does. Reported
  result: full pre-training of a 7B-parameter LLM on a 24GB consumer GPU.
  Not designed further because it solves a problem this project doesn't
  currently have: it's LoRA-only today, and LoRA's optimizer state is
  already small (proportional to the tiny adapter parameter count, not
  the frozen base) -- GaLore is the right answer for a hypothetical
  future *full-parameter* fine-tuning mode, not for improving on
  already-cheap LoRA optimizer state. Worth remembering if full
  fine-tuning is ever wanted; not relevant to today's actual training
  mode.
- **8-bit block-quantized optimizer moments** (Dettmers, Lewis, Shleifer,
  Zettlemoyer, "8-bit Optimizers via Block-wise Quantization,"
  arXiv:2110.02861, 2022 -- what `bitsandbytes`' `Adam8bit` implements).
  Real and mainstream, and the existing `Algorithm.init_state()` contract
  (iteration 1's nodes/optimizer/, unchanged by this document) already
  returns "a plain dict of named tensors" without mandating fp32 --
  nothing structurally blocks a quantized-state `Algorithm` variant.
  Not designed further because, same reasoning as GaLore: CAME and
  Adafactor were already chosen specifically as memory-frugal factored
  optimizers for a LoRA-sized parameter count, so the marginal win from
  further quantizing an already-small state is real but smaller than
  where the actual VRAM mass is (the frozen base, B.5). Lower priority
  than B.5 by a wide margin for this project specifically; not a
  contradiction of it being a fine idea in general.

---

## Updated composition walkthrough (delta from iteration 1)

Only the parts that changed -- everything else in iteration 1's
walkthrough (`ProjectLayout`, `DeviceContext`, `MemoryManager`,
`ResourceCoordinator`, `TrainingStepPipeline` construction) is unchanged
and not repeated here.

```python
policy = ManualResourcePolicy(
    checkpointing=FrozenParamSafeCheckpointing(placement=EveryBlockPlacement()),  # B.6
    optimizer_strategy=ChunkedScratchBufferStrategy,
    text_encoder_cache=True,
    adapter_strategy=PlainLoRAAdapter(),          # B.4 -- DoRAAdapter() to opt in
    lora_scaling_policy=RankStabilizedScaling(),  # B.2 -- adopted, see verdict
    frozen_weight_store=BF16WeightStore,          # B.5 -- NF4WeightStore is future work
    parameter_group_policy=UniformGroups(),       # B.3 -- LoRAPlusGroups(...) to opt in
)

schedule = RescaledZeroTerminalSNRSchedule()               # B.1
process = DiffusionProcess(schedule, VPredParameterization(), KarrasInputScaler())
# DiffusionProcess.__post_init__ would reject EpsParameterization here -- see B.1

model = build_trainable_model(weights, policy, device_ctx)   # now also wires
                                                               # FrozenWeightStore +
                                                               # AdapterStrategy + scaling
optimizer = build_optimizer(model.trainable_parameters(), policy, memory,
                             group_policy=policy.parameter_group_policy())
```

---

## Left for iteration 3 (fixes only, listed, not resolved)

Per the definition given for this pass: iteration 3 should be fixes, not
new capability. Collected here so it isn't lost, deliberately not
addressed in this document:

- **Finish `MinSNRLossWeighting`'s v-prediction branch.** Already
  documented as a known gap in the existing code's own docstring (not
  new scope this document adds) -- small, well-scoped, no new design
  needed, just completion.
- **Redo `docs/theoretical_pipeline_design.md`'s section 4 (gap analysis)
  properly**, after iteration 3, incorporating everything from this
  document -- the sequencing correction noted at the top of this file.
- **Reconcile the "Prioritized backlog"** in the original doc with this
  document's additions (where do `FrozenWeightStore`/`AdapterStrategy`/
  `ParameterGroupPolicy` actually land in the numbered order) -- a
  renumbering/merging task, not a design task.
- **Naming consistency pass** across both documents (e.g. confirm
  `ResourcePolicy`'s now-seven methods are named consistently with each
  other; `DiffusionProcess.__post_init__`'s validation should probably
  generalize to a small registry of "known-incompatible pairs" rather
  than one hardcoded `isinstance` check, once there's a second such
  pair to justify it -- not yet, per "don't overcomplicate").
- **`ResourceBudget`'s `vram_budget_mb` unit assumption** now gets used
  by `GreedyRatioPlacement` (B.6) in a way iteration 1 never exercised --
  worth a consistency check that every consumer of `ResourceBudget`
  agrees on what it's measuring (peak allocation? steady-state? -- not
  pinned down precisely anywhere yet).

---

## What's not changing (revalidated, not just carried over)

Explicitly unaffected by this pass, so the actual scope of iteration 2 is
clear: `Builder`/`Port`, `DeviceResident`, `DeviceContext`, `MemoryManager`
(interface, not adoption breadth), `ProjectLayout`, `StepPhase`/
`TrainingStepPipeline`'s overall shape, `ResourceCoordinator`/
`OffloadOrchestrator`, `ComponentRegistry`, `TrainingRecipe`/
`PipelineFactory`, the concurrency contract, and the Acyclic Domain
Dependency Rule. None of Part A's revisions touch any of these; Part B's
additions all compose on top of them without modification, which is
itself a small validation that iteration 1's foundational layer (Part
A.1's `NoiseSchedule` narrowing aside) was actually load-bearing enough
to build a real second pass on without cracking.
