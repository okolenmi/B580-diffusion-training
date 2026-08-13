# Training pipeline design: complete plan

A from-scratch design for the training pipeline -- VRAM-savings-first,
strict OOP, real composition, no singletons -- developed independently of
`nodes/`'s actual classes, then compared against them once the design was
settled. Originally written in three passes (foundational architecture,
then a review pass that added techniques with real published evidence
behind them, then a fixes-only pass), merged into one document instead of
left spread across three. The process that produced the original design
is in git history, not repeated here.

**Status update, 2026-08-12: the original numbered backlog (12 items) is
now fully implemented and equivalence-tested.** Every item that section
10 used to list -- `DiffusionProcess`/`DeviceContext`, the `DeviceResident`
ABC (with `OptimizerHandle`/`TrainableModel`/`TextEncoder` all conforming
to it), the `ParameterGroupPolicy` fix, `LoRAScalingPolicy`, Min-SNR's
v-prediction branch plus `P2LossWeighting`, `ActivationCheckpointingStrategy`,
`ProjectLayout`, the `AdapterStrategy`/`FrozenWeightStore` seam,
`TrainingStepPipeline`/`StepPhase`, `PrefetchingBatchSource`, and
`ResourceCoordinator`/`OffloadOrchestrator` -- is real, tested code in
`nodes/` today, not illustrative Python. This document has been edited
down accordingly: **sections describing something now implemented keep
their rationale but no longer repeat the illustrative class code --
that code is stale next to the real, tested version, so read the real
file (pointed to inline) instead of trusting a copy here that can drift.**
Real code that cites a section number in this document (e.g.
`nodes/model/adapter_strategy.py` citing 3.1) is citing the section for
its *rationale*, which is why that rationale is kept even where the
illustrative code it originally sat next to has been removed.

What's left in this document, in full, with illustrative code, is only
what's genuinely still open: two seam-only techniques not yet built out
(`DoRAAdapter`, `NF4WeightStore`), one policy blocked on missing
instrumentation (`CheckpointPlacementPolicy`), two consolidating
abstractions not yet needed (`ResourceBudget`/`ResourcePolicy`,
`ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`/`ResourceProfile`),
and two pieces of *validation* work that landing code alone can't finish
(`RescaledZeroTerminalSNRSchedule` needs a real end-to-end v-prediction
training run; `LoRAPlusGroups` needs a real tuned run, not just existing
as an opt-in policy). This remains a planning document for those pieces
-- their illustrative code is a strong proposal, not a spec set in stone,
same as before. Section 9 (Implementation status) is where this gets
compared against what `nodes/` actually is today; section 10 (Prioritized
backlog) is the ordered, concrete plan for what's left.

## Design goals and constraints

Stated once here, referenced rather than repeated throughout:

1. **VRAM first, speed second -- but not speed-blind.** Every VRAM-saving
   choice below either has no speed cost (pure refactor) or a named,
   estimable one (e.g. activation checkpointing's real recompute cost).
   Nothing here trades speed for VRAM silently or trades VRAM for a
   *bigger* speed cost than the current codebase already accepts,
   without saying so.
2. **Strict OOP.** Behavior lives on objects that implement declared
   interfaces (ABCs). No dict-of-flags threaded through a function, no
   `hasattr()`/`isinstance()` sniffing to decide behavior, no module-level
   mutable state reached for by import.
3. **No singletons.** A component that needs configuration, a device, a
   shared resource pool, or another component gets it through its
   constructor or a method argument -- never by importing a module and
   reading/writing its globals. This is the direct answer to "separation
   from old singleton code": `paths.py`'s `_comfy_dir_override` pattern and
   `core/noise_schedule.py`'s `ALPHA_T`/`SIGMA_T` module-level tensors are
   the two concrete existing instances of what this rule forbids going
   forward (see section 9).
4. **Composition over inheritance, and over rewriting.** Every new
   capability should be addable by writing a new small class that
   implements an existing interface, not by editing a big existing method.
   Where old, verified math is genuinely correct (UNet forward, LoRA
   layers), wrap it -- don't re-derive it -- exactly per the project's
   existing rule.
5. **One reviewed place for device-memory lifecycle.** Every reusable
   device buffer goes through a memory manager. This already exists
   (`nodes/memory/manager.py`'s `MemoryManager`) and is *correct* as a
   low-level primitive -- the design below reuses it unchanged and asks
   "what needs to start using it that doesn't yet," not "how should this
   be redesigned."
6. **Don't overcomplicate.** Every new abstraction below exists because a
   concrete, named problem needs it -- not because it's generically good
   practice. Section 7 lists what was considered and deliberately left
   out, with the reasoning.
7. **Modern techniques earn a place only with real evidence.** Every
   technique adopted below is backed by a specific, checked source (paper,
   arXiv ID) -- not a hunch or something recalled and trusted from memory.
   Each one also gets an honest calibration: adopt now, adopt with a
   stated caveat, or seam-only (room left, not built). A design that
   recommends every paper it read isn't more useful than one that reads
   none of them.

---

## 1. Foundational ontology

The base vocabulary: what kinds of objects exist, what each owns, and how
they're wired together at construction time vs. what they do at runtime.

### 1.1 Two different lifetimes, two different kinds of object

Everything in a training pipeline is either:

- A **builder**: takes configuration, produces a runtime object. Exists
  only during graph construction. Stateless with respect to training
  itself.
- A **runtime object**: the thing training actually calls methods on,
  every step, for the life of a run. Has real state (weights, optimizer
  moments, a cursor into a dataset).

Collapsing these two into one class is exactly what makes old-style
"trainer with 40 constructor args and 40 methods" code hard to test and
hard to extend -- a builder's only job is turning config into a runtime
object, so it can be swapped, mocked, or parameterized without the
runtime object's own logic ever being touched.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class Port:
    """Declarative metadata for one named input/output slot on a Builder."""
    name: str
    type: type
    required: bool = True
    default: Any = None
    doc: str = ""


class Builder(ABC):
    """Construction-time node: declared typed Ports in, one runtime object
    (or a small typed bundle of them) out. Pure with respect to training
    state -- calling build() twice with the same inputs produces two
    independent runtime objects, never shares hidden state between them."""

    INPUTS: ClassVar[dict[str, Port]] = {}
    OUTPUTS: ClassVar[dict[str, Port]] = {}

    @abstractmethod
    def build(self, **inputs) -> dict[str, Any]:
        ...
```

This is deliberately close to what already exists (`nodes/core.py`'s
`Node`/`Port`) -- arriving at it independently, from first principles,
before looking, is exactly the check this design process was for. See
section 9.1 for the actual comparison; the short version is "this part is
already right, keep it."

### 1.2 Runtime lifecycle: `DeviceResident`

Every runtime object that can hold device memory needs the same three
questions answerable, regardless of domain (optimizer, model, text
encoder, dataset prefetch buffer): how big is it right now, can it be
moved off-device without losing its identity, and can it be dropped
entirely. Before this, each domain answered this with its own ad hoc
method names (an optimizer had `offload_states_to_cpu`/`free_states`; a
model had `to()`; a text encoder had `unload()`) -- fine individually,
but nothing generic could coordinate across all of them.

**Implemented**, unchanged from the design above: `DeviceResident`
(`nodes/memory/handle.py`) -- three lifecycle tiers,
`footprint_bytes()`/`offload()`/`reload()`/`release()`, kept distinct on
purpose (collapsing them was exactly the mistake
`nodes/memory/manager.py`'s own docstring already documents once, the
reset-vs-free asymmetry bug class). `OptimizerHandle`
(`nodes/optimizer/handle.py`), `TrainableModel` (`nodes/model/handle.py`),
and `TextEncoder` (`nodes/model/text_encoder.py`, and through it
`CachingTextEncoder`) all conform to it now, each via thin alias methods
over their existing domain-specific ones rather than a rewrite --
`DeviceResident` is a floor, not a ceiling, so `decay_states`/
`reset_states`-style extras stayed as optimizer-specific additions beyond
the universal contract. `TrainableModel.footprint_bytes()` specifically
needed `FrozenWeightStore` (3.3) to exist first, for the frozen base's
contribution -- see `nodes/model/frozen_weight_store.py`.

This closed the *coordination* gap -- nothing generic could drive
offload/reload order across domains before this existed (5.1, 5.2 build
on it directly). It's still not, by itself, a fix for the still-open
"hang after VRAM pressure" report in `docs/suspicious_findings.md` --
that report's own leading hypothesis is a missing synchronize() on an
async offload path in `core/trainer.py`, a correctness bug this
lifecycle contract doesn't touch (see 5.2 and section 9.3 for why that's
explicitly out of scope here).

### 1.3 Pooled device buffers stay a separate, lower-level concern

`DeviceResident` is object-granularity ("offload this whole optimizer").
Underneath any one `DeviceResident`, there's often a need for
finer-granularity, reusable scratch buffers ("give me 4MB of float32
scratch, reuse it next step too") -- a different concern, already solved
correctly: a tag-keyed pool that grows lazily, never shrinks, and
distinguishes *released* (available for reuse, allocation kept) from
*freed* (allocation actually dropped). This is precisely
`nodes/memory/manager.py`'s `MemoryManager`, and this design reuses it
unchanged -- see section 9.1 for why no interface change was needed,
only a widened set of callers.

The relationship between the two: a `DeviceResident.release()`
implementation that owns pooled buffers is responsible for also calling
`MemoryManager.free()`/`free_all()` on whatever it acquired -- exactly the
pattern `ChunkedScratchBufferStrategy.free_extra()` already establishes.
`DeviceResident` doesn't replace `MemoryManager`; it's the object-level
contract that sits on top of it and on top of anything else a runtime
object owns (a model's parameters, an LRU cache's tensors) that isn't
itself a pooled scratch buffer. `MemoryManager` itself needed no
interface change to support this -- see 2.6.

### 1.4 The diffusion process: `NoiseSchedule`, `Parameterization`, `DiffusionProcess`

`core/noise_schedule.py` computed `ALPHA_T, SIGMA_T = make_schedule()` at
*import time*, as module-level tensors, with hardcoded default
`beta_start`/`beta_end`. Every caller reached for these two globals by
importing the module -- the concrete case the "no singletons" rule is
written against: two independent training runs with different noise
schedules couldn't coexist in one process, and nothing about the
dependency was visible in any constructor signature.

**Implemented**, unchanged from the design: `NoiseSchedule`/
`DiscreteLinearNoiseSchedule` (matching `core.noise_schedule.make_schedule()`'s
math, moved into an instance holding its own tensors, with a lazily-built
per-device cache instead of mutating a shared global),
`Parameterization`/`EpsParameterization`/`VPredParameterization`
(replacing `core/model_io.py`'s `raw_to_x0`/`raw_to_target` and
`core/noise_schedule.py`'s `eps_to_vpred`/`vpred_to_eps` four-way branch
of free functions with a two-member Strategy pair and a `convert_to()`
that's the identity for same-type conversion), `ModelInputTransform`/
`KarrasInputScaler` (replacing `comfy_input_transform`), and the
`DiffusionProcess` composite itself (rejecting the numerically-unsound
zero-terminal-SNR-plus-epsilon-prediction combination at construction,
per Lin et al. 2023 Sec 3.1, below) -- all in
`nodes/components/diffusion.py`, wired into `nodes/train/supervised.py`
in place of the three `core.*` imports `_run_step` used to make directly.
A continuous-time process (flow matching, section 7) would still need a
separate, smaller `Interpolant` contract as a sibling, not a subtype --
that seam is unchanged by anything below.

A second concrete schedule, `RescaledZeroTerminalSNRSchedule`
(also in `nodes/components/diffusion.py`, overriding only
`DiscreteLinearNoiseSchedule`'s one table-computing method), is
implemented too -- but the actual reason to want it is a real, published
train/inference mismatch worth restating: Lin et al., "Common Diffusion
Noise Schedules and Sample Steps are Flawed" (arXiv:2305.08891, WACV
2024) show that a standard linear beta schedule never reaches SNR=0 at
the final training timestep -- the model is trained on an input that
still contains a small amount of real signal (`x_T = 0.068265*x_0 +
0.997667*eps` for Stable Diffusion's actual schedule, per the paper),
while inference sampling starts from literal pure Gaussian noise. This is
a documented, real cause of generated images clustering around medium
brightness and an inability to generate very dark or very bright images.
The fix is a rescale so `sqrt(alphas_cumprod[-1]) == 0` exactly.

Two things worth being precise about, checked by hand rather than left
implicit or assumed from the paper's own claims: first,
`alphas_cumprod[-1]` is exactly `0.0` after this rescale, so `sigma_t[-1]`
is exactly `inf` (a real IEEE-754 division-by-zero-tensor result, not an
exception) -- correct, by construction, not a bug, but any code touching
raw `sigma_t` *outside* the `Parameterization` abstraction (a stray
`1 / sigma` somewhere) will hit that `inf` and needs to account for it.
Second, the paper states that enforcing zero terminal SNR requires
switching to v-prediction, because epsilon prediction's own math
(`x0 = x_t - sigma*eps`) becomes numerically degenerate as
`sigma -> infinity`, while v-prediction's `to_x0()` stays well-defined in
that limit (`x_t/denom -> 0` and `sigma/sqrt(denom) -> 1` as `sigma ->
inf`, so `x0 -> -raw`, a clean finite result) -- checked directly, not
trusted secondhand. This is exactly the incompatibility
`DiffusionProcess.__post_init__` rejects at construction now, for real:
pairing `RescaledZeroTerminalSNRSchedule` with `EpsParameterization`
raises `ValueError` before a run can start, rather than failing silently
mid-training.

**Still open, and this is real validation work, not construction.** This
project's current default remains epsilon prediction with the plain
`DiscreteLinearNoiseSchedule` --
`SupervisedLoRATrainerNode`'s `diffusion_process` port accepts any
`DiffusionProcess`, so nothing stops wiring
`RescaledZeroTerminalSNRSchedule` + `VPredParameterization` into a run
today, but doing so is a genuine training-behavior change with no old
code path to equivalence-test against, unlike everything else that
closed out of this document. It needs a real training run and
qualitative image-quality evaluation (does it actually fix
medium-brightness clustering on this project's own data) before it's
trustworthy as more than "the math is right and it doesn't crash." See
the backlog (section 10).

### 1.5 Device backend as a Strategy, not `hasattr()` calls

`core/comfy_setup.py`'s `xpu_empty_cache`/`xpu_synchronize`/
`xpu_memory_stats` each independently checked `hasattr(torch, "xpu") and
torch.xpu.is_available()`. Not a singleton in the mutable-global sense,
but the same category of problem as one: backend selection logic
duplicated at every call site instead of decided once and injected.

**Implemented**, unchanged from the design: `DeviceContext`
(`nodes/components/device.py`) -- `empty_cache()`/`synchronize()`/
`memory_stats()`, plus the `for_device()` factory dispatching to
`_XPUDeviceContext`/`_CUDADeviceContext`/`_NullDeviceContext` (the last a
correct, cheap no-op set for CPU or any backend without a cache/sync/
stats concept, so callers never need an `if device supports this` branch
of their own). Wired into `nodes/train/supervised.py` and
`nodes/train/step_pipeline.py`'s `TimedPhase` in place of the
`core.comfy_setup` imports `_run_step`'s profiling branch used to make
directly.

### 1.6 Configuration as an injected value object, not a mutable module global

`paths.py` is the other concrete singleton the original design was
written against: module-level `_comfy_dir_override`/
`_checkpoints_dir_override`/`_loras_dir_override`, mutated via
`set_comfy_dir()`/`set_checkpoints_dir()`/`set_loras_dir()`, read via
`get_*()` functions any file can call from anywhere -- process-global
state, set once by whoever calls the setters first (`server/config.py`),
silently shared by everything else running in the same process.

**Implemented**: `ProjectLayout` (`nodes/components/layout.py`) -- one
constructed, immutable value object (`comfy_dir`/`checkpoints_dir`/
`loras_dir`/`datasets_dir`/`runs_dir`, plus `resolve_model_path()`/
`resolve_safe_model_path()`), a deliberate, narrow exception to "no
singletons" stated precisely so it isn't confused with the pattern it
replaces: one long-lived configuration object, explicitly constructed and
explicitly passed down, is not the same thing as a mutable module-level
global reached for by import. Wired into the four `nodes/` Nodes that
called `paths.resolve_safe_model_path()`/`resolve_safe_dataset_path()`
directly (`nodes/model/checkpoint_loader.py`, `lora_saver.py`,
`lora_checkpoint_loader.py`, `nodes/dataset/managed.py`) via an optional
`project_layout` port, default `None` -> `ProjectLayout.from_paths_module()`.

**Deliberately a bridging period, not a clean swap, and this part is
still true and still open:** `paths.py` itself is untouched -- `server/*`,
`manager/*`, and `core/*` still read its module functions directly, not
`ProjectLayout`. `from_paths_module()` only snapshots that same global
state into an immutable object rather than replacing it. Migrating
`server/*`/`manager/*` off `paths.py` entirely is real, separate,
out-of-scope work for `nodes/` -- not attempted here, not blocking
anything above.

---

## 2. Orchestrating a training step

### 2.1 The step loop as a pipeline of phases, not one method

A training step is a fixed sequence of concerns -- fetch a batch, encode
conditioning, run the forward pass, compute loss, backward, apply the
optimizer update, report progress -- each with a genuinely different
reason to change (a new conditioning scheme, a new loss weighting, a new
optimizer family), but it used to live as sequential code inside one
~90-line method, which is exactly what made each "explicit v1 scope
reduction" item (CFG dual-pass, gradient accumulation, resume cadence)
mean editing that same method rather than adding something next to it.

**Implemented**, and more granular than the design's own illustrative
7-phase list: `StepState`/`StepPhase`/`TrainingStepPipeline` plus nine
concrete phases (`FetchBatchPhase`, `PrepareDiffusionInputsPhase`,
`EncodeConditioningPhase`, `OptimizerBeginStepPhase`, `ForwardPhase`,
`LossPhase`, `BackwardPhase`, `OptimizerStepPhase`, `MonitoringPhase`) and
`TimedPhase` (the generic profiling decorator, replacing `profile: bool`
manually wrapping five points with `xpu_synchronize()` + `perf_counter()`)
-- all in `nodes/train/step_pipeline.py`, driving
`SupervisedLoRATrainerNode` in `nodes/train/supervised.py`.
`update_lr()`/`zero_grad()`/`begin_step()` and diffusion-input prep
(`x_t`/`target`/`t`/`sigma`/`xc`) each got their own phase rather than
folding into `ForwardPhase`/`EncodeConditioningPhase`, since each is
genuinely its own reason to change -- the same test this design was built
around. The profiling output's *shape* genuinely changed as a result
(checked against every real consumer before shipping, per
`step_pipeline.py`'s own docstring) -- not a silent behavior change.

**Still not built, and this is the concrete, real payoff the refactor was
for:** `SupervisedLoRATrainerNode`'s own scope note still lists no CFG
cond/uncond dual pass, no gradient accumulation, no cyclic/teacher-rollout
caching, no DAgger, no adversarial pre-conditioning, and no resume/
checkpoint cadence beyond `on_step`. None of these are designed further
here -- each is now genuinely additive ("construct one more phase, insert
it in the list") rather than monolith surgery, which was the actual goal;
designing any one of them in detail is real, separate future work, sized
independently once actually needed.

### 2.2 Resource budget as a first-class value, resource policy as a Strategy

**Not implemented -- still open, and now the natural next step** (see the
backlog, section 10, item 1): "should this run use activation
checkpointing," "which `ExecutionStrategy` should the optimizer use,"
"should the text encoder cache be warmed," "which LoRA adapter family,
which weight-quantization scheme, which parameter-group policy, which
`DiffusionProcess`" are each an independent manual flag on an independent
node/port today -- more of them now than when this was first written,
since 3.1-3.4's seams landed as additional independent choices. That's
fine as a default (explicit is better than magic), but there's still no
single place that represents "how much VRAM headroom does this run
actually have," or the full set of VRAM/quality-affecting choices a run
makes, as values other components could consult.

```python
@dataclass(frozen=True)
class ResourceBudget:
    """A stated VRAM ceiling for one run, plus a safety margin. Purely
    descriptive -- constructing this doesn't enforce anything by itself;
    a ResourcePolicy is what turns it into actual choices (see below).

    vram_budget_mb measures against the allocator's *reserved* memory --
    the same figure DeviceContext.memory_stats()'s reserved_mb field
    reports (1.5), not allocated_mb. Reserved is what actually determines
    whether the OS hands back an out-of-memory error next, since it
    includes the allocator's own held-but-currently-unused pool, not just
    tensors presently live -- allocated_mb alone would let a caller
    believe it has headroom the allocator has already claimed and won't
    necessarily release. Every consumer of ResourceBudget (including
    CheckpointPlacementPolicy, 2.3) compares against reserved_mb
    consistently -- stated once, here, rather than left for each consumer
    to assume independently."""
    vram_budget_mb: float
    vram_reserve_mb: float = 512.0  # headroom kept free on purpose


class ResourcePolicy(ABC):
    """Decides the VRAM/speed/quality-affecting choices a run needs to
    make. Returns *descriptions* of what to build (which ExecutionStrategy
    class, which ActivationCheckpointingStrategy instance), not the built
    objects themselves -- keeps this a pure decision, testable without
    constructing real models/optimizers."""

    @abstractmethod
    def checkpointing_strategy(self) -> "ActivationCheckpointingStrategy":
        ...
    @abstractmethod
    def optimizer_execution_strategy(self) -> type:
        ...
    @abstractmethod
    def enable_text_encoder_cache(self) -> bool:
        ...
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


class ManualResourcePolicy(ResourcePolicy):
    """Today's actual behavior, made explicit: every choice is a
    constructor argument, no inspection of budget/hardware at all. This
    stays the default -- see section 7's note on why an inspect-and-decide
    AutoResourcePolicy is deliberately not designed in detail."""

    def __init__(self, checkpointing: "ActivationCheckpointingStrategy",
                 optimizer_strategy: type, text_encoder_cache: bool,
                 adapter_strategy: "AdapterStrategy" = None,
                 lora_scaling_policy: "LoRAScalingPolicy" = None,
                 frozen_weight_store: type = None,
                 parameter_group_policy: "ParameterGroupPolicy" = None):
        self._checkpointing = checkpointing
        self._optimizer_strategy = optimizer_strategy
        self._text_encoder_cache = text_encoder_cache
        self._adapter_strategy = adapter_strategy or PlainLoRAAdapter()
        self._lora_scaling_policy = lora_scaling_policy or ClassicLoRAScaling()
        self._frozen_weight_store = frozen_weight_store or BF16WeightStore
        self._parameter_group_policy = parameter_group_policy or UniformGroups()

    def checkpointing_strategy(self):
        return self._checkpointing
    def optimizer_execution_strategy(self):
        return self._optimizer_strategy
    def enable_text_encoder_cache(self):
        return self._text_encoder_cache
    def adapter_strategy(self):
        return self._adapter_strategy
    def lora_scaling_policy(self):
        return self._lora_scaling_policy
    def frozen_weight_store(self):
        return self._frozen_weight_store
    def parameter_group_policy(self):
        return self._parameter_group_policy
```

Every default above (`PlainLoRAAdapter`, `ClassicLoRAScaling`,
`BF16WeightStore`, `UniformGroups`) reproduces today's actual behavior
exactly -- nothing changes for an existing run unless it explicitly opts
into one of the techniques in section 3 or 4. The payoff of having one
`ResourcePolicy` type at all, even with only a manual implementation:
every VRAM/quality-affecting choice now has one shared shape a future
automatic policy could implement against, instead of being unrelated
flags on unrelated node classes. That future policy is explicitly not
designed here -- see section 7.

### 2.3 Activation checkpointing: strategy and placement

The underlying fix (`nodes/model/gradient_checkpointing.py`) was, and
remains, *correct*: filter `ctx.input_params` to `requires_grad=True`
entries before `torch.autograd.grad()`, reconstruct the full gradient
tuple with `None` at frozen positions. What was missing was that it used
to be exposed only as a global, process-wide monkeypatch triggered by a
bare `bool` port -- correct, but not itself an object another piece of
code could compose with or substitute.

**Implemented**, unchanged from the design: `ActivationCheckpointingStrategy`/
`NoCheckpointing`/`FrozenParamSafeCheckpointing` (same mechanism, now with
an `apply()` method; `NoCheckpointing` is the explicit "did nothing" case
replacing an implicit "the if just wasn't taken") -- in
`nodes/model/gradient_checkpointing.py`.
`ComfyUNetLoRANode`'s existing `use_checkpoint: bool` port stayed wired to
this internally, so nothing that already used it broke.
`FrozenParamSafeCheckpointing` takes no `placement` parameter yet --
deliberately: adding one with nothing real to pass it would be
scaffolding, not a feature. That's the policy below.

All-or-nothing checkpointing (every block, or none) is correct and
maximizes VRAM savings at maximum recompute cost. A more
principled middle ground has real published grounding: Chen et al.,
"Training Deep Nets with Sublinear Memory Cost" (arXiv:1604.06174, 2016)
show that checkpointing roughly every `sqrt(N)` layers achieves
near-optimal memory/recompute tradeoff for a *uniform*-cost network;
Korthikanti et al., "Reducing Activation Recomputation in Large
Transformer Models" (NVIDIA, 2022) generalize this to *selective*
recomputation -- ranking candidate checkpoint points by their actual
memory-saved-per-recompute-cost ratio, which is the more directly
applicable idea here since a UNet's blocks aren't uniform cost (attention
blocks vs. plain conv/resnet blocks differ in both activation size and
recompute time). **Not implemented -- still open:**

```python
@dataclass(frozen=True)
class BlockCost:
    """Per-block estimates a placement policy needs. Producing these
    accurately (profiling real activation sizes and real recompute time
    per block, on real hardware) is itself nontrivial work -- see
    calibration below."""
    activation_bytes: int
    recompute_ms: float


class CheckpointPlacementPolicy(ABC):
    @abstractmethod
    def select(self, blocks: list[BlockCost], budget: ResourceBudget) -> list[bool]:
        """One bool per block: True = checkpoint it (recompute during
        backward); False = keep its activations resident."""


class EveryBlockPlacement(CheckpointPlacementPolicy):
    """Today's actual, only behavior -- checkpoint everything."""
    def select(self, blocks, budget):
        return [True] * len(blocks)


class GreedyRatioPlacement(CheckpointPlacementPolicy):
    """Ranks blocks by activation_bytes/recompute_ms (memory saved per
    unit recompute cost) and checkpoints the best-ratio blocks first
    until the *remaining* uncompressed blocks' activation memory fits the
    budget. A direct, simplified reading of Korthikanti et al.'s
    cost-ratio ranking idea -- not their full method, which also reasons
    about which *operations within* a block to recompute, not just
    whole-block on/off."""
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

**Calibration.** `GreedyRatioPlacement` is only as good as `BlockCost`'s
numbers, and producing real, trustworthy per-block activation/recompute
estimates for this specific UNet needs actual profiling on real hardware
(extending `profile=True`, which already measures whole-step phases, down
to per-block granularity -- its own separate instrumentation task).
`EveryBlockPlacement` is the safe, typed default; `GreedyRatioPlacement`
should be treated as unvalidated until real `BlockCost` numbers exist --
shipping it with guessed costs would be worse than not having it, since a
bad placement decision gets neither the VRAM savings nor the untouched
speed of the two extremes. This calibration verdict is unchanged since
`nodes/model/gradient_checkpointing.py`'s own docstring cites it directly
-- **the actual blocker is the per-block profiling instrumentation, not
the policy class itself** (see the backlog, section 10).

### 2.4 Text encoder cache becomes visible to resource accounting

`CachingTextEncoder` (bounded LRU, default 512 entries, CPU-resident) was
already real, working, and self-contained -- the gap was that its memory
usage was invisible to anything outside itself.

**Implemented**: `TextEncoder` (`nodes/model/text_encoder.py`) extends
`DeviceResident` directly, so `CachingTextEncoder` gets `footprint_bytes()`/
`offload()`/`release()` -- landed as part of backlog item 12, the last
conformance gap left open once `ResourceCoordinator`/`OffloadOrchestrator`
(5.1, 5.2) actually needed a second real `DeviceResident` besides the
model to coordinate anything meaningful. The aggregate "where did my
memory go" report this enables (5.5, `ResourceProfile`) is itself still
not built -- see that section.

### 2.5 Dataset prefetching, kept honest about what it does and doesn't save

`SupervisedLoRATrainerNode`'s own `profile=True` output already reported
`data_wait_ms` (now `fetch_batch_ms`, see 2.1) -- so whether data loading
is a real bottleneck was already measurable, not a guess, before this was
built.

**Implemented**, unchanged from the design: `PrefetchingBatchSource`
(`nodes/dataset/prefetch.py`) -- a decorator over any
`TrainingBatchSource`, same pattern `nodes/dataset/renoise.py`'s
`RenoiseBatchSource` already established for this domain. One real
deviation worth noting: a fresh worker thread per `__iter__()` call
rather than one shared for the object's whole lifetime, since
`TrainingBatchSource.__iter__()` is expected to be restartable (a fresh
pass each call) and `FetchBatchPhase` relies on exactly that to wrap to a
new epoch. Explicitly not built: pinning the host-side buffers
(page-locked memory) -- a real platform-specific wrinkle (pinning
support/benefit isn't identical across CUDA and XPU), left for its own
follow-up once this is in real use. Not wired in by default -- still an
opt-in node, per the same "demand-driven, not speculative" reasoning as
before.

### 2.6 `MemoryManager`'s reach widens; its interface doesn't

Every new device-memory consumer identified in this design
(activation-checkpoint recompute scratch, if a future custom block needs
it; a text-encoder cache's tensors, if it's moved to device rather than
kept CPU-resident; a `PrefetchingBatchSource`'s pinned host buffers)
should acquire memory through the existing `MemoryManager.get_buffer()`/
`release()`/`free()` vocabulary, under its own tag, exactly the way
`ChunkedScratchBufferStrategy` already does for optimizer scratch. No new
method is being proposed on `MemoryManager` itself -- the design problem
it solves (tagged, lazily-grown, reuse-vs-drop-tracked buffers) is
domain-independent already; the gap is adoption, not capability.

---

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

`DoRAAdapter` below is **not implemented** -- this is real code cited
directly by `nodes/model/adapter_strategy.py`'s module docstring, so its
wording is preserved rather than reworded:

```python
class DoRAAdapter(AdapterStrategy):
    """Liu et al., 'DoRA: Weight-Decomposed Low-Rank Adaptation'
    (arXiv:2402.09353, ICML 2024 Oral). Decomposes each frozen weight
    matrix into a magnitude component (one learnable scalar per output
    channel) and a direction component (the weight-normalized matrix),
    applying LoRA only to the direction while training the magnitude
    directly -- reported to consistently outperform plain LoRA across
    LLaMA/LLaVA/VL-BART benchmarks, with no added inference cost (the
    decomposition folds back into a single weight matrix after training,
    same as plain LoRA). The extra trainable parameter count is one
    scalar per output channel -- negligible next to the LoRA matrices
    themselves, let alone the frozen base. Confirmed directly: DoRA and
    quantized-base training (3.3) already compose in published work
    ('QDoRA', referenced in the DoRA paper's own repo and in a public
    Answer.AI FSDP+QDoRA writeup) -- AdapterStrategy and FrozenWeightStore
    are genuinely orthogonal axes, not one combined 'quality mode' flag.

    Calibration: real, credible quality improvement at near-zero extra
    VRAM cost, but a genuine new forward-pass code path (weight
    normalization + magnitude scaling), not a formula tweak -- worth
    building and equivalence/quality-testing as a second AdapterStrategy
    once the seam exists, additive to PlainLoRAAdapter, not a
    replacement for it."""
    ...
```

The seam this needed (`AdapterStrategy` existing at all, with a real
second conformance checked against it) now exists -- see the backlog,
section 10, for why this is buildable now in a way it wasn't before.

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
`NF4WeightStore` below is **not implemented**; this is real code cited
directly by `nodes/model/frozen_weight_store.py`'s module docstring, so
its wording is preserved:

```python
class NF4WeightStore(FrozenWeightStore):
    """QLoRA-style blockwise NF4 + double quantization. Deliberately not
    designed in more detail here -- see calibration below. materialize()
    would dequantize to bf16 each call; a real implementation needs a
    genuine decision about caching that dequantized tensor per step vs.
    re-dequantizing per use (a real VRAM/speed tradeoff this design
    doesn't resolve for you)."""
    ...
```

**Calibration.** This is the single most valuable item left in this whole
document -- but still not designed in full, on purpose: dequantizing NF4
on the fly needs either a custom fused dequant-matmul kernel or an
explicit dequantize-then-matmul path with its own `MemoryManager`-backed
scratch-buffer story, real substantial systems work, plus the
diffusion-specific quality caveat above genuinely needs checking against
this project's own real UNet, not assumed to transfer from the LLM
literature. Design and validate `NF4WeightStore` as its own dedicated
effort, scoped like a `nodes/components/` migration with its own
equivalence-testing pass (see the backlog, section 10).

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

## 5. Coordination, registry, and observability

### 5.1 `ResourceCoordinator`: a registry of `DeviceResident`s, offload ordering made explicit

**Implemented**, unchanged from the design: `ResourceCoordinator`
(`nodes/memory/coordinator.py`) -- tracks every `DeviceResident` a run
has constructed via explicit `register()` calls, never reflection, never
a global registry reached for by import; `total_footprint_bytes()` and
`offload_all_except()` cover the "what do I own, offload everything but
these" operations that were otherwise easy to get subtly wrong by hand.
Sequenced correctly, per the original plan: it landed only once
`OptimizerHandle`, `TrainableModel`, and `TextEncoder` were all real,
tested `DeviceResident`s to actually coordinate.

### 5.2 `OffloadOrchestrator`: event-driven, reusing the existing pub/sub shape

The project already had a working, correctly-designed pub/sub mechanism
for cross-cutting concerns: `MonitorBus`/`MonitorHandle`, explicitly
injected rather than a singleton, already documented as safe to call from
a worker thread. Rather than invent a second event system for offload
orchestration, this reused that same shape.

**Implemented**, unchanged from the design: `TrainingLifecycleEvent`
(marker base -- `CacheRebuildStarting`, `PreviewGenerationStarting`,
`CheckpointSaveStarting`) and `OffloadOrchestrator` (subscribes to
events, drives a `ResourceCoordinator` in response) -- both in
`nodes/memory/coordinator.py`.

**Still true, and worth restating exactly as before:** this is the
principled version of what `core/trainer.py`'s hand-written offload calls
still do today, ad hoc, per call site -- for `nodes/`, not a claim that
it retroactively fixes `core/trainer.py`. It's also explicitly not, by
itself, a fix for the still-open "device lost"/hang-after-VRAM-pressure
report in `docs/suspicious_findings.md` -- that report's own leading
hypothesis is a missing explicit `synchronize()` on an async offload
path, a correctness bug this orchestrator's *existence* doesn't fix. It
fixes the *coordination* problem, which is necessary but not sufficient
-- see section 9.3.

### 5.3 `ComponentRegistry`: versioned, side-by-side registration

`server/nodegraph_registry.py` is already a plain name -> class registry,
which is fine for "the graph editor needs to resolve a class name." What's
missing, and what the project's own migration discipline actually needs,
is a way for a `nodes/components/`-style rewrite to be registered
*alongside* the legacy adapter it's replacing, both live, both usable, for
however long the equivalence-testing window takes -- exactly the pattern
`nodes/optimizer/`'s composed nodes vs. legacy-wrapping nodes already
follow *by convention*.

**Not implemented -- still not urgent, but for an updated reason.**
`nodes/components/` is no longer empty -- `diffusion.py`, `device.py`,
and `layout.py` are real, migrated content now (1.4-1.6) -- so the
original justification ("nothing has moved yet") is stale. The actual
current reason this still isn't needed: none of those three are
registered as selectable graph-editor `Node` types at all -- they're
plain constructed objects wired in through ports and constructor
arguments (`diffusion_process` on `SupervisedLoRATrainerNode`,
`project_layout` on the four Nodes that need it), not competing
named implementations a picker has to choose between. `ComponentRegistry`
solves "the graph editor needs to offer X-old and X-new side by side";
nothing built through `nodes/components/` so far needs that, since
nothing built through it is graph-editor-selectable in the first place.
Worth re-examining if a future `nodes/components/` migration *is* exposed
as a selectable `Node` (a `DiffusionProcess`-choosing Node in the graph
editor, say) -- listed for completeness, cheap to add exactly when that
happens.

### 5.4 `TrainingRecipe` / `PipelineFactory`: declarative composition

A value object describing a full run (dataset config, model config,
optimizer config, schedule, resource budget) plus a factory that turns it
into wired, constructed runtime objects -- the Abstract Factory pattern
applied to "build me a whole pipeline," rather than wiring every
`Builder` by hand each time. Not proposed as a replacement for the graph
editor (which has real value as an interactive, inspectable construction
UI) -- a second, programmatic entry point for the exact same underlying
`Builder`/runtime-object model, useful for tests, scripts, and
(longer-term, speculative) as a possible bridge toward driving `nodes/`
from a config file the way `core/` already is, without `nodes/` and
`core/` sharing any code to do it. **Not implemented -- still not
recommended as near-term work**, for the same reason as before: nothing
currently needs it (there's no test harness or script today that
constructs a full `nodes/` pipeline programmatically instead of through
the graph editor), and it depends on `ComponentRegistry` (5.3) existing
first.

```python
@dataclass(frozen=True)
class TrainingRecipe:
    dataset: dict
    model: dict
    optimizer: dict
    schedule: dict
    budget: ResourceBudget


class PipelineFactory:
    def __init__(self, registry: ComponentRegistry):
        self._registry = registry

    def build(self, recipe: TrainingRecipe) -> "TrainingStepPipeline":
        ...  # resolves each section against the registry, constructs,
             # wires Builders in dependency order -- the programmatic
             # equivalent of what GraphExecutor already does for a
             # graph-editor-submitted graph
```

### 5.5 `ResourceProfile`: one aggregate VRAM report

**Not implemented -- still open, and cheaper to build now than when
this was written**, since every `DeviceResident` it would aggregate
(`OptimizerHandle`, `TrainableModel`, `TextEncoder`) is now real:

```python
@dataclass(frozen=True)
class ResourceProfile:
    per_resident_bytes: dict[str, int]
    memory_manager_stats: dict[str, Any]
    allocator_stats: dict[str, float] | None  # from DeviceContext.memory_stats()

    @classmethod
    def capture(cls, coordinator: ResourceCoordinator, memory: "MemoryManager",
                device_ctx: DeviceContext) -> "ResourceProfile":
        return cls(
            per_resident_bytes={name: r.footprint_bytes()
                                 for name, r in coordinator._residents.items()},
            memory_manager_stats=memory.stats(),
            allocator_stats=device_ctx.memory_stats(),
        )
```

Directly generalizes what `profile=True` already reports (allocated/
reserved MB) into a per-component breakdown -- "how much of my VRAM is
the text encoder cache vs. optimizer scratch vs. the model itself," which
the current single allocator-level number can't answer. Real, standing
diagnostic value for the still-open VRAM-pressure investigation in
`docs/suspicious_findings.md` -- see the backlog, section 10.

### 5.6 Concurrency contract, stated explicitly

Stated once, precisely, rather than left implicit (which is how a
`PrefetchingBatchSource` worker thread, 2.5, could otherwise become a
real race someone finds the hard way) -- **this is no longer aspirational:
`nodes/dataset/prefetch.py` cites this section directly as the contract
its worker thread has to honor**:

- **Single-threaded by default.** `StepPhase.run()`, `DeviceResident`
  methods, `MemoryManager` methods, `ResourceCoordinator`/
  `OffloadOrchestrator` methods: none of these are safe to call from more
  than one thread concurrently, and none of them need to be -- a training
  run has exactly one thread driving the step loop.
- **Explicitly cross-thread-safe, by design, documented as such at the
  point of use:** `MonitorHandle.report()` (already true and already
  documented -- called from a FastAPI worker thread today);
  `ExecutionContext`'s cancel signal (already a `threading.Event` for
  exactly this reason); `PrefetchingBatchSource`'s internal queue (its
  *only* job is being a safe hand-off point between its worker thread and
  the training thread -- `queue.Queue` already gives this for free, so
  this isn't new design work, just a contract worth stating).
- Nothing else should grow a background thread without updating this
  list and justifying it the same way.

### 5.7 The Acyclic Domain Dependency Rule

The project's practiced discipline (`optimizer/` doesn't import `model/`;
domain ABCs live in each domain's own `handle.py`), including in
`nodes/components/` now that it's real, not just planned:

> A domain package (`dataset/`, `model/`, `optimizer/`, `train/`,
> `monitor/`, `components/`) may depend downward on `core.py` and
> `memory/` (and, for now, read-only on `core/`/`manager/` per the
> existing wrap-don't-copy rule) -- never sideways on another domain
> package's *implementation*. Cross-domain references go through that
> domain's `handle.py` ABCs only (e.g. `train/` depends on
> `model.handle.TrainableModel`, never on `model.lora_injector`
> directly). A dependency that seems to need to go sideways is a signal
> the shared piece belongs in `core.py`, `memory/`, or a new
> domain-independent module -- not that the rule should bend.

---

## 6. Composition walkthrough: one LoRA run, under this design

Concrete, to make sections 1-5 legible as a whole rather than a list of
classes. **Most of the classes below are real now** (see each section for
the exact file); `PipelineFactory` itself, `ManualResourcePolicy`, and
`ResourceBudget` are not, so this still isn't code that runs as shown --
it's the wiring a real script would do once section 10's remaining items
land, annotated below for which pieces exist today vs. which don't:

```python
layout = ProjectLayout(...)                              # 1.6, real
device_ctx = DeviceContext.for_device("xpu")              # 1.5, real

schedule = RescaledZeroTerminalSNRSchedule()               # 1.4, real class,
process = DiffusionProcess(schedule, VPredParameterization(), KarrasInputScaler())
# real class, but this specific combination is unvalidated -- see 1.4
# DiffusionProcess.__post_init__ rejects EpsParameterization here -- see 1.4

policy = ManualResourcePolicy(                            # 2.2, NOT implemented
    checkpointing=FrozenParamSafeCheckpointing(placement=EveryBlockPlacement()),  # 2.3:
    # FrozenParamSafeCheckpointing is real; EveryBlockPlacement/placement itself is not
    optimizer_strategy=ChunkedScratchBufferStrategy,
    text_encoder_cache=True,
    adapter_strategy=PlainLoRAAdapter(),          # 3.1, real -- DoRAAdapter() NOT implemented
    lora_scaling_policy=RankStabilizedScaling(),  # 3.2, real
    frozen_weight_store=BF16WeightStore,          # 3.3, real -- NF4WeightStore NOT implemented
    parameter_group_policy=UniformGroups(),       # 3.4, real -- LoRAPlusGroups(...) real but unvalidated
)
memory = MemoryManager()                                  # 1.3, unchanged, real
coordinator = ResourceCoordinator()                        # 5.1, real

model = build_trainable_model(weights, policy, device_ctx)   # a Builder; wires
                                                               # FrozenWeightStore +
                                                               # AdapterStrategy + scaling
coordinator.register("model", model)                         # 1.2 DeviceResident, real
optimizer = build_optimizer(model.trainable_parameters(), policy, memory,
                             group_policy=policy.parameter_group_policy())
coordinator.register("optimizer", optimizer)
text_encoder = CachingTextEncoder(build_text_encoder(weights))
coordinator.register("text_encoder", text_encoder)

pipeline = TrainingStepPipeline([                          # 2.1, real
    FetchBatchPhase(prefetching_source),
    EncodeConditioningPhase(text_encoder),
    ForwardPhase(process),
    LossPhase(P2LossWeighting()),                           # 4, real
    BackwardPhase(),
    OptimizerStepPhase(optimizer),
    MonitoringPhase(monitor_handle),
])

orchestrator = OffloadOrchestrator(coordinator, device_ctx)  # 5.2, real
orchestrator.on(CacheRebuildStarting, lambda e, c, d: c.offload_all_except({"model"}))

for step in range(total_steps):
    state = StepState(step=step, batch=None, model=model, device=device)
    state = pipeline.run_step(state)
```

Every object above is independently constructible and independently
testable; nothing is reached for by import; every device-memory owner is
a `DeviceResident` the coordinator actually knows about -- true of the
real pieces today, and the standard the remaining ones (section 10) are
held to as they land.

---

## 7. Deliberately deferred or rejected

Considered, left out on purpose -- listed with the actual reasoning, not
just "future work":

- **`AutoResourcePolicy` (inspect available VRAM, decide strategies
  automatically).** `ResourcePolicy` (2.2) is designed so this is
  *possible* to add later without touching anything else -- but designing
  it *now*, in detail, would mean guessing at a real, hard ML-systems
  heuristic problem (predicting a training step's peak VRAM from param
  counts + batch shape + checkpointing granularity, ahead of actually
  running it) with no real-hardware data to validate against. `nodes/`'s
  own existing rule -- equivalence-test before switching over -- can't be
  followed for a heuristic with nothing to compare it against yet. Left
  as an interface-shaped placeholder, not a designed algorithm.
- **Automatic eviction inside `MemoryManager`.** Already considered and
  rejected once, correctly, in the existing module docstring ("no
  automatic eviction under memory pressure... behavior stays predictable
  rather than depending on runtime memory conditions"). Nothing here
  changes that reasoning; an `OffloadOrchestrator`-driven, *event*-
  triggered offload (5.2) is a different thing from pressure-triggered
  eviction inside the allocator itself, and doesn't need the latter.
- **Layer-wise CPU offload of the frozen UNet base between steps**
  (ZeRO-Infinity-style). Real, large VRAM lever -- not designed here: no
  existing per-block streaming-offload primitive to build on, and the
  PCIe round-trip cost per step is a real, hardware-dependent question
  that needs actual measurement, not an interface guess. `DeviceResident`
  leaves room for a future `LayeredOffload` variant of `TrainableModel`
  without foreclosing it.
- **Continuous-time / flow matching** (Lipman et al., arXiv:2210.02747;
  Liu et al.'s rectified flow, arXiv:2209.03003 -- the formulation Stable
  Diffusion 3 and Flux actually train with). `NoiseSchedule`/
  `Parameterization` (1.4) already leave the seam open (a sibling
  `Interpolant` contract; `convert_to()` already generalizes to a third,
  velocity-target `Parameterization`). Not designed further because it's
  not a drop-in swap for this project's actual model: SDXL is a
  pretrained epsilon/v-prediction model, and converting an already-
  trained diffusion model's *sampling trajectory* into a flow-matching
  one is itself an active, nontrivial research question -- Schusterbauer
  et al.'s "Diff2Flow" (CVPR 2025) exists specifically to do this
  alignment, which wouldn't be a real research topic if it were simple.
- **GaLore** (Zhao et al., arXiv:2403.03507, ICML 2024) -- projects
  gradients into a low-rank subspace so a full-parameter optimizer's
  state costs close to what LoRA's optimizer state already costs, without
  restricting the actual weight updates to a low-rank subspace. Solves a
  problem this project doesn't currently have: it's LoRA-only today, and
  LoRA's optimizer state is already small (proportional to the tiny
  adapter parameter count, not the frozen base) -- the right answer for a
  hypothetical future *full-parameter* fine-tuning mode, not for
  improving on already-cheap LoRA optimizer state.
- **8-bit block-quantized optimizer moments** (Dettmers, Lewis, Shleifer,
  Zettlemoyer, arXiv:2110.02861, 2022 -- what `bitsandbytes`' `Adam8bit`
  implements). `Algorithm.init_state()`'s existing contract already
  returns "a plain dict of named tensors" without mandating fp32 --
  nothing structurally blocks a quantized-state `Algorithm` variant, but
  CAME and Adafactor were already chosen specifically as memory-frugal
  factored optimizers for a LoRA-sized parameter count, so the marginal
  win from further quantizing an already-small state is real but smaller
  than where the actual VRAM mass is (the frozen base, 3.3).
- **A second event bus for `OffloadOrchestrator`.** Reused `MonitorBus`'s
  existing shape (5.2) instead of inventing a parallel one -- two pub/sub
  systems in one codebase for two similar-but-different purposes would be
  duplication, not design.
- **Redesigning `Node`/`Port`/`ExecutionContext`.** Section 1.1 arrived at
  essentially the same shape independently; section 9.1 confirms it.
  Proposing changes to something already correct, just to have proposed
  something, would be the opposite of the "good code is the only metric"
  standard this design is held to.

---

## 8. Note on precedent and validation

A few small things worth naming because they're evidence the design holds
together, not just assertions that it does: `LossWeighting` (section 4)
accepted a second implementation (`P2LossWeighting`) with zero interface
change. `Parameterization`'s `convert_to()` (1.4) already generalizes to
a third, velocity-target implementer without modification. `Algorithm`'s
`init_state()` (`nodes/optimizer/`) already permits a non-fp32,
non-plain-tensor state representation without needing a new method.
`AdapterStrategy` and `FrozenWeightStore` (3.1, 3.3) turned out to be
genuinely orthogonal axes, confirmed by QDoRA existing in the published
literature as the combination of both. None of these were required to
hold -- each is a place the design could have needed a revision it didn't
turn out to need.

**A stronger form of the same evidence exists now that isn't just
about interface stability under paper study: every piece in section 9.1
below actually got built, real-hardware-adjacent, equivalence-tested
against the exact behavior it replaced, and landed without needing a
design revision along the way.** That's a different, harder bar than
"the interfaces look right on paper" -- it's "the interfaces were right
when actual code had to satisfy them."

---

## 9. Implementation status: this design vs. current `nodes/`

### 9.1 What's implemented -- no further action needed

Everything below is real, tested code, not illustrative Python.
Everything the original section 9.1 table listed as "already matched
independently" is included here too, since the distinction between
"matched before this design started" and "built because of it" doesn't
matter anymore -- both are equally done.

| Design piece | Real `nodes/` location | Status |
|---|---|---|
| `Builder`/`Port` (1.1) | `nodes/core.py`'s `Node`/`Port` | Pre-existing, arrived at independently -- confirmed, not changed. |
| `Algorithm` x `ExecutionStrategy` x `Handle` composition (referenced throughout) | `nodes/optimizer/` in full | Pre-existing reference implementation -- this design's generalization target, not a gap. |
| `DeviceResident` (1.2) | `nodes/memory/handle.py`, conformed to by `OptimizerHandle`, `TrainableModel`, `TextEncoder` | Backlog items 3, 9, 12. |
| Pooled device buffers (1.3) | `nodes/memory/manager.py`'s `MemoryManager` | Pre-existing, unchanged interface; adoption breadth closed by the `DeviceResident` rollout above. |
| `NoiseSchedule`/`Parameterization`/`DiffusionProcess`/`DeviceContext` (1.4, 1.5) | `nodes/components/diffusion.py`, `nodes/components/device.py` | Backlog items 1-2. |
| `ProjectLayout` (1.6) | `nodes/components/layout.py` | Backlog item 8. Bridging period still open -- see 1.6. |
| `TrainingStepPipeline`/`StepPhase` (2.1) | `nodes/train/step_pipeline.py` | Backlog item 10. |
| `ActivationCheckpointingStrategy` (2.3) | `nodes/model/gradient_checkpointing.py` | Backlog item 7. `CheckpointPlacementPolicy` itself still open -- see 9.2. |
| Text encoder cache as `DeviceResident` (2.4) | `nodes/model/text_encoder.py`, `nodes/model/text_encoder_cache.py` | Landed as part of item 12. |
| `PrefetchingBatchSource` (2.5) | `nodes/dataset/prefetch.py` | Backlog item 11. |
| `AdapterStrategy`/`PlainLoRAAdapter`, `LoRAScalingPolicy` (3.1, 3.2) | `nodes/model/adapter_strategy.py`, `nodes/model/lora_injector.py` | Backlog item 9 (part 2), item 5. `DoRAAdapter` itself still open -- see 9.2. |
| `FrozenWeightStore`/`BF16WeightStore` (3.3) | `nodes/model/frozen_weight_store.py` | Backlog item 9 (part 1). `NF4WeightStore` itself still open -- see 9.2. |
| `ParameterGroupPolicy`, `LoRAPlusGroups` (3.4) | `nodes/optimizer/composed.py` | Backlog item 4. `LoRAPlusGroups` real but unvalidated -- see 9.2. |
| `LossWeighting`/`LRSchedule` (section 4) | `nodes/train/loss.py`/`schedule.py` | Pre-existing clean ABCs, confirmed by `P2LossWeighting` needing zero interface change; v-pred branch + `P2LossWeighting` are backlog item 6. |
| `Algorithm.init_state()`'s state representation | `nodes/optimizer/algorithms/*.py` | Pre-existing -- contract already returns "a plain dict of named tensors," not specifically fp32; nothing structurally blocks a future quantized-state `Algorithm`. |
| `ResourceCoordinator`/`OffloadOrchestrator` (5.1, 5.2) | `nodes/memory/coordinator.py` | Backlog item 12. Doesn't by itself fix the still-open VRAM-hang report -- see 9.3. |
| Injected pub/sub, not a singleton bus (5.2's reasoning) | `nodes/monitor/`'s `MonitorHandle`/`LiveMonitorHandle` | Pre-existing house reference for "no singleton" done right -- explicitly reused, not redesigned, for `OffloadOrchestrator`. |
| Decorator-wrapped `TrainingBatchSource` (2.5) | `nodes/dataset/renoise.py`'s `RenoiseBatchSource` | Pre-existing pattern `PrefetchingBatchSource` followed. |
| Composition-over-mutation for stacked state (1.2) | `nodes/model/lora_phases.py`'s `LoRAGeneration` | Pre-existing, independent instance of the same principle `DeviceResident`'s offload-vs-free distinction is built on. |
| `server/graph_executor.py` | -- | Pre-existing, already matches this design's construction-time model closely: real topological execution, real `issubclass()`-based port compatibility checking, explicit `ExecutionContext` threading. No changes recommended. |

### 9.2 What's still missing, partial, or unvalidated

Everything below is real -- these are the only pieces of this document
still asking for something. See section 10 for the ordered plan.

**`ResourceBudget`/`ResourcePolicy` (2.2).** Doesn't exist even as a
manual-only shim. Every VRAM/quality-affecting choice today is an
independent port on an independent node -- more of them now than before
(`ComfyUNetLoRANode.use_checkpoint`, `ComposedCAMEOptimizerNode`'s
`strategy`, `CachingTextEncoderNode`'s presence/absence, LoRA's
adapter/scaling/weight-store/parameter-group choices, and now
`diffusion_process`, `gate_enabled`). Consolidating these into one
`ManualResourcePolicy` object is a real but low-risk change (it can wrap
the existing independent choices without changing any of their current
behavior) -- worth doing now more than ever, since there are more flags
to consolidate than when this was first written.

**`DoRAAdapter` (3.1).** Doesn't exist. The seam it needs
(`AdapterStrategy`, with a real second conformance already checked
against it) does now, which it didn't before -- this is buildable in a
way it wasn't before item 9 landed. Real quality win, bounded new
forward-pass code (weight normalization + magnitude scaling), its own
equivalence/quality-testing pass, additive to `PlainLoRAAdapter`.

**`NF4WeightStore` (3.3).** Doesn't exist. Still this document's single
highest-value remaining item -- needs a real dequantization
implementation (a fused kernel or a `MemoryManager`-backed
dequant-then-matmul scratch-buffer story) and verification against this
project's own real UNet, not assumed from the LLM literature. Its own
dedicated effort.

**`CheckpointPlacementPolicy`/`GreedyRatioPlacement` (2.3).** Doesn't
exist. Blocked on the same thing it always was: real per-block
activation/recompute measurements, which don't exist yet (extending
`profile=True` to block granularity is its own instrumentation task,
separate from the policy class itself).

**`RescaledZeroTerminalSNRSchedule` real end-to-end validation (1.4).**
The class itself is implemented and wired (unlike everything else in
this list) -- what's missing is a real training run with
`VPredParameterization` and qualitative image-quality evaluation. Unlike
every closed item, there's no old code path to equivalence-test against;
this needs real training runs to trust, not a unit test.

**`LoRAPlusGroups` real tuning (3.4).** The class itself is implemented
(unlike everything else in this list) -- what's missing is actually
running a LoRA training job with it wired in and comparing against a
`UniformGroups` baseline, at whatever `ratio` turns out to matter for
this project's own data.

**`ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`/`ResourceProfile`
(5.3, 5.4, 5.5).** None exist. `ComponentRegistry`/`TrainingRecipe` are
still not recommended as near-term work (see 5.3, 5.4 for the current
reasoning -- unlike when this was written, `nodes/components/` now has
real content, but nothing in it is graph-editor-selectable, so the
side-by-side-registration problem these solve still hasn't materialized).
`ResourceProfile` is smaller and now cheaper to build than when this was
written, since the `DeviceResident`s it would aggregate all exist --
real, standing diagnostic value for the VRAM-pressure investigation in
`docs/suspicious_findings.md`.

### 9.3 What's explicitly out of scope

`core/trainer.py` and the rest of `core/`/`manager/` are the production
path, reference material only, untouched by this design -- exactly the
existing project rule. The VRAM-pressure hang/device-lost report in
`docs/suspicious_findings.md` lives there today; this design's
`OffloadOrchestrator` is the *eventual*, principled home for that class
of coordination problem once `nodes/` is the production path, not a
claim that building it retroactively fixes `core/trainer.py`'s current
hand-rolled offload logic.

---

## 10. Prioritized backlog

**The original 12-item backlog is entirely complete** -- `DiffusionProcess`/
`DeviceContext`, `DeviceResident` (with `OptimizerHandle`/`TrainableModel`/
`TextEncoder` all conforming), the `ParameterGroupPolicy` fix,
`LoRAScalingPolicy`, Min-SNR's v-prediction branch + `P2LossWeighting`,
`ActivationCheckpointingStrategy`, `ProjectLayout`, the
`AdapterStrategy`/`FrozenWeightStore` seam, `TrainingStepPipeline`/
`StepPhase`, `PrefetchingBatchSource`, and `ResourceCoordinator`/
`OffloadOrchestrator` are all real, tested code -- see section 9.1 for
exactly where each one lives. What follows is a fresh list: only what's
actually still open, ordered by what unblocks what, sized to be
independently landable slices, each one equivalence-tested against
whatever it replaces (or, for the two validation-only items at the end,
tested by a real training run instead) before anything switches over to
it.

1. **`ResourceBudget`/`ResourcePolicy`/`ManualResourcePolicy`** (2.2).
   Doesn't exist even as a manual-only shim, and every seam built since
   this backlog was first written (adapter strategy, scaling policy,
   weight store, parameter group policy, plus `diffusion_process` and
   `gate_enabled`) is one more independent flag it would consolidate --
   there are more of them now than when this item was last considered,
   not fewer. Low risk: `ManualResourcePolicy` can wrap the existing
   independent choices without changing any of their current behavior.
   Sequenced first because everything else on this list (`DoRAAdapter`,
   `NF4WeightStore`, `CheckpointPlacementPolicy`) is a new choice this
   object would need a slot for -- landing it first means those don't
   each need their own ad hoc wiring into `SupervisedLoRATrainerNode`'s
   ports.
2. **`ResourceProfile`** (5.5). Small, and cheaper now than it would have
   been earlier -- every `DeviceResident` it aggregates
   (`OptimizerHandle`, `TrainableModel`, `TextEncoder`) already exists and
   is exercised in real runs. Real, standing diagnostic value for the
   still-open VRAM-pressure investigation in `docs/suspicious_findings.md`:
   "how much of my VRAM is the text encoder cache vs. optimizer scratch
   vs. the model itself" isn't answerable from the current single
   allocator-level number.
3. **Per-block profiling instrumentation, then `CheckpointPlacementPolicy`/
   `GreedyRatioPlacement`** (2.3). The actual blocker has always been the
   instrumentation, not the policy class -- extending `profile=True` (or
   its `nodes/`-native successor) to real per-block activation/recompute
   measurement is its own task, separate from and prior to the policy
   itself. `EveryBlockPlacement` stays the safe default until real
   `BlockCost` numbers exist; shipping `GreedyRatioPlacement` with guessed
   costs would be worse than not having it.
4. **`DoRAAdapter`** (3.1). Buildable now in a way it wasn't before item 9
   of the old backlog landed -- the seam (`AdapterStrategy`, with
   `PlainLoRAAdapter` as a real, tested conformance already checked
   against it) exists. Its own equivalence/quality-comparison pass, not
   just an equivalence test (it's a real quality claim, not a refactor).
5. **`NF4WeightStore`** (3.3). This document's single highest-value
   remaining item, and still its own dedicated effort, not a slice of
   anything else: needs a real dequantization implementation (a fused
   dequant-matmul kernel or an explicit dequantize-then-matmul path via
   `MemoryManager`) and verification against this project's actual UNet
   specifically, not assumed from the LLM literature -- the
   diffusion-specific quality caveat in 3.3 needs checking directly, not
   inherited from QLoRA's own LLM benchmarks. Sequenced after
   `DoRAAdapter` since the two are confirmed-compatible in published work
   (QDoRA) and `DoRAAdapter` is the smaller, faster piece to land first.

**Validation work, not construction -- real code exists for both, what's
missing is a real run:**

- **`RescaledZeroTerminalSNRSchedule` + `VPredParameterization`, end to
  end** (1.4). The classes are implemented and `DiffusionProcess` already
  rejects the unsound eps-prediction pairing at construction. What's
  missing is a real training run and qualitative image-quality evaluation
  -- does it actually fix medium-brightness clustering on this project's
  own data. Unlike everything numbered above, there's no old code path to
  equivalence-test against; this needs real training runs to trust, not a
  unit test.
- **`LoRAPlusGroups`, actually run** (3.4). Opting in is already a
  one-line `parameter_group_policy=LoRAPlusGroups(...)` change -- what's
  missing is running it on a real LoRA training job and comparing against
  a `UniformGroups` baseline, at whatever `ratio` (the `16.0` default is
  an unverified starting point) turns out to matter for this project's
  own data.

**Not recommended as near-term work, with reasoning kept where it's
argued in full:** `ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`
(5.3, 5.4 -- nothing built through `nodes/components/` so far is
graph-editor-selectable, so the side-by-side-registration problem these
solve hasn't materialized). **Deliberately deferred or rejected, for the
reasons section 7 gives in full:** `AutoResourcePolicy`, automatic
eviction inside `MemoryManager`, layer-wise base offload, flow matching,
GaLore, 8-bit optimizer moments.
