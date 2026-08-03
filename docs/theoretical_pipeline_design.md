# A from-scratch design for the training pipeline, then a gap analysis against `nodes/`

Requested directly: design a training pipeline as if `nodes/` didn't exist --
VRAM-savings-first, strict OOP, real composition, no singletons -- iterate on
it three times (each iteration is a full upgrade pass over the previous one,
not an alternative), *then* and only then compare it to what `nodes/`
currently is. Sections 1-3 below don't reference `nodes/`'s actual classes on
purpose, per that ordering. Section 4 (gap analysis) is where the comparison
happens.

This is a planning document, not implemented code. Every class shown is
illustrative Python (real syntax, real signatures, meant to be directly
usable as a starting point) -- none of it has been written as `.py` files or
tested. Treat interface details as a strong proposal, not a spec set in
stone; treat the architectural shape (what talks to what, what owns what
lifecycle) as the actual deliverable.

## Design goals and constraints

Stated once here, referenced rather than repeated through the three
iterations:

1. **VRAM first, speed second -- but not speed-blind.** Every VRAM-saving
   choice below either has no speed cost (pure refactor) or a named,
   estimable one (e.g. activation checkpointing's ~20-30% step-time cost,
   already measured in `docs/vram_and_lora_phase_split.md`). Nothing here
   trades speed for VRAM silently or trades VRAM for a *bigger* speed cost
   than the current codebase already accepts, without saying so.
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
   forward (see section 4).
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
   practice. Section 3.7 lists what was considered and deliberately left
   out, with the reasoning, the same way `docs/vram_and_lora_phase_split.md`
   documents "Considered, not implemented."

---

## Iteration 1 -- foundational ontology

The base vocabulary: what kinds of objects exist, what each owns, and how
they're wired together at construction time vs. what they do at runtime.
This iteration deliberately doesn't yet solve VRAM-specific orchestration
problems (that's iteration 2) -- it solves "what are the nouns," which
everything else depends on getting right first.

### 1.1 Two different lifetimes, two different kinds of object

Everything in a training pipeline is either:

- A **builder**: takes configuration, produces a runtime object. Exists only
  during graph construction. Stateless with respect to training itself.
- A **runtime object**: the thing training actually calls methods on, every
  step, for the life of a run. Has real state (weights, optimizer moments,
  a cursor into a dataset).

Collapsing these two into one class is exactly what makes old-style
"trainer with 40 constructor args and 40 methods" code hard to test and
hard to extend -- a builder's only job is turning config into a runtime
object, so it can be swapped, mocked, or parameterized without the runtime
object's own logic ever being touched.

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
before looking, is exactly the check this design process is for. See
section 4.1 for the actual comparison; the short version is "this part is
already right, keep it."

### 1.2 Runtime lifecycle: `DeviceResident`

Every runtime object that can hold device memory needs the same three
questions answerable, regardless of domain (optimizer, model, text
encoder, dataset prefetch buffer): how big is it right now, can it be
moved off-device without losing its identity, and can it be dropped
entirely. Today each domain answers this with its own ad hoc method names
(an optimizer has `offload_states_to_cpu`/`free_states`; a model has
`to()`; a text encoder has `unload()`) -- fine individually, but it means
nothing generic can coordinate across all of them, which is exactly the
gap behind the still-open "hang after VRAM pressure" report in
`docs/suspicious_findings.md`: nobody owns the *order* in which things get
offloaded and reloaded, because nothing exposes a uniform enough contract
for an orchestrator to drive.

```python
class DeviceResident(ABC):
    """Something that holds device memory as part of its normal operation.
    Three lifecycle tiers, kept distinct on purpose -- collapsing them is
    exactly the mistake nodes/memory/manager.py's module docstring already
    documents once (the reset-vs-free asymmetry bug class)."""

    @abstractmethod
    def footprint_bytes(self) -> int:
        """Best-effort current device-memory usage. Best-effort, not exact
        -- a model wrapping third-party layers may not be able to account
        for every buffer; document what's excluded rather than guessing."""

    @abstractmethod
    def offload(self) -> None:
        """Move to host memory. The object stays alive and identity-stable
        (same Python object, same optimizer momentum, same cache contents)
        -- this is the cheap, common, reversible operation."""

    @abstractmethod
    def reload(self, device: str | None = None) -> None:
        """Move back to device. None = wherever it was before offload()."""

    @abstractmethod
    def release(self) -> None:
        """Drop device (and possibly host) state entirely. Not reversible
        via reload() -- whatever built this object has to build it again.
        Used when a run is actually discarding something, not pausing it."""
```

`OptimizerHandle` (already exists, already correct) becomes a
`DeviceResident` by extension, not by rewrite -- its five existing
lifecycle methods map onto these three almost exactly
(`offload_states_to_cpu`/`reload_states_to_device` -> `offload`/`reload`;
`free_states` -> `release`; `decay_states`/`reset_states` stay
optimizer-specific extras beyond the universal contract, which is fine --
`DeviceResident` is a floor, not a ceiling). `TrainableModel`,
`TextEncoder`, and (iteration 2) a dataset prefetch buffer each get a thin
`DeviceResident` conformance the same way. See section 4 for exactly what
that retrofit looks like on the real classes.

### 1.3 Pooled device buffers stay a separate, lower-level concern

`DeviceResident` is object-granularity ("offload this whole optimizer").
Underneath any one `DeviceResident`, there's often a need for
finer-granularity, reusable scratch buffers ("give me 4MB of float32
scratch, reuse it next step too") -- a different concern, already solved
correctly: a tag-keyed pool that grows lazily, never shrinks, and
distinguishes *released* (available for reuse, allocation kept) from
*freed* (allocation actually dropped). This is precisely
`nodes/memory/manager.py`'s `MemoryManager`, and this design reuses it
unchanged -- see section 4.2 for why no interface change is being
proposed here, only a widened set of callers.

The relationship between the two: a `DeviceResident.release()`
implementation that owns pooled buffers is responsible for also calling
`MemoryManager.free()`/`free_all()` on whatever it acquired -- exactly the
pattern `ChunkedScratchBufferStrategy.free_extra()` already establishes.
`DeviceResident` doesn't replace `MemoryManager`; it's the object-level
contract that sits on top of it and on top of anything else a runtime
object owns (a model's parameters, an LRU cache's tensors) that isn't
itself a pooled scratch buffer.

### 1.4 Diffusion math as objects, not module globals

`core/noise_schedule.py` computes `ALPHA_T, SIGMA_T = make_schedule()` at
*import time*, as module-level tensors, with hardcoded default
`beta_start`/`beta_end`. Every caller reaches for these two globals by
importing the module. This is the concrete case the "no singletons" rule
is written against: two independent training runs with different noise
schedules can't coexist in one process, the values can't be constructed
with different parameters without monkeypatching the module, and nothing
about the dependency is visible in any constructor signature -- it's
invisible until something imports `core.noise_schedule` three call frames
deep.

The fix is straightforward: an object, constructed explicitly, holding its
own tensors.

```python
class NoiseSchedule(ABC):
    @abstractmethod
    def alpha_sigma(self, t) -> tuple["Tensor", "Tensor"]:
        """(alpha, sigma) for timestep index/indices t. Accepts int or
        Tensor, same as the free function it replaces."""


class DiscreteLinearNoiseSchedule(NoiseSchedule):
    """Matches ComfyUI's ModelSamplingDiscrete -- same math as
    core.noise_schedule.make_schedule(), moved into an instance instead of
    a module global. Precomputes alpha_t/sigma_t once, on CPU, at
    construction; caches a per-device copy lazily on first use rather than
    mutating a shared global (so a second schedule, or the same schedule
    used from two devices in one process, never step on each other)."""

    def __init__(self, n: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012):
        self._alpha_cpu, self._sigma_cpu = self._compute(n, beta_start, beta_end)
        self._device_cache: dict[str, tuple] = {}

    @staticmethod
    def _compute(n, beta_start, beta_end):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        alpha_t = alphas_cumprod.sqrt()
        sigma_t = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return alpha_t, sigma_t

    def alpha_sigma(self, t):
        if isinstance(t, torch.Tensor) and t.device.type != "cpu":
            key = str(t.device)
            if key not in self._device_cache:
                self._device_cache[key] = (
                    self._alpha_cpu.to(t.device), self._sigma_cpu.to(t.device))
            a_dev, s_dev = self._device_cache[key]
            return a_dev[t], s_dev[t]
        return self._alpha_cpu[t], self._sigma_cpu[t]
```

Prediction-type conversion (`core/model_io.py`'s `raw_to_x0`/`raw_to_target`,
`core/noise_schedule.py`'s `eps_to_vpred`/`vpred_to_eps`/etc.) becomes a
small `Parameterization` Strategy pair instead of a four-way branch of free
functions:

```python
class Parameterization(ABC):
    @abstractmethod
    def to_x0(self, raw, x_t, alpha, sigma) -> "Tensor":
        ...

    @abstractmethod
    def convert_to(self, raw, x_t, alpha, sigma, target: "Parameterization") -> "Tensor":
        """raw, expressed in this parameterization, converted to what
        `target`'s parameterization would have predicted for the same
        (x_t, alpha, sigma). Same-type conversion (isinstance(target,
        type(self))) is the identity -- a caller doesn't need to
        special-case 'both sides already agree' itself."""


class EpsParameterization(Parameterization):
    def to_x0(self, raw, x_t, alpha, sigma):
        return x_t - sigma * raw

    def convert_to(self, raw, x_t, alpha, sigma, target):
        if isinstance(target, EpsParameterization):
            return raw
        denom_sqrt = torch.sqrt(sigma ** 2 + 1.0)
        return (raw * (sigma ** 2 + 1.0) - x_t * sigma) / denom_sqrt


class VPredParameterization(Parameterization):
    def to_x0(self, raw, x_t, alpha, sigma):
        denom = sigma ** 2 + 1.0
        return x_t / denom - raw * sigma / torch.sqrt(denom)

    def convert_to(self, raw, x_t, alpha, sigma, target):
        if isinstance(target, VPredParameterization):
            return raw
        denom = sigma ** 2 + 1.0
        return x_t * sigma / denom + raw / torch.sqrt(denom)
```

And the third piece, `core/model_io.py`'s `comfy_input_transform`, becomes
a matching one-method `ModelInputTransform` Strategy. All three compose
into one object a trainer actually depends on:

```python
@dataclass(frozen=True)
class DiffusionProcess:
    """Everything a training step needs to know about the forward
    diffusion process and the model's I/O convention, as one injected
    dependency -- not three free-function module imports reached for
    individually inside a step loop (the current supervised.py._run_step
    shape; see section 4)."""
    schedule: NoiseSchedule
    parameterization: Parameterization
    input_transform: "ModelInputTransform"
```

### 1.5 Device backend as a Strategy, not `hasattr()` calls

`core/comfy_setup.py`'s `xpu_empty_cache`/`xpu_synchronize`/
`xpu_memory_stats` each independently check `hasattr(torch, "xpu") and
torch.xpu.is_available()`. Not a singleton in the mutable-global sense,
but the same category of problem as one: backend selection logic is
duplicated at every call site instead of decided once and injected.

```python
class DeviceContext(ABC):
    @abstractmethod
    def empty_cache(self) -> None: ...
    @abstractmethod
    def synchronize(self) -> None: ...
    @abstractmethod
    def memory_stats(self) -> dict[str, float] | None: ...

    @staticmethod
    def for_device(device) -> "DeviceContext":
        """Factory, called once at pipeline-construction time -- not a
        singleton lookup at every call site. The returned object is then
        passed around like anything else this design injects explicitly."""
        d = str(device)
        if d.startswith("xpu"):
            return _XPUDeviceContext()
        if d.startswith("cuda"):
            return _CUDADeviceContext()
        return _NullDeviceContext()


class _XPUDeviceContext(DeviceContext):
    def empty_cache(self) -> None:
        torch.xpu.empty_cache()
    def synchronize(self) -> None:
        torch.xpu.synchronize()
    def memory_stats(self) -> dict[str, float] | None:
        return {"allocated_mb": torch.xpu.memory_allocated() / 2**20,
                "reserved_mb": torch.xpu.memory_reserved() / 2**20}


class _NullDeviceContext(DeviceContext):
    """CPU, or any backend without a cache/sync/stats concept. Every
    method is a correct, cheap no-op -- callers don't need an `if device
    supports this` branch of their own."""
    def empty_cache(self) -> None: pass
    def synchronize(self) -> None: pass
    def memory_stats(self) -> dict[str, float] | None: return None
```

### 1.6 Configuration as an injected value object, not a mutable module global

`paths.py` is the other concrete singleton in the current codebase:
module-level `_comfy_dir_override`/`_checkpoints_dir_override`/
`_loras_dir_override`, mutated via `set_comfy_dir()`/`set_checkpoints_dir()`/
`set_loras_dir()`, read via `get_*()` functions that any file can call from
anywhere. This *works*, but it means "what directory does this run
against" is process-global state, set once by whoever calls the setters
first (`server/config.py`, in the current codebase) and silently shared by
everything else that happens to run in the same process.

The fix is not "delete configurability" -- a server genuinely does need
one, coherent, server-wide answer to "where are checkpoints" for its file
pickers. The fix is that this should be **one constructed value object**,
built once (at server/process startup) and threaded through explicitly
(via the execution context, or as a Port on any node that needs it) --
never a module reached into from arbitrary call sites, and never mutated
after construction.

```python
@dataclass(frozen=True)
class ProjectLayout:
    """Resolved, immutable set of directories for one server/process
    lifetime. Constructed once, from environment/config, at startup --
    not a module-global mutated by whichever setter runs first."""
    comfy_dir: "Path"
    checkpoints_dir: "Path"
    loras_dir: "Path"
    datasets_dir: "Path"
    runs_dir: "Path"

    def resolve_model_path(self, path_str: str, kind: str) -> "Path":
        ...

    def resolve_safe_model_path(self, relative_str: str, kind: str) -> "Path":
        """Sandboxed variant for untrusted input (the graph editor,
        reachable over the network) -- same distinction paths.py's
        resolve_model_path/resolve_safe_model_path already draw, kept
        here as two methods on one object instead of two free functions
        plus a private helper."""
        ...
```

This is a deliberate, narrow exception to "no singletons," stated
precisely so it isn't confused with the pattern being removed: **one
long-lived configuration object, explicitly constructed and explicitly
passed down, is not the same thing as a mutable module-level global
reached for by import.** The difference is visibility (every consumer's
constructor/method signature shows the dependency) and mutability (built
once, read-only afterward) -- both of which `paths.py`'s current shape
lacks and this shape has by construction.

### 1.7 What iteration 1 leaves unresolved

This iteration gives every domain a consistent runtime-lifecycle contract
and gets rid of the two concrete singletons, but it doesn't yet say
anything about:

- How a training *step* is actually structured, beyond "some runtime
  objects exist" -- today's step loop is one large method; nothing above
  addresses that.
- How VRAM budget gets *decided* (which optimizer strategy, whether
  checkpointing is on, how deep a prefetch queue is) -- today these are
  manual per-node flags, which is fine but not yet composable into a
  single coherent policy.
- How multiple `DeviceResident`s get offloaded/reloaded *together*, in the
  right order, at the right lifecycle points -- `DeviceResident` gives
  every object the right *verbs*, but nothing yet calls them in a
  coordinated way.

Iteration 2 addresses these three gaps directly.

---

## Iteration 2 -- orchestration and VRAM-focused composition

### 2.1 The step loop as a pipeline of phases, not one method

A training step is a fixed sequence of concerns -- fetch a batch, encode
conditioning, run the forward pass, compute loss, backward, apply the
optimizer update, report progress -- each with a genuinely different
reason to change (a new conditioning scheme, a new loss weighting, a new
optimizer family) but currently living as sequential code inside one
method. That shape is exactly what makes each of the "explicit v1 scope
reduction" items (CFG dual-pass, gradient accumulation, resume cadence)
require editing the same method rather than adding something next to it.

```python
@dataclass
class StepState:
    """Mutable, passed through every phase in sequence -- generalizes
    today's per-step local variables (x_t, target, ctx_emb, pred, loss,
    timing dict) into one object phases read from and write to, instead
    of a method-local variable soup only one function can see all of."""
    step: int
    batch: dict
    model: "TrainableModel"
    device: Any
    extras: dict[str, Any] = field(default_factory=dict)  # loss, pred, timing, etc.


class StepPhase(ABC):
    @abstractmethod
    def run(self, state: StepState) -> StepState:
        ...


class TrainingStepPipeline:
    """An ordered list of StepPhases. Owns nothing itself beyond that
    list -- adding CFG dual-pass, grad accumulation, or DAgger chain-
    mixing later is 'construct one more phase, insert it in the list',
    not 'edit the method that does everything'."""

    def __init__(self, phases: list[StepPhase]):
        self.phases = phases

    def run_step(self, state: StepState) -> StepState:
        for phase in self.phases:
            state = phase.run(state)
        return state
```

Concrete phases map directly onto what `SupervisedLoRATrainerNode._run_step`
already does, just separated: `FetchBatchPhase`, `EncodeConditioningPhase`
(wraps a `TextEncoder`; a CFG-dual-pass variant is a second class
implementing the same `StepPhase` contract, not a branch inside this one),
`ForwardPhase`, `LossPhase` (wraps `LossWeighting`), `BackwardPhase`,
`OptimizerStepPhase` (already has to branch on fused vs. non-fused --
that branch stays, it's genuine, but it's now the *entire* content of one
small class instead of interleaved with five other concerns),
`MonitoringPhase`.

Profiling (today: `profile: bool`, manually wrapping five points with
`xpu_synchronize()` + `perf_counter()`) becomes one generic decorator
applied uniformly instead of instrumentation duplicated five times:

```python
class TimedPhase(StepPhase):
    """Wraps any StepPhase, records wall time (and, if a DeviceContext is
    given, a real synchronize() before timing -- same correctness
    requirement the current profile=True feature already documented:
    async dispatch means an untimed op can finish after the Python call
    that launched it returns)."""

    def __init__(self, inner: StepPhase, device_ctx: DeviceContext, label: str):
        self.inner, self.device_ctx, self.label = inner, device_ctx, label

    def run(self, state: StepState) -> StepState:
        self.device_ctx.synchronize()
        t0 = time.perf_counter()
        state = self.inner.run(state)
        self.device_ctx.synchronize()
        state.extras.setdefault("timing_ms", {})[self.label] = (time.perf_counter() - t0) * 1000
        return state
```

Turning profiling on/off is then "wrap every phase in `TimedPhase` or
don't," decided once where the pipeline is assembled -- not a `profile:
bool` parameter threaded through every phase's own logic.

Gradient accumulation becomes a decorator around `OptimizerStepPhase`
(only actually calls the wrapped phase every Nth invocation) instead of a
documented gap -- genuinely additive, not a redesign, which is exactly
why it's worth restructuring the loop this way now rather than leaving it
until grad accumulation is actually needed and has to be retrofitted into
a monolith.

### 2.2 Resource budget as a first-class value, resource policy as a Strategy

Today, "should this run use activation checkpointing," "which
`ExecutionStrategy` should the optimizer use," "should the text encoder
cache be warmed" are each an independent manual flag on an independent
node. That's fine as a default (explicit is better than magic), but there's
no single place that represents "how much VRAM headroom does this run
actually have" as a value other components could consult if asked to.

```python
@dataclass(frozen=True)
class ResourceBudget:
    """A stated VRAM ceiling for one run, plus a safety margin. Purely
    descriptive -- constructing this doesn't enforce anything by itself;
    a ResourcePolicy is what turns it into actual choices (see below)."""
    vram_budget_mb: float
    vram_reserve_mb: float = 512.0  # headroom kept free on purpose


class ResourcePolicy(ABC):
    """Decides the VRAM/speed-affecting choices a run needs to make.
    Returns *descriptions* of what to build (which ExecutionStrategy
    class, which ActivationCheckpointingStrategy instance), not the
    built objects themselves -- keeps this a pure decision, testable
    without constructing real models/optimizers."""

    @abstractmethod
    def checkpointing_strategy(self) -> "ActivationCheckpointingStrategy":
        ...

    @abstractmethod
    def optimizer_execution_strategy(self) -> type:
        ...

    @abstractmethod
    def enable_text_encoder_cache(self) -> bool:
        ...


class ManualResourcePolicy(ResourcePolicy):
    """Today's actual behavior, made explicit: every choice is a
    constructor argument, no inspection of budget/hardware at all. This
    stays the default -- see section 3's note on why an
    inspect-and-decide AutoResourcePolicy is deliberately not designed
    in detail here."""

    def __init__(self, checkpointing: "ActivationCheckpointingStrategy",
                 optimizer_strategy: type, text_encoder_cache: bool):
        self._checkpointing = checkpointing
        self._optimizer_strategy = optimizer_strategy
        self._text_encoder_cache = text_encoder_cache

    def checkpointing_strategy(self):
        return self._checkpointing

    def optimizer_execution_strategy(self):
        return self._optimizer_strategy

    def enable_text_encoder_cache(self):
        return self._text_encoder_cache
```

The payoff of introducing `ResourcePolicy` even with only a manual
implementation: every VRAM-affecting choice now has *one type* a future
automatic policy could implement against, instead of being four unrelated
boolean/enum ports on four unrelated node classes with no shared shape.
That future policy is explicitly not designed here -- see section 3.7.

### 2.3 Activation checkpointing as a Strategy object

The current fix (`nodes/model/gradient_checkpointing.py`) is *correct*:
filter `ctx.input_params` to `requires_grad=True` entries before
`torch.autograd.grad()`, reconstruct the full gradient tuple with `None`
at frozen positions. What's missing is that it's exposed as a global,
process-wide monkeypatch triggered by a `bool` port
(`enable_frozen_param_safe_checkpointing()`, called once, mutating
ComfyUI's own `CheckpointFunction` class for the rest of the process) --
correct, but not itself an object another piece of code can compose with
or substitute.

```python
class ActivationCheckpointingStrategy(ABC):
    @abstractmethod
    def apply(self) -> None:
        """Install whatever's needed (a monkeypatch, a wrapper) before
        the model is built. Idempotent -- calling twice is a no-op,
        exactly like the current implementation already guarantees."""


class NoCheckpointing(ActivationCheckpointingStrategy):
    def apply(self) -> None:
        pass  # explicit "did nothing", not "wasn't asked"


class FrozenParamSafeCheckpointing(ActivationCheckpointingStrategy):
    """The current, verified fix -- filtering ctx.input_params by
    requires_grad before torch.autograd.grad(), reconstructing the full
    gradient tuple with None at frozen positions. Same mechanism, now an
    object with an apply() method instead of a bare function called from
    inside ComfyUNetLoRANode.build()."""
    def apply(self) -> None:
        ...  # unchanged monkeypatch body, moved here verbatim
```

Nothing about the underlying technique changes -- this is purely "give the
existing, correct fix a shape that composes," which is what makes it
possible to add a coarser-grained variant later (checkpoint every Nth
block instead of every block, trading less VRAM savings for less recompute
cost) as a second class implementing the same interface, without touching
`ComfyUNetLoRANode` at all.

### 2.4 Text encoder cache becomes visible to resource accounting

`CachingTextEncoder` (bounded LRU, default 512 entries, CPU-resident) is
real, working, and self-contained -- but its memory usage is invisible to
anything outside itself; `MemoryManager.stats()` has no idea it exists.
Making it a `DeviceResident` (`footprint_bytes()` sums cached tensor
sizes; `release()` clears the cache) means a future aggregate "where did
my memory go" report (section 3.5) covers it without a special case.

### 2.5 Dataset prefetching, kept honest about what it does and doesn't save

`SupervisedLoRATrainerNode`'s own `profile=True` output already reports
`data_wait_ms` -- so whether data loading is a real bottleneck is
*already measurable* today, not a guess. A prefetching decorator is worth
designing for when that number is shown to matter, not by default:

```python
class PrefetchingBatchSource(TrainingBatchSource):
    """Decorator over any TrainingBatchSource -- same pattern
    nodes/dataset/renoise.py's RenoiseBatchSource already establishes for
    this domain (wrap, don't reimplement iteration). A bounded background
    queue overlaps the *next* batch's host-side preparation with the
    *current* step's device compute.

    Explicitly NOT a VRAM optimization -- it trades a small, bounded
    amount of extra host (and, for pinned buffers, page-locked host)
    memory for reduced wall-clock stall, which is a different axis than
    this design's VRAM focus. Included here because it's the natural
    complement to the DeviceContext/StepPhase split above, not because it
    reduces device memory."""

    def __init__(self, inner: TrainingBatchSource, depth: int = 2):
        self._inner = inner
        self._depth = depth  # bounded on purpose -- see module docstring
        # ... worker thread + bounded queue.Queue(maxsize=depth) ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        return len(self._inner)

    def invalidate(self) -> None:
        self._inner.invalidate()
        # ... drain and restart the prefetch queue ...
```

### 2.6 `MemoryManager`'s reach widens; its interface doesn't

Every new device-memory consumer identified above (activation-checkpoint
recompute scratch, if a future custom block needs it; a text-encoder
cache's tensors, if it's moved to device rather than kept CPU-resident;
a `PrefetchingBatchSource`'s pinned host buffers) should acquire memory
through the existing `MemoryManager.get_buffer()`/`release()`/`free()`
vocabulary, under its own tag, exactly the way
`ChunkedScratchBufferStrategy` already does for optimizer scratch. No new
method is being proposed on `MemoryManager` itself in this iteration --
the design problem it solves (tagged, lazily-grown, reuse-vs-drop-tracked
buffers) is domain-independent already; the gap is adoption, not
capability.

### 2.7 What iteration 2 leaves unresolved

- Multiple `DeviceResident`s still don't have anything coordinating them
  *together* -- `TrainableModel.offload()`, `OptimizerHandle.offload()`,
  and a cache's `release()` are each callable, but nothing decides *when*
  to call which, in what order, in response to what training-lifecycle
  event. This is exactly the shape of the still-open VRAM-pressure
  hang/device-lost report.
- There's no registry letting a rewritten (`nodes/components/`-style)
  implementation of something coexist, side by side, with the legacy
  adapter it's replacing, during the equivalence-testing window the
  project's own discipline requires.
- Nothing yet aggregates `MemoryManager.stats()` plus every
  `DeviceResident.footprint_bytes()` into one coherent picture --
  `profile=True`'s VRAM numbers today come from one `DeviceContext.
  memory_stats()` call, which only sees the allocator's own totals, not
  a breakdown by component.

Iteration 3 addresses these three.

---

## Iteration 3 -- coordination, registry, contracts, observability

### 3.1 `ResourceCoordinator`: a registry of `DeviceResident`s, offload ordering made explicit

```python
class ResourceCoordinator:
    """Tracks every DeviceResident a run has constructed (explicit
    register() calls at construction time -- never reflection, never a
    global registry reached for by import). Doesn't decide *when* to
    offload anything by itself in this base form -- that's
    OffloadOrchestrator below, layered on top. This class only answers
    'what do I currently own, and what's my current total footprint,'
    and provides the one bulk operation ('offload everything except
    these') that's otherwise easy to get subtly wrong by hand (forgetting
    one resident, offloading in the wrong order and causing an
    intermediate OOM that wouldn't happen with the right order)."""

    def __init__(self):
        self._residents: dict[str, DeviceResident] = {}

    def register(self, name: str, resident: DeviceResident) -> None:
        self._residents[name] = resident

    def total_footprint_bytes(self) -> int:
        return sum(r.footprint_bytes() for r in self._residents.values())

    def offload_all_except(self, keep: set[str]) -> None:
        for name, resident in self._residents.items():
            if name not in keep:
                resident.offload()

    def reload(self, name: str, device: str | None = None) -> None:
        self._residents[name].reload(device)
```

### 3.2 `OffloadOrchestrator`: event-driven, reusing the existing pub/sub shape

The project already has a working, correctly-designed pub/sub mechanism
for cross-cutting concerns: `MonitorBus`/`MonitorHandle`, explicitly
injected rather than a singleton, already documented as safe to call from
a worker thread. Rather than invent a second event system for offload
orchestration, this design reuses that same shape:

```python
class TrainingLifecycleEvent(ABC):
    """Marker base -- CacheRebuildStarting, PreviewGenerationStarting,
    CheckpointSaveStarting, etc. Each is a plain, immutable value; no
    behavior of its own."""


class OffloadOrchestrator:
    """Subscribes to TrainingLifecycleEvents, drives a ResourceCoordinator
    in response. This is the principled version of what core/trainer.py's
    hand-written offload calls at specific points in the training loop
    are doing today, ad hoc, per call site -- made into one reviewed
    place with an explicit, testable event -> action mapping, rather than
    N scattered `.to('cpu')` calls that each have to remember to exist.

    This is a genuine, non-trivial piece of new design, not a small
    refactor -- flagged as such (see section 5's priority ordering) and
    explicitly NOT a claim that it fixes the open 'device lost' report on
    its own. That report needs its root cause found first (the
    docs/suspicious_findings.md entry's own leading hypothesis is a
    missing explicit synchronize() on an async offload path, which is a
    correctness bug this orchestrator's *existence* doesn't fix by
    itself -- it fixes the *coordination* problem, which is necessary
    but not sufficient)."""

    def __init__(self, coordinator: ResourceCoordinator, device_ctx: DeviceContext):
        self._coordinator = coordinator
        self._device_ctx = device_ctx
        self._handlers: dict[type, list] = {}

    def on(self, event_type: type, handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: TrainingLifecycleEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event, self._coordinator, self._device_ctx)
```

### 3.3 `ComponentRegistry`: versioned, side-by-side registration

`server/nodegraph_registry.py` is already a plain name -> class registry,
which is fine for "the graph editor needs to resolve a class name." What's
missing, and what the project's own migration discipline actually needs,
is a way for a `nodes/components/`-style rewrite to be registered
*alongside* the legacy adapter it's replacing, both live, both usable, for
however long the equivalence-testing window takes -- exactly the pattern
`nodes/optimizer/`'s composed nodes vs. legacy-wrapping nodes already
follow *by convention* (both exist, unretired, until real-hardware
validation). Formalizing it:

```python
class ComponentRegistry:
    """Generalizes a plain name->class dict with an explicit
    'replaces' relationship, so a caller (or a future admin UI) can ask
    'what's the recommended implementation of X right now' without
    needing to know the migration is in progress by reading a doc."""

    def __init__(self):
        self._entries: dict[str, list[tuple[type, bool]]] = {}  # name -> [(cls, is_default)]

    def register(self, name: str, cls: type, *, default: bool = False) -> None:
        entries = self._entries.setdefault(name, [])
        if default:
            entries = [(c, False) for c, _ in entries]
            self._entries[name] = entries
        entries.append((cls, default))

    def resolve(self, name: str) -> type:
        """The current default -- what the graph editor's picker shows
        first, what a Recipe (3.4) resolves to when it names a component
        by role rather than by exact class."""
        for cls, is_default in self._entries.get(name, []):
            if is_default:
                return cls
        raise KeyError(name)
```

This is a genuine addition, not a rename of the existing registry -- and,
honestly, not urgent: nothing in `nodes/components/` has moved yet (its
own README says so), so there's no live side-by-side migration this would
help with *today*. Listed here for completeness and because it's cheap to
add later exactly when the first `nodes/components/` migration actually
needs it; not recommended as near-term work (see section 5).

### 3.4 `TrainingRecipe` / `PipelineFactory`: declarative composition

A value object describing a full run (dataset config, model config,
optimizer config, schedule, resource budget) plus a factory that turns it
into wired, constructed runtime objects -- the Abstract Factory pattern
applied to "build me a whole pipeline," rather than wiring every
`Builder` by hand each time. This is *not* proposed as a replacement for
the graph editor (which has real value as an interactive, inspectable
construction UI) -- it's a second, programmatic entry point for the exact
same underlying `Builder`/runtime-object model, useful for tests, scripts,
and (longer-term, speculative) as a possible bridge toward driving
`nodes/` from a config file the way `core/` already is, without `nodes/`
and `core/` sharing any code to do it.

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

### 3.5 `ResourceProfile`: one aggregate VRAM report

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
reserved MB from `xpu_memory_stats`) into a per-component breakdown --
"how much of my VRAM is the text encoder cache vs. optimizer scratch vs.
the model itself," which the current single allocator-level number can't
answer.

### 3.6 Concurrency contract, stated explicitly

Stated once, precisely, rather than left implicit (which is how a future
`PrefetchingBatchSource` worker thread could otherwise become a real race
someone finds the hard way):

- **Single-threaded by default.** `StepPhase.run()`, `DeviceResident`
  methods, `MemoryManager` methods, `ResourceCoordinator`/
  `OffloadOrchestrator` methods: none of these are safe to call from more
  than one thread concurrently, and none of them need to be -- a training
  run has exactly one thread driving the step loop.
- **Explicitly cross-thread-safe, by design, documented as such at the
  point of use:** `MonitorHandle.report()` (already true and already
  documented -- called from a FastAPI worker thread today);
  `ExecutionContext`'s cancel signal (already a `threading.Event` for
  exactly this reason); a `PrefetchingBatchSource`'s internal queue
  (its *only* job is being a safe hand-off point between its worker
  thread and the training thread -- `queue.Queue` already gives this for
  free, so this isn't new design work, just a contract worth stating).
- Nothing else should grow a background thread without updating this
  list and justifying it the same way.

### 3.7 The Acyclic Domain Dependency Rule, formalized

Already the project's practiced discipline (`optimizer/` doesn't import
`model/`; domain ABCs live in each domain's own `handle.py`); worth naming
so it stays true as `nodes/components/` grows:

> A domain package (`dataset/`, `model/`, `optimizer/`, `train/`,
> `monitor/`, the future `components/` submodules) may depend downward on
> `core.py` and `memory/` (and, for now, read-only on `core/`/`manager/`
> per the existing wrap-don't-copy rule) -- never sideways on another
> domain package's *implementation*. Cross-domain references go through
> that domain's `handle.py` ABCs only (e.g. `train/` depends on
> `model.handle.TrainableModel`, never on `model.lora_injector`
> directly). A dependency that seems to need to go sideways is a signal
> the shared piece belongs in `core.py`, `memory/`, or a new
> domain-independent module -- not that the rule should bend.

---

## Composition walkthrough: one LoRA run, under this design

Concrete, to make sections 1-3 legible as a whole rather than a list of
classes. Not code that runs -- the actual wiring a `PipelineFactory` or a
hand-written script would do:

```python
layout = ProjectLayout(...)                             # 1.6, built once
device_ctx = DeviceContext.for_device("xpu")             # 1.5
schedule = DiscreteLinearNoiseSchedule()                 # 1.4
process = DiffusionProcess(schedule, EpsParameterization(), KarrasInputScaler())

policy = ManualResourcePolicy(                           # 2.2
    checkpointing=FrozenParamSafeCheckpointing(),         # 2.3
    optimizer_strategy=ChunkedScratchBufferStrategy,
    text_encoder_cache=True,
)
memory = MemoryManager()                                 # 1.3, unchanged
coordinator = ResourceCoordinator()                       # 3.1

model = build_trainable_model(weights, policy, device_ctx)  # a Builder
coordinator.register("model", model)                        # 1.2 DeviceResident
optimizer = build_optimizer(model.trainable_parameters(), policy, memory)
coordinator.register("optimizer", optimizer)
text_encoder = CachingTextEncoder(build_text_encoder(weights))
coordinator.register("text_encoder", text_encoder)

pipeline = TrainingStepPipeline([                          # 2.1
    FetchBatchPhase(prefetching_source),
    EncodeConditioningPhase(text_encoder),
    ForwardPhase(process),
    LossPhase(UniformLossWeighting()),
    BackwardPhase(),
    OptimizerStepPhase(optimizer),
    MonitoringPhase(monitor_handle),
])

orchestrator = OffloadOrchestrator(coordinator, device_ctx)  # 3.2
orchestrator.on(CacheRebuildStarting, lambda e, c, d: c.offload_all_except({"model"}))

for step in range(total_steps):
    state = StepState(step=step, batch=None, model=model, device=device)
    state = pipeline.run_step(state)
```

Every object above is independently constructible and independently
testable; nothing is reached for by import; every device-memory owner is
a `DeviceResident` the coordinator actually knows about.

---

## Deliberately deferred or rejected (avoiding overcomplication)

Considered while writing this design, left out on purpose -- listed with
the actual reasoning, not just "future work," matching the standard this
project already holds itself to elsewhere:

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
  rather than depending on runtime memory conditions"). Nothing in this
  design changes that reasoning; an `OffloadOrchestrator`-driven,
  *event*-triggered offload (3.2) is a different thing from
  pressure-triggered eviction inside the allocator itself, and doesn't
  need the latter.
- **Layer-wise CPU offload of the frozen UNet base between steps**
  (ZeRO-Infinity-style). Real, large VRAM lever -- the single biggest
  static allocation is exactly the frozen base, per
  `docs/vram_and_lora_phase_split.md`'s own "Considered, not implemented"
  section. Not designed here for the same reason that doc gave: no
  existing per-block streaming-offload primitive to build on, and the
  PCIe round-trip cost per step is a real, hardware-dependent question
  that needs actual measurement, not an interface guess. `DeviceResident`
  leaves room for a future `LayeredOffload` variant of `TrainableModel`
  without foreclosing it -- that's as far as this design goes.
- **Quantized (int8) frozen base weights.** Same status as the item
  above in `docs/vram_and_lora_phase_split.md` -- flagged there as the
  next real lever after activation checkpointing, and still out of scope
  here for the same reason (real numerical-drift risk against a frozen
  "ground truth," substantial standalone effort). This design's
  `TrainableModel` doesn't currently name an explicit `FrozenWeightStore`
  seam for this because one hasn't been justified by a concrete need yet
  -- adding an unused seam "just in case" is exactly the kind of
  speculative machinery the "don't overcomplicate" instruction rules out.
  Worth revisiting as its own design pass if/when this becomes the
  priority, not bolted on here.
- **A second event bus for `OffloadOrchestrator`.** Reused `MonitorBus`'s
  existing shape (3.2) instead of inventing a parallel one -- two pub/sub
  systems in one codebase for two similar-but-different purposes would be
  duplication, not design.
- **Redesigning `Node`/`Port`/`ExecutionContext`.** Section 1.1 arrived
  at essentially the same shape independently; section 4.1 confirms it.
  Proposing changes to something already correct, just to have proposed
  something, would be the opposite of the "good code is the only metric"
  standard this document is held to.

---

## Gap analysis: this design vs. current `nodes/`

Now the comparison the earlier sections deliberately avoided making until
the design was settled.

### 4.1 What already matches -- no changes recommended

| Design piece | Current `nodes/` equivalent | Verdict |
|---|---|---|
| `Builder`/`Port` (1.1) | `nodes/core.py`'s `Node`/`Port` | Already this, arrived at independently. Keep as-is. |
| `Algorithm` x `ExecutionStrategy` x `Handle` composition (referenced throughout) | `nodes/optimizer/` in full | This design's generalization target *is* this subpackage, already done correctly. Reference implementation, not a gap. |
| Pooled device buffers (1.3) | `nodes/memory/manager.py`'s `MemoryManager` | Interface is correct; gap is adoption breadth, not design (see 4.2). |
| `LossWeighting`/`LRSchedule` (referenced in 2.1) | `nodes/train/loss.py`/`schedule.py` | Already clean Strategy-pattern ABCs. No changes. |
| Injected pub/sub, not a singleton bus (3.2's reasoning) | `nodes/monitor/`'s `MonitorHandle`/`LiveMonitorHandle` | Already the house reference example for "no singleton" done right -- explicitly reused, not redesigned, for `OffloadOrchestrator`. |
| Decorator-wrapped `TrainingBatchSource` (2.5) | `nodes/dataset/renoise.py`'s `RenoiseBatchSource` | Same pattern this design's `PrefetchingBatchSource` should follow -- prior art already exists, reuse the idiom. |
| Composition-over-mutation for stacked state (referenced in 1.2) | `nodes/model/lora_phases.py`'s `LoRAGeneration` | Independent instance of the same principle `DeviceResident`'s offload-vs-free distinction is built on. Good existing precedent, worth citing when this design is implemented. |

### 4.2 What's missing or partial, by subpackage

**`paths.py` / `core/noise_schedule.py` / `core/comfy_setup.py` (not in
`nodes/` at all today, but directly imported by it).** The two concrete
singletons this design is written against (1.4, 1.6) and the
`hasattr()`-scattered device-backend logic (1.5) all live here, not
inside `nodes/` itself -- but `nodes/train/supervised.py`'s `_run_step`
imports from all three directly (`core.model_io.comfy_input_transform`,
`core.noise_schedule.get_alpha_sigma`, `core.comfy_setup.xpu_synchronize`/
`xpu_empty_cache`/`xpu_memory_stats`). This is the literal, current
"dependency on old singleton code" the next step is about. **Highest
priority** -- see section 5's backlog, items 1-2.

**`nodes/components/`.** Empty except a README. This is the design's
intended home for `NoiseSchedule`/`Parameterization`/`DiffusionProcess`
(1.4), `DeviceContext` (1.5), and arguably `ProjectLayout` (1.6, though
its blast radius is bigger -- see 5's notes). Nothing to change about the
package's own stated discipline (equivalence-test before switching over)
-- it's exactly right, just not used yet.

**`DeviceResident` (1.2).** Doesn't exist. `OptimizerHandle` has the
closest existing equivalent (`offload_states_to_cpu`/
`reload_states_to_device`/`free_states`), semantically identical to
`offload`/`reload`/`release` but under different names and declared only
for optimizers. Recommend introducing `DeviceResident` as a shared ABC
`OptimizerHandle` extends (either by renaming, or by adding thin
alias methods -- the latter is lower-risk, doesn't touch any existing
call site). `TrainableModel` and `TextEncoder` currently have no
`footprint_bytes()`/`offload()`/`release()` at all -- `TrainableModel`'s
`to()`/`train()`/`eval()` cover part of the same ground but don't report
size, and `TextEncoder.unload()` is a `release()`-shaped operation with
no `offload()` tier (it either holds the model or doesn't -- no
"paused, still resident on CPU, cheap to bring back" state distinct from
"fully unloaded"). Real, addressable gaps, not large ones.

**`ActivationCheckpointingStrategy` (2.3).** Doesn't exist as an object.
The current fix in `nodes/model/gradient_checkpointing.py` is correct and
verified (`smoke_test_gradient_checkpointing.py`) -- this is purely about
exposing it as a composable `apply()`-shaped object instead of a function
`ComfyUNetLoRANode.build()` calls conditionally on a `bool` port. This is
this design's highest-value *small* change: the biggest existing VRAM
lever (per `docs/vram_and_lora_phase_split.md`'s own framing), currently
the least composable piece of code that implements it.

**`TrainingStepPipeline`/`StepPhase` (2.1).** Doesn't exist.
`SupervisedLoRATrainerNode._run_step` is one ~90-line static method
covering fetch/encode/forward/loss/backward/optimizer-step/monitor/
profiling in sequence. This is the largest, riskiest recommended change
in this whole document -- riskiest because it's a real behavior-preserving
refactor of the only working end-to-end training loop that exists, not an
additive change. It's also the change that unblocks the most currently-
documented "explicit v1 scope" gaps (CFG dual-pass, grad accumulation,
resume cadence) from needing monolith surgery each. Recommend doing this
*after* the smaller, additive changes above have landed and settled (see
priority order in section 5) -- not first, precisely because it's the
highest-blast-radius item and shouldn't be the one thing this design's
adoption is first judged on.

**`ResourceBudget`/`ResourcePolicy` (2.2).** Doesn't exist even as a
manual-only shim. Every VRAM-affecting choice today is an independent
port on an independent node (`ComfyUNetLoRANode.use_checkpoint`,
`ComposedCAMEOptimizerNode`'s `strategy` argument, `SDXLTextEncoderNode`
having no cache toggle of its own -- caching is a separate node,
`CachingTextEncoderNode`, wired in front of it). Consolidating these into
one `ManualResourcePolicy` object is a real but low-risk change (it can
wrap the existing independent choices without changing any of their
current behavior); worth doing once there are three or more such flags to
consolidate, which is already true today.

**`ResourceCoordinator`/`OffloadOrchestrator` (3.1, 3.2).** Doesn't
exist. Nothing today tracks "every live device-resident object in this
run" as a set; offload/reload calls that do happen (e.g. around cyclic
cache rebuilds, in `core/trainer.py` -- explicitly legacy, out of
`nodes/`'s scope) are hand-written per call site. This is the piece most
directly aimed at the still-open VRAM-pressure hang/device-lost report,
and also the piece with the least existing prior art to build from --
correctly sequenced last in the backlog (section 5) because it needs
several real `DeviceResident`s to exist and be exercised individually
first, or there's nothing concrete to coordinate yet and the abstraction
risks being speculative.

**`PrefetchingBatchSource` (2.5).** Doesn't exist. `data_wait_ms`
(already reported by `profile=True`) is the number that should decide
whether this is worth building for a given real dataset/hardware
combination -- not built preemptively here.

**`ComponentRegistry`/`TrainingRecipe` (3.3, 3.4).** Doesn't exist; not
recommended as near-term work either (see section 3.3/3.7's own
reasoning) -- both solve problems `nodes/components/` doesn't have yet
because nothing has migrated there.

**`server/graph_executor.py`.** Already matches this design's
construction-time model closely: real topological execution, real
`issubclass()`-based port compatibility checking (not string comparison),
explicit `ExecutionContext` threading (not a singleton). No changes
recommended.

### 4.3 What's explicitly out of scope, restated

`core/trainer.py` and the rest of `core/`/`manager/` are the production
path, reference material only, untouched by this design -- exactly the
existing project rule. The VRAM-pressure hang/device-lost report in
`docs/suspicious_findings.md` lives there today; this design's
`OffloadOrchestrator` is the *eventual*, principled home for that class of
coordination problem once `nodes/` is the production path, not a claim
that building it retroactively fixes `core/trainer.py`'s current
hand-rolled offload logic.

---

## Prioritized backlog

Concrete, ordered, sized to be independently landable slices -- each one
equivalence-tested against whatever it replaces, per the project's
existing discipline, before anything switches over to it.

1. **`nodes/components/diffusion.py`**: `NoiseSchedule` /
   `Parameterization` / `DiffusionProcess` (design section 1.4).
   Equivalence-test against `core.noise_schedule`'s free functions
   (bit-exact, CPU, same discipline as the optimizer `Algorithm`
   equivalence tests). Removes `core.noise_schedule`/`core.model_io`
   imports from `nodes/train/supervised.py`'s `_run_step`. Small,
   self-contained, zero behavior change if done correctly -- good first
   slice.
2. **`nodes/components/device.py`**: `DeviceContext` (1.5). Removes
   `core.comfy_setup` imports from `_run_step`'s profiling branch.
   Equivalence: same `hasattr`/`is_available` gating, moved, not changed.
3. **`DeviceResident` ABC** in `nodes/core.py` or a new
   `nodes/memory/handle.py`; retrofit `OptimizerHandle` to satisfy it
   (additive alias methods, no existing call site touched).
4. **`ActivationCheckpointingStrategy`** in `nodes/model/` (2.3), wrapping
   today's monkeypatch as `FrozenParamSafeCheckpointing`. Additive --
   `ComfyUNetLoRANode`'s existing `use_checkpoint: bool` port can stay,
   mapped internally to `NoCheckpointing()`/`FrozenParamSafeCheckpointing()`,
   so nothing wired to it today breaks.
5. **`ProjectLayout`** replacing `paths.py`'s module-global pattern (1.6).
   Larger blast radius than 1-4 -- `server/*` and `manager/*` also depend
   on `paths.py` today, so this needs a bridging period (both the old
   module functions and the new object correct and in sync) rather than
   a clean swap. Sequenced after the smaller wins above so the pattern
   ("equivalence-test, land small, keep old path until validated") is
   well-practiced on lower-stakes changes first.
6. **`TrainingStepPipeline`/`StepPhase` refactor** of
   `SupervisedLoRATrainerNode` (2.1). The biggest single change here --
   deliberately last among the "make the existing thing correct-shaped"
   items, once 1-5 are stable and the phase boundaries they'd need
   (device context, diffusion process, checkpointing strategy) already
   exist as real objects instead of being invented at the same time as
   the pipeline that uses them.
7. **`PrefetchingBatchSource`** (2.5) -- only once a real dataset/hardware
   combination shows `data_wait_ms` actually matters; the measurement
   already exists, so this is demand-driven, not speculative.
8. **`ResourceCoordinator`/`OffloadOrchestrator`** (3.1, 3.2) -- sequenced
   last on purpose: needs several real `DeviceResident`s (item 3, plus
   `TrainableModel`/`TextEncoder` conformance) in place and individually
   exercised first, or it's coordinating nothing concrete yet.

Not on this list, deliberately: `ComponentRegistry`, `TrainingRecipe`,
`AutoResourcePolicy`, layer-wise base offload, quantized base weights --
see "Deliberately deferred or rejected" above for why each one waits.
