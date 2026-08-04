# Training pipeline design: complete plan

A from-scratch design for the training pipeline -- VRAM-savings-first,
strict OOP, real composition, no singletons -- developed independently of
`nodes/`'s actual classes, then compared against them once the design was
settled. This is the single, current, complete version: everything that
survived three passes over the design (foundational architecture, then a
review pass that added techniques with real published evidence behind
them, then a fixes-only pass), merged into one document instead of left
spread across three. The process that produced it is in git history, not
repeated here; this document is the result, not a log of how it was
reached.

This is a planning document, not implemented code. Every class shown is
illustrative Python (real syntax, real signatures, meant to be directly
usable as a starting point) -- none of it has been written as `.py` files
or tested. Treat interface details as a strong proposal, not a spec set
in stone; treat the architectural shape (what talks to what, what owns
what lifecycle) as the actual deliverable. Section 9 (Gap analysis) is
where this gets compared to what `nodes/` actually is today, and section
10 (Prioritized backlog) is the ordered, concrete plan for closing that
gap -- nothing described here has been implemented.

## Design goals and constraints

Stated once here, referenced rather than repeated throughout:

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
   out, with the reasoning, the same way `docs/vram_and_lora_phase_split.md`
   documents "Considered, not implemented."
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
`TextEncoder`, and a dataset prefetch buffer (2.5) each get a thin
`DeviceResident` conformance the same way. See section 9 for exactly what
that retrofit looks like on the real classes.

`TrainableModel.footprint_bytes()` is worth being concrete about, since
it's the one `DeviceResident` implementation this design can't fully
account for from `DeviceResident`'s own contract alone: it needs
`sum(p.numel() * p.element_size() for p in self.trainable_parameters()) +
self._frozen_store.footprint_bytes()`, where `_frozen_store` is a
`FrozenWeightStore` (section 3.3) -- without that object, there's nothing
to compute the frozen base's contribution from, since it's otherwise an
opaque quantity inside the wrapped third-party UNet.

### 1.3 Pooled device buffers stay a separate, lower-level concern

`DeviceResident` is object-granularity ("offload this whole optimizer").
Underneath any one `DeviceResident`, there's often a need for
finer-granularity, reusable scratch buffers ("give me 4MB of float32
scratch, reuse it next step too") -- a different concern, already solved
correctly: a tag-keyed pool that grows lazily, never shrinks, and
distinguishes *released* (available for reuse, allocation kept) from
*freed* (allocation actually dropped). This is precisely
`nodes/memory/manager.py`'s `MemoryManager`, and this design reuses it
unchanged -- see section 9.2 for why no interface change is being
proposed here, only a widened set of callers.

The relationship between the two: a `DeviceResident.release()`
implementation that owns pooled buffers is responsible for also calling
`MemoryManager.free()`/`free_all()` on whatever it acquired -- exactly the
pattern `ChunkedScratchBufferStrategy.free_extra()` already establishes.
`DeviceResident` doesn't replace `MemoryManager`; it's the object-level
contract that sits on top of it and on top of anything else a runtime
object owns (a model's parameters, an LRU cache's tensors) that isn't
itself a pooled scratch buffer.

### 1.4 The diffusion process: `NoiseSchedule`, `Parameterization`, `DiffusionProcess`

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

The fix: an object, constructed explicitly, holding its own tensors.
Scoped deliberately to discrete-time, variance-preserving diffusion
(DDPM-style, which is what SDXL actually is) -- not "any noising
process." A continuous-time process (flow matching, section 7) would
implement a separate, smaller `Interpolant` contract instead, as a
sibling, not a subtype forced to fake having a discrete alpha/sigma
table.

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

A second concrete schedule fixes a real, published train/inference
mismatch: Lin et al., "Common Diffusion Noise Schedules and Sample Steps
are Flawed" (arXiv:2305.08891, WACV 2024) show that the standard linear
beta schedule above never reaches SNR=0 at the final training timestep --
the model is trained on an input that still contains a small amount of
real signal (`x_T = 0.068265*x_0 + 0.997667*eps` for Stable Diffusion's
actual schedule, per the paper), while inference sampling starts from
literal pure Gaussian noise. This is a documented, real cause of generated
images clustering around medium brightness and an inability to generate
very dark or very bright images. The fix is a rescale so
`sqrt(alphas_cumprod[-1]) == 0` exactly:

```python
class RescaledZeroTerminalSNRSchedule(DiscreteLinearNoiseSchedule):
    """Same construction as DiscreteLinearNoiseSchedule, but rescales the
    computed alphas_cumprod sequence (Lin et al. 2023, Sec 3.1) so the
    final value is exactly zero, before deriving alpha_t/sigma_t.
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

Two things worth being precise about, checked by hand rather than left
implicit or assumed from the paper's own claims: first,
`alphas_cumprod[-1]` is exactly `0.0` after this rescale, so `sigma_t[-1]`
is exactly `inf` (a real IEEE-754 division-by-zero-tensor result, not an
exception) -- correct, by construction, not a bug, but any code touching
raw `sigma_t` *outside* the `Parameterization` abstraction below (a stray
`1 / sigma` somewhere) will hit that `inf` and needs to account for it.
Second, the paper states that enforcing zero terminal SNR requires
switching to v-prediction, because epsilon prediction's own math
(`x0 = x_t - sigma*eps`) becomes numerically degenerate as
`sigma -> infinity`. Checked directly rather than trusted secondhand:
v-prediction's `to_x0()` (below) stays well-defined in that limit
(`x_t/denom -> 0` and `sigma/sqrt(denom) -> 1` as `sigma -> inf`, so
`x0 -> -raw`, a clean finite result). This project's current default is
epsilon prediction, so adopting this schedule means finishing
v-prediction training support end to end, not just swapping the schedule
in isolation.

Prediction-type conversion (`core/model_io.py`'s `raw_to_x0`/
`raw_to_target`, `core/noise_schedule.py`'s `eps_to_vpred`/`vpred_to_eps`)
becomes a small `Parameterization` Strategy pair instead of a four-way
branch of free functions:

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

Worth being explicit about, rather than leaving it implicit: the `x_t`
these formulas take is in the k-diffusion/`ModelSamplingDiscrete`
convention `NoiseSchedule` already uses (`sigma = sqrt((1-alphas_cumprod)
/alphas_cumprod)`, an alpha-normalized noise-to-signal ratio) --
`x0 + sigma*eps`, not the raw DDPM `x_t = alpha*x0 + sqrt(1-alphas_cumprod)*eps`
a dataset loader would actually produce. Converting between the two is
`ModelInputTransform`'s job, immediately below -- `Parameterization` and
`NoiseSchedule` never see raw DDPM-space `x_t` directly, only the
already-transformed value.

A future continuous-time process (flow matching, section 7) would need a
third `Parameterization` member -- a velocity target, in the rectified-
flow sense -- which the `convert_to()` signature above already
generalizes to without any interface change.

`core/model_io.py`'s `comfy_input_transform` becomes a matching
one-method `ModelInputTransform` Strategy. All three compose into one
object a trainer actually depends on, with one compatibility check baked
in: a schedule that enforces zero terminal SNR is silently wrong when
paired with epsilon prediction (above), so the composite rejects that
combination at construction time rather than only in a comment:

```python
@dataclass(frozen=True)
class DiffusionProcess:
    """Everything a training step needs to know about the forward
    diffusion process and the model's I/O convention, as one injected
    dependency -- not three free-function module imports reached for
    individually inside a step loop (the current supervised.py._run_step
    shape; see section 9)."""
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

---

## 2. Orchestrating a training step

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
`ForwardPhase`, `LossPhase` (wraps a `LossWeighting`, section 4),
`BackwardPhase`, `OptimizerStepPhase` (already has to branch on fused vs.
non-fused -- that branch stays, it's genuine, but it's now the *entire*
content of one small class instead of interleaved with five other
concerns), `MonitoringPhase`.

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
why it's worth structuring the loop this way rather than retrofitting it
into a monolith once grad accumulation is actually needed.

### 2.2 Resource budget as a first-class value, resource policy as a Strategy

"Should this run use activation checkpointing," "which `ExecutionStrategy`
should the optimizer use," "should the text encoder cache be warmed,"
"which LoRA adapter family, which weight-quantization scheme" are each,
absent this design, an independent manual flag on an independent node.
That's fine as a default (explicit is better than magic), but there's no
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
    inside ComfyUNetLoRANode.build(). Accepts an optional `placement`
    (below) it consults per block instead of applying uniformly -- the
    patch logic itself is unchanged either way; placement only changes
    *which* blocks it gets applied to."""

    def __init__(self, placement: "CheckpointPlacementPolicy | None" = None):
        self._placement = placement or EveryBlockPlacement()

    def apply(self) -> None:
        ...  # unchanged monkeypatch body, moved here verbatim
```

All-or-nothing checkpointing (every block, or none) is correct and
maximizes VRAM savings at maximum recompute cost (the ~20-30% measured in
`docs/vram_and_lora_phase_split.md`). A more principled middle ground has
real published grounding: Chen et al., "Training Deep Nets with Sublinear
Memory Cost" (arXiv:1604.06174, 2016) show that checkpointing roughly
every `sqrt(N)` layers achieves near-optimal memory/recompute tradeoff for
a *uniform*-cost network; Korthikanti et al., "Reducing Activation
Recomputation in Large Transformer Models" (NVIDIA, 2022) generalize this
to *selective* recomputation -- ranking candidate checkpoint points by
their actual memory-saved-per-recompute-cost ratio, which is the more
directly applicable idea here since a UNet's blocks aren't uniform cost
(attention blocks vs. plain conv/resnet blocks differ in both activation
size and recompute time):

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
speed of the two extremes.

### 2.4 Text encoder cache becomes visible to resource accounting

`CachingTextEncoder` (bounded LRU, default 512 entries, CPU-resident) is
real, working, and self-contained -- but its memory usage is invisible to
anything outside itself; `MemoryManager.stats()` has no idea it exists.
Making it a `DeviceResident` (`footprint_bytes()` sums cached tensor
sizes; `release()` clears the cache) means the aggregate "where did my
memory go" report (5.5) covers it without a special case.

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
    memory for reduced wall-clock stall, a different axis than this
    design's VRAM focus. The bounded queue is also the one place besides
    MonitorHandle where this design crosses a thread boundary -- see
    5.6's concurrency contract."""

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
`core.lora`-wrapping code hardcodes it as the only option; this design
makes it an explicit choice:

```python
class AdapterStrategy(ABC):
    """How a trainable delta composes with a frozen weight."""
    @abstractmethod
    def wrap(self, frozen: "FrozenWeightStore", rank: int,
              scaling_policy: "LoRAScalingPolicy") -> "AdaptedLayer":
        ...

class PlainLoRAAdapter(AdapterStrategy):
    """Today's core.lora.LoRALinear/LoRAConv2d math, wrapped -- unchanged,
    per the existing rule: genuinely-correct legacy math gets wrapped,
    not re-derived."""
    ...

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

### 3.2 `LoRAScalingPolicy`

Standard LoRA scales its output by `alpha/r`. Kalajdzievski, "A Rank
Stabilization Scaling Factor for Fine-Tuning with LoRA" (arXiv:2312.03732,
2023) proves this causes the adapter's output and gradient magnitude to
collapse as rank `r` grows -- which is why LoRA in practice is usually
kept at low rank, since higher ranks "should" add capacity but
empirically don't help, because the scaling itself suppresses them. The
fix is a one-line change: scale by `alpha/sqrt(r)` instead. Proven, not
just observed, and costs nothing extra at inference or training time.

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

**Calibration.** Adopt -- zero VRAM cost, zero inference cost, the
closest thing in this design to a strict improvement with no tradeoff.
Its actual value depends on training at higher rank than this project's
current default (`rank: 64`) to have anything to stabilize -- worth
pairing with a rank increase, not independently useful at the current
default rank by itself.

### 3.3 `FrozenWeightStore`

The frozen base is this project's own documented single biggest static
VRAM allocation (`docs/vram_and_lora_phase_split.md`'s "Considered, not
implemented" section). Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA:
Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314, NeurIPS 2023)
is the concrete, published, extensively-benchmarked answer: **NF4**
(4-bit NormalFloat), a quantile-based 4-bit type shaped for the
near-Gaussian distribution of pretrained weights, plus **double
quantization** of the per-block scale factors themselves (another ~0.37
bits/parameter saved on average). The frozen base stays 4-bit in storage;
every forward/backward dequantizes on the fly to bf16 for the actual
matmul, so numerical compute happens at full working precision -- only
storage shrinks (4x vs. bf16, before double quantization's further
saving). The paper reports NF4 + double quantization *fully recovering*
16-bit LoRA's benchmark accuracy on models up to 65B parameters.

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

```python
class FrozenWeightStore(ABC):
    @abstractmethod
    def footprint_bytes(self) -> int:
        ...
    @abstractmethod
    def materialize(self) -> "Tensor":
        """bf16 view for one forward pass -- may allocate a fresh
        dequantized tensor each call (NF4WeightStore) or just return the
        stored tensor directly (BF16WeightStore)."""


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
    designed in more detail here -- see calibration below. materialize()
    would dequantize to bf16 each call; a real implementation needs a
    genuine decision about caching that dequantized tensor per step vs.
    re-dequantizing per use (a real VRAM/speed tradeoff this design
    doesn't resolve for you)."""
    ...
```

**Calibration.** This is the single most valuable *future* item in this
whole design -- but not designed in full on purpose: dequantizing NF4 on
the fly needs either a custom fused dequant-matmul kernel or an explicit
dequantize-then-matmul path with its own `MemoryManager`-backed
scratch-buffer story, real substantial systems work, plus the
diffusion-specific quality caveat above genuinely needs checking against
this project's own real UNet, not assumed to transfer from the LLM
literature. Build `BF16WeightStore` now -- small, behavior-preserving,
closes the `TrainableModel.footprint_bytes()` gap (1.2). Design and
validate `NF4WeightStore` as its own dedicated follow-up effort, scoped
like a future `nodes/components/` migration with its own
equivalence-testing pass.

### 3.4 Per-parameter-group learning rates

Checked against the actual current code, not assumed:
`nodes/optimizer/composed.py`'s `ComposedOptimizerHandle` already stores
`param_lr` as a *list*, one entry per parameter --
`self.param_lr = [lr] * len(self.params)` at construction. Per-parameter
rates are almost already representable. The real gap is that
`update_lr()`, called by the LR schedule every step, unconditionally
overwrites every entry with the same value:

```python
def update_lr(self, new_lr: float) -> None:  # today's actual code
    self._lr = new_lr
    self.param_lr = [new_lr] * len(self.params)
```

Anything that set a per-group ratio at construction would have it
silently erased on the very next step -- worth fixing as infrastructure
before anything needs it, not after it ships as a hard-to-notice bug:

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
exactly today's `[lr] * len(params)`). This unlocks Hayou, Ghosh, Yu,
"LoRA+: Efficient Low Rank Adaptation of Large Models" (arXiv:2402.12354,
ICML 2024): standard LoRA trains both adapter matrices (`A`, random-
initialized; `B`, zero-initialized) at the same rate, which an
infinite-width scaling argument proves is inefficient for large-width
models. Using a fixed ratio `lr_B = lambda * lr_A` with `lambda > 1`
(tuned per task, not theoretically pinned to one value -- the theorem
gives an asymptotic relationship, not a constant) restores efficient
feature learning; the paper reports up to ~2x finetuning speedup and
1-2% task-performance improvement at identical computational cost:

```python
class LoRAPlusGroups(ParameterGroupPolicy):
    """B matrices at `ratio`x the base rate, everything else at 1x.
    is_b_matrix is a predicate over a parameter, decoupled from any one
    LoRA implementation's own naming -- this project's core.lora would
    supply it via each layer's own lora_B reference. ratio=16.0 is a
    commonly-used starting point in public implementations (e.g. Hugging
    Face PEFT's LoraPlusModel), not independently verified as optimal for
    SDXL LoRA here -- a reasonable default to tune from, not a
    proven-correct constant."""

    def __init__(self, is_b_matrix, ratio: float = 16.0):
        self._is_b_matrix = is_b_matrix
        self._ratio = ratio

    def group_ratios(self, params) -> list[float]:
        return [self._ratio if self._is_b_matrix(p) else 1.0 for p in params]
```

**Calibration.** Adopt both -- the `ComposedOptimizerHandle` fix is
small, precise, and worth landing regardless of whether LoRA+ is ever
used, since it closes a real latent bug. `LoRAPlusGroups` itself is
genuinely free (same parameter count, same forward/backward cost) once
the fix exists.

---

## 4. Loss weighting

`nodes/train/loss.py`'s `LossWeighting` ABC is already a clean
Strategy-pattern interface and needs no change -- confirmed by adding a
second implementation to it and finding zero friction. The existing
`MinSNRLossWeighting`'s own docstring already documents a known,
already-scoped gap: it's "only correct for an eps-predicting student...
the v-prediction form, not yet implemented here" -- small, existing-code
completion work, not new design (see the backlog, section 10, item 6).

A genuinely new addition: Choi et al., "Perception Prioritized Training
of Diffusion Models" (P2 weighting, CVPR 2022) weight by
`1 / (k + SNR)^gamma` -- a smoother, more aggressive de-emphasis of the
highest-SNR (near-clean-image, most imperceptible-detail) steps than
Min-SNR's hard clamp, worth having as an available option rather than
assuming Min-SNR is the only reasonable choice:

```python
class P2LossWeighting(LossWeighting):
    def __init__(self, k: float = 1.0, gamma: float = 1.0):
        self.k, self.gamma = k, gamma

    def weight(self, sigma: float) -> float:
        snr = 1.0 / (sigma ** 2 + 1e-8)
        return 1.0 / ((self.k + snr) ** self.gamma)
```

**Calibration.** Adopt as an available option -- cheap, well-precedented,
orthogonal to the eps/v-pred completion work.

---

## 5. Coordination, registry, and observability

### 5.1 `ResourceCoordinator`: a registry of `DeviceResident`s, offload ordering made explicit

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

### 5.2 `OffloadOrchestrator`: event-driven, reusing the existing pub/sub shape

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

    A genuine, non-trivial piece of design, not a small refactor -- and
    explicitly NOT a claim that it fixes the open 'device lost' report on
    its own. That report needs its root cause found first (the
    docs/suspicious_findings.md entry's own leading hypothesis is a
    missing explicit synchronize() on an async offload path, a
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

### 5.3 `ComponentRegistry`: versioned, side-by-side registration

`server/nodegraph_registry.py` is already a plain name -> class registry,
which is fine for "the graph editor needs to resolve a class name." What's
missing, and what the project's own migration discipline actually needs,
is a way for a `nodes/components/`-style rewrite to be registered
*alongside* the legacy adapter it's replacing, both live, both usable, for
however long the equivalence-testing window takes -- exactly the pattern
`nodes/optimizer/`'s composed nodes vs. legacy-wrapping nodes already
follow *by convention*. Formalizing it:

```python
class ComponentRegistry:
    """Generalizes a plain name->class dict with an explicit 'replaces'
    relationship, so a caller (or a future admin UI) can ask 'what's the
    recommended implementation of X right now' without needing to know
    the migration is in progress by reading a doc."""

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
        first, what a Recipe (5.4) resolves to when it names a component
        by role rather than by exact class."""
        for cls, is_default in self._entries.get(name, []):
            if is_default:
                return cls
        raise KeyError(name)
```

Not urgent: nothing in `nodes/components/` has moved yet (its own README
says so), so there's no live side-by-side migration this would help with
*today*. Listed for completeness; cheap to add exactly when the first
`nodes/components/` migration actually needs it.

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
`core/` sharing any code to do it.

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

### 5.6 Concurrency contract, stated explicitly

Stated once, precisely, rather than left implicit (which is how a
`PrefetchingBatchSource` worker thread, 2.5, could otherwise become a
real race someone finds the hard way):

- **Single-threaded by default.** `StepPhase.run()`, `DeviceResident`
  methods, `MemoryManager` methods, `ResourceCoordinator`/
  `OffloadOrchestrator` methods: none of these are safe to call from more
  than one thread concurrently, and none of them need to be -- a training
  run has exactly one thread driving the step loop.
- **Explicitly cross-thread-safe, by design, documented as such at the
  point of use:** `MonitorHandle.report()` (already true and already
  documented -- called from a FastAPI worker thread today);
  `ExecutionContext`'s cancel signal (already a `threading.Event` for
  exactly this reason); a `PrefetchingBatchSource`'s internal queue (its
  *only* job is being a safe hand-off point between its worker thread and
  the training thread -- `queue.Queue` already gives this for free, so
  this isn't new design work, just a contract worth stating).
- Nothing else should grow a background thread without updating this
  list and justifying it the same way.

### 5.7 The Acyclic Domain Dependency Rule

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

## 6. Composition walkthrough: one LoRA run, under this design

Concrete, to make sections 1-5 legible as a whole rather than a list of
classes. Not code that runs -- the actual wiring a `PipelineFactory` or a
hand-written script would do:

```python
layout = ProjectLayout(...)                              # 1.6, built once
device_ctx = DeviceContext.for_device("xpu")              # 1.5

schedule = RescaledZeroTerminalSNRSchedule()               # 1.4
process = DiffusionProcess(schedule, VPredParameterization(), KarrasInputScaler())
# DiffusionProcess.__post_init__ rejects EpsParameterization here -- see 1.4

policy = ManualResourcePolicy(                            # 2.2
    checkpointing=FrozenParamSafeCheckpointing(placement=EveryBlockPlacement()),  # 2.3
    optimizer_strategy=ChunkedScratchBufferStrategy,
    text_encoder_cache=True,
    adapter_strategy=PlainLoRAAdapter(),          # 3.1 -- DoRAAdapter() to opt in
    lora_scaling_policy=RankStabilizedScaling(),  # 3.2
    frozen_weight_store=BF16WeightStore,          # 3.3 -- NF4WeightStore is future work
    parameter_group_policy=UniformGroups(),       # 3.4 -- LoRAPlusGroups(...) to opt in
)
memory = MemoryManager()                                  # 1.3, unchanged
coordinator = ResourceCoordinator()                        # 5.1

model = build_trainable_model(weights, policy, device_ctx)   # a Builder; wires
                                                               # FrozenWeightStore +
                                                               # AdapterStrategy + scaling
coordinator.register("model", model)                         # 1.2 DeviceResident
optimizer = build_optimizer(model.trainable_parameters(), policy, memory,
                             group_policy=policy.parameter_group_policy())
coordinator.register("optimizer", optimizer)
text_encoder = CachingTextEncoder(build_text_encoder(weights))
coordinator.register("text_encoder", text_encoder)

pipeline = TrainingStepPipeline([                          # 2.1
    FetchBatchPhase(prefetching_source),
    EncodeConditioningPhase(text_encoder),
    ForwardPhase(process),
    LossPhase(P2LossWeighting()),                           # 4
    BackwardPhase(),
    OptimizerStepPhase(optimizer),
    MonitoringPhase(monitor_handle),
])

orchestrator = OffloadOrchestrator(coordinator, device_ctx)  # 5.2
orchestrator.on(CacheRebuildStarting, lambda e, c, d: c.offload_all_except({"model"}))

for step in range(total_steps):
    state = StepState(step=step, batch=None, model=model, device=device)
    state = pipeline.run_step(state)
```

Every object above is independently constructible and independently
testable; nothing is reached for by import; every device-memory owner is
a `DeviceResident` the coordinator actually knows about.

---

## 7. Deliberately deferred or rejected

Considered, left out on purpose -- listed with the actual reasoning, not
just "future work," matching the standard `docs/vram_and_lora_phase_split.md`
already holds itself to:

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
  (ZeRO-Infinity-style). Real, large VRAM lever -- not designed here for
  the same reason `docs/vram_and_lora_phase_split.md` gave: no existing
  per-block streaming-offload primitive to build on, and the PCIe
  round-trip cost per step is a real, hardware-dependent question that
  needs actual measurement, not an interface guess. `DeviceResident`
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

---

## 9. Gap analysis: this design vs. current `nodes/`

### 9.1 What already matches -- no changes recommended

| Design piece | Current `nodes/` equivalent | Verdict |
|---|---|---|
| `Builder`/`Port` (1.1) | `nodes/core.py`'s `Node`/`Port` | Already this, arrived at independently. Keep as-is. |
| `Algorithm` x `ExecutionStrategy` x `Handle` composition (referenced throughout) | `nodes/optimizer/` in full | This design's generalization target *is* this subpackage, already done correctly. Reference implementation, not a gap. |
| Pooled device buffers (1.3) | `nodes/memory/manager.py`'s `MemoryManager` | Interface is correct; gap is adoption breadth, not design (see 9.2). |
| `LossWeighting`/`LRSchedule` (section 4) | `nodes/train/loss.py`/`schedule.py` | Already clean Strategy-pattern ABCs -- confirmed by `P2LossWeighting` needing zero interface change. No changes. |
| `Algorithm.init_state()`'s state representation | `nodes/optimizer/algorithms/*.py` | Contract already returns "a plain dict of named tensors," not specifically fp32 plain tensors -- nothing structurally blocks a future block-quantized-state `Algorithm`. No interface change needed if that's ever built. |
| Injected pub/sub, not a singleton bus (5.2's reasoning) | `nodes/monitor/`'s `MonitorHandle`/`LiveMonitorHandle` | Already the house reference example for "no singleton" done right -- explicitly reused, not redesigned, for `OffloadOrchestrator`. |
| Decorator-wrapped `TrainingBatchSource` (2.5) | `nodes/dataset/renoise.py`'s `RenoiseBatchSource` | Same pattern this design's `PrefetchingBatchSource` should follow -- prior art already exists, reuse the idiom. |
| Composition-over-mutation for stacked state (1.2) | `nodes/model/lora_phases.py`'s `LoRAGeneration` | Independent instance of the same principle `DeviceResident`'s offload-vs-free distinction is built on. Good existing precedent. |

### 9.2 What's missing or partial, by subpackage

**`paths.py` / `core/noise_schedule.py` / `core/comfy_setup.py` (not in
`nodes/` at all today, but directly imported by it).** The two concrete
singletons this design is written against (1.4, 1.6) and the
`hasattr()`-scattered device-backend logic (1.5) all live here, not
inside `nodes/` itself -- but `nodes/train/supervised.py`'s `_run_step`
imports from all three directly (`core.model_io.comfy_input_transform`,
`core.noise_schedule.get_alpha_sigma`, `core.comfy_setup.xpu_synchronize`/
`xpu_empty_cache`/`xpu_memory_stats`). This is the literal, current
"dependency on old singleton code" the next step is about. **Highest
priority** -- see the backlog (section 10), items 1-2.

**`nodes/components/`.** Empty except a README. This is the design's
intended home for `NoiseSchedule`/`Parameterization`/`DiffusionProcess`
(1.4), `DeviceContext` (1.5), and arguably `ProjectLayout` (1.6, though
its blast radius is bigger -- see the backlog's notes). Nothing to change
about the package's own stated discipline (equivalence-test before
switching over) -- it's exactly right, just not used yet.

**`DeviceResident` (1.2).** Doesn't exist. `OptimizerHandle` has the
closest existing equivalent (`offload_states_to_cpu`/
`reload_states_to_device`/`free_states`), semantically identical to
`offload`/`reload`/`release` but under different names and declared only
for optimizers. Recommend introducing `DeviceResident` as a shared ABC
`OptimizerHandle` extends (either by renaming, or by adding thin alias
methods -- the latter is lower-risk). `TrainableModel` and `TextEncoder`
currently have no `footprint_bytes()`/`offload()`/`release()` at all --
`TrainableModel`'s `to()`/`train()`/`eval()` cover part of the same
ground but don't report size, and `TextEncoder.unload()` is a
`release()`-shaped operation with no `offload()` tier. Real, addressable
gaps, not large ones -- and `TrainableModel.footprint_bytes()`
specifically waits on `FrozenWeightStore` (9.2 below) existing, not on
anything in `DeviceResident` itself.

**`ParameterGroupPolicy` (3.4).** Doesn't exist -- and checked directly
against the real code: `nodes/optimizer/composed.py`'s
`ComposedOptimizerHandle.update_lr()` currently does
`self.param_lr = [new_lr] * len(self.params)` unconditionally, every
step. Nothing is broken *today*, but the moment a run wants per-group
rates, that line silently erases them on the very next scheduler tick.
Worth landing before it's needed, not after something breaks because of
it -- small, precise, behavior-preserving for every current caller.

**`AdapterStrategy` / `LoRAScalingPolicy` / `FrozenWeightStore` (3.1-3.3).**
None exist. Today, `core.lora`'s LoRA math is wrapped directly by
`nodes/model/lora_injector.py` with one fixed scaling formula
(`alpha/rank`) and no alternative to plain LoRA -- correct and verified,
but with no seam for `RankStabilizedScaling` (free, provably-motivated),
`DoRAAdapter` (real quality win, bounded new forward-pass code), or a
`FrozenWeightStore` seam that would let a future `NF4WeightStore` (this
design's single highest-value future VRAM lever) exist without
restructuring `TrainableModel` again later. Recommend building the seam
(`AdapterStrategy`/`FrozenWeightStore` interfaces, `PlainLoRAAdapter`/
`BF16WeightStore` as the exactly-today-behavior implementations) as its
own slice; `RankStabilizedScaling` is cheap enough to land in the same
slice; `DoRAAdapter` and `NF4WeightStore` are each their own, later,
dedicated efforts (see backlog).

**`ActivationCheckpointingStrategy` and `CheckpointPlacementPolicy`
(2.3).** Neither exists as an object. The current fix in
`nodes/model/gradient_checkpointing.py` is correct and verified
(`smoke_test_gradient_checkpointing.py`) -- this is purely about exposing
it as a composable `apply()`-shaped object instead of a function
`ComfyUNetLoRANode.build()` calls conditionally on a `bool` port. This
design's highest-value *small* change: the biggest existing VRAM lever
(per `docs/vram_and_lora_phase_split.md`'s own framing), currently the
least composable piece of code that implements it.
`CheckpointPlacementPolicy` (selective, cost-ratio-ranked placement) is a
separate, later concern -- it needs real per-block activation/recompute
measurements that don't exist yet (extending `profile=True` to block
granularity is its own instrumentation task).

**`TrainingStepPipeline`/`StepPhase` (2.1).** Doesn't exist.
`SupervisedLoRATrainerNode._run_step` is one ~90-line static method
covering fetch/encode/forward/loss/backward/optimizer-step/monitor/
profiling in sequence. The largest, riskiest recommended change in this
whole document -- riskiest because it's a real behavior-preserving
refactor of the only working end-to-end training loop that exists, not
an additive change. It's also the change that unblocks the most
currently-documented "explicit v1 scope" gaps (CFG dual-pass, grad
accumulation, resume cadence) from needing monolith surgery each.
Recommend doing this *after* the smaller, additive changes above have
landed and settled (see backlog) -- not first, precisely because it's
the highest-blast-radius item and shouldn't be the one thing this
design's adoption is first judged on.

**`ResourceBudget`/`ResourcePolicy` (2.2).** Doesn't exist even as a
manual-only shim. Every VRAM/quality-affecting choice today is an
independent port on an independent node (`ComfyUNetLoRANode.use_checkpoint`,
`ComposedCAMEOptimizerNode`'s `strategy` argument, `SDXLTextEncoderNode`
having no cache toggle of its own -- caching is a separate node,
`CachingTextEncoderNode`, wired in front of it, plus now LoRA's
adapter/scaling/weight-store choices). Consolidating these into one
`ManualResourcePolicy` object is a real but low-risk change (it can wrap
the existing independent choices without changing any of their current
behavior); worth doing once there are several such flags to consolidate,
which is already true today.

**`MinSNRLossWeighting`'s v-prediction branch, and `P2LossWeighting`
(section 4).** Different in kind from every other item in this section:
the v-prediction gap is already documented in the *existing* code's own
docstring, doesn't need a new class or a design decision -- just
finishing a branch that's already scoped, in code that already exists.
`P2LossWeighting` is a genuinely new, tiny, optional addition to the same
file. Grouped together here because they land in the same file, not
because they're the same kind of work.

**`ResourceCoordinator`/`OffloadOrchestrator` (5.1, 5.2).** Doesn't
exist. Nothing today tracks "every live device-resident object in this
run" as a set; offload/reload calls that do happen (e.g. around cyclic
cache rebuilds, in `core/trainer.py` -- explicitly legacy, out of
`nodes/`'s scope) are hand-written per call site. This is the piece most
directly aimed at the still-open VRAM-pressure hang/device-lost report,
and also the piece with the least existing prior art to build from --
correctly sequenced last in the backlog because it needs several real
`DeviceResident`s to exist and be exercised individually first, or
there's nothing concrete to coordinate yet and the abstraction risks
being speculative.

**`PrefetchingBatchSource` (2.5).** Doesn't exist. `data_wait_ms`
(already reported by `profile=True`) is the number that should decide
whether this is worth building for a given real dataset/hardware
combination -- not built preemptively here.

**`ComponentRegistry`/`TrainingRecipe` (5.3, 5.4).** Doesn't exist; not
recommended as near-term work either (see those sections' own reasoning)
-- both solve problems `nodes/components/` doesn't have yet because
nothing has migrated there.

**`server/graph_executor.py`.** Already matches this design's
construction-time model closely: real topological execution, real
`issubclass()`-based port compatibility checking (not string comparison),
explicit `ExecutionContext` threading (not a singleton). No changes
recommended.

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

Concrete, ordered, sized to be independently landable slices -- each one
equivalence-tested against whatever it replaces, per the project's
existing discipline, before anything switches over to it.

1. **`nodes/components/diffusion.py`**: `NoiseSchedule` /
   `Parameterization` / `DiffusionProcess` (1.4). Equivalence-test against
   `core.noise_schedule`'s free functions (bit-exact, CPU, same discipline
   as the optimizer `Algorithm` equivalence tests). Removes
   `core.noise_schedule`/`core.model_io` imports from
   `nodes/train/supervised.py`'s `_run_step`. Small, self-contained, zero
   behavior change if done correctly -- good first slice.
2. **`nodes/components/device.py`**: `DeviceContext` (1.5). Removes
   `core.comfy_setup` imports from `_run_step`'s profiling branch.
   Equivalence: same `hasattr`/`is_available` gating, moved, not changed.
3. **`DeviceResident` ABC** in `nodes/core.py` or a new
   `nodes/memory/handle.py`; retrofit `OptimizerHandle` to satisfy it
   (additive alias methods, no existing call site touched).
4. **`ParameterGroupPolicy` fix to `ComposedOptimizerHandle`** (3.4) --
   separate base rate from per-group ratio in `update_lr()`.
   Behavior-preserving (`UniformGroups` reproduces today's exact
   `[lr] * len(params)`); closes a real latent bug before anything
   actually needs per-group rates, not after.
5. **`LoRAScalingPolicy`** (`ClassicLoRAScaling`/`RankStabilizedScaling`,
   3.2) in the LoRA-construction path. Zero risk, zero cost, opt-in --
   `ClassicLoRAScaling` stays the default, reproducing today's
   `alpha/rank` exactly.
6. **Finish `MinSNRLossWeighting`'s v-prediction branch; add
   `P2LossWeighting`** (section 4) in `nodes/train/loss.py`. The v-pred
   completion needs no new design (already scoped in the existing
   docstring); `P2LossWeighting` is additive. Both small, independent of
   everything else on this list.
7. **`ActivationCheckpointingStrategy`** in `nodes/model/` (2.3), wrapping
   today's monkeypatch as `FrozenParamSafeCheckpointing`. Additive --
   `ComfyUNetLoRANode`'s existing `use_checkpoint: bool` port can stay,
   mapped internally to `NoCheckpointing()`/`FrozenParamSafeCheckpointing()`,
   so nothing wired to it today breaks.
8. **`ProjectLayout`** replacing `paths.py`'s module-global pattern (1.6).
   Larger blast radius than 1-7 -- `server/*` and `manager/*` also depend
   on `paths.py` today, so this needs a bridging period (both the old
   module functions and the new object correct and in sync) rather than
   a clean swap. Sequenced after the smaller wins above so the pattern
   ("equivalence-test, land small, keep old path until validated") is
   well-practiced on lower-stakes changes first.
9. **`AdapterStrategy`/`FrozenWeightStore` seam** (3.1, 3.3 -- seam only:
   `PlainLoRAAdapter`/`BF16WeightStore`, both exactly reproducing today's
   behavior). Medium effort -- touches `core.lora`'s wrapping path in
   `nodes/model/`. Not yet: `DoRAAdapter` or `NF4WeightStore` themselves
   (see the "further out" items below) -- this slice is the seam, not
   the new techniques it enables.
10. **`TrainingStepPipeline`/`StepPhase` refactor** of
    `SupervisedLoRATrainerNode` (2.1). The biggest single change here --
    deliberately this late among the "make the existing thing
    correct-shaped" items, once 1-9 are stable and the phase boundaries
    it needs (device context, diffusion process, checkpointing strategy,
    adapter/weight-store seam) already exist as real objects instead of
    being invented at the same time as the pipeline that uses them.
11. **`PrefetchingBatchSource`** (2.5) -- only once a real dataset/hardware
    combination shows `data_wait_ms` actually matters; the measurement
    already exists, so this is demand-driven, not speculative.
12. **`ResourceCoordinator`/`OffloadOrchestrator`** (5.1, 5.2) --
    sequenced last on purpose: needs several real `DeviceResident`s
    (item 3, plus `TrainableModel`/`TextEncoder` conformance) in place
    and individually exercised first, or it's coordinating nothing
    concrete yet.

**Further out -- real, valuable, deliberately not numbered into the
sequence above** because each needs something the numbered list doesn't
provide by itself (real profiling data, a larger validation effort, or a
genuine behavioral change rather than a behavior-preserving refactor):

- **`DoRAAdapter`** (3.1) -- once item 9's seam exists. Its own
  equivalence/quality-comparison pass, not just an equivalence test (it's
  a real quality claim, not a refactor).
- **`NF4WeightStore`** (QLoRA, 3.3) -- this design's single highest-value
  future VRAM lever, and explicitly *not* squeezed into the numbered
  list: needs a real dequantization implementation (kernel or
  scratch-buffer-backed dequant-then-matmul, via `MemoryManager`) and
  verification against this project's actual UNet specifically, not
  assumed from the LLM literature. Its own dedicated effort.
- **`CheckpointPlacementPolicy`/`GreedyRatioPlacement`** (2.3) -- blocked
  on extending `profile=True` to real per-block activation/recompute
  measurements, which doesn't exist yet and is its own instrumentation
  task before this policy has real numbers to rank against.
- **`RescaledZeroTerminalSNRSchedule` plus finishing real v-prediction
  training end to end** (1.4) -- unlike everything else in this backlog,
  this isn't equivalence-tested against existing behavior (there's no
  old code path to match); it's a genuine training-behavior change that
  needs real training runs and qualitative image-quality evaluation to
  validate, not a unit test. Sequenced separately for that reason, not
  because it's unimportant.
- **`LoRAPlusGroups` actually wired into a run** (3.4) -- once item 4
  lands, opting in is a one-line `parameter_group_policy=
  LoRAPlusGroups(...)` change, not worth its own numbered slot; listed
  here only so it isn't forgotten as the actual payoff of item 4.

Not on this list, deliberately: `ComponentRegistry`, `TrainingRecipe`,
`AutoResourcePolicy`, layer-wise base offload, flow matching, GaLore,
8-bit optimizer moments -- see "Deliberately deferred or rejected"
(section 7) for why each one waits.
