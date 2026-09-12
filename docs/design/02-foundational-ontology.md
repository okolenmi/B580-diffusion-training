*[← docs/design index](README.md)*

# 1. Foundational ontology

The base vocabulary: what kinds of objects exist, what each owns, and how
they're wired together at construction time vs. what they do at runtime.

## 1.1 Two different lifetimes, two different kinds of object

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

## 1.2 Runtime lifecycle: `DeviceResident`

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

## 1.3 Pooled device buffers stay a separate, lower-level concern

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

## 1.4 The diffusion process: `NoiseSchedule`, `Parameterization`, `DiffusionProcess`

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

## 1.5 Device backend as a Strategy, not `hasattr()` calls

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

## 1.6 Configuration as an injected value object, not a mutable module global

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
