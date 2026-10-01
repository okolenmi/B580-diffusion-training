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

This is deliberately close to what already exists (`nodes/core.py`'s
`Node`/`Port`) -- arriving at it independently, from first principles,
before looking, is exactly the check this design process was for: this
part is already right, keep it.

## 1.2 Runtime lifecycle: `DeviceResident`

Every runtime object that can hold device memory needs the same three
questions answerable, regardless of domain (optimizer, model, text
encoder, dataset prefetch buffer): how big is it right now, can it be
moved off-device without losing its identity, and can it be dropped
entirely.

`DeviceResident` (`nodes/memory/handle.py`) -- `OptimizerHandle`,
`TrainableModel` and `TextEncoder` all conform to it.

This closed the *coordination* gap -- nothing generic could drive
offload/reload order across domains before this existed (5.1, 5.2 build
on it directly). It's still not, by itself, a fix for the still-open
"hang after VRAM pressure" report in `docs/known-issues/open.md` --
that report's own leading hypothesis is a missing synchronize() on an
async offload path in `core/trainer.py`, a correctness bug this
lifecycle contract doesn't touch (see 5.2 and section 9.3 for why that's
explicitly out of scope here).

## 1.3 Pooled device buffers stay a separate, lower-level concern

The relationship between the two: a `DeviceResident.release()`
implementation that owns pooled buffers is responsible for also calling
`MemoryManager.free()`/`free_all()` on whatever it acquired -- exactly
the pattern `ChunkedScratchBufferStrategy.free_extra()` already
establishes.

## 1.4 The diffusion process: `NoiseSchedule`, `Parameterization`, `DiffusionProcess`

A second concrete schedule, `RescaledZeroTerminalSNRSchedule`, is
implemented -- but the actual reason to want it is a real, published
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

## 1.6 Configuration as an injected value object, not a mutable module global

**Deliberately a bridging period, not a clean swap, and this part is
still true and still open:** `paths.py` itself is untouched -- `server/*`,
`manager/*`, and `core/*` still read its module functions directly, not
`ProjectLayout`. `from_paths_module()` only snapshots that same global
state into an immutable object rather than replacing it. Migrating
`server/*`/`manager/*` off `paths.py` entirely is real, separate,
out-of-scope work for `nodes/` -- not attempted here, not blocking
anything above.

---
