# What `archive/core/` uniquely held

`core/` was ~7,700 lines across 23 modules. It moved to `archive/core/`
when the supervised-subprocess route was retired; what remains runnable
is `python -m archive.core.cli`. See
[`design/11-core-removal.md`](design/11-core-removal.md) for what the move
touched.

This file answers one question: **what did `core/` have that nothing else
in the repository provides?** It matters for the day someone wants `cyclic`,
`distillation` or `full` back — see
[`design/12-training-modes.md`](design/12-training-modes.md) — and for
anyone reading the archive to decide what to port.

The answer is much shorter than the module count suggests, and it is
shorter than an earlier version of this file claimed, because several things
once listed here have since been reimplemented in `nodes/`. Each entry below
was checked against the current tree.

## Two things this file has been wrong about

Recorded because both errors were made here, and both are the kind that
survive review.

**It once said `nodes/` had no dependency on `core/` at all.** True of the
library, and the wrong question. `nodes/smoke_tests/` holds equivalence
tests that import `archive.core` as the reference implementation, and they
are the yardstick the rewrite is measured with.

**A correction written the same day then said `core/` "is the trainer the
backend spawns".** It is not. That was inferred from "the backend launches
`python -m core.cli`", which is a fact about *which binary ran*, not about
who did the work. Measured rather than argued: **all 102 `nodes/` modules
import with `archive.core` blocked from `sys.meta_path`**, and
`smoke_test_managed_trainer.py` passes that way — real forward, real
backward, a real `optimizer.step()`, LoRA written to `.safetensors`.
`nodes/train/node.py` is the step loop.

A third claim, also since disproved by the move itself: that deleting
`core/` would break dataset ingestion and the dataset `t_mode` Port,
because `manager/` reached six `core/` modules through two indirections.
`manager/` now imports none of it — `sample_timestep` lives in
`nodes/components/noise_schedule.py`, `vae_decode` in
`nodes/model/vae_decode.py`, `make_init_noise` in
`nodes/components/model_io.py`. That dependency was measured, found absent,
and is why the deletion was a move rather than a project.

## Genuinely unique to `archive/core/`

Each exists only there. `nodes/` has no equivalent, and where `nodes/`
mentions one it is to say it deliberately does *not* replicate it.

1. **Mid-run image previews** — `preview_sampler.py` samples from the live
   UNet mid-training. Also an orphan: the images are still written by every
   run that enables them, but the endpoint that served them went with the
   retired `server/` and nothing replaced it. Surface them again or stop
   spending the time generating them.
2. **Six optimizer *batching* strategies** — `FusedXPUAdafactor`,
   `ChunkedXPUAdafactor`, `ForeachXPUAdafactor`, `ChunkedXPUCAME`,
   `ForeachXPUCAME`, `CPUAdamW`, all in `archive/core/optimizers.py`. The
   per-parameter *maths* were reimplemented and equivalence-tested, and
   fused execution exists in `nodes/`; the *batching* was never rewritten.
   `nodes/optimizer/algorithms/` names them only to record which behaviour
   it replicates and which it does not. This is the one unmeasured
   performance trade the rewrite accepted; it is tracked in
   [`known-issues/open.md`](known-issues/open.md).
3. **Adversarial pre-conditioning** — `train_step.py`, low-power
   cross-conditioning drafts between the conditioned and unconditioned
   passes with a clean-step ratio. `nodes/train/supervised.py` says so in its
   own first paragraph: *"no DAgger, no adversarial pre-conditioning"*.
4. **Cyclic training** — `trainer.py`: N-step cycles with cache rebuild and
   teacher offload/reload between them. Configurable through
   `tuning.method` and implemented nowhere; see `12-training-modes.md`.
5. **Radial per-UNet-block LR grouping** — `optimizer_builder.py`,
   interpolated LR across `input_blocks`/`output_blocks` with separate
   `time_embed`/`label_emb` multipliers. LoRA mode ignores it, which the
   module warns about.
6. **Polynomial LR decay** — `schedules.py::make_poly_lr`. Cosine, constant
   and warmup are in `nodes/train/schedule.py`; poly is not.
7. **Welford loss window and a background GC worker** — `train_step.py`.
   Only the per-t loss breakdown was rebuilt (`nodes/train/loss.py`).
8. **Mid-run checkpointing and optimizer-state persistence** — `save.py`,
   including the on-disk optimizer-state safetensors schema (`__step__`,
   `vr_i`/`vc_i`/`vs_i`/`ea_i`, `resr_i`/`resc_i`, `__tiny_vs_*__`). Nothing
   in `nodes/` reads or writes those keys, so it is a resume-compatibility
   contract nothing else can honour.
9. **The `.progress.jsonl` phase protocol** — `progress_writer.py`:
   `cache_start`/`cache_done`/`training_start`/`step`/`done`, with a
   flush-interval choice (0.4 s) that is easy to lose. Legacy consumers
   only, but `backend/infrastructure/workspace.py` still knows the path.
10. **Parallel `pin_memory()` and an O(1) cache re-batch** —
    `cache_utils.py`. The re-batch function understands the v1/v2/v3/v4
    cache tuple layouts and has **zero call sites**.

## Since reimplemented — not unique any more

Listed because an earlier version of this file claimed them as unique, and
that claim was the stated obstacle to moving `core/`.

| Was | Lives in |
|---|---|
| `make_init_noise` | `nodes/components/model_io.py` |
| `comfy_input_transform`, `raw_to_denoised`, `raw_to_target` | `nodes/components/diffusion.py` |
| Latent caching (`cache_trajectory`) | `nodes/components/model_io.py` |
| `sample_timestep` and the five `T_MODES` | `nodes/components/noise_schedule.py`, with the mode list in `nodes/dataset/timestep_modes.py` |
| VAE decode and ComfyUI's `DEFAULT_SCALE_FACTOR = 0.13025` | `nodes/model/vae_decode.py` |
| The LoRA timestep gate (`compute_lora_gate`, `set_lora_gate`) | `nodes/model/lora_gate.py`. Still the project's own contribution and still depended on by every adapter layer — plain, DoRA and NF4 — but no longer unique. |
| The XPU performance environment contract | `nodes/xpu_env.py` |

## Dead code inside `archive/core/`

So it is not mistaken for load-bearing by anyone reading the archive:

* `cache_utils.py::shuffle_and_rebatch_cache` — zero call sites anywhere.
* `unet_wrapper.py::clear_embedder_cache` (now
  `nodes/model/unet_wrapper.py`) — imported by `archive/core`, never
  called. Left in place rather than removed, because the archive
  re-exports it and the archive is meant to stay runnable; deleting a
  dead function is not worth breaking `python -m archive.core.cli`.
* `unet_wrapper.py::ComfyUNetWrapper.enable_gradient_checkpointing` — was
  dead, and has been removed from `nodes/model/unet_wrapper.py`.
  Checkpointing is now a strategy applied around the forward
  (`nodes/model/gradient_checkpointing.py`), chosen per call site rather
  than by mutating the model once.
* `lora.py::GroupedLoRALinear.forward` (now `nodes/model/lora.py`) — an
  explicit `pass`.
* `__init__.py`'s `load_config` alias — kept for compatibility, unused.

## Two hazards worth carrying forward

**The XPU env ordering was silently wrong.** `backend/cli.py` imported
`core.xpu_env` under a comment reading "Pure os.environ writes, no torch
import — safe before any child". The comment was false: importing any
`core.*` submodule runs `core/__init__.py`, which eagerly re-exported
`optimizers` and `unet_wrapper`, so `torch` was loaded two lines *before* the
SYCL variables were set. `archive/core/__init__.py` is now a lazy PEP 562
facade and the same import no longer pulls torch. It was probably harmless —
SYCL reads those variables at its own runtime init, not at `import torch` —
but "probably" is not a contract.

**`lora.py::_current_gate` is still a module global, and still open.** It is
acknowledged in `nodes/model/lora.py` and safe only because every trainer is
single-threaded. Fixing it means passing the gate to each layer instead of
having every `forward()` read a global: a real change to four layer classes
(plain, DoRA, NF4, plus the phase-split generation). Not attempted.
`archive/core/lora.py` deliberately does not re-export the global, since a
re-exported global is a stale snapshot pretending to be live.