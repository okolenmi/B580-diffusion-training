# What `core/` is, and what only it has

`core/` gets called "legacy" because of when it was written, not
because of what it does. It is **the training engine**: 7,707 lines
across 23 modules, and every training path in this repository runs
through it.

The question "what is unique in `core/`?" has a shorter answer than
expected, and a more important one attached.

## The three things that depend on it

| Consumer | How it reaches `core/` | Breaks if `core/` goes |
|---|---|---|
| `convert.py` (CLI) | `from core.cli import main` | The command-line trainer, entirely |
| `backend/` (web) | **spawns `<venv python> -m core.cli --config ...`** as the training subprocess (`infrastructure/subprocess_gateway.py:124`) | Every run started from the UI |
| `manager/builder.py` | imports 9 modules (`lora`, `clip_encode`, `model_io`, `noise_schedule`, `seed`, `unet_wrapper`, `vae_decode`, `comfy_setup`) | Dataset ingestion, and therefore every run that has data |
| `nodes/` (rewrite) | lazy imports of `core.lora`, `core.optimizers`, `core.clip_encode`, `core.unet_wrapper` | LoRA injection, the text encoder, the UNet wrapper, one optimizer node |

The second row is the one that surprises people. The node-graph rewrite
is a *graph around* the trainer, not a replacement for it: pressing
**Start** in the web UI launches `python -m core.cli` as a child
process and then supervises it. The dataset path is the same shape --
the backend's ingest worker calls `manager.builder.DataTaskRunner`,
which calls into `core/`.

23 of the 66 `nodes/` smoke tests import `core.*` directly (every LoRA
test, every optimizer equivalence test, the diffusion equivalence test),
so the test suite would go red immediately.

## Module by module

### Reimplemented in `nodes/` (legacy-only copies remain)

Only the optimizer *algorithms* got this treatment, and the design docs
are right about it: `nodes/optimizer/algorithms/` holds pure
per-parameter implementations of Adafactor and CAME, each verified
equivalence-tested against the `core/` version it replaces.

| `core/` module | Replacement | Notes |
|---|---|---|
| `optimizers.py` (1,410 lines) | `nodes/optimizer/algorithms/{adafactor,came,adamw}.py` | Algorithm math only. The *batching* and *fused* strategies still live in `core/` and are still wrapped. |
| `noise_schedule.py` (schedule + conversions only) | `nodes/components/diffusion.py` | A fresh `NoiseSchedule`/`Parameterization` pair, and the rewrite is strictly *ahead* -- `RescaledZeroTerminalSNRSchedule` has no `core/` equivalent. `sample_timestep` and `T_MODES` did **not** move. |
| `schedules.py` (cosine/constant/warmup only) | `nodes/train/schedule.py` | `make_poly_lr` did not move. |

### Wrapped at runtime (the rewrite cannot work without them)

| `core/` module | Who calls it |
|---|---|
| `lora.py` (556) | `nodes/model/adapter_injection.py` **monkeypatches `core.lora.LoRALinear`/`LoRAConv2d`** to swap in the rewrite's own layer classes; `lora_class_cache.py`, `lora_gate.py`, `lora_injector.py`, `lora_checkpoint_loader.py`, `train/step_pipeline.py`, `train/managed.py`, `train/t_probe.py` all import from it. 150 references. |
| `unet_wrapper.py` | `nodes/model/lora_injector.py:228`, `manager/builder.py:25` -- the ComfyUI UNet adapter itself |
| `clip_encode.py` | `nodes/model/text_encoder.py:168`, `sdxl_architecture.py:51`, `manager/builder.py:20` -- the SDXL text encoder |
| `optimizers.py` (strategies) | `nodes/optimizer/adafactor.py:103` still builds a `ChunkedXPUAdafactor`; that node is the last un-retired legacy wrapper (tracked in `docs/design/09-prioritized-backlog.md`), and it is still graph-reachable via `pkgutil` discovery |
| `config_io.py`, `config_model.py` | `backend/infrastructure/{core_config_inspector,subprocess_gateway,config_schema}.py` -- the backend reads and validates configs through them |
| `comfy_setup.py` | `backend/infrastructure/graph/runtime.py:62` (`xpu_empty_cache` as the graph runtime's memory releaser), `manager/builder.py` |
| `xpu_env.py` | `backend/cli.py:41` -- the backend calls this before anything spawns a child, because SYCL reads these at its own runtime init |
| `noise_schedule.py` (partly) | the schedule and eps/vpred conversions **are** reimplemented in `nodes/components/diffusion.py`; but `sample_timestep` and the five `T_MODES` distributions are **not**, and they reach the rewrite one hop away through `manager/t_sampling.py` (see below) |
| `model_io.py` (partly) | the transforms are reimplemented as `KarrasInputScaler` / `Parameterization` objects; `make_init_noise` is not, and `manager/builder.py` uses it |

### A second tier that is easy to miss: `manager/` pulls six more in

`manager/` is not legacy-only -- the backend imports it directly for
dataset ingestion (`backend/infrastructure/dataset_task_worker.py:79`
runs `manager.builder.DataTaskRunner`). So `core/` is load-bearing for
the rewrite *through* `manager/` as well:

| `core/` module | Reached via | Consequence if gone |
|---|---|---|
| `vae_decode.py` | `manager/builder.py:26`, `manager/preview.py:11` | **Backend dataset ingestion breaks** -- both preview decoding and the LoRA-raw-image encode path |
| `noise_schedule.py` (`sample_timestep`, `T_MODES`) | `manager/t_sampling.py:50` <- `manager/loader.py` <- `nodes/dataset/managed.py:110` | The five timestep distributions die, taking `ManagedDatasetSourceNode`'s `t_mode` Port with them. `nodes/dataset/timestep_modes.py` is a deliberate *copy* of the constant, not a replacement |
| `noise_schedule.py` (`get_alpha_sigma`, `eps_to_vpred`) | `manager/loader.py:12` | same live path |
| `model_io.py` (`make_init_noise`) | `manager/builder.py:22` | teacher-trajectory initial noise |
| `seed.py` (`derive_seed`) | `manager/builder.py:24` | deterministic seeding across four call sites |
| `comfy_setup.py` | `manager/builder.py:21`, `manager/preview.py:12` | device/XPU primitives during ingest |

A plan of "delete `core/`, keep the rewrite" therefore breaks dataset
ingestion and dataset `t_mode`, not just the node graph.

### Legacy-only (reachable only via `convert.py` -> `core.cli`, or `manager/` alone)

These are *not* imported by the rewrite: `trainer.py`, `train_step.py`,
`optimizer_builder.py`, `progress_writer.py`, `save.py`, `timer.py`,
`cache_utils.py`, `cache_random.py`, `cache_trajectory.py`,
`preview_sampler.py`. The last three are where the unique capabilities
below live.

### Genuinely unique -- nothing else in the repo provides it

This is the answer to the question asked. Each of these exists *only*
in `core/`, and the rewrite either borrows it or has no equivalent:

1. **Mid-run image previews.** `core/preview_sampler.py` samples images
   from the live UNet during training when `preview.enabled` is set, and
   `core/trainer.py:254` calls it. Nothing in `nodes/` or `backend/` can
   do this.
   *Currently an orphan:* the images are still written by every run that
   enables previews, but the endpoint that served them
   (`GET /runs/{id}/previews`) was dropped with the legacy server and
   the new UI never gained one. Either surface them again or stop
   spending the time generating them.
2. **The fused / chunked / foreach optimizer execution strategies.**
   `core/optimizers.py` holds six optimizer *classes* whose value is
   *how* the per-parameter math is batched on XPU
   (`FusedXPUAdafactor`, `ChunkedXPUAdafactor`, `ForeachXPUAdafactor`,
   `ChunkedXPUCAME`, `ForeachXPUCAME`, `CPUAdamW`). `nodes/` did
   reimplement the per-parameter *math* (`algorithms/{adamw,adafactor,
   came}.py`, verified equivalent) and fused execution
   (`composed_fused.py`) -- but it kept borrowing the *batching*
   strategies, which is why one live node (`AdafactorOptimizerNode`)
   still wraps `ChunkedXPUAdafactor`.
3. **The LoRA timestep gate.** `core/lora.py`'s `compute_lora_gate` /
   `set_lora_gate` / `lora_gate_override` restrict LoRA updates to the
   timestep range actually present in the data. The rewrite *consumes*
   it from three call sites and reimplements nothing -- it is the
   project's own contribution, documented in
   `docs/design/04-...md` section 3.1, and it lives here.
4. **Latent caching.** `core/cache_trajectory.py` (646) and
   `cache_random.py` (186) precompute VAE latents for teacher and random
   trajectories. The rewrite's `ManagedLoRATrainerNode` uses its own
   single-latent path and does not replace these.
5. **Adversarial pre-conditioning.** `core/train_step.py:265-301` --
   low-power cross-conditioning drafts between the conditioned and
   unconditioned passes, with a clean-step ratio. No `nodes/` equivalent.
6. **The five timestep distributions.** `core/noise_schedule.py`'s
   `sample_timestep` (`uniform`/`low`/`mid`/`high`/`logit`) plus
   `T_MODES`. The rewrite copies the *constant* but not the
   implementation, and reaches the function through `manager/`.
7. **Latent target math.** `core/model_io.py`'s `comfy_input_transform`
   / `raw_to_denoised` / `raw_to_target` -- the conversion from raw
   model output to the training target per parameterization -- plus
   `make_init_noise`, which is *not* reimplemented.
8. **VAE decode.** `core/vae_decode.py`, including ComfyUI's
   `DEFAULT_SCALE_FACTOR = 0.13025` pairing -- and it is load-bearing
   for backend dataset ingestion, not just core.
9. **Mid-run checkpointing and optimizer-state persistence.**
   `core/save.py` -- and the on-disk optimizer-state safetensors schema
   (`__step__`, `vr_i`/`vc_i`/`vs_i`/`ea_i`, `resr_i`/`resc_i`,
   `__tiny_vs_*__`) is a *resume-compatibility contract*: nothing else can
   read a run already on disk.
10. **Cyclic training.** `core/trainer.py:577-658` -- N-step cycles with
    cache rebuild and teacher offload/reload between them.
11. **Radial per-UNet-block LR grouping.** `core/optimizer_builder.py:91-134`
    -- interpolated LR across `input_blocks`/`output_blocks` with separate
    `time_embed`/`label_emb` multipliers. LoRA mode silently ignores it,
    which the module warns about.
12. **Polynomial LR decay.** `core/schedules.py`'s `make_poly_lr`.
    Cosine, constant and warmup *are* in `nodes/train/schedule.py`; poly
    is not.
13. **Welford loss window, `dW` weight-drift tracking and a background
    GC worker.** `core/train_step.py`. Of the three, only the per-t
    loss breakdown was rebuilt (`nodes/train/loss.py`).
14. **The `.progress.jsonl` phase protocol.**
    `core/progress_writer.py` -- `cache_start`/`cache_done`/
    `training_start`/`step`/`done`. Legacy consumers only, but a defined
    contract with a flush-interval choice (0.4 s) that is easy to lose.
15. **The XPU performance environment contract.** `core/xpu_env.py` sets
    the `SYCL_*` / `UR_L0_*` / `IGC_*` variables before torch loads. The
    symptom it targets (fast steps, then a long stall on a new
    resolution) is recorded nowhere else.
16. **Parallel `pin_memory()` and an O(1) cache re-batch.**
    `core/cache_utils.py` -- the re-batch function understands the
    v1/v2/v3/v4 cache tuple layouts and has **zero call sites** (see
    dead code below).

## So why does it read as legacy?

Because of its *ownership model*, not its algorithms: `core/trainer.py`
is a 865-line class with the training loop, the caching, the optimizer
construction and the preview generation in one place, configured by a
TOML file rather than composed. `docs/architecture.md` records the
position this project took: `core/` is the production path, its bugs get
fixed in place, and where `nodes/` builds a verified-equivalent version
that version becomes canonical and the old wrapper retires. That has
happened for the optimizer algorithms. It has not happened for LoRA
injection, text encoding, dataset ingestion or the trainer loop --
`docs/architecture.md` says so explicitly.

## Dead code found inside `core/`

Worth knowing before any refactor, so it is not mistaken for load-bearing:

* `core/cache_utils.py::shuffle_and_rebatch_cache` -- zero call sites
  anywhere in the repo (exported from `__init__`, never called).
* `core/unet_wrapper.py::clear_embedder_cache` -- imported at
  `trainer.py:38` and `train_step.py:28`, never called.
* `core/unet_wrapper.py::ComfyUNetWrapper.enable_gradient_checkpointing`
  -- no call sites; superseded by `nodes/model/gradient_checkpointing.py`.
* `core/lora.py::GroupedLoRALinear.forward` -- an explicit `pass`.
* `core/__init__.py`'s `load_config` alias -- kept for compatibility,
  unused.

## Two hazards the audit surfaced

**The XPU env ordering was silently wrong, and is now fixed.**
`backend/cli.py:41` imported `core.xpu_env` with the comment "Pure
os.environ writes, no torch import -- safe before any child". That
comment was false: importing any `core.*` submodule runs
`core/__init__.py`, which eagerly re-exported `optimizers` and
`unet_wrapper` -- so `torch` was already loaded two lines *before* the
SYCL variables were set. Verified: `torch` appeared in `sys.modules`
across that single import. It was probably harmless in practice (SYCL
reads those variables at its own runtime init, not at `import torch`),
but "probably" is not a contract.

`core/__init__.py` is now a lazy PEP 562 facade: the same names resolve
identically, and `from core.xpu_env import ...` no longer pulls torch.
Every re-export in that file was verified unused outside `core/`, and the
only external `from core import ...` uses submodules, so nothing else
could regress.

**Two process-global concurrency hazards, documented but not fixed:**

* `core/lora.py::_current_gate` is a module global. It is acknowledged as
  a hazard in `core/lora.py:41` and in
  `nodes/train/step_pipeline.py:201-207`, and it is safe only because
  both trainers are single-threaded.
* `nodes/model/adapter_injection.py` monkeypatches the module globals
  `core.lora.LoRALinear` / `LoRAConv2d` for the duration of
  `adapter_strategy_scope`, so two concurrent `build()` calls would race.
  The module's own docstring names the risk.

## What deletion would actually mean

Not "removing a dead folder". It would mean:

1. `convert.py` stops working, and `run_server.sh`'s Start button stops
   producing a training run.
2. Dataset ingestion stops, so no run has data.
3. The rewrite loses LoRA injection, the text encoder, the UNet wrapper
   and one optimizer node -- roughly the domains the design docs list as
   "not yet rewritten".
4. 23 of 66 smoke tests fail.

Removing it is only meaningful as a *migration* -- port the remaining
domains across first, which is exactly the work
`docs/design/09-prioritized-backlog.md` still lists.