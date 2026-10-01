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
| `noise_schedule.py` | `nodes/components/diffusion.py` | A fresh `NoiseSchedule`/`Parameterization` pair. `manager/` and `backend/` still import `core.noise_schedule` directly. |

### Wrapped at runtime (the rewrite cannot work without them)

| `core/` module | Who calls it |
|---|---|
| `lora.py` (556) | `nodes/model/adapter_injection.py` **monkeypatches `core.lora.LoRALinear`/`LoRAConv2d`** to swap in the rewrite's own layer classes; `lora_class_cache.py`, `lora_gate.py`, `lora_injector.py`, `lora_checkpoint_loader.py`, `train/step_pipeline.py`, `train/managed.py`, `train/t_probe.py` all import from it. 150 references. |
| `unet_wrapper.py` | `nodes/model/lora_injector.py:228`, `manager/builder.py:25` -- the ComfyUI UNet adapter itself |
| `clip_encode.py` | `nodes/model/text_encoder.py:168`, `sdxl_architecture.py:51`, `manager/builder.py:20` -- the SDXL text encoder |
| `optimizers.py` (strategies) | `nodes/optimizer/adafactor.py:103` still builds a `ChunkedXPUAdafactor`; that node is the last un-retired legacy wrapper (tracked in `docs/design/09-prioritized-backlog.md`) |
| `config_io.py`, `config_model.py` | `backend/infrastructure/{core_config_inspector,subprocess_gateway,config_schema}.py` -- the backend reads and validates configs through them |
| `comfy_setup.py` | `backend/infrastructure/graph/runtime.py:62` (`xpu_empty_cache` as the graph runtime's memory releaser), `manager/builder.py` |
| `xpu_env.py` | `backend/cli.py:41` -- **the backend calls this before anything touches torch**, because SYCL reads these at its own runtime init |

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
   `ChunkedXPUCAME`, `ForeachXPUCAME`, `CPUAdamW`). `nodes/`
   reimplemented the per-parameter math and kept borrowing these.
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
5. **Latent target math.** `core/model_io.py`'s
   `comfy_input_transform` / `raw_to_denoised` / `raw_to_target` -- the
   conversion from raw model output to the training target for each
   parameterization. `manager/builder.py` depends on it.
6. **VAE decode.** `core/vae_decode.py` -- used by preview sampling and
   by `manager/preview.py`.
7. **Mid-run checkpointing.** `core/save.py`'s `save_midrun`, plus the
   optimizer-state serialization the resume path depends on.
8. **The XPU performance environment contract.** `core/xpu_env.py` sets
   the `SYCL_*` / `UR_L0_*` / `IGC_*` variables before torch loads. The
   symptom it targets (fast steps, then a long stall on a new
   resolution) is recorded nowhere else.

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