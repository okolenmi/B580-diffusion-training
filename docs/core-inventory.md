# What `core/` is, and what only it has

`core/` is **the training engine**: ~7,700 lines across 23 modules,
and every training path in this repository runs through it -- including
the web UI's, which launches `python -m core.cli` as a supervised
subprocess.

It was long described as "legacy", which described when it was written
rather than what it does, and the label was actively misleading: it
implied dead or superseded code, and it is the trainer every real run
uses. As of 2026-10-02 the `nodes/` rewrite no longer imports any of
it, so what is left here is a record of the parts nothing else
provides.

The question "what is unique in `core/`?" has a shorter answer than
expected, and a more important one attached.

## The three things that depend on it

| Consumer | How it reaches `core/` | Breaks if `core/` goes |
|---|---|---|
| `python -m core.cli` (CLI) | direct; also the backend's spawned subprocess | The command-line trainer, entirely |
| `backend/` (web) | **spawns `<venv python> -m core.cli --config ...`** as the training subprocess (`infrastructure/subprocess_gateway.py:124`) | Every run started from the UI |
| `manager/builder.py` | imports 8 modules (`lora`, `model_io`, `noise_schedule`, `seed`, `unet_wrapper`, `vae_decode`, `comfy_setup`, and `clip_encode` -- all via the shims below) | Dataset ingestion, and therefore every run that has data |
| `nodes/` (rewrite) | **nothing** -- imports no `core.*` at all since 2026-10-02 | nothing |

The second row is the one that surprises people. The node-graph rewrite
is a *graph around* the trainer, not a replacement for it: pressing
**Start** in the web UI launches `python -m core.cli` as a child
process and then supervises it. The dataset path is the same shape --
the backend's ingest worker calls `manager.builder.DataTaskRunner`,
which calls into `core/`.

**Shrinking fast, as of 2026-10-02.** In one day the optimizer domain,
text encoding, and LoRA/UNet injection were all unwired:
`nodes/optimizer/` imports nothing from `core.optimizers`,
`SDXLClipEncoder` moved to `nodes/model/clip_encoder.py`, and
`LoRALinear`/`LoRAConv2d`/`_inject_lora`/`ComfyUNetWrapper`/`derive_seed`
moved to `nodes/model/{lora,unet_wrapper}.py` and
`nodes/components/seed.py`. `nodes/` now imports nothing from `core/`.

`core/lora.py`, `core/unet_wrapper.py`, `core/clip_encode.py` and
`core/seed.py` survive only as re-export shims, because `core/`'s own
trainer, its two cache builders and `manager/builder.py` still construct
these objects and `core/` is the production training path. They are not
dependencies of the node graph.

Smoke tests still import `core.optimizers` directly as *reference
implementations* -- that is the point of an equivalence test, and those
references remain valid precisely because `core/` is unchanged.

What still reaches into `core/` is `backend/`, for things that are
genuinely shared rather than unwired: `core.config_io`/`core.config_model`
(the TOML schema the config editor round-trips and the trainer reads),
`core.comfy_setup.xpu_empty_cache` (the graph runtime's memory releaser),
and `core.xpu_env` (SYCL variables, which must be set before anything
touches a device).

## Module by module

### Reimplemented or relocated into `nodes/` (the `core/` copies remain)

Only the optimizer *algorithms* got this treatment, and the design docs
are right about it: `nodes/optimizer/algorithms/` holds pure
per-parameter implementations of Adafactor and CAME, each verified
equivalence-tested against the `core/` version it replaces.

| `core/` module | Replacement | Notes |
|---|---|---|
| `optimizers.py` (1,410 lines) | `nodes/optimizer/` (Algorithm + ExecutionStrategy) | **Fully unwired 2026-10-02.** `nodes/optimizer/` imports nothing from `core.optimizers`; the last holdout node is deleted. |
| `noise_schedule.py` (schedule + conversions only) | `nodes/components/diffusion.py` | A fresh `NoiseSchedule`/`Parameterization` pair, and the rewrite is strictly *ahead* -- `RescaledZeroTerminalSNRSchedule` has no `core/` equivalent. `sample_timestep` and `T_MODES` did **not** move. |
| `schedules.py` (cosine/constant/warmup only) | `nodes/train/schedule.py` | `make_poly_lr` did not move. |

### Was wrapped at runtime (the rewrite could not work without them)

Strikethrough marks the entries the rewrite no longer depends on. They
are kept because the history is the useful part: it shows which
dependencies were genuinely load-bearing and what each one cost. What
remains live is the `backend/` and `manager/` traffic, which is shared
infrastructure rather than an unwired domain.

| `core/` module | Who calls it |
|---|---|
| ~~`lora.py` (556)~~ | **unwired 2026-10-02.** Now `nodes/model/lora.py`; `core/lora.py` is a re-export shim for `core/`'s and `manager/`'s use. The rewrite used to monkeypatch its `LoRALinear`/`LoRAConv2d` names to swap in its own layers; it now passes them to `_inject_lora` as an argument, and `lora_class_cache.py` is gone. |
| ~~`unet_wrapper.py`~~ | **unwired 2026-10-02.** Now `nodes/model/unet_wrapper.py`; `core/unet_wrapper.py` is a re-export shim. Still built directly by `manager/builder.py`. |
| ~~`clip_encode.py`~~ | **unwired 2026-10-02** -- the implementation now lives at `nodes/model/clip_encoder.py`; `core/clip_encode.py` is a re-export shim for `core/`'s and `manager/`'s own use. `manager/builder.py:20` is the remaining non-`core/` caller |
| ~~`optimizers.py` (strategies)~~ | **No longer wrapped.** The `AdafactorOptimizerNode` that built a `ChunkedXPUAdafactor` is deleted; see `docs/known-issues/open.md` for the one unmeasured performance trade that retirement accepted |
| `config_io.py`, `config_model.py` | `backend/infrastructure/{core_config_inspector,subprocess_gateway,config_schema}.py` -- the backend reads and validates configs through them |
| `comfy_setup.py` | `backend/infrastructure/graph/runtime.py:62` (`xpu_empty_cache` as the graph runtime's memory releaser), `manager/builder.py`. `nodes/` does **not** use this -- `nodes/components/device.py`'s `_XPUDeviceContext` is the rewrite, and `smoke_test_device_context_equivalence.py` proves the two agree |
| `xpu_env.py` | `backend/cli.py:41` -- the backend calls this before anything spawns a child, because SYCL reads these at its own runtime init |
| `noise_schedule.py` (partly) | the schedule and eps/vpred conversions **are** reimplemented in `nodes/components/diffusion.py`; but `sample_timestep` and the five `T_MODES` distributions are **not**, and they reach the rewrite one hop away through `manager/t_sampling.py` (see below) |
| `model_io.py` (partly) | the transforms are reimplemented as `KarrasInputScaler` / `Parameterization` objects; `make_init_noise` is not, and `manager/builder.py` uses it |

### A second tier that is easy to miss: `manager/` pulls six more in

`manager/` is not just `core/`'s own -- the backend imports it directly for
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

### Not reachable from the rewrite at all (only via `core.cli`, or `manager/`)

`trainer.py`, `train_step.py`, `optimizer_builder.py`, `progress_writer.py`,
`save.py`, `timer.py`, `cache_utils.py`, `cache_random.py`,
`cache_trajectory.py`, `preview_sampler.py`. The last three hold the
unique capabilities listed below.

### Genuinely unique -- nothing else in the repo provides it

This is the answer to the question asked. Each of these exists *only*
in `core/`, and the rewrite either borrows it or has no equivalent:

1. **Mid-run image previews.** `core/preview_sampler.py` samples images
   from the live UNet during training when `preview.enabled` is set, and
   `core/trainer.py:254` calls it. Nothing in `nodes/` or `backend/` can
   do this.
   *Currently an orphan:* the images are still written by every run that
   enables previews, but the endpoint that served them
   (`GET /runs/{id}/previews`) was dropped with the retired `server/` and
   the new UI never gained one. Either surface them again or stop
   spending the time generating them.
2. **The fused / chunked / foreach optimizer execution strategies.**
   `core/optimizers.py` holds six optimizer *classes* whose value is
   *how* the per-parameter math is batched on XPU
   (`FusedXPUAdafactor`, `ChunkedXPUAdafactor`, `ForeachXPUAdafactor`,
   `ChunkedXPUCAME`, `ForeachXPUCAME`, `CPUAdamW`). `nodes/` did
   reimplement the per-parameter *math* (`algorithms/{adamw,adafactor,
   came}.py`, verified equivalent) and fused execution
   (`composed_fused.py`) -- but the *batching* strategies were never
   rewritten, and none are now. `nodes/` no longer imports any of these;
   `docs/known-issues/open.md` records the one unmeasured performance
   trade that retiring the last wrapper accepted.
3. **The LoRA timestep gate.** `compute_lora_gate` / `set_lora_gate` /
   `lora_gate_override` restrict LoRA updates to the timestep range
   actually present in the data. This is the project's own contribution
   (documented in `docs/design/04-...md` section 3.1) and it still has
   no equivalent elsewhere; as of 2026-10-02 it lives in
   `nodes/model/lora.py` rather than `core/`, so "only `core/` has it"
   no longer applies -- what is true is that both trainers and every
   adapter layer (plain, DoRA, NF4) depend on it, and nothing else
   implements the concept.
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

## Why it still exists at all

Because of its *ownership model*, not its algorithms: `core/trainer.py`
is an 865-line class with the training loop, the caching, the optimizer
construction and the preview generation in one place, configured by a
TOML file rather than composed. Adopting a domain means moving the
*shared* implementation into `nodes/` -- which is now done for the
optimizer domain, text encoding, and LoRA/UNet injection -- but the
trainer loop itself has no `nodes/` counterpart yet, and `backend/`
spawns `core.cli` to run it.

So the honest position today: `core/` is simultaneously (a) the
production trainer, (b) the home of capabilities listed above that
nothing else provides, and (c) no longer a dependency of the node
graph. Retiring it means giving the trainer loop and the trainer-only
capabilities a home of their own -- which is
`docs/design/09-prioritized-backlog.md`'s remaining dataset-ingestion
item plus the trainer loop itself.

## Dead code found inside `core/`

Worth knowing before any refactor, so it is not mistaken for load-bearing:

* `core/cache_utils.py::shuffle_and_rebatch_cache` -- zero call sites
  anywhere in the repo (exported from `__init__`, never called).
* `unet_wrapper.py::clear_embedder_cache` (now
  `nodes/model/unet_wrapper.py`) -- imported at `core/trainer.py:38` and
  `core/train_step.py:28`, never called.
* `unet_wrapper.py::ComfyUNetWrapper.enable_gradient_checkpointing`
  (now `nodes/model/unet_wrapper.py`) -- no call sites; superseded by
  `nodes/model/gradient_checkpointing.py`.
* `lora.py::GroupedLoRALinear.forward` (now `nodes/model/lora.py`) -- an
  explicit `pass`.
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

**One process-global hazard remains; the other is fixed.**

* **Fixed -- the LoRALinear/LoRAConv2d monkeypatch.** `nodes/` used to
  rebind those two module globals for the duration of a build, which
  meant two concurrent `ComfyUNetLoRANode.build()` calls would corrupt
  each other's layers, and needed `lora_class_cache.py` to hand out the
  real classes to anything that recursed. Both are gone: `_inject_lora`
  takes the classes to build as an argument
  (`nodes/model/lora.py::inject_lora_into_unet`'s `layer_classes`), so
  there is no shared state and no cache to defeat. This also fixed four
  `isinstance` gates in `extract_lora_weights`/`load_lora_into_model`/
  `merge_lora_into_unet`/`lora_param_count` that had been silently
  checking the patched name and so skipped every DoRA and NF4 layer.
* **Still open -- `lora.py::_current_gate` is a module global.** It is
  acknowledged in `nodes/model/lora.py` and in
  `nodes/train/step_pipeline.py:201-207`, and it is safe only because
  both trainers are single-threaded. Fixing it properly means passing the
  gate to each layer instead of having every `forward()` read a global,
  which is a real change to four layer classes (plain, DoRA, NF4,
  plus the phase-split generation) and not attempted here. Note that
  `core/lora.py` deliberately does *not* re-export `_current_gate`, since
  a re-exported global is a stale snapshot pretending to be live.

## What deletion would actually mean

Not "removing a dead folder". It would mean:

1. The TOML trainer stops working (`python -m core.cli`, which is also
   what `run_server.sh`'s Start button spawns), so no run is produced at
   all.
2. Dataset ingestion stops, so no run has data.
3. Nothing in the node graph breaks -- `nodes/` imports no `core.*` at
   all. What breaks is `manager/builder.py`'s dataset ingestion, which
   goes through six `core/` modules (see the table above), and
   `backend/`'s config handling, graph-runtime memory releaser, and XPU
   environment setup.
4. Smoke tests fail wherever they use `core.*` as an equivalence
   reference -- which is most of the optimizer and LoRA suites, and is
   correct: those references are the point.

Removing it is only meaningful as a *migration* -- port the remaining
domains across first, which is exactly the work
`docs/design/09-prioritized-backlog.md` still lists.