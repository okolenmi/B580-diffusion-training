# Audit: model loading and SDXL coupling (read from code, origin/main a7cc71e)

## 1. Loading paths today

**Route A, node graph ("main").**
`SafetensorsCheckpointNode(path)` -> `ModelWeights` (lazy handle) ->
`ComfyUNetLoRANode` (injects LoRA into the UNet) and `SDXLTextEncoderNode`
(builds the text encoder from the same `ModelWeights`).
`nodes/model/checkpoint_loader.py:25`, `nodes/model/handle.py`.

* The node only resolves the path inside the sandbox and opens the safetensors
  header (`safe_open`, `next(iter(keys))`). It does **not** detect what the
  file contains; "is this SDXL" is assumed.
* `ModelWeights` loads the full file on first access and splits it into
  `unet_sd` and `non_unet_sd` with one prefix test
  (`resource_inspection._is_unet_key`: `model.diffusion_model.`). CLIP and VAE
  live together in `non_unet_sd`; the text encoder node later pulls CLIP out of
  that bucket itself.
* Output type is `ModelWeights`: **a single handle that means "an SDXL-layout
  single-file checkpoint"**. Separate UNet / CLIP / VAE files cannot be
  expressed.

**Route B, managed.**
`ResourcesControllerNode(preset, checkpoint_path, device, unet_dtype,
continue_lora..., frozen_lora...)` -> `LoRATrainingResources`
(`SDXL_LoRATrainingResources`) -> `LoRATrainingConfigNode` -> `LoRATrainingSkeleton`
(`SDXL_LoraTrainer`) -> `ManagedLoRATrainerNode`.

* `ResourcePreset` (`resources_controller.py:81`) is a prior attempt at the
  family idea: "one concrete (task, architecture) pair"; one concrete preset,
  `LoRASDXLPreset`; the docstring says a second preset is a second instance.
  The validator for it is the string check `'model.diffusion_model.' prefix`
  (`resources_controller.py:155-165`).
* `SDXLArchitecture` (`nodes/model/sdxl_architecture.py`) is a **mixin**
  providing `split_checkpoint`, `build_text_encoder`, `inject_lora`, combined
  with a task ABC by inheritance (`SDXL_LoraTrainer(SDXLArchitecture,
  LoRATrainingSkeleton)`; base order matters or abstract stubs shadow the real
  methods). Its docstring: "No generic multi-architecture abstraction exists;
  deliberately specific to SDXL".
* So the codebase already has the *shape* of a family (three methods) but it
  is fused with the task (LoRA training) and with a single file.

**VAE.** Not a graph resource at all. `manager/builder.py` loads the VAE from
the checkpoint (`first_stage_model.` prefix stripped) in the dataset ingest
step, encodes images to latents (`VAEDecoder.encode`, scale factor 0.13025,
`px = latent_size * 8`, noise shape `(1, 4, ls, ls)`) and writes **latent
shards** (`manager/storage.py`, layout `single_latent`). The preview generator
decodes with the same VAE. The trainer consumes shards and never touches a VAE.
So "Data - VAE" in the owner's mapping is right, but today it runs in the
`manager/` ingest, outside the node graph.

**Folders / scopes.** `ProjectLayout` has `checkpoints_dir`, `loras_dir`,
`datasets_dir`, `runs_dir` (+ comfy_dir). `path_kind` values in use:
`checkpoint` (3 ports), `lora` (3), `lora_output` (1), `dataset` (2). There is
**no scope for UNet-only files, CLIPs or VAEs.** The UI picker lists the whole
`checkpoint` folder.

**What detection exists.** Header-only inspection:
`inspect_checkpoint_dtypes` (per-component dtype and key count by prefix) and
`inspect_lora` (dtype and rank). Prefix classes: `model.diffusion_model.` ->
unet, `first_stage_model.` -> vae, everything else -> "clip".

## 2. SDXL facts and where they live

| Fact | Where | General or SDXL-only |
|---|---|---|
| UNet key prefix `model.diffusion_model.`; VAE `first_stage_model.`; CLIP = rest | `resource_inspection.py` `classify_key`, `_is_unet_key`, `_is_vae_key`; `ModelWeights._ensure_loaded` | SDXL single-file (also SD1.5/SD2 layouts) |
| CLIP-L + OpenCLIP-G (`conditioner.embedders.0/1`) extracted together | `clip_encoder.SDXLClipEncoder`, `clip.py` `SDXLClipModel`, `SDXLClipG` | SDXL-only; `CLIPTextModel`, `CLIPEncoder`, `SDClipModel` are reusable |
| Conditioning pair `(context (B,77,2048), pooled y)`; `y` = pooled 1280 + six 256-wide size embeddings | `text_encoder.py` `TextEncoder` ABC (`encode_prompt_only`, `resolution_embedding`, `encode`, `encode_per_sample`, `encode_per_sample_prompts`, `unload`), `unet_wrapper.py` `_embedder`, `make_rand_cond` | ABC is shaped by SDXL (resolution embedding is part of the contract) |
| Cache key `(prompt, batch_size)` and `(h, w, n)`; sizes taken as `x_t.shape * 8` | `text_encoder_cache.py`, `text_encoder_prewarm.py:269` | cache logic general; the `* 8` and the key shape are SDXL |
| UNet architecture (`UNetModel`, ResBlock, SpatialTransformer path, `input_blocks`, `middle_block`, `output_blocks`) | `nodes/model/unet.py`, `attention.py` | SDXL / SD-family UNet |
| Forward signature `forward(x_t, timestep, context, y)` | `TrainableModel` ABC (`handle.py`), `ComfyUNetWrapper.forward` | UNet-style; a DiT with 5-D latents and no `y` would not fit as is |
| LoRA targets `to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2 proj_in proj_out (+ label_emb, time_embed)`; "722 / 724 / 726 modules on SDXL" | `lora.py:128-145` | names are UNet/diffusers-style, family-specific |
| LoRA file keys `lora_unet_<path with _>` (kohya/ComfyUI) | `lora_phases.lora_key`, `lora_checkpoint_loader` | the *file format* is general; the `lora_unet_` prefix and the path are family-specific |
| VAE `AutoencoderKL`, scale factor 0.13025, 4 latent channels, spatial factor 8 | `vae.py`, `vae_decode.py:32`, `manager/builder.py`, `unet_wrapper.py:241` | SDXL |
| Noise schedule (scaled-linear betas, `alphas_cumprod`, zero-terminal-SNR rescale option), eps and v-prediction converters | `components/diffusion.py`, `components/noise_schedule.py`, `components/model_io.py`, `manager/loader.py:476`, shard DB column `model_type eps|vpred` | the eps/vpred machinery is general to ddpm-style models; flow matching is absent |
| Teacher/student eps<->vpred conversion | `nodes/config_model.py`, `supervised.py` | project origin: distillation; still present |
| LoRA injection function with 12 parameters | `lora_injector.build_lora_injected_unet` | the options (rank, alpha, dropout, strategy, scaling, targets, block weights, checkpointing, weight store) are general; the walk over `ComfyUNetWrapper` modules is not |
| Activation checkpointing replacement | `checkpoint.py` | general |
| Frozen weight store, DeviceResident (offload/reload/release/footprint) | `memory/handle.py`, `model/frozen_weight_store.py` | general |

## 3. The method surface the rest of the code calls

What trainers, phases, cache, saver and probes need from "a model family"
(derived from signatures above, not invented):

1. **Inspect** (cheap, header-only): `classify_key`-style partition of a file's
   keys into parts, per-part dtype and key count.
2. **Split / load parts**: `split_checkpoint(state_dict) -> {unet, clip, vae}`.
3. **Build text encoder(s)**: `build_text_encoder(clip_sd, device) -> TextEncoder`.
4. **Build the trainable denoiser with an adapter**: `inject_lora(unet_sd,
   **options) -> TrainableModel` with `forward`, `trainable_parameters`,
   `train/eval/to`, `trained_state_dict`, `footprint_bytes/offload/reload/release`.
5. **Conditioning contract**: how a prompt and an image size become what
   `forward` takes (`encode`, `resolution_embedding`).
6. **Latent contract**: encode image -> latent (scale, channels, spatial factor),
   decode latent -> image (previews).
7. **Parameterization**: what the network predicts (eps, v, flow velocity) and
   how to build the training target from `x0`, noise, timestep.
8. **Adapter layout**: which modules can carry an adapter, their names, and the
   save key scheme.

## 4. Assumptions that would break for a DiT such as Anima

(Anima facts are collected in step 4; these are the assumptions in *our* code.)

* Latents are 4-D `(B, 4, H/8, W/8)`; a Cosmos-style model works on a video
  tensor layout (to be verified).
* `y` (pooled + size embeddings) is part of the call and of the cache; a model
  without a vector conditioning has no `y`.
* Timesteps are integers in a DDPM schedule; flow-matching models sample a
  continuous `t` and predict velocity.
* `TextEncoder.encode` returns a fixed pair; a Qwen encoder returns a variable
  length sequence of hidden states, possibly through an adapter.
* Module names for LoRA targets, the block-weight scheme (`input_blocks`,
  `middle_block`, `output_blocks`), and the `time_embed`/`label_emb` special
  cases are UNet-specific.

## 5. Coupling hot spots (what a second family forces us to edit today)

1. `resource_inspection.classify_key` and `_is_unet_key` (prefix constants).
2. `ModelWeights` (name and meaning: unet/non-unet split baked in).
3. `SafetensorsCheckpointNode` (single path, `checkpoint` scope, no detection).
4. `SDXLArchitecture` mixin and the `SDXL_*` classes plus the dict mapping
   `SDXL_LoRATrainingResources -> SDXL_LoraTrainer` (`lora_training_config.py:71`).
5. `ResourcePreset` / `LoRASDXLPreset` and the `preset` port.
6. `TextEncoder` ABC (SDXL pair and resolution embedding in the contract).
7. `text_encoder_cache.py`, `text_encoder_prewarm.py` (key shape, `* 8`).
8. `TrainableModel.forward(x_t, timestep, context, y)` and `ComfyUNetWrapper`.
9. `lora.py` target sets and block weights; `lora_phases.lora_key`.
10. `vae_decode.py`, `vae.py`, `manager/builder.py` (scale, channels, factor,
    `first_stage_model.`), and `manager/storage.py` shard layout.
11. `components/diffusion.py` / `noise_schedule.py` / `model_io.py` (no flow
    matching) and `manager/loader.py` target construction.
12. `ProjectLayout` (no UNet / CLIP / VAE scopes) and `path_kind` values.
13. `step_pipeline.py` phases that call the model with the SDXL signature
    (to be read in detail in plan 70).
