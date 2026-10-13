# Research: Anima (facts for the loader and the family design)

Collected 2026-10-13 by reading the sources below. Each claim says where it
came from. **V** = stated by a source, **R** = recollection from general
knowledge, not verified here (check before relying on it).

## Sources

* Model card: https://huggingface.co/circlestone-labs/Anima
* Diffusers pipeline docs: https://huggingface.co/docs/diffusers/en/api/pipelines/anima
* sd-scripts training guide:
  https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md
* ComfyUI detection code:
  https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_detection.py

## 1. What it is (V, model card)

* Released by CircleStone Labs with Comfy Org. About 2B parameters, text to
  image, anime and illustration, built on `nvidia/Cosmos-Predict2-2B-Text2Image`.
* Versions: **Anima-Base** (pretrained; **LoRAs should be trained on this**),
  Anima-Aesthetic (fine-tuned for style), Anima-Turbo (distilled, CFG 1, 8 to
  12 steps).
* License: CircleStone Labs Non-Commercial License, and the NVIDIA Open Model
  License to the extent it is a derivative of Cosmos-Predict2-2B.
* Resolution 512 to 1536 px. Training data: several million anime images and
  about 800k non-anime art, no synthetic data; random tag dropout was used.

## 2. Files and folders (V, model card)

| File | ComfyUI folder | Part |
|---|---|---|
| `anima-base-v1.0.safetensors` | `models/diffusion_models` | DiT (diffusion transformer); the LLM adapter keys live inside this file |
| `qwen_3_06b_base.safetensors` | `models/text_encoders` | Qwen3 0.6B base |
| `qwen_image_vae.safetensors` | `models/vae` | Qwen-Image VAE |

So the normal Anima setup is **three separate files**, not one checkpoint.
A "Diffusers" layout also exists (`circlestone-labs/Anima-Base-v1.0-Diffusers`).
sd-scripts: "If [the adapter path is] omitted, it loads from the DiT file when
the key `llm_adapter.out_proj.weight` exists" (V).

## 3. Architecture (V unless marked)

* Denoiser: Cosmos transformer (diffusers `CosmosTransformer3DModel`).
  ComfyUI detection for the 2B case: `model_channels` 2048, `num_heads` 16,
  `patch_spatial` 2, `patch_temporal` 1, `crossattn_emb_channels` 1024,
  `pos_emb_cls` rope3d, `use_adaln_lora` true with `adaln_lora_dim` 256,
  `in_channels` 16 (text to image; 17 for the image-to-video variant), a
  padding-mask channel concatenated. Block count is counted from keys.
  sd-scripts mentions a "28-block model (Anima-Preview)".
* Text path: **Qwen3-0.6B** (`Qwen3Model`, `Qwen2Tokenizer`) gives hidden
  states; an **LLM adapter** (diffusers `AnimaTextConditioner`: 6 layers,
  16 heads, model dim 1024, `target_vocab_size` 32128) cross-attends from
  learned **T5 token embeddings** (a `T5Tokenizer` encodes the same prompt;
  its ids act as query tokens) to the Qwen3 hidden states, and its output is
  the cross-attention context of the transformer. Minimum sequence length 512,
  `max_sequence_length` default 512. How the two token sequences are paired is
  not described in the docs read (gap).
* The model card warns that the adapter "has outsized influence" and says
  **do not train it** (`llm_adapter_lr=0` in diffusion-pipe; sd-scripts has
  options; guide suggests about 5e-5 if it is trained).
* VAE: Qwen-Image VAE, `AutoencoderKLQwenImage`; sd-scripts: fixed config
  `dim=96, z_dim=16`; **16 latent channels, 8x spatial downscale**. A 2-D
  image-only conversion exists (about 2x faster, about 1/3 peak VRAM, single
  image latents match the 3-D VAE numerically per the guide). Latent scale
  factor: **not stated** in any source read (gap).
* Diffusion formulation: **rectified flow**; diffusers uses
  `FlowMatchEulerDiscreteScheduler` (shift and other parameters not stated).
  sd-scripts timestep sampling: `sigma`, `uniform`, `sigmoid` (default),
  `shift`, `flux_shift`; `discrete_flow_shift` default 1.0,
  `t_shifted = t*shift / (1 + (shift-1)*t)`; weighting schemes `uniform`,
  `sigma_sqrt`, `cosmap`, `none`, `logit_normal`, `mode`.
* Latents are video-style (the transformer has a temporal patch size; a still
  image is a single frame). Tensor layout is 5-D `(B, C, T, H, W)` (R; the
  "3D" in the VAE conversion note supports it).

## 4. Fingerprints (what a loader can test without loading tensors)

Anima DiT (V, ComfyUI `detect_unet_config`):

* `"<prefix>blocks.0.mlp.layer1.weight"` present -> Cosmos-Predict2 family.
* `"<prefix>llm_adapter.blocks.0.cross_attn.q_proj.weight"` present -> **Anima**
  (otherwise plain Cosmos-Predict2).
* `x_embedder.proj.1.weight` carries width and channel count.
* Prefix: ComfyUI's `unet_prefix_from_state_dict` tries candidates including
  `"net."` (annotated cosmos) and falls back to `"model."`; it picks the
  prefix with most keys (needs more than 5). So a file may use `net.` or
  `model.diffusion_model.`; the detector must try both.

Qwen3-0.6B encoder (R): HuggingFace layout `model.embed_tokens.weight`,
`model.layers.N.self_attn.{q,k,v,o}_proj.weight`, `q_norm`/`k_norm`,
`model.norm.weight`, possibly `lm_head.weight`. 28 layers, hidden size 1024
(R).

Qwen-Image VAE (R): Wan-2.1 style keys (`encoder.downsamples.*`,
`decoder.upsamples.*`, `encoder.middle`, `conv1`, `conv2`, `gamma` tensors).

SDXL single file (V from this repo's code, `resource_inspection.py`):
`model.diffusion_model.`, `first_stage_model.`,
`conditioner.embedders.0.` (CLIP-L) and `conditioner.embedders.1.` (OpenCLIP-G).

Plausible next families (R; each needs a verified fingerprint when it is
added, none is needed now): SD 1.5 (`model.diffusion_model.` with cross
attention width 768, a single `cond_stage_model.`), Flux (`double_blocks.`,
`single_blocks.`), Wan (video DiT with 3-D patch embedding), Qwen-Image
(MMDiT with Qwen2.5-VL encoder).

## 5. Training facts that shape the design (V, sd-scripts guide)

* LoRA targets: inside each `Block` the `self_attn`, `cross_attn` and `mlp`
  linears (module names like `blocks.0.self_attn.q_proj`). Default exclusions
  by regex: `.*(_modulation|_norm|_embedder|final_layer).*`. Optional:
  `PatchEmbed`, `TimestepEmbedding`, `FinalLayer`; adapter blocks only if
  `train_llm_adapter` is on. `include_patterns` / `exclude_patterns` use
  `re.fullmatch`; per-module rank and LR overrides exist. Also LoRA+, rank
  dropout, module dropout.
  **This is the pattern the owner asked for: the family defines module groups
  and the user picks which groups train.**
* Rank example 8 with LR 1e-4 (alpha 1.0); model card: about 2e-5 for rank 32.
  Optimizer example AdamW8bit, constant schedule, bf16.
* **fp8 base is not supported** for Anima in sd-scripts (disabled with a
  warning). Memory levers: `blocks_to_swap` (offload blocks to host; max 26 of
  28), gradient checkpointing, cached latents, cached text-encoder outputs,
  `--qwen_image_vae_2d`, `--vae_chunk_size`.
* Cached text-encoder outputs are incompatible with training a text-encoder
  LoRA; caption dropout with cached outputs stores the rate in the cache
  (change it -> regenerate the cache). This matches our prewarm/cache design.
* ComfyUI does not directly load sd-scripts-format LoRAs that include Qwen3
  weights; a conversion script exists (`networks/convert_anima_lora_to_comfy.py`).
  LoRA key naming is **not documented** in the guide (gap): needed before the
  saver can be written.
* Captions: lowercase tags with spaces; quality/meta/year/safety first, then
  subject, character, series, artist (`@` prefix), general tags; dataset tags
  on a first line then a newline, optional second line with alt text;
  natural-language prompts should be at least two sentences. This is relevant
  to caption stages (plan 50): the Anima prompt format is structured text, not
  one tag list.
* Whether a 12 GB card can train it: the model is about 2B parameters; with
  block swap and no fp8 the guide's own recommendations apply. Nothing was
  measured here (gap).

## 6. ComfyUI loader conventions (R, for the scope design)

`CheckpointLoaderSimple` (folder `checkpoints`, returns MODEL, CLIP, VAE),
`UNETLoader` (`diffusion_models`, formerly `unet`), `CLIPLoader` /
`DualCLIPLoader` (`text_encoders`, formerly `clip`, with a `type` selector),
`VAELoader` (`vae`). Folder categories are configured in
`extra_model_paths.yaml`. The user-facing rule: **the folder says what you
may pick; the loader decides what comes out.**

## 7. Gaps to close before implementing the Anima family

1. LoRA key naming that ComfyUI accepts for Anima.
2. Latent scale/shift constants for the Qwen-Image VAE.
3. Exact pairing of Qwen tokens and T5 ids in the adapter.
4. Scheduler shift used at inference, and which sampling to train with.
5. Memory measurements on the B580 with block swap.
All five are checkable by reading the open-source implementations
(sd-scripts `library/anima_*`, ComfyUI `comfy/ldm/cosmos`,
`comfy/text_encoders/anima.py`) in a session that has network access to them.
