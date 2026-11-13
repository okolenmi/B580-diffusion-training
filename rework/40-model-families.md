# 40. Model families (SDXL, Anima, and room for more)

Status: **proposal, not decided.** Inputs: `audit/model-loading-and-sdxl-coupling.md`,
`audit/anima-model-facts.md`. Related plans: 41 (Load Model), 42 (text
encoders), 43 (adapters), 50 (dataset and latents), 70 (trainer).

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* One class (or folder of classes) per supported model: `SDXL` and `Anima`.
* Nodes use a family's own methods "like an import of a function with the same
  structure, but from a different library". Methods must be broad enough to
  allow compatibility.
* Things that are not model-specific (CLIP, Qwen) live in a **general** folder.
* LoRA and other adapter structures are defined by the family; which parts get
  trained must be choosable.

## 2. The idea in one paragraph

A **family** is a package that answers the same fixed set of questions about
one kind of model: *what does a file of this family look like, how do I load its
parts, how does a prompt become conditioning, what does the latent space look
like, what does the network predict, where can an adapter go and how is it
saved, how does memory split into blocks.* Nodes never import a family. They
receive a **`FamilyRef`** inside the handles that travel on wires (a loaded
part knows its family), and call `part.family.<area>.<method>(...)`. That is
the "same function, different library": the structure is the interface in
`families/api.py`, the library is whichever family the handle came from.

## 3. Package layout

```
families/                       # new top-level package, peer of nodes/ and manager/
  api/                          # PURE PYTHON, no torch import. Protocols, dataclasses, registry
    registry.py                 # family id -> lazy loader; list_families()
    detection.py                # Detection, PartInfo, PartKind, header reader
    conditioning.py             # Conditioning protocol (opaque tensors + generic ops)
    latents.py                  # LatentSpec
    parameterization.py         # Parameterization, TimestepPolicy
    adapters.py                 # AdapterLayout, ModuleGroup, TrainableSelection
    memory.py                   # BlockLayout, MemoryProfile
  general/                      # torch allowed; reused by several families
    clip/                       # CLIP text model, tokenizer (L and G are configs, not classes)
    qwen/                       # Qwen3 text model, tokenizer
    t5/                         # T5 tokenizer (ids only)
    vae/                        # AutoencoderKL base; Wan-style VAE
    adapters/                   # LoRA, DoRA layers, strategies, scaling policies, checkpointing
    schedules/                  # ddpm schedules, flow-matching sampling, eps/v/velocity targets
    safetensors_io/             # header reading, lazy part loading, dtype inspection
  sdxl/
    fingerprint.py              # PURE: how to recognise SDXL files (no torch)
    denoiser/                   # UNet (moved from nodes/model/unet.py and friends)
    conditioning.py             # SDXL pair (ctx, y), size embeddings, packing
    latents.py                  # 4 channels, factor 8, scale 0.13025
    parameterization.py         # eps / v-pred, scaled-linear schedule
    adapters.py                 # module groups, key scheme lora_unet_*
    memory.py                   # input/middle/output blocks
    family.py                   # SDXLFamily: wires the areas above together
  anima/
    (same file names; the denoiser is a Cosmos DiT, plus the LLM adapter)
```

Rules:

1. `families/api` imports nothing heavy. `backend/` and node `resolve_*` hooks
   may import it (they run in the server process that must stay light).
2. `families/<name>/fingerprint.py` is also torch-free. The catalog and the
   Load Model node call fingerprints during editing without importing any
   model code.
3. Only `registry.py` knows the family names; it imports a family **lazily**,
   so a graph that never touches Anima never imports Anima code.
4. Families import `general/` and `api/`. They never import each other and
   never import `nodes/`, `backend/` or `manager/`.
5. `nodes/` imports `families.api` and `families.registry`, never a concrete
   family. This is what keeps node code family-agnostic.

Where the current files go (the per-file ledger is `PORTING.md`; files are rewritten into `next/families/`, plan 90): `nodes/model/unet.py`,
`attention.py`, `unet_wrapper.py`, `lora.py` targets -> `families/sdxl/` and
`families/general/adapters/`; `clip.py`, `tokenizer.py` -> `general/clip`;
`vae.py`, `vae_decode.py` -> `general/vae` plus family constants;
`resource_inspection.py` -> `general/safetensors_io` plus `sdxl/fingerprint.py`;
`components/diffusion.py`, `noise_schedule.py`, `model_io.py` ->
`general/schedules`. `nodes/model/*` keeps only thin node classes.

## 4. The interface

All of these are `typing.Protocol`s in `families/api` (structural typing: a
family does not have to inherit anything, which keeps the "different library,
same function names" feel). A `Family` object exposes the areas as attributes:

```python
class Family(Protocol):
    id: str                        # "sdxl", "anima"
    label: str
    detect:     Detector           # header-only, pure
    parts:      PartLoader         # lazy load of denoiser / encoders / vae
    conditioning: ConditioningKit  # encoders -> Conditioning
    latents:    LatentKit
    objective:  ObjectiveKit       # parameterization + timestep policy + loss targets
    adapters:   AdapterKit
    memory:     MemoryKit
    settings:   tuple[SettingSpec, ...]   # family-specific graph settings (plan 20)
```

### 4.1 `detect` (pure, cheap, header-only)

```python
class Detector(Protocol):
    def match(self, index: HeaderIndex) -> Match | None: ...
        # HeaderIndex = key names, shapes, dtypes, __metadata__ (no tensor data)
        # Match: score 0..1, parts: tuple[PartInfo], warnings
class PartInfo:  kind: PartKind; prefix: str; key_count: int; dtype: str|None
                 config: dict   # e.g. {"model_channels": 2048, "blocks": 28}
class PartKind(Enum): DENOISER, TEXT_ENCODER, VAE, ADAPTER, OTHER
                      # TEXT_ENCODER parts carry a `role`: clip_l, clip_g, qwen3
```

Every family answers independently; the registry asks all and returns all
matches ranked. Several matches means "ambiguous" and the Load Model node asks
the user (plan 41). Examples from the audit: SDXL = prefixes
`model.diffusion_model.`, `conditioner.embedders.0/1`, `first_stage_model.`;
Anima DiT = `blocks.0.mlp.layer1.weight` plus
`llm_adapter.blocks.0.cross_attn.q_proj.weight` under prefix `net.` or
`model.diffusion_model.`.

### 4.2 `parts` (lazy loading)

```python
class PartLoader(Protocol):
    def open(self, source: PartSource, kind: PartKind, role: str|None, *, device, dtype): ...
        # PartSource = (path, prefix) - a slice of a file; nothing is read until used
```

The general handle is **`ModelPart`**: `(family_id, kind, role, source, info)`;
it replaces `ModelWeights` as the thing wired out of a loader. The loader node
never loads tensors; the first consumer does (as `ModelWeights` already does
lazily).

### 4.3 `conditioning` (the biggest difference between families)

The current `TextEncoder` ABC returns `(context, pooled_y)` and carries a
resolution embedding. That is SDXL's shape, not a general one. New contract:

```python
class Conditioning(Protocol):          # opaque to the trainer
    def to(self, device) -> "Conditioning": ...
    def cat(items) -> "Conditioning": ...        # stack samples into a batch
    def index(self, i|slice) -> "Conditioning": ...
    def blend(self, other, p) -> "Conditioning": ...   # p scalar or per-sample
    def segments(self) -> dict[str, SegmentInfo]       # named parts a blend may treat apart
    def nbytes(self) -> int
class ConditioningKit(Protocol):
    def cache_key(self, prompt: str, size: tuple[int,int]|None) -> Hashable
    def encode(self, prompts, sizes, *, encoders) -> list[Conditioning]
    def blank(self, size) -> Conditioning
    def bytes_per_prompt(self) -> int                  # sizes the prewarm cache
    def required_encoders(self) -> tuple[EncoderSpec]  # roles that must be wired
```

* SDXL `Conditioning` holds `ctx (77, 2048)` and `y (2816)`;
  `segments()` = `{"clip_l": ctx[..., :768], "clip_g": ctx[..., 768:] + pooled}`
  so the **per-half blend** the owner asked for earlier is a generic
  per-segment blend, not SDXL code in a trainer. Its `cache_key` includes the
  size because `y` holds size embeddings.
* Anima `Conditioning` holds the adapter output (`~512 x 1024`, per diffusers
  docs; to verify). `segments()` may be a single segment, or expose the
  pre-adapter Qwen hidden states if blending there proves better (open
  question). `cache_key` has no size.
* The cache, prewarm and blend code (today written against SDXL tuples)
  become family-agnostic users of this protocol. `ConditioningBlend`
  (already shipped) becomes `Conditioning.blend` with a per-segment keep range.

### 4.4 `latents`

```python
class LatentSpec:  channels: int; spatial_factor: int; temporal: bool
                   scale: float; shift: float
def noise_shape(self, height, width, batch) -> tuple[int, ...]   # (B,4,h,w) vs (B,16,1,h,w)
def encode(self, vae_part, images) -> Tensor ; def decode(self, vae_part, latents) -> Tensor
```

SDXL: 4 channels, factor 8, scale 0.13025. Anima: 16 channels, factor 8,
single-frame 5-D layout (R), scale/shift **unknown, must be found** (audit gap 2).

### 4.5 `objective` (what the network predicts and how it is trained)

Absorbs the existing handles `DiffusionProcess` and `Parameterization` (both
have no producer node today):

```python
class ObjectiveKit(Protocol):
    def default_process(self) -> DiffusionProcess       # ddpm schedule | rectified flow
    def default_parameterization(self) -> Parameterization   # eps | v | velocity
    def sample_timesteps(self, n, policy) -> Tensor     # incl. flow shift, sigmoid sampling
    def make_target(self, x0, noise, t) -> tuple[x_t, target]
    def to_x0(self, pred, x_t, t) -> Tensor              # for previews and some losses
    def loss_weight(self, t, name) -> Tensor             # uniform / min-snr / sigma_sqrt ...
```

SDXL keeps eps and v-pred and the scaled-linear schedule it has now. Anima:
rectified flow, `sigmoid` timestep sampling default, optional `shift`
(sd-scripts options listed in the audit). The *nodes* for loss weighting and
timestep sampler (plan 70) call this kit; the family contributes the valid
choices (so the UI offers "min-SNR" for SDXL and "sigma_sqrt, cosmap" for
Anima without the node knowing).

### 4.6 `adapters` - see plan 43

`AdapterKit` exposes `layout()` (named module groups), `select(model,
selection)`, `inject(model, spec)`, `export(model) -> dict`, `import_(dict)`.

### 4.7 `memory`

```python
class MemoryKit(Protocol):
    def blocks(self, denoiser) -> list[BlockRef]    # units for offload / swap / checkpoint
    def estimate(self, info: PartInfo, dtype) -> int   # bytes, from header only
```

SDXL: input / middle / output blocks. Anima: the 28 transformer blocks, enabling
a `blocks_to_swap`-style policy. The *policy* lives in the runtime (plan 70);
the family only says where the seams are. `estimate` powers the pre-run memory
preview from the header alone, with no load.

## 5. What stays general, and the test

A component is **general** if two families already need it or the literature
says another popular family will: CLIP text model (SD1.5, SDXL, Flux), Qwen
(Anima, Qwen-Image), T5 tokenizer, VAE bases, LoRA/DoRA layers, schedules.
Test: "does its code mention a family?" If it does, it belongs in the family.
`general/` classes take a config object; the family supplies the config
(`CLIPConfig(hidden=768, layers=12)` for L, `(1280, 32, projection)` for G).

## 6. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| F1 | Interface style | ABC to inherit; Protocol (structural); module of functions | Protocols + one `Family` object per family: nodes see the same attribute names, tests can fake a family without inheritance |
| F2 | Where families live | inside `nodes/`; new top-level `families/` | Top-level. `nodes/` stays a thin layer, and `backend/` may import the pure `api` and `fingerprint` parts without torch |
| F3 | Family travels how | node imports it; handle carries `FamilyRef`; global "current family" | Handle carries it. A graph with an SDXL and an Anima chain would break a global |
| F4 | Conditioning shape | keep `(ctx, y)`; opaque `Conditioning` | Opaque with generic ops; it is the only way cache, prewarm, blend and per-half blend stay family-agnostic |
| F5 | Task (LoRA vs full finetune) in the family? | fused as today (`ResourcePreset`); separate | Separate. Family = model; adapter kind = task. This dissolves `ResourcePreset` and the `SDXL_*` class pairs |
| F6 | Flow matching in `general/schedules` or Anima | | `general/schedules`: Flux, Wan, Qwen-Image will want it |
| F7 | Unknown family | refuse; load as generic | Refuse with a readable report of what was found (key counts, nearest family); never guess |

## 7. Anima specifics to design around

* Three separate files (DiT, Qwen3, VAE); the DiT file also contains the **LLM
  adapter**, which must not be trained by default (model card). So Anima's
  `parts` splits the DiT file into `DENOISER` and `ADAPTER` parts by key
  prefix.
* Two tokenizers feed the adapter (Qwen ids for hidden states, T5 ids as
  queries): the Anima `ConditioningKit` owns both tokenizations; callers see
  only a prompt string.
* fp8 base unsupported (sd-scripts note); block swap is the main memory lever.
* Open facts (audit section 7): LoRA key naming, VAE scale and shift,
  adapter token pairing, scheduler shift, B580 memory. **The Anima family
  cannot be finished without reading the open-source reference implementations
  or running on hardware.**

### 7.1 Built so far (phase 8)

Checked key-for-key against the owner's headers (`next/tests/fixtures/anima_headers.json.gz`) and number-for-number against
ComfyUI's and Hugging Face's code on shared random weights (`next/tests/reference/`):

* `families/anima/dit.py`, `adapter.py`: the transformer (3-D rotary, AdaLN-LoRA, per-block checkpointing) and the text adapter.
* `text.py`: Qwen3-0.6B (last layer after its final norm; the file's keys under `model.`).
* `tokens.py`: Qwen byte-level BPE and T5 Unigram through the `tokenizers` package; vocabularies are found through
  `QWEN_TOKENIZER_DIR` / `T5_TOKENIZER_DIR`, `assets/{qwen,t5}_tokenizer` or a ComfyUI checkout (`$COMFY_DIR/comfy/text_encoders`).
  Captions are literal text: no emphasis syntax (booru tags carry parentheses), no padding.
* `conditioning.py`: what is cached per prompt is the adapter's *input* (Qwen hidden states and T5 ids), so the cache does not
  depend on the DiT file and the adapter, which lives in that file, runs inside each step. A batch is always 512 positions wide
  with masks (constant shapes for captured step graphs; prompts beyond 512 tokens are cut). Answer to open question 2: the blend
  acts before the adapter, on Qwen's states; T5 ids cannot be mixed and switch to the partner's at p >= 0.5.
* `general/vae/wan.py`: the Wan 2.x / Qwen-Image autoencoder for single images. One frame meets only the last temporal slice of
  every 3-D kernel and the temporal resampling does nothing for a first frame, so it runs as 2-D layers; `to_image_state` converts
  the file's tensors (the real 3-D network gives the same numbers, `equivalence_wan_vae.py`). Latents are `(z - mean) / std` per
  channel (`LatentSpec.mean/std`, part of the latent cache identity), images are cut to multiples of 16 px (`LatentSpec.grid`).
* `objective.py`: rectified flow, velocity target; `shift` (default 3.0, ComfyUI's value) and the sampler (`sigmoid`, `logit_normal`,
  `uniform`, `cosmap`) are Noiser options, and a timestep range cuts the sampler's distribution on its cumulative probability.
  `sigma` is the noise-to-signal ratio `s / (1 - s)`, so loss weightings written for it keep their meaning (uniform is the default).
* `adapters.py`: groups attention (self, cross), feed-forward on by default (280 modules for the base DiT); timestep modulation,
  embeddings and the text adapter off. Saved as kohya `lora_unet_<path>` keys, which ComfyUI maps onto `diffusion_model.<path>`.
* `denoiser.py`, `family.py`; the `anima-lora` starter (CLI `python -m next.run anima-lora`, backend starter list) wires three Load
  Model nodes (DiT, Qwen3, VAE).
* Not built: block swap (the base DiT is about 4 GB in bf16, so the first runs do not need it; the 40-block file will), the
  self-perceptual loss tap, previews while training (decode works through the codec).

## 8. Risks

* The old `nodes/model/` stays until the swap, so nothing that imports it breaks
  while `next/families/` is written (plan 90). The risk moves to the swap, where
  imports in `scripts/` and tests are rewritten once.
* An interface designed from two families will have mistakes a third exposes.
  Mitigation: keep each Kit small, and mark the Anima kit methods that are
  guesses (`# unverified`) until measured.
* Opaque `Conditioning` costs one indirection in the hot path; measure once
  with the existing step benchmark before and after (plan 70).

## 9. Open questions for the owner

1. Is `families/` as a top-level package acceptable (it moves a lot of
   `nodes/model/`)?
2. For Anima, should a blend act on the final adapter output, or on the Qwen
   hidden states before the adapter? Decided: before (7.1); an experiment can still compare.
3. Do you want SD 1.5 or Flux stubs created now to test the interface, or only
   SDXL and Anima? Recommendation: only the two, but write the interface tests
   against a fake third family.
