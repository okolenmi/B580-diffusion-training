# 42. Text encoders, conditioners, and the general library

Status: **proposal.** Needs 40 (the `Conditioning` contract) and 41 (parts and
role types). Touches 50 (prompt channels), 70 (residency), 20 (cache and
prewarm settings). Inputs: audit files; `nodes/model/text_encoder*.py`,
`clip*.py`, `tokenizer.py` as read in `audit/model-loading-and-sdxl-coupling.md`.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* SDXL has two encoders, so a checkpoint gives **4 outputs**; an SDXL-specific
  node **packs** two of them. That allows "unique routes for different CLIPs"
  or one simple route with both.
* Encoders that are not designed for one model go to a **general** folder.
  Obvious case: Anima uses a Qwen model.
* Open from the earlier talk: CLIP_L usually reads tag-like text, CLIP_G
  descriptive text (dual captions); per-half blend.

## 2. Vocabulary

| Word | Meaning |
|---|---|
| **Part** | a loaded-on-demand slice of a file with a role (`clip_l`, `clip_g`, `qwen3`...), from Load Model (41) |
| **Encoder** | a part made usable: tokenizer + text model, with options (clip skip, max tokens, dtype, optional adapter). Family-agnostic |
| **Conditioner** | a family's packed set of encoders, owning how a prompt becomes `Conditioning` (40 section 4.3), the cache and the blank prompt |
| **Prompt** | a mapping of **channels** to text (`text` by default; `tags`, `description` when the dataset has them). A plain string is `{"text": s}` |

## 3. Graph shapes

SDXL, simplest route (the pack node accepts raw parts):

```
Load Model --clip_l--+
(checkpoint)         +--> SDXL Conditioner --conditioner--> trainer / cache / blend
            --clip_g-+
```

SDXL, per-encoder route (an Encoder node in between, for options or a
text-encoder adapter):

```
Load Model --clip_l--> Text Encoder (clip skip 2, trainable) --+
            --clip_g--> Text Encoder (clip skip 2)             +--> SDXL Conditioner
```

Anima:

```
Load Model (text_encoders)  --qwen3---->+
Load Model (diffusion_models) --adapter->+--> Anima Conditioner
```

An Encoder node returns the same type as the part it was given (a configured
`ClipLPart` is still a `ClipLPart`), so a port that wants `ClipLPart` accepts
either. That keeps the simple route one node shorter without a second type
system.

## 4. Nodes

| Node | Lives in | Inputs (grouped) | Outputs |
|---|---|---|---|
| `TextEncoderOptionsNode` (new, general, optional) | `nodes/` | `part` (any text-encoder part); group **Options**: `clip_skip`, `max_tokens`, `dtype`; group **Training**: `adapter` (an adapter spec, plan 43) | the same part type, configured |
| `SDXLConditionerNode` (replaces `SDXLTextEncoderNode`) | `families/sdxl/nodes/` | `clip_l`, `clip_g` (each optional); group **Channels**: `l_channel`, `g_channel` (prompt channel each encoder reads); group **Missing half**: `fill` = `blank` or `zeros` when only one is wired; group **Size conditioning**: on/off | `conditioner` |
| `AnimaConditionerNode` (new) | `families/anima/nodes/` | `qwen3`, `adapter`; group **Tokens**: max Qwen tokens, max T5 tokens; group **Format**: prompt layout (`tags_then_text`, `text_only`) | `conditioner` |
| `ConditioningBlendNode` (exists) | `nodes/` | `conditioner`; per-segment keep ranges (generic over `Conditioning.segments()`) | `conditioner` (wrapped, plan 70 section 8) |

Not ported (replaced by this plan, no shims, plan 90): `SDXLTextEncoderNode`,
`CachingTextEncoderNode`, `PrewarmedTextEncoderNode`.

**Where family-specific nodes live.** Discovery (`scan_nodes`) today walks
`nodes/`. Extend it to also walk `families/*/nodes/`. Rule: a node file inside a
family folder may import its family; a node in `nodes/` never imports a
concrete family. The palette domain comes from the module path, so these appear
under "sdxl" and "anima" with no catalog change.

## 5. The general encoder layer (`families/general/`)

```
general/clip/    CLIPTextModel (config: hidden, layers, heads, projection), tokenizer
general/qwen/    Qwen3 text model (config), Qwen2 BPE tokenizer
general/t5/      T5 SentencePiece tokenizer (ids only; weights not needed for Anima)
```

* CLIP-L and CLIP-G are **two configs of one class**, not two classes. Today
  `SDClipModel`, `SDXLClipG`, `SDXLClipModel` already follow that shape inside
  `nodes/model/clip.py`; the move formalises it.
* A general encoder exposes one small runtime interface:

```python
class TextEncoder(Protocol):        # general, role-aware
    role: str; max_tokens: int
    def tokenize(self, text: str) -> Tokens: ...
    def encode(self, tokens) -> EncoderOutput: ...   # hidden states, pooled (or None), mask
    # DeviceResident: to / offload / reload / release / footprint
```

  A family's `ConditioningKit` composes encoders; the trainer never calls an
  encoder directly. This is what `TextEncoder.encode_prompt_only` /
  `resolution_embedding` split today (`text_encoder.py`) becomes: the first half
  moves into the general encoder, the SDXL-specific second half (size
  embedding) moves into the SDXL kit.
* **Tokenizer assets.** SDXL's CLIP vocabulary is already bundled. Anima needs
  the Qwen2 tokenizer files and a T5 tokenizer vocabulary (the guide says only
  tokenizer files are needed, not weights; sd-scripts bundles them under
  `configs/t5_old/`). These are small text assets; fetching them is possible
  from GitHub in a session with network, but the **licences must be checked and
  recorded** before they are committed (open question E3).

## 6. Prompt channels and dual captions

* The data layer (plan 50) delivers each sample's text as a `Prompt`
  (channel -> string). The conditioner decides how channels map to encoders:
  SDXL `l_channel` / `g_channel` (default both read `text`, so nothing
  changes for existing datasets); the owner's idea "tags for CLIP_L,
  descriptions for CLIP_G" is `l_channel=tags`, `g_channel=description`.
* **Cache key** therefore includes both strings: `kit.cache_key(prompt, size)`
  (40 section 4.3). Two channels with identical text collapse to one entry.
* Anima's `format` option joins channels as the model card describes: tags on
  the first line, a newline, optional natural-language second line.
* `AutoCaptionNode` (exists) gets an `output_channel` so a caption model
  writes into `description` while a tagger writes `tags` (plan 50).

## 7. Cache and prewarm stop being nodes

Today three nodes decorate each other (`SDXLTextEncoder` -> `Caching...` ->
`Prewarmed...`), and the prewarm plus the "empty cache after" interplay caused
the OOM described by the owner. Both are *environment and lifetime*, not
process, so:

* **Cache**: built into every `Conditioner`; capacity set by one graph setting
  `conditioning.cache_mb` (default derived from `bytes_per_prompt()` and free
  host RAM), replacing the constant `prompt_capacity() = 5000` and the
  unrelated `max_entries = 512`. Keys come from the kit.
* **Prewarm**: a graph setting `conditioning.prewarm = auto | off`. With
  `auto` the runtime, at run start and on each `prompts_changed` event
  (plan 10), encodes the keys the data layer says it will need, then asks the
  residency manager to unload the encoders (plan 70). Incremental: only new or
  changed prompts are encoded, which removes the "universe" computation that
  the pool/blend patch had to add.
* Encoders are `DeviceResident`s; **who loads and unloads them is the runtime**,
  not the node. The "empty cache after" flag moves with the rest of residency
  to graph settings (plan 20).

## 8. Text-encoder adapters

`TextEncoderOptionsNode.adapter` attaches an adapter spec (plan 43) to an encoder, so
encoder LoRA is the same mechanism as UNet LoRA with a different layout (the
family or general library supplies `AdapterLayout` for CLIP and Qwen layers).
Constraint from the Anima guide: **cached text-encoder outputs are incompatible
with training an encoder adapter**, and the same is true for any family. The
conditioner must refuse to cache (or refuse the adapter) and say so; this is a
validation rule, not a runtime surprise.

## 9. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| C1 | Pack node accepts | parts only; encoders only; either | Either (encoders are subtypes of parts): short graphs by default, an Encoder node when options or an adapter are needed |
| C2 | Cache and prewarm | stay nodes; become settings plus events | Settings plus events: they are lifetime/environment, and the OOM history shows nodes cannot coordinate them |
| C3 | Family-specific nodes | in `nodes/`; in `families/<name>/nodes/` | In the family folder, with discovery extended; keeps `nodes/` family-agnostic |
| C4 | Missing half (one CLIP wired) | forbid; fill zeros; fill blank encoding | Offer `blank` (default) and `zeros`; both labelled, since this is exactly the unseen-pairing question from the CLIP_L-only investigation |
| C5 | Prompt channels | one string; channel mapping | Channels (default one), because dual captions and AutoCaption both need it |
| C6 | Tokenizer assets | download at runtime; commit to repo | Commit small assets with their licences (to be checked) |

## 10. Open questions for the owner

1. Is "pack takes either parts or encoders" the behaviour you want, or should
   the Encoder node always be required (more nodes, more explicit)?
2. May tokenizer assets for Qwen2 and T5 be added to the repository once
   their licences are confirmed?
3. Should `fill=blank` or `fill=zeros` be the default for a single-CLIP SDXL
   route? (I recommend blank, because zeros is a conditioning no SDXL model
   was trained on; it remains an unverified hypothesis.)
