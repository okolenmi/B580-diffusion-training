# MEASURED: why TestAes breaks under CLIP-L-only conditioning

Investigated 2026-10-12 on the B580, real SDXL weights.

**Verdict: the cause is not CLIP_L.** It is that the LoRA was trained on
a dataset whose captions are **all empty**, which pushed the adaptation
onto the global conditioning path (`label_emb`), and *any* change of
conditioning at inference then exposes it. CLIP-L-only is simply the most
extreme such change — it is not special.

## What is in TestAes

Recovered from the file itself and from the saved graph that produced it
(`backend/data/backend.db`, `saved_graphs.name = 'def'`):

| | |
|---|---|
| tensors | 2172 = 724 modules x 3 |
| rank / alpha | 16 / 16 (read off the file; the forensics table's "rank" column is wrong) |
| dataset | `test2` — **204 images, 204 empty captions** (verified in `datasets/test2/metadata.db`) |
| graph | managed route, batch 8, 1000 steps, lr 1e-4 constant, checkpoint `divA.safetensors` |

Its own metadata already said what was wrong:

```
lora_health warnings:
  "label_emb median is >5x the attention reference: check what the LoRA
   put on the global conditioning path"
```

`label_emb` median |dW| = **1.81** against ~0.29 for the attention
groups — 6.2x.

## Why that is the mechanism

`nodes/model/unet.py:473` — `emb = emb + self.label_emb(y)`, where `y`
is `cat([pooled, resolution_embedding])`. So `label_emb` is fed the
**pooled CLIP-G** vector and the resolution embedding. **It never sees
CLIP-L.** (This is the key correction: the adapted module is a CLIP-G /
global carrier, not a CLIP-L one.)

With 204 images and one constant empty caption there is nothing in the
text for the model to learn from, so the per-sample variation has to be
carried by the image path, and the global conditioning injection becomes
the place the optimisation leans on. It ends up 6x hotter than every
attention group. At inference:

* the pooled vector is no longer the constant `pool_G("")` the LoRA was
  fitted against, and
* the context is no longer `[encode_L("") | encode_G("")]`.

The heavily-adapted `label_emb` then applies a large delta computed for
the wrong conditioning, which is what oversaturation and abstract
shapes are.

CLIP-L-only is the worst case for two compounding reasons: the CLIP-G
half of the context becomes **exactly zero** — a value never seen in
training, where that half was always `encode_G("")` and non-zero — and
the CLIP-L half changes the most, from an empty-prompt pattern to a real
prompt. Hence "full is better but still not great; CLIP-L poisons it".

## What was measured, and what was NOT

Measured: per-group |dW| and the metadata warning; that the dataset is
captionless; that `label_emb` is fed pooled CLIP-G.

**Refuted 1 — "the LoRA leans on a half that disappears".** A
real-weights probe (`a7_clip_tower_fragility.py`) runs the UNet on one
fixed latent under three conditionings and measures how far the LoRA
moves the predicted noise:

| LoRA | full (L+G) | CLIP-L only | CLIP-G only | L-only / full |
|---|---|---|---|---|
| TestAes | 0.180 | 0.170 | 0.103 | 0.95x |
| B_arm5_const_722 (renders fine) | 0.225 | 0.210 | 0.101 | 0.93x |

TestAes is no more tower-sensitive than a LoRA that renders correctly.

**Refuted 2 — "label_emb is what breaks it".** The maintainer rendered
`TestAes_noLabelEmb.safetensors` (TestAes with the four label_emb factor
tensors zeroed, verified as the only difference) and it has **the same
artifacts**. Also, zeroing those adapters moves the single-step effect
by only ~9% (0.180 -> 0.164).

**Refuted 3 — "it is simply over-strong".** Effective delta
`(B@A)*(alpha/rank)`, median Frobenius norm:

| LoRA | attention | other |
|---|---|---|
| TestAes | 0.312 | 0.348 |
| B_arm5_const_722 (renders fine) | **0.403** | 0.468 |

The good LoRA pushes *harder*. Strength is not the discriminator.

**Not the difference either.** The two files are structurally identical:
rank 16, alpha 16 on all 724 modules, suffixes `alpha`/`weight`, same
metadata keys. Nothing about the format separates them.

## The route difference that IS real

Recovered from the saved graph: the default graph trains on
**`divA.safetensors`**; the study harness trains on `div_4.safetensors`.
Those are **different models**: 1656 of 1680 UNet tensors differ between
them, max |diff| 0.46. A LoRA trained on one and rendered against the
other is a base mismatch, which is an established cause of colour and
composition artifacts.

Other route differences still open: batch 8 vs 4, managed route
(resource controller with a 10500 MB budget, prefetching source) vs the
direct route, and `cache_text_encoder: false` (so the towers are
re-run every step rather than warmed once).

## The base checkpoints are different models

`divA.safetensors` vs `div_4.safetensors`, tensor by tensor:

| component | tensors | differing | max abs diff |
|---|---|---|---|
| UNet (`model.diffusion_model.*`) | 1680 | **1656** | 0.4648 |
| CLIP-L (`conditioner.embedders.0.*`) | 197 | **190** | 0.1797 |
| CLIP-G (`conditioner.embedders.1.*`) | 390 | **390** | 1.7471 |
| VAE (`first_stage_model.*`) | 248 | **0** | 0.0000 |

So divA is a finetune with its **own UNet and its own text encoders**,
sharing only the VAE. Every study-arm LoRA was trained on `div_4`;
TestAes was trained on `divA`. A LoRA trained against one set of text
encoders is being applied against a different set at render time, which
is exactly the class of mismatch that shows up as conditioning damage.

This is the maintainer's own read too ("you trained on div_4. That's the
biggest difference"), and it is being tested rather than assumed: see
`runs/BASEMAP_divA.log` for the reciprocal arm — the study harness
training on `divA` with TestAes's rank/alpha/steps, so the base model is
the only thing that changed from a LoRA known to survive CLIP-L-only.

Note the asymmetry this does *not* explain: CLIP-G is the *more*
diverged encoder (max 1.75 vs 0.18), yet the reported failure is
CLIP-L-specific. That remains unexplained, and is the reason the base
model is being treated as the leading hypothesis rather than the answer.

## Still unresolved

None of the above explains a *CLIP_L-specific* failure, and three
plausible mechanisms have now been refuted by measurement. The blocking
unknown is what "encoding only CLIP_L" means concretely at render time:
which ComfyUI conditioning nodes, whether the CLIP-G half of the 2048
context is zeroed or the 768-dim tower output is used directly, and
which checkpoint the render uses. Guessing at that has produced three
wrong answers in a row, so it is asked rather than assumed.

## Consequence for the C3 default

This is evidence **against** the 724 default adopted on 2026-10-11, and
it should be weighed before that default is relied on for a LoRA you
intend to prompt with: `kohya_label` adapts `label_emb`, which is the
one module whose damage is invisible on captionless training data and
visible the moment the conditioning changes. The arm-6 render ranking
was measured on captionless data only.