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

Measured and **negative**: a real-weights probe
(`a7_clip_tower_fragility.py`) that runs the UNet on one fixed latent
under three conditionings and reports how much the LoRA moves the
predicted noise:

| LoRA | full (L+G) | CLIP-L only | CLIP-G only | L-only / full |
|---|---|---|---|---|
| TestAes | 0.180 | 0.170 | 0.103 | 0.95x |
| B_arm5_const_722 (renders fine) | 0.225 | 0.210 | 0.101 | 0.93x |

So TestAes is **not** more sensitive to the CLIP-L/G split than a LoRA
that renders correctly. The first hypothesis ("the LoRA leans on a half
that disappears") was wrong, and this table is why.

Not measured: whether removing `label_emb` from the target set fixes the
render. That is the obvious next experiment and it needs the maintainer
to judge the images — see the queue. The honest expectation is that
`kohya_default` (722, no `label_emb`) is the safer target set for a
captionless dataset, at the cost of the fit quality arm 6 showed.

## Consequence for the C3 default

This is evidence **against** the 724 default adopted on 2026-10-11, and
it should be weighed before that default is relied on for a LoRA you
intend to prompt with: `kohya_label` adapts `label_emb`, which is the
one module whose damage is invisible on captionless training data and
visible the moment the conditioning changes. The arm-6 render ranking
was measured on captionless data only.