# MEASURED: conditioning breaks because the training captions are all empty

Final diagnosis 2026-10-12, after the base model, `label_emb`, strength,
file format, conditioning deviation and CLIP loading were each tested and
each ruled out.

**Evidence class: read in code + measured (dataset contents, conditioning
deviations, checkpoint deltas).** The confirming render is the
maintainer's.

## The cause

`datasets/test2` has **204 images and 204 empty captions** — one distinct
caption, the empty string. Every training step therefore presents the
model with a *constant* `(ctx, y)`: `[encode_L("") | encode_G("")]`
plus `pool_G("")`.

A cross-attention projection `W_k · c` whose input `c` never changes
receives a gradient of zero from any loss that depends on it. So the
attention's key/query/value projections get **no learning signal at
all** — they are not underfit, they are unconstrained, and whatever the
optimizer leaves there is arbitrary. At render, when conditioning moves
off that one constant value, the model meets key/query projections that
were never shaped to interpret any context, and the failure is exactly
what is reported: oversaturation, abstraction.

That also explains why no tuning knob fixes it. The weight is not
*wrong*; it is *unconstrained*, and nothing in a 100%-captionless run
ever constrains it.

## Why every earlier hypothesis failed

| hypothesis | verdict | evidence |
|---|---|---|
| `label_emb` is 6x hot | **refuted** | maintainer rendered `TestAes_noLabelEmb` (the 4 `label_emb` factor tensors zeroed) — same artifacts; zeroing moves the UNet output ~9% |
| the LoRA is over-strong | **refuted** | effective `(B@A)·(alpha/rank)` median ‖dW‖: TestAes 0.312 vs B_arm5 0.403, and arm5 renders correctly |
| file format differs | **refuted** | both files rank 16 / alpha 16 / 724 modules / same suffixes / same metadata keys |
| conditioning goes out-of-distribution | **refuted as the mechanism** | measured deviation from the training baseline: CLIP-L only **0.081**, CLIP-G only 0.307, full 0.318 — the failing mode moves the conditioning *least* |
| divA vs div_4 text encoders | **real but not the cause** | UNet 1656/1680 differ, CLIP-L 190/197, CLIP-G 390/390, VAE identical; but a div_4-trained LoRA fails on div_4 too |
| CLIP loaded from the wrong checkpoint | **refuted** | `_extract_and_convert_clip_state_dict` yields 715 keys for both, same 197/518 split, and the tensors genuinely differ — divA's own encoders are in use |

The CLIP-L-only deviation being the *smallest* is the tell: the failure
is not "conditioning moved too far", it is "there was never any signal
that made conditioning matter".

## Confirming experiment

`CAPTION_div4_724`: div_4, rank 16 / alpha 16, `kohya_label`, cond off,
1000 steps, constant 1e-4 — identical to TestAes except the dataset is
`multi-caption` (273 images, **8 distinct captions, 0 empty**).

If it survives CLIP-L-only where every captionless LoRA failed, the cause
is settled.

Note a trap this arm had to avoid: the loader groups samples by
`(prompt, size)`, so with few distinct captions and `batch_size=4`, most
groups are smaller than a batch. Measured on multi-caption before
fixing it: **only 28 of 273 samples were trained on per epoch**, with
the loader warning. `keep_incomplete_batches` is now required for such
datasets; that flag is what makes this a control rather than a smaller
version of the same mistake.

## Fixes, cheapest first

1. **Real captions** — the actual fix. A style LoRA is applied to
   arbitrary prompts, so it must be conditioning-invariant, and only
   varied conditioning teaches that.
2. **Caption dropout** — with probability *p* per step, replace a
   sample's caption with another from the dataset. One trainer option,
   no new data, and it forces conditioning-invariance directly. This is
   the option worth adding.
3. **Target set** — `kohya_default` (722) adapts no global prompt
   carrier. Weaker for a captionless fit (arm 6 rendered best *because*
   the captions were empty), but the fit quality it buys is exactly the
   part that does not survive contact with a prompt.
4. **Save-time detector (not a fix)** — measure how far the LoRA moves
   the UNet output across empty / L-only / G-only / full conditioning
   and report the spread. One number; a large spread means fragile.
   This would have flagged TestAes automatically at save time, instead
   of needing forensics after the fact.

## Note on the div_4 -> divA workaround

The one combination reported to work is a div_4-trained LoRA rendered on
divA. Since matched base fails and mismatched works, the likely reading
is that the mismatch weakens the delta and a weaker LoRA simply does not
do enough damage to show artifacts. That is a mitigation to use while
waiting, not a fix, and it is not evidence that the mismatch is correct.