# Maintainer queue

Things only the maintainer can do (renders, judgment, approvals). Each
item states exactly what to run and what decision it unblocks. Resolved
items move out, not piled below.

## 1. B renders + collage verdict (blocks B4, C3)

Files (in ComfyUI `models/loras/`, base checkpoint `div_4.safetensors`):

| file | schedule | targets |
|---|---|---|
| `B_arm1_cosine_attn` | cosine | attention + cond (564, old defaults) |
| `B_arm2_const_attn` | constant | attention + cond (564) |
| `B_arm3_cosine_plus` | cosine | kohya_plus (726) |
| `B_arm4_const_plus` | constant | kohya_plus (726) |
| `B_arm5_const_722` | constant | kohya_default, cond off (722) |

Recipe: `archive/bad-LoRA-investigation/B_RENDER_CHECKLIST.md` (fixed
seed of your choice, strengths 0/0.5/1.0 x empty/novel-prompt/1024px +
the arm-4-vs-5 cond check). Keep every image under `runs/B_renders/`,
run `archive/bad-LoRA-investigation/blur_metric.py` over them.

Needed back: the contact sheets (or the images), the blur table, and one
verdict word per arm (*cured / partly / not cured*). Only you can judge
the collage; the agent must not declare it fixed.

## 2. ~~C3 follow-up: should new-run defaults move to 724?~~ RESOLVED — yes

The maintainer approved the move. New runs default to constant +
`kohya_label` (724) + cond off; `kohya_default` (722) remains available
as a preset for stored values. Landed with the decision note and
`CHANGELOG.md`.

Still open under it: the B3 far-prompt/size check. If out-of-distribution
renders show a global shift on 724, flip back to 722 by editing one
default.

## 3. ~~Real-weights meta-init confirmation (A2 follow-up)~~ DONE

Ran on real SDXL weights (CPU fp32/fp16, `div_4.safetensors`): UNet
1680/1680 tensors bit-identical, **forward bit-identical (0.000e+00)**,
and the same after LoRA injection (724 adapters). CLIP encode likewise
bit-identical; the only 2 differing tensors are the allowlisted-missing
ones, which the meta route zero-fills by design — and the bit-identical
encode now *demonstrates* they are unused rather than arguing it in a
comment. See
`archive/bad-LoRA-investigation/MEASURED-real-weights-meta-init.md`.

Not covered by that run: non-SDXL checkpoints, and bf16 device forwards.

## 4. ~~Block-weight metadata on real weights (D2 follow-up)~~ DONE (the
strength half); render check is yours

The defect is fixed and proven numerically: the export writes the
effective alpha (`alpha * weight`), so an outside reader's
`(B@A)*(alpha/rank)` reproduces the trained delta exactly for weights
0.5/1.0/2.0 on LoRA and DoRA. Two LoRAs were trained to exercise it
(`BW_armA_plain`, `BW_armB_weighted` in ComfyUI `models/loras/`).

What only you can confirm: open **both** in ComfyUI at the same seed and
strength. They must look **different** — `BW_armB_weighted` was trained
with `input_blocks.7:0.25` and `middle_block:1.0`, so its mid-UNet
contribution is deliberately a quarter strength. If the two renders are
identical, the weight is not reaching the file and this is a fail.
