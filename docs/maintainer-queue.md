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

## 3. Real-weights meta-init confirmation (A2 follow-up)

Tiny-weight CPU proofs exist (`smoke_test_meta_init.py`); the same claim
on real weights (meta-built UNet/CLIP + checkpoint load + one forward
vs the eager path) was verified for weights (1680 tensors bit-identical,
CLIP encodes 0.00e+00) but a real-weights *forward* comparison was never
run. If you want it: build both ways in one process on CPU fp32 (needs
~20 GB host RAM free), compare one forward exactly. Say the word and the
agent will run it here.
