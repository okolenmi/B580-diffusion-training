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

## 2. C3 follow-up: should new-run defaults move to 724? (decided: see below)

C3 was approved and landed as (b): constant + kohya_default, cond off
(`CHANGELOG.md`, `docs/decisions/0006-lora-defaults.md`). Since then the
arm-6 isolation run showed 722 + label_emb alone (724) renders best of
all six arms. Question for you: move new-run defaults to
constant + `kohya_default,label_emb` + cond off (724), or keep (b) and
leave label_emb opt-in? Evidence for the move: best renders, label_emb
at 1.73 with no health warning, diffusers loads the keys. Against:
one dataset, seeds unset across arms, far-field B3 check still open.
`B_arm6_const_724.safetensors` (+ step-250/500/750 checkpoints) is in
ComfyUI loras for your own comparison.

## 3. Real-weights meta-init confirmation (A2 follow-up)

Tiny-weight CPU proofs exist (`smoke_test_meta_init.py`); the same claim
on real weights (meta-built UNet/CLIP + checkpoint load + one forward
vs the eager path) was verified for weights (1680 tensors bit-identical,
CLIP encodes 0.00e+00) but a real-weights *forward* comparison was never
run. If you want it: build both ways in one process on CPU fp32 (needs
~20 GB host RAM free), compare one forward exactly. Say the word and the
agent will run it here.
