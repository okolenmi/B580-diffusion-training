# B render checklist (maintainer renders, maintainer judges)

Arms (all: test2, 204 imgs, rank 64 / alpha 48, batch 8, 1000 steps,
seed 1234, AdamW 1e-4):

| file | schedule | targets |
|---|---|---|
| `B_arm1_cosine_attn.safetensors` | cosine (5% floor) | attention + cond (564) = old defaults |
| `B_arm2_const_attn.safetensors` | constant | attention + cond (564) |
| `B_arm3_cosine_plus.safetensors` | cosine | kohya_plus (726) |
| `B_arm4_const_plus.safetensors` | constant | kohya_plus (726) |
| `B_arm5_const_722.safetensors` | constant | kohya_default, cond off (722) |

Base checkpoint: the same file training used (`div_4.safetensors`).
Per arm, same seed (pick one, e.g. 1234), same sampler/steps/CFG:

- (a) training caption = empty prompt, 512x512, strengths 0 / 0.5 / 1.0
- (b) novel prompt (not in the dataset, e.g. "a lighthouse on a cliff at
  dusk, waves"), 512x512, strengths 0 / 0.5 / 1.0
- (c) empty prompt, **1024x1024**, strengths 0 / 0.5 / 1.0
- B3 extras: arm 4 vs arm 5 with a prompt AND size far from training, at
  0.5 (cond-path baking check); arm 4 at 0.5 vs arm 5 for global
  colour/composition shifts.

Keep every image (`runs/B_renders/<arm>_<case>_s<strength>.png`).
Then run `blur_metric.py` over them and paste the table into
`MEASURED-original-symptom.md`. Verdict words (*cured / partly /
not cured*) are yours.
