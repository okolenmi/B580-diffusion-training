# Decision note: LoRA training defaults for new runs (C3 — APPROVED, landed 2026-10-11)

Status: **approved and implemented; revised the same day to 724.** New
graphs/configs without stored values default to constant LR +
`kohya_label` (724 = the kohya 722 set + `label_emb`) + cond path off.
Saved runs keep stored values. `kohya_plus` (726) stays opt-in. See
`CHANGELOG.md`.

## Revision (2026-10-11, after the arm-6 isolation run)

C3 first landed option (b): constant + `kohya_default` (722), chosen when
722 was the best-rendering arm. The maintainer then approved the arm-6
isolation run, which rendered best of all six:

| arm | targets | verdict |
|---|---|---|
| 6 | 722 + `label_emb` only (724) | **best** — backgrounds most detailed, dataset vibe best represented |
| 5 | 722, no cond path | second |
| 4 | 726 (`kohya_plus`) | third |
| 2 | 560 attention + cond | fourth |

Holding label constant, `time_embed` *hurts* (726 < 724): the
maintainer's timing hypothesis, confirmed by isolation. So the default
moved to 724, and `TARGETS_KOHYA_LABEL` is a named preset rather than the
comma-joined string it started as. `kohya_plus` stays opt-in — its
`time_embed` is the tower the isolation indicted.

**Evidence class:** the render ranking is the maintainer's own judgement
on one dataset (`test2`, 204 images, empty caption, 512 px) with seeds
unset across arms, so batch order differed per arm. `label_emb` runs at
1.73 median |dW| with no health warning, and diffusers loads the keys.
What is *not* proven: the B3 far-prompt/size check is still open, so
this rests on in-distribution renders only.

## What B established (evidence, not argument)

- Constant vs cosine at the same nominal LR: ~2x trained magnitude
  (forensics), renders rank constant first (A1 + B arms 2>1, 4>3).
- Targets: attention-only forces attn_output to ~1.35x attn_input;
  722 lands at 1.08; renders rank 724 > 722 > 726 > 560.
- `label_emb` helps, `time_embed` hurts (arm 6 vs 5 and 4).
- No arm damages any t bucket (B2); fit-metrics don't rank renders.

## Options for *new* graphs/configs

- (a) Keep defaults (cosine, attention+cond). Cheapest, preserves the
  undertraining + narrow-target behaviour the study just indicted.
- (b) Default new runs to **constant + kohya_default** (722). Landed
  first; superseded by arm 6 the same day.
- (c) Default new runs to constant + kohya_plus. Bakes dataset constants
  into the global conditioning path — and the isolation showed
  `time_embed` specifically is the harmful half. Not recommended.
- (d) Default new runs to **constant + kohya_label** (724). **Chosen:**
  matches the best-rendering arm, and keeps the cond-path exposure to
  the one tower that measured *helpful*.

## Rules for whichever is chosen

- Existing saved graphs/configs keep stored values: defaults apply only
  where no value is stored (write the values explicitly at save time so
  a default change cannot alter an old run). `kohya_default` remains a
  valid stored value, so an arm-5-era config is unchanged.
- One-line changelog entry.
- Reachability (C1) is landed but dead: config keys parse/validate yet
  reach no node. Whichever default is chosen must be wired to an actual
  control, or C1's deadness note stays true.

## Maintainer: pick (a), (b), or (c)

(b) was recommended first and approved; the maintainer then approved the
arm-6 isolation run and moved the default to 724 (d). If the B3
far-prompt/size check or a second dataset reverses 724, say so and this
note flips back to (b) — 722 — which stays available as a preset.
