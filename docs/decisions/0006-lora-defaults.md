# Decision note: LoRA training defaults for new runs (C3 — awaiting approval)

Status: **proposed, not implemented. Do not change any default until the
maintainer approves one option below.**

## What B established (evidence, not argument)

- Constant vs cosine at the same nominal LR: ~2x trained magnitude
  (forensics), renders rank constant first (A1 + B arms 2>1, 4>3).
- Targets: attention-only forces attn_output to ~1.35x attn_input;
  722 lands at 1.08; renders rank 722 > 726 > 560.
- kohya_plus (726) renders second-best on captionless data, but its
  label_emb runs hot (5x warning fired on arm 3) and the far-prompt/size
  B3 comparison is still open. diffusers loads all 726 keys (compat OK;
  semantics unproven).
- No arm damages any t bucket (B2); fit-metrics don't rank renders.

## Options for *new* graphs/configs

- (a) Keep defaults (cosine, attention+cond). Cheapest, preserves the
  undertraining + narrow-target behaviour the study just indicted.
- (b) Default new runs to **constant + kohya_default**, kohya_plus
  opt-in. Recommended: matches the best-rendering arm (722) with no
  cond-path exposure.
- (c) Default new runs to constant + kohya_plus. Faster renders on
  captionless data today; bakes dataset constants into the global
  conditioning path until B3 clears it. Not recommended yet.

## Rules for whichever is chosen

- Existing saved graphs/configs keep stored values: defaults apply only
  where no value is stored (write the values explicitly at save time so
  a default change cannot alter an old run).
- One-line changelog entry.
- Reachability (C1) is landed but dead: config keys parse/validate yet
  reach no node. Whichever default is chosen must be wired to an actual
  control, or C1's deadness note stays true.

## Maintainer: pick (a), (b), or (c)

(b) is the agent's recommendation subject to your B render judgment
standing (722 first). If your sheets later reverse 722 vs 726, say so
and this note flips to (c) with the B3 caveat attached.
