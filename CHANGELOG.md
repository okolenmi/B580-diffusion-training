# Changelog

## 2026-10-12 — Varied text conditioning: caption pool, conditioning blend, auto-caption

Three opt-in tools for the captionless-conditioning problem (a dataset whose
captions never vary gives cross-attention one constant conditioning; see
`archive/bad-LoRA-investigation/MEASURED-captionless-conditioning-break.md`).
All are off unless wired; a graph that does not use them is unchanged.

- **Auto-caption node** (`AutoCaptionNode`): fills the dataset's *empty*
  captions with a caption model (local Hugging Face image-to-text, BLIP base by
  default; more backends via `manager.captioning.register_captioner`). Never
  replaces a caption you wrote unless `overwrite`. Wire its `dataset_root`
  output to the dataset source so the source loads the result.
- **Caption pool node** (`CaptionPoolNode` -> source `caption_pool` port): each
  image trains on a prompt built from `n_min..n_max` of the dataset's real
  captions (`.`/`,` as dividers; `captions` or `parts` mode). `variants=K`
  draws a bounded seeded prompt set so text-encoder prewarm covers it. Needs
  `shape_bucket_multiple > 1` and at least two distinct captions.
- **Conditioning blend node** (`ConditioningBlendNode` -> trainer
  `conditioning_blend` port): `ctx = p*ctx(caption) + (1-p)*ctx("")`, p drawn
  per sample per step from `keep_min..keep_max`. Default 1..1 is exactly off.

Fixed on the way: `ManagedDatasetLoader.__len__` loaded samples without the
setup `__iter__` did, so calling `len()` first left caption dropout a silent
no-op; and the text-encoder prewarm only looked at each batch's first caption,
so mixed-caption batches (dropout, pool) were all cache misses that re-loaded
CLIP.

Not shown: that any of this fixes the CLIP_L-only break. It is the
mechanism, built so the experiment can be run (see `docs/maintainer-queue.md`).

## 2026-10-11 — C3: new-run defaults are constant LR + kohya_label targets (724), cond path off

New graphs and configs without stored values now train with a constant
learning rate, the `kohya_label` target set (724 modules: kohya's 722
plus `label_emb`), and the conditioning path off. Saved graphs/configs
keep their stored values and behave exactly as before; pin old values
explicitly if you want the old behaviour on a new run (`kohya_default`
= 722 still parses). `kohya_plus` stays opt-in.

This supersedes the same-day C3 entry below, which set `kohya_default`
(722) before the arm-6 isolation run existed. `label_emb` measured
helpful and `time_embed` harmful (726 < 724 with label held constant),
so the default takes the one conditioning tower that helped. Ranked by
the maintainer's render judgment: 724 > 722 > 726 > 560.

## 2026-10-11 — C3: new-run defaults are constant LR + kohya_default targets, cond path off

New graphs and configs without stored values now train with a constant
learning rate, the `kohya_default` target set (722 modules), and the
conditioning path off -- the B-study arm-5 configuration, which rendered
best at the time. Saved graphs/configs keep their stored values and behave
exactly as before; pin old values explicitly if you want the old
behaviour on a new run. `kohya_plus` stays opt-in pending the far-field
B3 check.
