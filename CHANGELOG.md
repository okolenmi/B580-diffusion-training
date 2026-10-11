# Changelog

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
