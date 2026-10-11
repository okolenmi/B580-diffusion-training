# Changelog

## 2026-10-11 — C3: new-run defaults are constant LR + kohya_default targets, cond path off

New graphs and configs without stored values now train with a constant
learning rate, the `kohya_default` target set (722 modules), and the
conditioning path off -- the B-study arm-5 configuration, which rendered
best. Saved graphs/configs keep their stored values and behave exactly
as before; pin old values explicitly if you want the old behaviour on a
new run. `kohya_plus` stays opt-in pending the far-field B3 check.
