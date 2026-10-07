Per-axis padding and modal-side preservation, for `pad_fraction_stats()` /
`report_pad_fraction()` in `manager/loader.py`.

**Why.** The `shape_bucket_multiple` knob rounds both axes up independently, so
the multiple has to divide the dataset's modal side or it pads the side nobody
needed padded. On `non-square` that is invisible from the existing report, which
gives total pad fraction only: x32 reports "13.3% mean pad" and x24 reports
"25.1%", which reads as a 2x difference rather than as *x24 pads the width of
every single sample and x32 pads none*. The per-axis numbers make the actual
failure legible:

| mult | preserves width 64 | padded on both axes | mean pad |
|---|---|---|---|
| x8  | 210/273 | 0%  | 4.8%  |
| x16 | 242/273 | 0%  | 10.5% |
| x24 | 0/273   | 93%  | 25.1% |
| x32 | 246/273 | 0%  | 13.3% |
| x48 | 0/273   | 94%  | 50.6% |
| x64 | 246/273 | 0%  | 23.4% |

Four additions, all cheap (one pass over `self.trajectories`, which the report
already does):

1. `modal_side_latent` -- the most common stored side, per axis, and its count.
   This is the number a multiple has to divide, and nothing currently reports it.
2. `modal_side_preserved` -- fraction of samples whose modal side is already a
   multiple, i.e. untouched on that axis. 246/273 for x32, **0/273** for x24.
3. `padded_both_axes` -- fraction of samples padded on *both* axes. 0% for every
   multiple except x24 (93%) and x48 (94%), which is the property that makes the
   model see a noisy border on four sides rather than two.
4. `divisible_by` -- whether the multiple divides the modal side at all, so the
   report can say "does not divide the modal width 64" in words rather than
   leaving the reader to subtract.

Deliberately not added: a recommendation. Which multiple is right depends on
four things that fit the data equally well (total pad, two-axis padding,
modal-side preservation, and whether the resulting buckets land on the base
model's own training resolutions) and `shapes diversity problem 2/` has runs
queued to separate them. A recommendation now would be a guess wearing a
number's clothes.
