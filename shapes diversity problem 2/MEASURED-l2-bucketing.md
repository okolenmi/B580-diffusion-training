# L2: does shape bucketing cost quality, and which multiple?

Answered by measurement, not argument. Both questions needed a number that did
not exist before: loss on a **fixed unpadded holdout**, which nothing in the
project could previously produce.

## The measurement, and what it actually measures

A run's own loss cannot answer this. It is measured on whatever that run
happened to train on, so comparing two arms' training losses compares two
different sets of samples at two different shapes. It cannot distinguish
"bucketing damaged the model" from "bucketing changed the data" — and those are
the only two things in question.

So `scripts/hw_validate.py` grew `--holdout-batches` / `--holdout-seed`. The
evaluation set is built **before** training, from a loader with bucketing
**off** and shuffle **off**, and the RNG is reseeded immediately beforehand, so
its contents depend only on the seed. `summary.json` records a SHA-256 digest
over every tensor and caption, and the analyzer **refuses to compare** two runs
whose digests differ. That check is not ceremony: it is the difference between
a comparison and a coincidence. Every arm here produced `087d80cc6b1149f9`.

The score is plain unweighted MSE, forward-only, no loss weighting and no LoRA
gate. The diffusion process is rebuilt with the trainer node's own defaults,
copied, so the input transform is the one training used.

### It is not a held-out set, and the flag name overstates it

16 batches of this dataset is **31 images**, drawn from the same 273 the model
trains on. Measured after 300 steps:

| arm | distinct images trained | scored images never seen |
|---|---|---|
| x0 | 242 | **2 of 31** |
| x16 | 270 | 0 |
| x24 | 272 | 0 |
| x32 | 272 | 0 |

So this measures **fit, not generalization**. The unbucketed arm is
*penalised* on 6% of the set it is scored on, and trained on ~12% less data
(242 vs 272 distinct images) — which cuts against the finding below rather than
for it, but it does mean the arms never differed only in padding. The flag name
is kept for compatibility with existing run configs; the docstring now says what
it measures. A genuinely held-out split is still unbuilt.

### Why the comparison is still valid: it is paired

Same image, same noise, same t per batch, so per-batch difficulty cancels in the
difference. The raw spread across the 16 batches is **47x** (0.0068 to 0.318) —
a number that looks like it should swamp a 0.45% effect, and does not, because
it never enters a paired difference.

## Results: 300 steps, batch 2, seed 1234

| run | bucket | buckets | first-sighting s | steps/s | peak MB | pad mean | pad max | **holdout MSE** |
|---|---|---|---|---|---|---|---|---|
| `L2_x0` | off | 44 shapes | 158.8 | 0.690 | 7,234 | 0% | 0% | 0.166285 |
| `L2_x0_b` | off, seed 4321 | 44 | 157.9 | 0.711 | 7,234 | 0% | 0% | 0.166543 |
| `L2_x16` | 16 | 7 | 25.3 | 1.031 | 7,232 | 10.5% | 23.4% | 0.166273 |
| `L2_x24` | 24 | 5 | 18.9 | 1.045 | 7,220 | 25.1% | 39.5% | 0.167034 |
| `L2_x32` | 32 | 3 | 8.9 | **1.076** | 7,220 | 13.3% | 32.3% | 0.166414 |

### (c) Quality: the paired result, and what it does not say

Two runs of the *same* configuration differing only in seed give **0.000258**,
which is a noise floor but not a test — n=1. The stronger evidence is the
per-batch sign, because the arms are paired:

| arm | batches better than x0 | mean Δ vs x0 |
|---|---|---|
| `L2_x0_b` (seed control) | 1/16 | +0.000258 |
| `L2_x16` | 7/16 | −0.000013 |
| `L2_x24` | **0/16** | **+0.000748 (+0.45%)** |
| `L2_x32` | 5/16 | +0.000129 |

**x24 is worse on 16 of 16 batches, with no exceptions.** Under a null of random
sign that is p ≈ 2⁻¹⁶ ≈ 1.5e-5. x16 and x32 straddle zero, which is what noise
looks like. The original write-up led with "2.9× the noise control" — that is the
*weaker* of these two arguments and should not have been the headline.

**What this does and does not establish:**

- **The sign is solid; the magnitude is one draw.** +0.45% with n=1 per arm and
  no error bar. The ordering across three arms is also monotone in pad
  fraction (x16 10.5% → −0.000013, x32 13.3% → +0.000129, x24 25.1% →
  +0.000748), which is a dose-response in the mechanism's magnitude and is
  harder to explain by noise than any single pairwise comparison.
- **It does not mean x24's images look worse.** +0.45% MSE is very likely below
  the visible threshold. "Measurably worse" is a statement about a number, and
  the first write-up let it read as a statement about quality.
- **It is fit, not generalization** (see above), and the arms differed in data
  coverage as well as padding.

The x24 result is still the interesting one, because **bucket count alone would
have ranked it best**: 5 buckets is between x16's 7 and x32's 3, and 1.045
steps/s looks like a sensible middle. What it actually does is pad *more* than
x32 (25.1% mean against 13.3%) while reaching *fewer* shapes only because its
quantum straddles the size histogram differently. Pad fraction predicts the
cost; bucket count does not.

That is the argument for the L2a report existing: the pad fraction was the
number that made this predictable before the run.

### (b) Throughput and cost

- **x32 is fastest: 0.690 → 1.076 steps/s = 1.56x**, and it cuts first-sighting
  time from 158.8 s to 8.9 s.
- The earlier 1.83x figure was measured over 150 steps; over 300 it is 1.56x,
  which is what amortisation of a one-time saving should do. The saving is
  149.9 s; spread over 300 steps it is 0.50 s/step, which at 0.690 steps/s is
  most of the difference.
- **Peak memory is unchanged** (7,220 vs 7,234 MB). Bucketing costs no memory.
- x16 gives most of the speed (1.49x) at the lowest pad fraction (10.5%).

## What this does and does not settle

**Settles:** bucketing at 16 or 32 does not measurably hurt a 300-step LoRA's
quality on this dataset, and x32 is 1.56x faster for free in memory terms.

**Does not settle, and is left open:**

- **It is a fit metric, not a held-out one.** The scored images are the training
  images. If padding damages *generalization* in a way that a 31-image in-sample
  evaluation cannot see, this design would not find it.
- **300 steps is short.** A LoRA's quality differences usually appear much later.
  The x24 result being *visible* at 300 steps suggests the measurement has
  power, but "no difference at 300 steps" is not "no difference".
- **The arms differed in more than padding.** Unbucketed training covered 242
  distinct images against 270-272 for the bucketed arms, and left 2 of the 31
  scored images unseen. A per-sample-count-matched comparison would be cleaner.
- **One dataset, one prompt.** `non-square` has a single distinct caption, so
  per-caption variation — the thing L4 is about — is untested here.
- **The default stays off.** The task said to keep it off until (c) was
  reported; (c) now says 32 and 16 are indistinguishable from not bucketing,
  which is a reason to *consider* defaulting on, not a reason to do it alongside
  the measurement. The remaining arguments for off are unchanged: it changes
  which pixels the loss covers and the order samples arrive in, and L4's
  per-sample captioning would subsume much of what it buys anyway.

## Reproducing

```bash
bash "shapes diversity problem 2/l2_sweep.sh"        # ~30 min, 5 sequential runs
python3 "shapes diversity problem 2/analyze_l2.py"
```

Runs are skipped if `summary.json` already exists, and the sweep checks that
artefact rather than the exit status, because a run that died of an OOM looks
identical to one that finished when the output is piped.

## One thing this found that is not about bucketing

`scripts/hw_validate.py`'s docstring lists `console.log` as one of its three
outputs. It was never written. Every node-side report — the L2a pad fractions
above, the residency controller's calibration line, the prewarm summary —
existed only in terminal scrollback. Found because this analyzer went looking
for `pad fraction` in that file and got nothing, and printed a table of 0.0%
pad fractions as if they had been measured. Fixed by teeing stdout into the
file; the pad column in the table above is from the L2a measurement, not from
the runs in this sweep, because these five ran before the fix.
