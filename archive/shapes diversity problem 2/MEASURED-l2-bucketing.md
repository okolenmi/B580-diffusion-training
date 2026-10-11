# L2: does shape bucketing cost quality, and which multiple?

> ## EVERY QUALITY NUMBER IN THE "FIRST SWEEP" SECTION BELOW IS VOID
>
> Found 2026-10-07, after those runs were taken. The **managed** trainer
> route's `LossPhase` had no bucketing-mask handling at all: it took a plain
> mean over the whole canvas, so every bucketed run scored (and trained on) the
> padding, **and had its loss and gradient scaled by the valid fraction** — a
> 25%-padded batch trained at 0.75x the effective rate. `hw_validate.py
> managed` is the route every measurement here was taken on.
>
> The fix is `0a7f6a2`. The five first-sweep runs are kept at
> `/tmp/opencode/void_runs/` rather than deleted, because their **throughput**
> numbers survive — the bug was in the loss, not in the shapes or the launches
> — but not one quality conclusion does.
>
> The tell was in data this file already contained and read as interesting:
> bucketed runs' *training* loss was up to 42.7% below unbucketed's while their
> unpadded evaluation was 0.45% above it. That divergence was not a mechanism.
> At x24's 25.1% mean pad, the rescaling alone predicts a 0.749x loss ratio
> against the 0.573x that was measured.
>
> **A re-run of all five, plus x8 / x48 / x64, is what this document now
> waits on.** The analysis sections below are kept because the *shapes* are
> properties of the dataset and are unaffected — but the verdicts are not.

## What is settled regardless of the loss bug: which shapes each multiple produces

This is arithmetic on the dataset, not a measurement of training, so it stands:

| mult | buckets | latent sides | px buckets | SDXL buckets | pad W | pad H | **2-axis** | mean pad | width kept at 64 |
|---|---|---|---|---|---|---|---|---|---|
| off | 63 | 48-96 | — | 2/63 | 0% | 0% | 0% | 0% | 273/273 |
| x8 | 13 | 48,56,64,72,80,88,96 | 384…768 | 3/13 | 17% | 57% | **0%** | 4.8% | 210/273 |
| x16 | 7 | 48,64,80,96 | 384…768 | 3/7 | 24% | 63% | **0%** | 10.5% | 242/273 |
| x24 | 5 | 48,72,96 | 384…768 | **0/5** | **98%** | 95% | **93%** | 25.1% | **0/273** |
| x32 | 3 | 64,96 | **512×512, 512×768, 768×512** | **3/3** | 25% | 68% | **0%** | 13.3% | 246/273 |
| x48 | 3 | 48,96 | 384…768 | 0/3 | **99%** | 95% | **94%** | 50.6% | **0/273** |
| x64 | 3 | 64,128 | 512×512, 512×1024, 1024×512 | 1/3 (3/3 incl. 1024-px sides) | 25% | 69% | **0%** | 23.4% | 246/273 |

**The dataset's modal width is exactly 64 latent (512 px), 204 of 273 samples.**
64 is divisible by 8, 16 and 32 and by neither 24 nor 48. So:

- **x24 and x48 are the only multiples that never preserve that width** (0 of
  273 samples keep it) and the only ones that **pad on both axes** (93-94%).
  Every other multiple pads at most one axis, so content stays full-bleed on
  one side and the model sees a noisy border on two edges rather than four.
- **x32 lands every bucket on one of SDXL's own training resolutions** —
  512×512, 512×768, 768×512 — and x24 lands on none of them.

### The lesson is about the interaction, not the number

24 and 48 are not bad multiples. They are bad *on this dataset*, because its
modal side is 64 and they do not divide it. On a dataset whose modal width were
72, x24 would preserve it, x32 would pad it, and the ranking would invert.

The rule that falls out: **prefer a multiple that divides the dataset's modal
side.** `shape_bucket_multiple` is a single global integer that rounds both
axes up independently, so it cannot express that — a user who picks 24 has no
way to learn that they just padded the width of 75% of their dataset. The
build-time pad report (below) is where that should say so.

### Four candidate mechanisms, and what separated them

x32 was good on all four counts and x24 bad on all four, which is why the first
sweep's single comparison could not distinguish anything:

1. **total pad fraction** — 13.3% vs 25.1%
2. **two-axis padding** — 0% vs 93%
3. **modal-width preservation** — 246/273 vs 0/273
4. **SDXL-bucket affinity** — 3/3 vs 0/5

| run | designed to test | prediction if (1) pad dominates | if (4) shapes dominate |
|---|---|---|---|
| **x8** | lowest pad (4.8%), 0% two-axis, but 3/13 SDXL buckets | fine | worse |
| **x64** | high pad (23.4%), width preserved, all SDXL buckets | worse | fine |
| **x48** | 94% two-axis, 50.6% pad, 0/3 buckets | much worse | worse |

**x8 and x48 ran; x64 was queued but the elimination is already complete without
it** (see "What the seven arms settle"). x8 came back fine, killing mechanism 4.
x48 came back worst while x24 — identical to it on mechanisms 2 and 3 — came
back fine, killing mechanism 2 and leaving mechanism 1.

The cleanest separator had been cropping to SDXL buckets: 100% in-distribution
shapes *and* 0% padding, which no multiple can achieve because padding always
introduces a mismatch. `archive/shapes diversity problem/shape_policy.py` implements the
cropping half; the loader only pads today. It is no longer needed to answer the
question, but it remains the untested option if pad volume is ever the binding
constraint.

---

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

## Results after the fix: 300 steps, batch 2, seed 1234

| run | bucket | buckets | first-sighting s | steps/s | peak MB | pad mean | **holdout MSE** | Δ vs x0 |
|---|---|---|---|---|---|---|---|---|
| `L2_x0` | off | 44 | 153.9 | 0.720 | 7,234 | 0% | 0.166322 | — |
| `L2_x0_b` | off, seed 4321 | 44 | 152.4 | 0.728 | 7,234 | 0% | 0.166765 | **+0.000443** |
| `L2_x8` | 8 | 13 | 51.8 | 0.948 | 7,232 | 4.8% | 0.166166 | −0.000155 |
| `L2_x16` | 16 | 7 | 25.3 | 1.020 | 7,232 | 10.5% | 0.166065 | −0.000257 |
| **`L2_x24`** | 24 | 5 | 18.4 | 1.049 | 7,220 | 25.1% | 0.166569 | +0.000247 |
| `L2_x32` | 32 | 3 | 8.7 | **1.078** | 7,220 | 13.3% | 0.166188 | −0.000133 |
| **`L2_x48`** | 48 | 3 | 8.7 | 0.985 | 7,220 | 50.6% | **0.168623** | **+0.002302** |

All seven arms produced digest `087d80cc6b1149f9`.

**The noise control is +0.000443.** Against it, only **x48 is worse** (5.2× the
control). Every other multiple is within noise, including **x24**, which the
void sweep called "measurably worse" — that was the loss bug, and it is
withdrawn.

### What the seven arms settle, by elimination

**The SDXL-bucket hypothesis is dead.** `x8` lands on only **3 of 13** buckets
and shows no degradation. If affinity to SDXL's training resolutions mattered,
x8 would be the worst arm and it is among the best.

**Two-axis padding is dead too.** x24 shares *every* one of x48's suspicious
properties — 0 SDXL buckets, 0 of 273 samples keeping the modal width, 93-94%
padded on both axes — and differs only in pad volume. It is within noise. So
those three properties are not what makes x48 bad.

**Pad fraction is the mechanism**, with the threshold between 25% (fine) and 51%
(worse). That also has a plain reading: at 50.6% pad, half the canvas is noise
the model is not scored on, so half the compute per step buys nothing and the
real-pixel batch is halved.

The four hypotheses this document opened with are now one. Pad fraction predicts
every arm; bucket count, SDXL affinity, two-axis padding and modal-width
preservation each predict at least one arm backwards.

### Throughput

x32 is **1.50x** (0.720 → 1.078 steps/s) with peak memory unchanged (7,220 vs
7,234 MB) and first-sighting time cut from 153.9 s to 8.7 s. x16 gives 1.42x at
the lowest pad that still collapses the shape count. x8 is 1.32x but keeps 13
buckets, so it pays 51.8 s of compiles to save 32.7.

**x32 is the right default for this dataset** on all three of: throughput, pad
fraction, and SDXL-bucket alignment.

## Results before the fix — VOID, kept as the record

These are the runs that found the managed-route mask bug, so they are kept as
the record of what was measured on a broken loss rather than as findings.

| run | bucket | buckets | first-sighting s | steps/s | peak MB | pad mean | pad max | **holdout MSE** |
|---|---|---|---|---|---|---|---|---|
| `L2_x0` | off | 44 shapes | 158.8 | 0.690 | 7,234 | 0% | 0% | 0.166285 |
| `L2_x0_b` | off, seed 4321 | 44 | 157.9 | 0.711 | 7,234 | 0% | 0% | 0.166543 |
| `L2_x16` | 16 | 7 | 25.3 | 1.031 | 7,232 | 10.5% | 23.4% | 0.166273 |
| `L2_x24` | 24 | 5 | 18.9 | 1.045 | 7,220 | 25.1% | 39.5% | 0.167034 |
| `L2_x32` | 32 | 3 | 8.9 | **1.076** | 7,220 | 13.3% | 32.3% | 0.166414 |

### (c) Quality — VOID. The paired *method* still stands.

The paired per-batch comparison is still the right way to read this metric, and
the 16-of-16 result below is what made the x24 signal worth chasing. But every
number was produced by a run whose loss excluded nothing and was rescaled by
the valid fraction, so none of it is evidence about bucketing.



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
