# L2: does shape bucketing cost quality, and which multiple?

Answered by measurement, not argument. Both questions needed a number that did
not exist before: loss on a **fixed unpadded holdout**, which nothing in the
project could previously produce.

## The measurement, and why it needed building

A run's own loss cannot answer this. It is measured on whatever that run
happened to train on, so comparing the bucketed run's loss to the unbucketed
run's compares two different sets of samples at two different shapes. It cannot
distinguish "bucketing damaged the model" from "bucketing changed the data" —
and those are the only two things in question.

So `scripts/hw_validate.py` grew `--holdout-batches` / `--holdout-seed`. The
holdout is built **before** training, from a loader with bucketing **off** and
shuffle **off**, and the RNG is reseeded immediately beforehand, so its
contents depend only on the seed and not on how much randomness each arm
consumed first (which differs between arms by construction). `summary.json`
records a SHA-256 `holdout.digest` over every tensor and caption, and the
analyzer **refuses to compare** two runs whose digests differ.

That check is not ceremony: it is the difference between a comparison and a
coincidence. Every arm here produced `087d80cc6b1149f9`.

The score is plain unweighted MSE, forward-only, no loss weighting and no LoRA
gate — a yardstick that does not depend on the run's configuration. The
diffusion process is rebuilt with the trainer node's own defaults, copied, so
the input transform is the one training used.

## Results: 300 steps, batch 2, seed 1234

| run | bucket | buckets | first-sighting s | steps/s | peak MB | pad mean | pad max | **holdout MSE** |
|---|---|---|---|---|---|---|---|---|
| `L2_x0` | off | 44 shapes | 158.8 | 0.690 | 7,234 | 0% | 0% | 0.166285 |
| `L2_x0_b` | off, seed 4321 | 44 | 157.9 | 0.711 | 7,234 | 0% | 0% | 0.166543 |
| `L2_x16` | 16 | 7 | 25.3 | 1.031 | 7,232 | 10.5% | 23.4% | 0.166273 |
| `L2_x24` | 24 | 5 | 18.9 | 1.045 | 7,220 | 25.1% | 39.5% | 0.167034 |
| `L2_x32` | 32 | 3 | 8.9 | **1.076** | 7,220 | 13.3% | 32.3% | 0.166414 |

### (c) Quality: the noise control is what makes this readable

Two runs of the *same* configuration differing only in seed give
**|0.000258|**. That is the yardstick. Read against it:

| arm | vs `L2_x0` | verdict |
|---|---|---|
| `L2_x16` | −0.000013 (−0.01%) | **within noise** |
| `L2_x32` | +0.000129 (+0.08%) | **within noise** |
| `L2_x24` | +0.000748 (+0.45%) | **WORSE — 2.9x the noise** |

**A multiple of 32 and 16 are indistinguishable from not bucketing. A multiple
of 24 is measurably worse.** Without `L2_x0_b` the +0.000129 for x32 would have
looked like a small regression and the +0.000748 for x24 like a large one; the
control is what separates them.

The x24 result is the interesting one, because **bucket count alone would have
ranked it best**: 5 buckets is between x16's 7 and x32's 3, and 1.045 steps/s
looks like a sensible middle. What it actually does is pad *more* than x32
(25.1% mean against 13.3%) while reaching *fewer* shapes only because its
quantum straddles the size histogram differently. Pad fraction predicts the
quality cost; bucket count does not.

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

- **300 steps is short.** A LoRA's quality differences usually appear much later
  than 300 steps. The x24 result being *visible* at 300 steps suggests the
  measurement has power, but "no difference at 300 steps" is not "no
  difference".
- **One dataset, one prompt.** `non-square` has a single distinct caption, so
  per-caption variation — the thing L4 is about — is untested here.
- **The default stays off.** The task said to keep it off until (c) was
  reported; (c) now says 32 and 16 are within noise, which is a reason to
  *consider* defaulting on, not a reason to do it in the same breath as the
  measurement. The remaining arguments for off are unchanged: it changes which
  pixels the loss covers and the order samples arrive in, and L4's per-sample
  captioning would subsume most of what it buys anyway.

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
