*[← docs/known-issues index](README.md)*

# Pending user testing

Fixes that are believed correct but have only been exercised on CPU, or
whose *effect* has not been measured. Each entry says what is unproven.

A fix leaves this file by being run on real hardware -- Intel Arc B580,
12 GB, torch 2.12.1+xpu -- and moving to [`resolved.md`](resolved.md)
with the numbers that confirmed it. `runs/` and `datasets/` are
gitignored, so those measurements exist nowhere else, which is the whole
reason the resolved entries keep them.

## Confirming one

- `scripts/hw_validate.py` -- a single-experiment harness that builds
  and runs a real training graph on the real XPU (main or managed
  route, real checkpoint, real dataset), capturing per-step loss,
  wall time, and torch-reported reserved/allocated/peak memory to
  `runs/hw_validation/<label>/{steps.jsonl,summary.json}`. One process
  per experiment; an OOM or a `strict=True` raise is recorded as a
  *result* (exit 2 / outcome classification), not a harness failure.
- `scripts/hw_validation_batch.sh` -- the standard set of experiments
  (attention-checkpointing before/after, non-square ratchet,
  budget-pressure + strict, managed-route perf, managed-route
  escalation), runnable as a batch or one label at a time.

A fix that needs hardware confirmation should get its own experiment
there, and an entry here until it has been run.

## Pending

- **[2026-09-30] Loader silently drops images whose (caption, size) group
  is smaller than `batch_size` (shuffle on).** `ManagedDatasetLoader` now
  prints a one-time warning with exact counts, and
  `ManagedDatasetSourceNode` has a new `keep_incomplete_batches` Port
  (default False = old behavior) that keeps those samples as smaller
  batches. Covered by `manager/smoke_tests/smoke_test_loader_incomplete_
  batches.py` (real sqlite + shard).

  **Both open questions answered on hardware, 2026-10-04.**

  *The warning's counts are correct on every real dataset.* Verified by
  recomputing the grouping independently from the loader's own buckets --
  `never` (samples in groups smaller than a batch) and `partial` (the
  remainder `len(group) % batch_size`) -- and comparing all four printed
  numbers (`never`, `total`, `partial`, and the derived "only N of M are
  used") plus the number of samples actually yielded:

  | dataset | bs | samples | never | partial | usable | yielded | match |
  |---|---|---|---|---|---|---|---|
  | `1024 aes` | 2 | 201 | 0 | 1 | 200 | 200 | yes |
  | `1024 aes` | 4 | 201 | 0 | 1 | 200 | 200 | yes |
  | `non-square` | 2 | 273 | 19 | 12 | 242 | 242 | yes |
  | `non-square` | 3 | 273 | 49 | 20 | 204 | 204 | yes |
  | `non-square` | 4 | 273 | 70 | 19 | 184 | 184 | yes |

  `1024 aes` and `test2` are a single `(prompt, size)` group each, so only
  the remainder is ever lost. **`non-square` is the one that matters**: 63
  groups, and at batch 4 **89 of its 273 samples (33%) are never trained
  on** without the flag. With `keep_incomplete_batches=True` all 273 are
  used, at both batch sizes.

  *But it costs throughput, and the warning does not say so.* Measured with
  `scripts/hw_validate.py main --dataset "non-square"`, 30 steps:

  | bs | shapes off | shapes on | steps/sec off | steps/sec on | change |
  |---|---|---|---|---|---|
  | 2 | 44 | 75 | 0.412 | 0.467 | +13% |
  | 4 | 22 | 73 | 0.412 | 0.288 | **-30%** |

  So the flag trades throughput for coverage, and at batch 4 on a
  many-shaped dataset it is a 30% throughput cost to train on 33% more
  images -- worth taking deliberately, not by default. `scripts/hw_validate.py`
  grew `--keep-incomplete-batches` for these measurements.

- **[2026-10-04] `test_graph_task_gateway.py`'s cooperative-stop test fails
  about one gate run in eight under load, unrooted.** 2026-10-04's gate:
  `the child got past startup and started building nodes` and
  `having built some but not all: 0 of 4000 nodes`. Zero node records means
  the child *exited*, not that it was slow -- `_collect` already polls with a
  90 s budget and a bare `import torch` in a child costs 1.5 s.

  Could not reproduce: **0 failures in 24 runs at 4-way parallelism**, and
  3 of 3 clean `backend/tests/run_all.py` runs. So it needs the full gate's
  context, which is not yet characterised. The child builds CPU-only
  `FloatConstantNode`s but is given `COMFY_DIR`, so it does the same node
  discovery a real run does -- a discovery-time failure would look exactly
  like this.

  What changed: the two failing checks now carry `_why_silent(launch)`,
  which the file already had for this class of failure and which these two
  simply were not calling. A repeat will now report the child's last three
  log lines instead of a bare count. That is a diagnosis, not a fix.

- **[2026-09-30] Fixed-probe / gradient-alignment diagnostics**
  (`probe_every_n_steps` etc. on `ManagedLoRATrainerNode`; see
  [`../training-diagnostics.md`](../training-diagnostics.md)). Unit-tested
  on toy models only. **Not run on a real SDXL UNet or XPU:** confirm the
  step-1 record reads rel~1.000 / drift~0 on a real LoRA-injected model
  (DoRA and NF4 included), measure the probe's real wall/VRAM cost, and
  that `probe_grad_alignment`'s backward fits next to the training step's
  VRAM peak at your operating point.
