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
  is smaller than `batch_size` (shuffle on).** Reproduced on CPU with a
  synthetic 8-image dataset at `batch_size=2`: only 4 of 8 images were used
  per epoch, and every image with a unique caption was never trained on.
  Change: `ManagedDatasetLoader` now prints a one-time warning with exact
  counts, and `ManagedDatasetSourceNode` has a new `keep_incomplete_batches`
  Port (default False = old behavior) that keeps those samples as smaller
  batches. Covered by `manager/smoke_tests/smoke_test_loader_incomplete_
  batches.py` (real sqlite + shard). **Not run on real data or hardware:**
  confirm the warning's counts on a real dataset, and that
  `keep_incomplete_batches=True` doesn't trigger a new-shape stall on XPU
  (each extra batch shape is a fresh kernel set).
- **[2026-09-30] Fixed-probe / gradient-alignment diagnostics**
  (`probe_every_n_steps` etc. on `ManagedLoRATrainerNode`; see
  [`../training-diagnostics.md`](../training-diagnostics.md)). Unit-tested
  on toy models only. **Not run on a real SDXL UNet or XPU:** confirm the
  step-1 record reads rel~1.000 / drift~0 on a real LoRA-injected model
  (DoRA and NF4 included), measure the probe's real wall/VRAM cost, and
  that `probe_grad_alignment`'s backward fits next to the training step's
  VRAM peak at your operating point.
