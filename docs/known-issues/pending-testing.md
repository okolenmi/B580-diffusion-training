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

The two entries this section used to hold are now in
[`resolved.md`](resolved.md) with the hardware numbers that closed them:
`keep_incomplete_batches`, and the training diagnostics.
