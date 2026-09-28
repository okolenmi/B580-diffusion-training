*[← docs/known-issues index](README.md)*

# Pending user testing

**Queue is empty as of 2026-09-28.** All five previously-pending
fixes were run on real hardware (Intel Arc B580, 12 GB,
torch 2.12.1+xpu) and moved to [`resolved.md`](resolved.md), each with
the measured result that confirmed it.

How that happened, and how to repeat it:

- `scripts/hw_validate.py` -- a single-experiment harness that builds
  and runs a real training graph on the real XPU (main or managed
  route, real checkpoint, real dataset), capturing per-step loss,
  wall time, and torch-reported reserved/allocated/peak memory to
  `runs/hw_validation/<label>/{steps.jsonl,summary.json}`. One process
  per experiment; an OOM or a `strict=True` raise is recorded as a
  *result* (exit 2 / outcome classification), not a harness failure.
- `scripts/hw_validation_batch.sh` -- the standard set of experiments
  behind the five old entries (attention-checkpointing before/after,
  non-square ratchet, budget-pressure + strict, managed-route perf,
  managed-route escalation), runnable as a batch or one label at a time.

A new fix that needs hardware confirmation should get its own
experiment there, and an entry here until it's been run -- then move it
to `resolved.md` with the numbers.
