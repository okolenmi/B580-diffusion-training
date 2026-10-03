# Round-3 review: outcomes

The third external review produced 9 findings (`N3-01`..`N3-09`) and 11
improvements. This is the record of what was decided for each and why.

The review's own material is at
[`archive/review-r3/`](../../archive/review-r3/), which also says which
findings were verified against this tree and which did not reproduce. It
was written at `13566c2`; this repository has moved since, so each item was
re-verified before being worked on. Two of them did not survive contact
with the current code.

The review's rules were sound. Its picture of the state often was not, for
the same reason round 2's was: it was reviewing work in progress, so some
items were already fixed when it was written. One — N3-02 — was fixed days
earlier, and its own reproduction now prints the fixed behaviour.

## Findings

| ID | Severity | Outcome | What holds it now |
| --- | --- | --- | --- |
| N3-01 | Med-High | fixed, in two parts | `backend/tests/test_graph_adoption.py`. A transient error inside the watcher used to mark the row failed while the child ran on, unsignalled, which released the single-active check and invited a second trainer onto the same card. The loop now rides out transient failures and a watcher that genuinely gives up stops its child before the row goes terminal. The second part was worse and less visible: `tail.poll()` consumes a batch as it hands it over, so a node result whose *write* failed was gone rather than deferred, and the row finished normally with results missing and nothing saying so. `_record_node` now retries its own write, idempotently — see below. |
| N3-02 | Med-High | already fixed | The startup sweep failed any row with no live process, so a run that completed while the server was down was reported as an error with its results discarded. Settled from the run's own outcome record; `scripts/repro/r16_graph_finished_while_down.py` prints `status=finished results=3/3`. |
| N3-03 | Medium | fixed, and it found something | `finish()` reported ALL CHECKS PASSED having run nothing, and two test files here had already been bitten that way. It now counts every check and treats zero as a failure. The static half is a gate step, `scripts/check_test_wiring.py`, which immediately found a test that had never executed — and a mutation check reported for it that was therefore vacuous. |
| N3-04 | Low-Med | **open** | A `Last-Event-ID` from a previous server process is accepted as valid once the new process has published that far. The event sequence is process-local and carries no epoch. `scripts/repro/r15_stale_event_id.py` reproduces it. |
| N3-05 | Low-Med | fixed, both halves | One half wrote into the developer's real model directory on every run and had already done so; measured, then fixed by giving the test a directory it owns. The other half is that the suite needs a configured ComfyUI at all: `support.use_temporary_comfy_dir()`. |
| N3-06 | Low | **open** | `pyproject.toml` targets py314 for ruff and mypy but nothing declares a floor or checks one, and `get_origin(t) is Union` is only true from 3.14. Only tests and tooling reach it at runtime, so the impact is low. |
| N3-07 | Low-Med | **open** | Per-execution scratch files — graph, event history, log — are never deleted, and `DeleteGraphExecutions` removes rows, not files. Adoption reads a whole file in one call. |
| N3-08 | Low | fixed | The child handled SIGINT but not SIGTERM, and SIGTERM is what `kill`, `docker stop` and systemd send. Measured mid-run: SIGINT wrote an outcome record and exited 0, SIGTERM was killed by signal 15 with no record at all, so the row was told a run the user had deliberately stopped had suffered a device fault. |
| N3-09 | Info | known | Every run pays full node discovery and a torch import in a fresh child: ~2 s, measured, and the price of the per-run process that exists so a device fault cannot take the server down. Documented rather than fixed; a warm worker is the lever if it becomes annoying. |

## The part of N3-01 worth reading twice

`ExecutionLifecycleWriter.commit` compare-and-swaps the row and *then*
publishes. So "the write failed" does not mean "the row is unchanged", and
the first version of the retry — which simply repeated the whole write —
turned one injected post-swap failure into **35 stored results for 12
nodes**. The retry now re-reads the row and skips when it already carries
that `node_id`, which is never legitimate: the entity already refuses a row
holding more results than its graph has nodes.

That was measured against the real SQLite repository, not the test suite's
doubles, because the safety of a retried write depends on `get` being
uncached, and `StubExecutions` hands the same mutable object to every
caller. The double double-counts under this retry; the real one does not,
and only the real one can show that.

## Two claims withdrawn

Recorded because a measurement taken before a later fix stopped being
evidence, and the conclusion drawn from it had to be given back.

* **The poll cycle does not need reordering.** Reading the row before
  consuming a batch, rather than after, was tried on the theory that a
  failure at that read arrived after delivery and lost the batch. It changes
  nothing, because every path inside `_apply` is already guarded per record
  once `_record_node` retries its own write. Reverted; the "4 of 40 results
  lost" figure in the comment came from a measurement taken *before* that
  retry existed.
* **The late handler install in the child is not covered by a test.** Moving
  `signal.signal` to the top of `main` was measured at 37 ms of imports and
  argument parsing — too short to hit by hand, which is presumably why it
  survived — and it is three lines. But both signal tests deliberately wait
  for the child's first node record before signalling, precisely so they
  test the signal path rather than the startup race, and that record
  arrives after the handlers are installed either way. Reintroducing the
  late install leaves every check green. The SIGTERM half is tested and
  mutation-verified; the reordering is a cheap correctness change with
  nothing behind it.

## Also found while fixing these

Three things the review did not report, all in code it had looked at:

* A test was writing `m.safetensors` — 2 bytes — into the developer's real
  `models/checkpoints/`, on every run. A real checkpoint of that name would
  have been overwritten by them. The stray file was removed and the blast
  radius measured: exactly one file, in the model directory, nothing else
  anywhere.
* Fixing that test exposed a real bug: `bootstrap` passed
  `layout.checkpoints_dir` into `StartDatasetTask` as a value read once at
  wiring time, so a `checkpoints_dir` changed in Settings was reported by
  the API and not used until the server restarted — and the refusal named
  the *old* directory, pointing the user at the setting they had just
  changed. `StartDatasetTask` now resolves per use.
* A `def test_*` that nothing called had been sitting in
  `test_graph_event_stream.py` since it was written, and the mutation check
  reported for it was vacuous for the same reason: both runs "passed"
  because nothing ran.