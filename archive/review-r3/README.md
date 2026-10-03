# Round-3 review material

The external review this repository's `docs/status/fixes-r3.md` works
through, kept for provenance rather than for use.

* `material/review-round-3.md` -- the review itself: nine findings
  (N3-01..N3-09) and eleven improvements, each marked `[R]` reproduced or
  `[C]` read from the source.
* `material/TASK-round-3-fixes.md` -- the reviewer's own task breakdown.
* `material/0001-Docs-third-review-....patch` -- the documentation patch
  that came with it.
* `material/repro-scripts-round3.tar.gz` -- reproduction scripts for round
  3. `scripts/repro/` holds r14..r16 extracted from it.

## What was reproduced, and what was not

Verified against this tree rather than taken on trust, because the review
was written at `13566c2` and this repository has moved since.

| Finding | Outcome here |
|---|---|
| N3-01 watcher error leaves a live child | **Real.** One transient read error marked the row failed while the child ran, with no stop and no kill. Fixed, and the second half of it (a lost node result) turned out to be the more serious half. |
| N3-02 finished while the server was down | **Already fixed** before this review was read, by `13563d6`. Its own reproduction now prints `status=finished results=3/3`. |
| N3-03 vacuous pass | **Real.** `finish()` reported ALL CHECKS PASSED having run nothing, and a gate check for orphaned tests immediately found one that had never executed. |
| N3-04 stale `Last-Event-ID` | **Real.** The id is now a cursor, `{epoch}:{seq}`, and a foreign epoch is refused whatever its number. |
| N3-05 not hermetic; writes to the model directory | **Real, both halves.** The model-directory write had already happened. |
| N3-06 undeclared Python 3.14 floor | **Real.** The floor is stated in `backend/python_floor.py` and checked by both process entry points. |
| N3-07 scratch files grow without bound | **Real**, and the estimate was high by an order of magnitude: measured, 5 MB for a 12 h run, not tens of MB. Swept at startup and on delete. |
| N3-08 child signal handling | **Real.** SIGTERM had no handler at all; measured, the run lost its outcome record and every result it had produced. |
| N3-09 per-run startup cost | Known and documented. |

`test_process_identity` was named in N3-05 but does not fail on a bare
checkout -- it spawns `/bin/sleep` and `/bin/true`, nothing that resolves a
ComfyUI path. Two of that finding's three files reproduce, not three.