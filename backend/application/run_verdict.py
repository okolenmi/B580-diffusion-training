"""One rule for turning a trainer's own last word into a run status.

Extracted because the recovery path and the normal path had drifted
apart, which is exactly the shape of defect docs 08 N-03 describes:
`RunSupervisor._finalize` consults the trainer's terminal progress line
when it cannot read an exit code, while `ReconcileRuns` did not -- so a
run that finished *while the server was down* was recorded `failed` at
0 steps, even though the progress file it never opened said otherwise.

The rule itself, for a process this server did not spawn:

- ``finished`` -> completed
- ``error``    -> failed ("trainer reported an error")
- anything else, including silence -> failed. A missing last word is
  never read as success: the evidence is absent, not favourable.

`exit_code is not None` means we spawned the process and can read its
real exit status, which outranks whatever it wrote -- a trainer that
crashed after writing ``finished`` is a failure, and only the exit code
says so.
"""

from __future__ import annotations

from ..domain.entities.run import Run
from .ports.clock import Clock
from .ports.progress_source import ProgressSample


def fold_samples(run: Run, samples: list[ProgressSample], clock: Clock) -> str | None:
    """Apply every sample in order to ``run``, return the trainer's last
    verdict (``"finished"``/``"error"``) or None if it never wrote one.

    No persistence and no events: the caller owns the commit, because
    both callers need a different compare-and-swap policy around it.
    ``done_steps`` only ever grows (the last sample that carries a step
    wins), and ``total_steps`` is whatever the trainer reported -- a
    plan only grows, which is the entity's rule, not a caller's
    (docs 08 S-15).

    A pure terminal line contributes its verdict and nothing else, the
    same distinction `ProgressSample.is_terminal_only` exists for.
    """
    verdict: str | None = None
    for sample in samples:
        if sample.terminal is not None:
            verdict = sample.terminal
        if sample.is_terminal_only:
            continue
        run.record_progress(
            done_steps=run.done_steps if sample.step is None else sample.step,
            at=clock.now(),
            current_loss=sample.loss,
            avg_loss=sample.avg,
            phase=sample.phase,
            total_steps=sample.total,
            cache_done=sample.cache_done,
            cache_total=sample.cache_total,
        )
    return verdict


def apply_verdict(run: Run, *, verdict: str | None, exit_code: int | None,
                  clock: Clock, unreadable_exit_note: str) -> str:
    """Put the terminal status on ``run`` and return the status word, for
    the log marker.

    Does not persist; the caller decides whether its CAS still holds.
    """
    at = clock.now()
    if exit_code is not None:
        # We spawned it, so the real exit status is authoritative.
        if exit_code == 0:
            run.mark_completed(at=at)
            return "completed"
        run.mark_failed(at=at, error=f"Exit code {exit_code}", exit_code=exit_code)
        return "failed"

    if verdict == "finished":
        run.mark_completed(at=at)
        return "completed"
    run.mark_failed(
        at=at,
        error=(
            "trainer reported an error"
            if verdict == "error"
            else unreadable_exit_note
        ),
    )
    return "failed"