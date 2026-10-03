"""SweepExecutionScratch -- remove per-execution scratch once its run is over.

Round-3 N3-07. Every execution the supervisor starts leaves three files in
the scratch directory: ``execution_N.graph.json``, ``execution_N.events.jsonl``
and ``execution_N.log``. Nothing ever removed them. `DeleteGraphExecutions`
deleted rows, and a row is the only record that a run existed -- so deleting
the history freed nothing on disk, and the directory grew by one run's worth
forever.

Measured here, real child and real writer:

    one 4000-node execution        845,756 bytes
      execution_1.events.jsonl     523,692  (62%)   131 bytes/node
      execution_1.graph.json       321,817  (38%)
      execution_1.log                  247
    monitor reports                 126 bytes each
      one per second              ~0.5 MB/hour, ~5 MB over a 12 h run

So the review's "order of tens of MB per 12 h run" is an over-estimate by
roughly an order of magnitude -- it labelled the figure an estimate, and the
measured number is smaller. What makes this worth fixing is not the size of
one run but that there is no upper bound on the *number* of them: a hundred
runs is 85 MB, a thousand is 850 MB, and deleting the history from the UI
appears to be tidying up while leaving all of it.

**When a file is safe to delete.** The event file is read in two situations:
by the watcher while the run is live, and by `adopt` on a server restart,
for a run that is still going. Once the row is terminal the run's results
are in the row and its outcome is in the row, so the file is read by nobody.
A file whose row has *vanished* -- deleted, or from a database that was
replaced -- is read by nobody either, and is the case that otherwise
survives forever with no way to tell what it was.

So the rule is deliberately not age-based: terminal or orphaned, and nothing
else. Retention by age would be a second rule covering a case this one
already catches, and would keep files this rule would have taken.

What is *not* handled here: `ExecutionEventTail.poll` reads from its offset
to end-of-file in a single call, so replaying a large file parses all of it
at once. Bounding that changes the hot path of every poll, and it is a
memory bound rather than a growth problem -- the file is bounded by the run
that is currently in flight. Recorded rather than fixed.

**What is kept, and why.** The event file and the graph go for every
terminal run, exactly as before. The *log* of a run that ended `error` or
`stopped` stays, up to `GRAPH_FAILED_LOG_KEEP` of the most recent.

That is a change from the first version of this file, which deleted all
three unconditionally, and it was wrong: for a child that died without
reporting an outcome the row says "exited without reporting an outcome
(crashed, or a device fault killed it)" and points at the execution log --
which the next server start then deleted. The only place a user could look
for the traceback was destroyed by the tool that was tidying up, and the
row said nothing more than "it died".

Two mitigations rather than one, because either alone leaves a hole. The
row now carries a bounded tail of the log (see
`GraphExecutionSupervisor._with_log_tail`), so the evidence survives in the
database; and the file survives on disk for the newest N, so a user can
read a traceback in full rather than through a 4 KiB summary.

A run whose row has *vanished* is not one of the N -- nothing claims it
failed, and orphan logs would spend the slots on runs nobody can identify.
A row that cannot be read is, because unreadable is not absent.

A `finished` run's log is still removed immediately: it succeeded, so there
is nothing in it that the row does not already say.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..limits import GRAPH_FAILED_LOG_KEEP
from ..ports.graph_execution_repository import GraphExecutionRepository
from ...domain.value_objects import ExecutionId, GraphStatus

logger = logging.getLogger(__name__)

#: `execution_12.events.jsonl`, `execution_12.graph.json`, `execution_12.log`.
#: Anchored and non-greedy on the digits so `execution_1.log` cannot match
#: `execution_12` and take the wrong run's file.
_FILE = re.compile(r"^execution_(\d+)\.(?:events\.jsonl|graph\.json|log)$")


@dataclass(frozen=True, slots=True)
class SweepExecutionScratchResult:
    """What was removed, so a caller can say so out loud.

    `bytes` is here because "cleaned up some scratch" is not an
    operational fact; "freed 840 MB" is.
    """

    runs: int = 0
    files: int = 0
    bytes: int = 0
    kept: int = 0


class SweepExecutionScratch:
    """Delete the scratch of runs that are over, and report what that freed."""

    def __init__(
        self,
        executions: GraphExecutionRepository,
        scratch_dir: Path,
        failed_log_keep: int = GRAPH_FAILED_LOG_KEEP,
    ) -> None:
        self._executions = executions
        self._scratch_dir = scratch_dir
        self._failed_log_keep = max(0, failed_log_keep)

    @property
    def scratch_dir(self) -> Path:
        """Where this sweeps. Public because a caller may reasonably want
        to know -- the composition root logs against it, and a test that
        wants to assert the directory is empty should not have to reach
        into a private attribute to find out where it is."""
        return self._scratch_dir

    def execute(self) -> SweepExecutionScratchResult:
        if not self._scratch_dir.is_dir():
            # Nothing has ever run here. Not an error, and not worth a
            # directory: creating it would make every fresh checkout look
            # like it had done work.
            return SweepExecutionScratchResult()

        removed_files = removed_bytes = removed_runs = kept = 0
        # Which runs failed, and therefore which logs are worth keeping.
        # Collected first because the retention cap is "the newest N", and
        # deciding that while deleting in id order would keep whichever N
        # happened to sort first.
        failed = sorted(
            run_id for run_id in self._by_run()
            if self._ends_badly(run_id)
        )
        # `failed[-0:]` is `failed[0:]` -- the whole list. A retention of 0 would
        # therefore keep everything, which is the opposite of what 0 means.
        # Written as a positive slice for that reason, and tested: the check
        # "retention of 0 keeps nothing" is what found it.
        keep_logs = (
            set(failed[-self._failed_log_keep:])
            if failed and self._failed_log_keep > 0
            else set()
        )
        dropped_logs = 0

        for run_id, paths in sorted(self._by_run().items()):
            if self._is_live(run_id):
                kept += 1
                continue
            if run_id in keep_logs:
                # Keep the log, take the rest. The event file is the 62% and
                # the graph is the 38% of what a run leaves behind; both are
                # read by nobody once the row is terminal.
                log = self._paths_for(run_id).get("log")
                files, freed = self._remove(
                    [p for p in paths if p != log]
                )
                if files:
                    removed_runs += 1
                    removed_files += files
                    removed_bytes += freed
                continue
            if run_id in failed:
                # Failed, but outside the newest N. Counted so the number is
                # visible rather than silently differing between runs.
                dropped_logs += 1
            files, freed = self._remove(paths)
            if files:
                removed_runs += 1
                removed_files += files
                removed_bytes += freed

        if removed_files:
            logger.info(
                "removed scratch for %d finished graph execution(s): "
                "%d file(s), %.1f MB", removed_runs, removed_files,
                removed_bytes / 1_000_000,
            )
        if dropped_logs:
            logger.info(
                "removed %d old failed-run log(s) beyond the newest %d",
                dropped_logs, self._failed_log_keep,
            )
        return SweepExecutionScratchResult(
            runs=removed_runs, files=removed_files,
            bytes=removed_bytes, kept=kept,
        )

    def _paths_for(self, run_id: int) -> dict[str, Path]:
        base = self._scratch_dir / f"execution_{run_id}"
        return {"graph": base.with_suffix(".graph.json"),
                "events": base.with_suffix(".events.jsonl"),
                "log": base.with_suffix(".log")}

    def _ends_badly(self, run_id: int) -> bool:
        """Whether this run's row says it failed or was stopped.

        Only a row that *says so*. A row that has vanished is not treated as
        a failure, and that is deliberate on both counts: nothing claims it
        failed, and keeping orphan logs would spend the retention slots on
        runs nobody can identify, in preference to named ones that a user is
        actually debugging. An orphan's files are removed as before.

        A row that cannot be *read* is a different question and is treated as
        badly-ended: unreadable is not the same as absent, and deleting the
        only copy of a log because the database hiccuped is worse than
        keeping one extra.
        """
        try:
            record = self._executions.get(ExecutionId(run_id))
        except Exception:  # noqa: BLE001 -- unreadable is not absent
            logger.debug("could not read execution %s for the sweep", run_id,
                         exc_info=True)
            return True
        if record is None:
            return False
        # STOPPED, not CANCELLED: GraphStatus has no CANCELLED -- a
        # stop-requested run ends STOPPED, and it is worth keeping the log
        # of for the same reason a failure's is.
        return record.status in (GraphStatus.ERROR, GraphStatus.STOPPED)

    def _by_run(self) -> dict[int, list[Path]]:
        """The scratch files we wrote, grouped by the run they belong to.

        Grouped so one directory listing settles each run once, and so a
        run's files go together rather than leaving a graph.json whose
        events are gone.
        """
        grouped: dict[int, list[Path]] = {}
        for path in sorted(self._scratch_dir.iterdir()):
            if not path.is_file():
                continue
            match = _FILE.match(path.name)
            if match is None:
                # Not one of ours. The scratch directory is not private to
                # this class, and deleting a file because its name looked
                # nearly right is not a thing to do.
                continue
            grouped.setdefault(int(match.group(1)), []).append(path)
        return grouped

    def _remove(self, paths: list[Path]) -> tuple[int, int]:
        """Unlink these files. Returns how many went, and their bytes.

        A file that has already gone is not an error -- another sweep, or a
        user, and either way not this sweep's problem to report. A file
        that cannot be removed is worth a warning and worth continuing
        past, because one unwritable file should not strand the rest.
        """
        files = 0
        freed = 0
        for path in paths:
            try:
                freed += path.stat().st_size
            except OSError:
                pass
            try:
                path.unlink()
                files += 1
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning(
                    "could not remove execution scratch %s", path,
                    exc_info=True,
                )
        return files, freed

    def _is_live(self, run_id: int) -> bool:
        """Is this run's scratch still somebody's to read?

        True only while the row exists and is not terminal.

        An absent row is *not* live: nothing can adopt it, no watcher holds
        it, and keeping its file would keep it forever. That includes an
        id no SQLite INTEGER could hold, which the repository already
        answers as "not found" -- and correctly so, since no row can carry
        one. A file named exactly like ours in a directory we created is
        ours to judge; nothing here needs to guess at ownership beyond the
        three names the supervisor itself writes.
        """
        execution = self._executions.get(ExecutionId(run_id))
        return execution is not None and not execution.status.is_terminal