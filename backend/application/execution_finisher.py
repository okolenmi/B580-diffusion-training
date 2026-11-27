"""The terminal write for a watched run, and the evidence that goes into a failed one."""

from __future__ import annotations

import logging
from pathlib import Path

from ..domain.value_objects import ExecutionId, GraphStatus
from .memory_admission import LedgerSource, graph_owner, release
from .lifecycle_writer import ExecutionLifecycleWriter
from .ports.clock import Clock
from .ports.graph_execution_repository import GraphExecutionRepository

from collections.abc import Callable

logger = logging.getLogger(__name__)


class ExecutionFinisher:
    """Decides how a run ended and writes it, once, through the repository CAS."""

    def __init__(self, *, executions: GraphExecutionRepository, writer: ExecutionLifecycleWriter, clock: Clock,
                 memory_ledger: LedgerSource | None, log_path_for: Callable[[ExecutionId], Path]) -> None:
        self._executions = executions
        self._writer = writer
        self._clock = clock
        # None only where the container has no ledger at all, and then no claim can exist either.
        self._memory_ledger = memory_ledger
        self._log_path_for = log_path_for

    def release_claim(self, execution_id: ExecutionId) -> None:
        """Hand the run's admission claim back; idempotent."""
        release(self._memory_ledger, graph_owner(execution_id))

    #: How much of a crashed child's log is copied into its row. Four
    #: kilobytes is a traceback and a bit; a full training log is megabytes
    #: of progress lines, and a row is not a log file.
    LOG_TAIL_BYTES = 4096

    def finish(
        self,
        execution_id: ExecutionId,
        saw_outcome: bool,
        outcome_error: str | None,
        *,
        abandoned: bool = False,
        stopped: bool = False,
        exit_note: str | None = None,
    ) -> None:
        """Terminal write for one watched run. ``stopped``: a stop was requested for it; ``exit_note``: how the child's process ended, when known.

        Three sources of truth about how it ended, in priority order, and
        the order is the whole point:

        * a row already terminal was written by a stop or a reconcile --
          do nothing, its writer's result stands;
        * the child's own ``outcome`` record says whether the graph
          failed, and with what;
        * no record at all means the process died, which is a different
          answer from "the graph failed" and must not be reported as a
          success just because nobody wrote down a problem.

        Note that the record's *existence* is not the same as its
        contents: a run that finished and a run whose graph failed both
        wrote one, and only the error inside it separates them.
        """
        try:
            execution = self._executions.get(execution_id)
            if execution is None or execution.status is not GraphStatus.RUNNING:
                return  # stop request or reconcile beat us to the row
            at = self._clock.now()
            # Hand the claim back *before* the row can read terminal, so
            # the invariant an observer sees is strict: this run's row
            # only reaches a terminal status once the capacity is free.
            # The supervisor releases again afterwards (idempotent), which covers the early-return path where a
            # stop or reconcile already wrote the row.
            release(self._memory_ledger, graph_owner(execution_id))
            if not saw_outcome:
                # Which of the two "no outcome" stories is true depends on
                # whether the child was still there. Saying "crashed, or a
                # device fault killed it" about a process we ourselves
                # stopped -- and that was running perfectly well -- is the
                # wrong answer twice over: it sends the reader to the
                # execution log for a crash that never happened.
                #
                # The log's tail goes INTO the row, not behind a "see the
                # execution log" pointer: the startup sweep deletes that log, so
                # the traceback of a crashed run would not survive the next
                # server start. The row is what survives; the evidence belongs
                # in it.
                execution.mark_failed(
                    at=at,
                    error=self.with_log_tail(execution_id, (
                        "supervision of this run failed, so it was stopped; "
                        "it reported no outcome -- see the server log and "
                        "the execution log"
                        if abandoned else
                        "execution process exited without reporting an "
                        "outcome (crashed, or a device fault killed it)"
                        + (f": {exit_note}" if exit_note else "") + " -- see the execution log"
                    )),
                )
                self._writer.commit(execution, expected=GraphStatus.RUNNING)
            elif outcome_error is not None:
                execution.mark_failed(at=at, error=outcome_error)
                self._writer.commit(execution, expected=GraphStatus.RUNNING)
            elif stopped:
                # The child returned normally because it was asked to
                # stop, and that is not a completed graph.
                execution.stop(at=at, reason="stop requested")
                self._writer.commit(execution, expected=GraphStatus.RUNNING)
            else:
                execution.mark_finished(at=at)
                self._writer.commit(execution, expected=GraphStatus.RUNNING)
        except Exception:  # noqa: BLE001 -- already at the end of the run
            logger.exception("finalising graph execution %s failed", execution_id)

    def with_log_tail(self, execution_id: ExecutionId, error: str) -> str:
        """`error` plus the end of the child's log, or just `error`.

        Bounded, because a row is not a log file and the sweep's retention
        counts rows. Read from the end: the tail is where a traceback is,
        and a crashed process can have written a lot before dying.

        Every failure here degrades to "no tail" rather than raising. This
        runs inside the `finally` of a run that has already failed, and a
        reader that throws while reporting an unrelated crash would replace
        a real error message with an internal one. Decoded with
        ``errors="replace"`` because a traceback from a dying process can
        contain a partial line, and this must never be the thing that raises.
        """
        try:
            log_path = self._log_path_for(execution_id)
            size = log_path.stat().st_size
            with log_path.open("rb") as handle:
                if size > self.LOG_TAIL_BYTES:
                    handle.seek(size - self.LOG_TAIL_BYTES)
                raw = handle.read()
        except Exception:  # noqa: BLE001 -- evidence is best-effort
            logger.debug("could not read the log tail for execution %s",
                         execution_id, exc_info=True)
            return error

        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return error
        return (
            f"{error}\n\n"
            f"--- last {min(size, self.LOG_TAIL_BYTES)} bytes of the "
            f"execution log ---\n{text}"
        )
