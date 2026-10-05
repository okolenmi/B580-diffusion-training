"""The one channel between the server and a graph-execution child process.

Why a file and not a pipe
--------------------------
A pipe couples the two lifecycles in exactly the way isolation is meant
to break: close the parent and the child gets EPIPE (or SIGPIPE) on its
next write, so a server restart would kill the run it was supposed to
survive. A file has no such coupling -- the child keeps writing whether
or not anyone is reading, which is also what makes adoption possible
(step 4: a restarted server re-reads the file from the start and rebuilds
the history it missed).

What is on it
-------------
Four record kinds, append-only, one JSON object per line:

``node``
    A node finished. Carries the same ``NodeResult`` the in-process path
    used to hand to ``on_node_done``, so the supervisor's persist-and-
    publish step is unchanged -- only its source moved.

``monitor``
    A live monitor report. This is the part with no in-process
    equivalent: ``SharedMonitorBus`` keeps its history in a per-process
    deque and hands each subscriber an ``asyncio.Queue``, none of which a
    child can reach. Routing reports through the file keeps the server's
    own bus as the single source of what a dashboard sees, including the
    "opened the page mid-run and still saw the history" property, since
    the file *is* the history.

``memory``
    One memory telemetry frame: ``reserved_mb``, ``allocated_mb``,
    ``peak_mb``, ``budget_mb`` (null = no budget stated). The child
    reports numbers; the server's watcher files ``peak_mb`` into the
    peak store under the fingerprint key admission stored on the row
    (MEM-04 #2) -- one configuration-keyed high-water mark per graph,
    written by the one writer. The fixed-interval producer is MEM-05 #4.

``outcome``
    The run finished -- successfully or not. Written by the child, so the
    parent knows a clean exit apart from a crash, which is the difference
    between "failed with this message" and "died".

Torn tails
----------
The parent consumes only whole lines. A child killed mid-write leaves a
partial line at the end, and consuming it would raise on every poll and,
worse, could be read as a real record with missing fields. The deleted
``JsonlProgressSource`` had the same rule for the same reason (docs 07
F-07), and it is repeated here rather than assumed because the failure
mode is silent: a truncated record is not a crash, it is a run that
quietly stops reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..application.ports.graph_task_stream import EventKind, ExecutionEvent
from ..json_safe import sanitize

logger = logging.getLogger(__name__)


class ExecutionEventWriter:
    """Child side: append records as they happen.

    Written per record rather than batched because the whole point is that
    the parent sees progress while the run is still going. ``close`` is
    idempotent and, now that nothing is buffered, does almost nothing --
    kept so callers do not have to know that.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._closed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Opened once to find out now whether records can be written at
            # all, then closed again -- a real open in append mode, not a
            # ``touch``, because "exists" is not "writable" (a directory
            # sitting at this path satisfies touch and rejects every write,
            # which would leave available() lying).
            with open(path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            # A run whose event file cannot be opened is a run the server
            # cannot supervise -- but it can still *run*, and the child's
            # job is to run it. Better to lose progress reporting than
            # the run, so this is a warning and not an exit.
            logger.warning("graph execution event file unavailable at %s: %s", path, exc)
            self._closed = True

    @property
    def available(self) -> bool:
        return not self._closed

    def _write(self, record: dict[str, Any]) -> None:
        """Append one record with a single open/write/close.

        Reopening per record costs one syscall trio per report -- a handful
        per run for node results, about one a second for monitor frames --
        and buys two things a held-open handle does not:

        * there is no user-space buffer, so a SIGKILL cannot lose a record
          that was written but not yet flushed;
        * each append is one ``write(2)`` under ``O_APPEND``, which the
          kernel does as a single step, so a record cannot be interleaved
          with another writer's and the reader sees whole lines far more
          often than not. The reader still tolerates a torn tail because
          "far more often" is not "always".
        """
        if not self.available:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            # ValueError is what allow_nan=False raises; sanitize() below
            # is meant to prevent it, and if it does not, losing this one
            # record is still better than losing the run.
            logger.warning("could not write graph execution record: %s", exc)

    def node(self, result: dict[str, Any]) -> None:
        self._write({"kind": EventKind.NODE.value, **result})

    def monitor(self, monitor_id: str, data: dict[str, Any]) -> None:
        # Sanitized here rather than at the far end: the in-process path
        # sanitizes in SharedMonitorBus.report, which this child has no
        # access to, and a bare NaN would make the frame unparseable in
        # every browser (docs 07 F-03).
        self._write(
            {
                "kind": EventKind.MONITOR.value,
                "monitor_id": monitor_id,
                "data": sanitize(data),
            }
        )

    def memory(
        self,
        *,
        reserved_mb: float,
        allocated_mb: float,
        peak_mb: float,
        budget_mb: float | None,
    ) -> None:
        """One memory telemetry frame (MEM-04 #2).

        All four are allocator MB. ``budget_mb`` None means no budget was
        stated for this run -- never 0.0, which would claim the run was
        given nothing on purpose (task rule 2: an absent value is an
        explicit unknown, not a zero).
        """
        self._write(
            {
                "kind": EventKind.MEMORY.value,
                "reserved_mb": reserved_mb,
                "allocated_mb": allocated_mb,
                "peak_mb": peak_mb,
                "budget_mb": budget_mb,
            }
        )

    def outcome(self, *, error: str | None, results_count: int) -> None:
        self._write(
            {
                "kind": EventKind.OUTCOME.value,
                "error": error,
                "results_count": results_count,
            }
        )

    def close(self) -> None:
        self._closed = True


class ExecutionEventTail:
    """Parent side: hand back records written since the last poll.

    Offset-based rather than read-whole-file, because a long run's file
    is the run's history and re-reading it on every poll would make
    watching an N-node run cost O(N^2).

    ``reset`` exists for adoption: a restarted server starts from zero
    on purpose, to rebuild the history a dashboard would otherwise have
    missed.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def caught_up(self) -> bool:
        """Has everything written so far been consumed as whole records?

        Not the same as "the last poll returned nothing". A poll returns
        nothing both when the writer is idle and when it is mid-line, and
        the difference matters to adoption: the watcher uses this to close
        a replay window, and closing it early would replay node results
        the row already holds.
        """
        try:
            return self._path.stat().st_size == self._offset
        except OSError:
            return True  # gone: there is nothing left to catch up to

    def reset(self) -> None:
        self._offset = 0

    def poll(self) -> list[ExecutionEvent]:
        """Complete records written since the last call."""
        try:
            size = self._path.stat().st_size
        except OSError:
            return []  # not created yet, or gone
        if size < self._offset:
            # Truncated or replaced -- re-read from the top rather than
            # silently skipping the gap.
            logger.warning(
                "graph execution event file shrank (%d < %d); re-reading",
                size, self._offset,
            )
            self._offset = 0
        if size == self._offset:
            return []

        with open(self._path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            return []  # torn tail: wait for the rest of the line
        complete = chunk[: last_newline + 1]
        self._offset += len(complete)

        events: list[ExecutionEvent] = []
        for raw in complete.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                # One bad line must not end the stream: the rest of the
                # run's records are still good.
                logger.warning("skipping non-JSON graph execution record: %r", raw[:160])
                continue
            if not isinstance(data, dict):
                logger.warning("skipping non-object graph execution record: %r", raw[:160])
                continue
            kind = data.get("kind")
            try:
                kind = EventKind(kind)
            except ValueError:
                logger.warning("skipping record of unknown kind %r", kind)
                continue
            events.append(ExecutionEvent(kind=kind, payload=data))
        return events