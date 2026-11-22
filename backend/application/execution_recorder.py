"""What a run's event records turn into: stored node results, progress announcements, republished telemetry, filed peaks."""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from ..domain.events import GraphExecutionProgressed
from ..domain.graph import GraphDefinition, NodeResult
from ..domain.modules import GRAPH_STREAM, node_stream_id, run_stream_id
from ..domain.value_objects import ExecutionId, GraphStatus
from .ports.graph_task_stream import EventKind
from .ports.peak_store import PeakStore
from .event_publisher import EventPublisher
from .lifecycle_writer import ExecutionLifecycleWriter
from .ports.clock import Clock
from .ports.graph_execution_repository import GraphExecutionRepository


logger = logging.getLogger(__name__)


class ExecutionRecorder:
    """Applies the records a watcher reads from a run's event file. Every method contains its own failures: a record that cannot be applied costs that record, never the run's supervision."""

    def __init__(self, *, executions: GraphExecutionRepository, writer: ExecutionLifecycleWriter, events: EventPublisher, clock: Clock,
                 monitor_bus=None, peak_store: PeakStore | None = None, max_record_attempts: int = 3) -> None:
        self._executions = executions
        self._writer = writer
        self._events = events
        self._clock = clock
        self._monitor_bus = monitor_bus
        # None only in tests that do not exercise peaks: a run whose peaks are not recorded still runs.
        self._peak_store = peak_store
        # Separate from the poll bound: a failed poll can simply be repeated, a failed *record* cannot (see record_node).
        self._max_record_attempts = max_record_attempts
        # The peak this run has already had filed, so frames that repeat it cost no statement (see record_peak).
        self._peak_seen: dict[ExecutionId, float] = {}

    def forget(self, execution_id: ExecutionId) -> None:
        self._peak_seen.pop(execution_id, None)

    def apply(
        self, execution_id: ExecutionId, events, catch_up: bool
    ) -> list[dict]:
        """Apply one batch of records; return the outcome records in it.

        A batch, rather than a record, because that is what a poll
        returns and because the two calls the watcher makes of it -- the
        steady one and the final drain -- must apply records identically.
        """
        outcomes: list[dict] = []
        for event in events:
            kind = event.kind
            if kind is EventKind.NODE:
                if not catch_up:
                    self.record_node(execution_id, event.payload)
            elif kind is EventKind.MONITOR:
                self.republish_monitor(execution_id, event.payload)
            elif kind is EventKind.MEMORY:
                self.record_peak(execution_id, event.payload)
                self.republish_memory(execution_id, event.payload)
            else:  # OUTCOME
                outcomes.append(event.payload)
        return outcomes

    def record_node(self, execution_id: ExecutionId, payload: dict) -> None:
        """Persist a finished node's result and announce its progress.

        Re-fetches the row on every attempt (the authoritative status lives
        there, not in this thread's memory) and never raises into the
        watcher.

        **Why it retries.** A record that fails to persist is gone, not
        deferred. `tail.poll()` advances its offset as it hands over a
        batch, so by the time a node's result is being written the record
        exists only in this call -- the watcher cannot re-read it and the
        retry above it re-polls an empty tail. So a single "database is
        locked" during a write cost the run that step's result, silently:
        the row still finished, the child's own `outcome` still reported
        the full count, and the two simply disagreed.

        Losing one step of a long run to a locked database is not acceptable, so the write is retried.

        The retry is idempotent rather than blind, because "the write
        failed" does not mean "the row is unchanged": `commit` swaps the row
        and *then* publishes, so a failure in the second half leaves the
        first half done. Re-reading the row and skipping when this node's
        result is already on it is what makes repeating safe -- and the
        event is emitted outside the retry for the same reason.
        """
        node_id = str(payload.get("node_id", ""))
        duration_ms = float(payload.get("duration_ms") or 0.0)
        committed = False
        already_stored = False
        failure: Exception | None = None

        for attempt in range(1, self._max_record_attempts + 1):
            try:
                execution = self._executions.get(execution_id)
                if execution is None or execution.status is not GraphStatus.RUNNING:
                    return  # stopped/deleted mid-run; drop this sample
                if any(r.node_id == node_id for r in execution.results):
                    # Already stored: `commit` is a compare-and-swap *and then* a publish, so it can fail after the
                    # row was updated, and the retry would append a second copy. A node runs once per execution, so a
                    # duplicate node_id is never legitimate and skipping on sight is both the fix and the invariant.
                    already_stored = True
                    break
                execution.record_result(
                    NodeResult(
                        node_id=node_id,
                        ok=bool(payload.get("ok")),
                        outputs=dict(payload.get("outputs") or {}),
                        error=payload.get("error"),
                        duration_ms=duration_ms,
                    ),
                    at=self._clock.now(),
                )
                if not self._writer.commit(execution, expected=GraphStatus.RUNNING):
                    return  # terminal writer won between get and update
                committed = True
                break
            except Exception as exc:  # noqa: BLE001 -- one step, not the run
                failure = exc
                logger.warning(
                    "recording result for node %r (execution %s) failed on "
                    "attempt %d/%d: %s: %s",
                    node_id, execution_id, attempt, self._max_record_attempts,
                    type(exc).__name__, exc,
                )
                if attempt < self._max_record_attempts:
                    threading.Event().wait(min(0.05 * attempt, 0.5))

        if not committed and not already_stored:
            # The one honest thing left: say which result is missing, loudly,
            # rather than leave a short row that looks complete.
            logger.error(
                "giving up on node %r of execution %s after %d attempts; "
                "this run's row will be missing that result (%s: %s)",
                node_id, execution_id, self._max_record_attempts,
                type(failure).__name__ if failure else "unknown",
                failure,
            )
            return

        try:
            self._events.emit(
                GraphExecutionProgressed(
                    execution_id=execution_id,
                    node_id=node_id,
                    ok=bool(payload.get("ok")),
                    duration_ms=duration_ms,
                    occurred_at=self._clock.now(),
                )
            )
        except Exception:  # noqa: BLE001 -- a missed progress ping, not a lost result
            logger.exception(
                "announcing progress for node %r (execution %s) failed; the "
                "result itself is stored", node_id, execution_id,
            )

    def republish_monitor(self, execution_id: ExecutionId, payload: dict) -> None:
        """Put a child's monitor report onto this server's own bus.

        The child's bus is unreachable -- a per-process deque of asyncio
        queues -- so this is the bridge that makes a dashboard opened
        mid-run work, including its history, because the file the watcher
        replays *is* the history.
        """
        if self._monitor_bus is None:
            return
        try:
            stream, data = str(payload.get("monitor_id", "")), dict(payload.get("data") or {})
            self._monitor_bus.report(stream, data)
            self._monitor_bus.report(run_stream_id(execution_id, stream), data)
        except Exception:  # noqa: BLE001 -- telemetry must not abort the run
            logger.exception("republishing a monitor report failed")

    #: The memory-frame keys the graph-scope memory modules read.
    _MEMORY_FRAME_KEYS = (
        "t", "reserved_mb", "allocated_mb", "peak_mb", "budget_mb", "event", "mb",
    )

    def republish_memory(self, execution_id: ExecutionId, payload: dict) -> None:
        """Put a child's memory frame onto the graph-scope stream.

        This is the data behind the "Device memory" chart a page can show
        (``domain/modules.GRAPH_DISPLAYS``): the same frames the peak
        store already listens to, now also visible live. Guarded like the
        monitor republish -- telemetry never costs the run's supervision.
        """
        if self._monitor_bus is None:
            return
        try:
            frame = {k: payload[k] for k in self._MEMORY_FRAME_KEYS if k in payload}
            self._monitor_bus.report(GRAPH_STREAM, frame)
            self._monitor_bus.report(run_stream_id(execution_id, GRAPH_STREAM), frame)
        except Exception:  # noqa: BLE001 -- telemetry must not abort the run
            logger.exception("republishing a memory frame failed")

    def clear_streams(self, graph: GraphDefinition) -> None:
        """A new run starts with empty widget streams.

        The child's own ``clear`` is a no-op across the process boundary
        (its bus is not this server's), so the run that is *starting* is
        the one place that can honestly say "drop the last run's history":
        otherwise a page opened on the same node ids would draw the new
        run's line over the old one -- the overlay bug MonitorBus.clear
        exists to prevent, here for widget streams.
        """
        if self._monitor_bus is None:
            return
        for stream in (GRAPH_STREAM, *(node_stream_id(n.uid) for n in graph.nodes)):
            try:
                self._monitor_bus.clear(stream)
            except Exception:  # noqa: BLE001 -- a stale chart is not worth a failed launch
                logger.exception("clearing stream %r failed", stream)

    def record_peak(self, execution_id: ExecutionId, payload: dict) -> None:
        """File one child-reported peak in the peak store (MEM-04 #2).

        The child reports numbers; the row's ``memory_json`` carries the
        fingerprint key admission computed for this run, and the store is
        keyed by it -- so the same key admission *read* the past under is
        the key this run's own high-water mark is written under. One
        statement per rise, and the store's ``MAX`` semantics make a
        duplicate write harmless anyway.

        Guarded like ``_record_node`` and ``_republish_monitor``: a path
        inside ``_apply`` never raises into the watcher, so telemetry
        trouble costs the record, not the run's supervision.

        Two numbers are deliberately *not* filed. A peak of 0 or less
        means the run never allocated anything (it died before touching
        the device), and storing it would have admission read back a
        claim that this configuration needs nothing -- unknown, never
        zero (task rule 2). And a peak at or below what this run has
        already had filed is not a rise: the fixed-interval frames after
        a run stops growing repeat the same number (MEM-05 #4), and they
        cost nothing once the run has plateaued.
        """
        if self._peak_store is None:
            return
        # The frame is a child's JSON: its values are untyped, and
        # float() raising TypeError on None *is* this guard -- the
        # except below is where "no peak was stated" is handled.
        raw: Any = payload.get("peak_mb")
        try:
            peak = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "execution %s: memory record without a usable peak_mb (%r); "
                "not recorded",
                execution_id, payload.get("peak_mb"),
            )
            return
        if not math.isfinite(peak) or peak <= 0.0:
            logger.debug(
                "execution %s: memory record reports no measured peak (%s); "
                "not recorded",
                execution_id, peak,
            )
            return
        try:
            execution = self._executions.get(execution_id)
            key = (
                None
                if execution is None or execution.memory is None
                else execution.memory.fingerprint_key
            )
            if key is None:
                # Exactly what admission saw: an unknown fingerprint has
                # no configuration to file a peak under.
                logger.debug(
                    "execution %s: peak %s MB not recorded, the graph's "
                    "fingerprint is unknown",
                    execution_id, peak,
                )
                return
            risen = self._peak_seen.get(execution_id)
            if risen is not None and peak <= risen:
                return  # this run has already reported this much
            self._peak_store.record(key, peak)
            self._peak_seen[execution_id] = peak
        except Exception:  # noqa: BLE001 -- telemetry must not abort the run
            # The frame itself is gone (the offset already advanced), so
            # the next frame's number -- no lower than this one for a
            # healthy run -- is what will be filed instead.
            logger.exception(
                "recording a peak for execution %s failed", execution_id
            )
