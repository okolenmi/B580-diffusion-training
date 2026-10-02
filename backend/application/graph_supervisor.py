"""GraphExecutionSupervisor -- background thread for one graph run.

The application-side twin of ``RunSupervisor`` with a simpler job: the
graph runs *in-process* (nodes build real objects in this interpreter),
so watching means owning the thread and the cancellation event, not
tail-ing a child.

Concurrency posture (every write goes through the repository CAS, so
races resolve deterministically):

1. ``launch`` registers the execution's ``threading.Event`` *before*
   starting the thread, so a stop arriving while the row is still
   ``queued`` both sets the event and wins the CAS -- the thread then
   fails its ``queued -> running`` claim and exits having run nothing.
2. The thread claims the row (CAS), publishes ``Started``, and runs
   ``GraphRuntime.execute`` with a per-node callback that CAS-persists
   partial results and publishes ``Progressed``. Callback exceptions
   are contained: losing progress reporting must never abort the run
   (the final write carries the authoritative results anyway).
3. Finalisation picks the terminal status -- ``error`` if the outcome
   carries one, else ``stopped`` if the cancel event is set, else
   ``finished`` -- and CASes ``running -> final``. A lost CAS means the
   stop request already wrote ``stopped``; this thread publishes
   nothing.
4. The cancel event is always unregistered. Device memory is *not*
   this class's business: ``GraphRuntime.execute`` releases it in its
   own ``finally``, so the supervisor cannot forget it (docs 08 S-05).

A crashed supervisor thread best-effort fails its row instead of
leaving ``running`` stuck (which would block the single-active check
until the next restart); startup reconciliation covers whatever even
that misses.
"""

from __future__ import annotations

import logging
import threading

from ..domain.events import GraphExecutionProgressed
from ..domain.graph import GraphDefinition, NodeResult
from ..domain.value_objects import ExecutionId, GraphStatus
from .event_publisher import EventPublisher
from .lifecycle_writer import ExecutionLifecycleWriter
from .ports.clock import Clock
from .ports.execution_launcher import ExecutionLauncher
from .ports.graph_execution_repository import GraphExecutionRepository
from .ports.graph_runtime import GraphOutcome, GraphRuntime

logger = logging.getLogger(__name__)


class GraphExecutionSupervisor(ExecutionLauncher):
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        writer: ExecutionLifecycleWriter,
        runtime: GraphRuntime,
        events: EventPublisher,
        clock: Clock,
    ) -> None:
        self._executions = executions
        # CAS-then-announce lives in the writer; ``events`` is for
        # telemetry only.
        self._writer = writer
        self._runtime = runtime
        self._events = events
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel_events: dict[ExecutionId, threading.Event] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(
        self, execution_id: ExecutionId, graph: GraphDefinition
    ) -> None:
        """Register cancellation, then start the worker daemon thread."""
        event = threading.Event()
        with self._lock:
            self._cancel_events[execution_id] = event
        thread = threading.Thread(
            target=self._guard,
            args=(execution_id, graph, event),
            name=f"backend-graph-{execution_id}",
            daemon=True,
        )
        thread.start()

    def cancel(self, execution_id: ExecutionId) -> None:
        """Set the execution's cancel event; no-op if not running here
        (a queued row never launched, or the thread already finished) --
        the stop use case's row CAS decides the outcome either way."""
        with self._lock:
            event = self._cancel_events.get(execution_id)
        if event is not None:
            event.set()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _guard(
        self,
        execution_id: ExecutionId,
        graph: GraphDefinition,
        event: threading.Event,
    ) -> None:
        try:
            self._supervise(execution_id, graph, event)
        except Exception:  # noqa: BLE001 -- thread must not die silently
            logger.exception("supervisor for graph execution %s crashed", execution_id)
            self._fail_leftover(execution_id)
        finally:
            with self._lock:
                self._cancel_events.pop(execution_id, None)

    def _supervise(
        self,
        execution_id: ExecutionId,
        graph: GraphDefinition,
        event: threading.Event,
    ) -> None:
        execution = self._executions.get(execution_id)
        if execution is None or execution.status is not GraphStatus.QUEUED:
            return  # stopped/reconciled before the claim; that writer won
        execution.mark_running(at=self._clock.now())
        if not self._writer.commit(execution, expected=GraphStatus.QUEUED):
            return  # lost the claim race (stop on a queued row, mostly)

        outcome = self._runtime.execute(
            graph,
            cancel_event=event,
            on_node_done=lambda result: self._record_result(execution_id, result),
        )
        self._finalize(execution_id, event, outcome)

    def _record_result(self, execution_id: ExecutionId, result: NodeResult) -> None:
        """Per-node callback: persist partial results + publish progress.

        Re-fetches the row every time (the authoritative status lives
        there, not in this thread's memory) and never raises into the
        executor.
        """
        try:
            execution = self._executions.get(execution_id)
            if execution is None or execution.status is not GraphStatus.RUNNING:
                return  # stopped/deleted mid-run; drop this sample
            execution.record_result(result, at=self._clock.now())
            if not self._writer.commit(execution, expected=GraphStatus.RUNNING):
                return  # terminal writer won between get and update
            self._events.emit(
                GraphExecutionProgressed(
                    execution_id=execution_id,
                    node_id=result.node_id,
                    ok=result.ok,
                    duration_ms=result.duration_ms,
                    occurred_at=self._clock.now(),
                )
            )
        except Exception:  # noqa: BLE001 -- progress must not abort the run
            logger.exception(
                "recording result for node %r (execution %s) failed",
                result.node_id,
                execution_id,
            )

    def _finalize(
        self, execution_id: ExecutionId, event: threading.Event, outcome: GraphOutcome
    ) -> None:
        execution = self._executions.get(execution_id)
        if execution is None or execution.status is not GraphStatus.RUNNING:
            return  # stop request or reconcile beat us to the row
        at = self._clock.now()
        if outcome.error is not None:
            execution.mark_failed(at=at, error=outcome.error)
        elif event.is_set():
            execution.stop(at=at, reason="stop requested")
        else:
            execution.mark_finished(at=at)
        if not self._writer.commit(execution, expected=GraphStatus.RUNNING):
            return  # lost the terminal CAS; the stop writer's result stands

    def _fail_leftover(self, execution_id: ExecutionId) -> None:
        """Last-ditch row repair after a supervisor crash: never leave
        the row active (it would block the next start until restart)."""
        try:
            execution = self._executions.get(execution_id)
            if execution is None:
                return
            self._writer.fail_if_unfinished(
                execution,
                error="execution supervisor crashed (see server log)",
            )
        except Exception:  # noqa: BLE001 -- already in the crash path
            logger.exception(
                "could not fail leftover execution %s after supervisor crash",
                execution_id,
            )


