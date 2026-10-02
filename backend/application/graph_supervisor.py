"""GraphExecutionSupervisor -- watches one graph run, wherever it runs.

WP-22 moved graph execution out of this process. The run itself happens in
a child (``SubprocessGraphTaskGateway``) or on a thread here
(``InProcessGraphTaskGateway``); either way the supervisor's job is the
same and is the same in one implementation:

1. write the graph somewhere the child can read it, spawn, and record the
   pid **before** starting the watcher -- so a stop arriving while the row
   is still ``queued`` reaches a process that exists;
2. watch the run by *tailing its event file*. Node records are
   CAS-persisted and announced exactly as they were when the callback
   fired in-process; monitor records go onto this server's own bus, which
   is what keeps a dashboard opened mid-run able to see the history;
3. finalise from the child's own ``outcome`` record -- so "the graph
   failed" is distinguishable from "the child died", which without the
   record is the difference between a message and a shrug;
4. escalate a stop that the child ignored to a hard kill after the grace
   period, then best-effort fail the row rather than leaving it
   ``running`` (which blocks the single-active check until the next
   restart). Startup reconciliation covers whatever even that misses.

Everything durable goes through the repository CAS, so races resolve
deterministically: a stop that wins the row first means this watcher
finds the row already terminal and publishes nothing.

Device memory is not this class's business. ``run_execution`` releases it
in its own ``finally``, in whichever process ran the graph, so neither the
supervisor nor the child can forget it (docs 08 S-05).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..domain.events import GraphExecutionProgressed
from ..domain.graph import GraphDefinition, NodeResult
from ..domain.value_objects import ExecutionId, GraphStatus
from .ports.graph_task_stream import EventKind
from .event_publisher import EventPublisher
from .lifecycle_writer import ExecutionLifecycleWriter
from .ports.clock import Clock
from .ports.execution_launcher import ExecutionLauncher
from .ports.graph_execution_repository import GraphExecutionRepository
from .ports.graph_task_gateway import GraphTaskGateway, GraphTaskLaunch

logger = logging.getLogger(__name__)


class GraphExecutionSupervisor(ExecutionLauncher):
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        writer: ExecutionLifecycleWriter,
        gateway: GraphTaskGateway,
        events: EventPublisher,
        clock: Clock,
        monitor_bus=None,
        scratch_dir: Path,
        poll_interval: float = 0.1,
        stop_grace: float = 15.0,
        make_tail=None,
    ) -> None:
        self._executions = executions
        # CAS-then-announce lives in the writer; ``events`` is for
        # telemetry only.
        self._writer = writer
        self._gateway = gateway
        self._events = events
        self._clock = clock
        # Where the per-execution graph.json and events.jsonl live. Not
        # the runs dir: these are supervision scratch, and a run row's
        # artifacts are a different thing with a different lifetime.
        self._scratch_dir = scratch_dir
        self._poll_interval = poll_interval
        # Long enough for a stopping run to flush its event file and let
        # the runtime release device memory, short enough that a wedged
        # run does not hold a card forever. The retired training route
        # used the same figure for the same reason (docs 07 F-12).
        self._stop_grace = stop_grace
        self._monitor_bus = monitor_bus
        # How to open a reader over a run's event file. Injected because
        # "a file" is infrastructure and this is application: the class that
        # reads JSONL off disk does not belong in a file that must not
        # import infrastructure. The composition root passes the real one.
        self._make_tail = make_tail
        self._lock = threading.Lock()
        self._pids: dict[ExecutionId, int] = {}
        self._stop_requested: set[ExecutionId] = set()
        self._replaying: set[ExecutionId] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(
        self, execution_id: ExecutionId, graph: GraphDefinition
    ) -> None:
        """Spawn the run, record its pid, then start watching it.

        The pid is registered *before* the watcher starts, and the watcher
        before anything is claimed: a stop arriving while the row is still
        ``queued`` has to find something to signal, and has to find it
        before the child's first CAS -- otherwise the stop is delivered to
        a process that does not exist yet and the run starts anyway.
        """
        paths = self._paths_for(execution_id)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["graph"].write_text(json.dumps(graph.as_dict()), encoding="utf-8")
        # The event file starts empty, every time, before the child is
        # spawned. Records are appended (one write(2) per record, so a
        # SIGKILL cannot tear the last one), which means a *new* run at a
        # path that still holds an old run's records would inherit them --
        # and that is not the unreachable case it looks like: delete the
        # database and row ids start again at 1, so the next execution 1
        # lands on the previous execution 1's file. The watcher then reads
        # 6000 node records for a 3000-node graph, the row grows past the
        # domain's "no more results than nodes" rule, and every subsequent
        # read *and* write of it raises, leaving the row stuck running
        # forever. Observed live.
        #
        # Adoption is the case this must not touch, and does not: `adopt`
        # never comes through here, and it is exactly the one path that
        # needs the old records kept.
        paths["events"].write_text("", encoding="utf-8")

        launch = GraphTaskLaunch(
            execution_id=execution_id,
            graph_path=paths["graph"],
            event_path=paths["events"],
            log_path=paths["log"],
        )
        try:
            pid = self._gateway.spawn(launch)
        except Exception:
            # The row was already inserted by the use case, and a row left
            # queued blocks the single-active check until a restart. This
            # is the one failure no watcher can see, because there is no
            # watcher: repair it here, then let the caller see the error.
            self._fail_leftover(execution_id)
            raise
        with self._lock:
            self._pids[execution_id] = pid
        logger.info(
            "graph execution %s running as pid %s (%s)", execution_id, pid, paths["events"],
        )
        threading.Thread(
            target=self._watch,
            args=(execution_id, pid, self._make_tail(paths["events"])),
            name=f"backend-graph-watch-{execution_id}",
            daemon=True,
        ).start()

    def _paths_for(self, execution_id: ExecutionId) -> dict[str, Path]:
        base = self._scratch_dir / f"execution_{execution_id}"
        return {
            "graph": base.with_suffix(".graph.json"),
            "events": base.with_suffix(".events.jsonl"),
            "log": base.with_suffix(".log"),
        }

    def cancel(self, execution_id: ExecutionId) -> None:
        """Ask the run to stop; no-op if it is not running here.

        Whether the row actually stops is decided by the stop use case's
        status CAS, not by this call.
        """
        with self._lock:
            pid = self._pids.get(execution_id)
            if pid is not None:
                # Remembered, because a cooperative stop makes the child
                # return *normally*: by its exit status alone a stopped run
                # and a finished one are the same event, and the row must
                # not be labelled "finished" because the child was polite.
                self._stop_requested.add(execution_id)
        if pid is None:
            return
        self._gateway.request_stop(pid)
        # The child's runtime notices at its next step boundary. If that
        # never comes -- a node wedged in an uninterruptible kernel -- the
        # grace period ends and the watcher escalates itself, so there is
        # exactly one place that decides a run is not stopping.
        threading.Thread(
            target=self._escalate,
            args=(execution_id, pid),
            name=f"backend-graph-stop-{execution_id}",
            daemon=True,
        ).start()

    def adopt(self, execution_id: ExecutionId) -> int | None:
        """Re-attach to a run that outlived the server that started it.

        A child lives in its own session, so it survives its parent's
        death; a restart is therefore not by itself a reason to throw a
        run away. Asked once per unfinished row at startup, before the
        reconciler gives up on it.

        Replays the event file from the start rather than resuming at the
        end, and the point of that replay is the *monitor* history: it is
        the only trace of the run that lives outside the row, and it is
        what makes a dashboard opened after the restart show the run's
        curve instead of an empty chart until the next report.

        Node records are skipped during the replay for the opposite reason.
        The row already holds their results -- they were CAS-persisted
        before the restart -- so replaying them would append each one
        twice, and the domain refuses to load a row with more results than
        nodes. It would also re-announce progress events the event store
        is about to replay to reconnecting clients on its own (docs 07
        F-08).

        Returns the pid, or ``None`` when there is nothing to adopt -- no
        live child, no event file, or a run that had not yet started
        writing. All three mean the same thing to the caller.
        """
        pid = self._gateway.find_running(execution_id)
        if pid is None:
            return None
        paths = self._paths_for(execution_id)
        if not paths["events"].exists():
            # A live child whose event file is missing means the scratch
            # was cleaned out from under it. Its output would go
            # nowhere, so the run is not observable and not adoptable.
            logger.warning(
                "graph execution %s has a live child (pid %s) but no event "
                "file at %s; not adopting", execution_id, pid, paths["events"],
            )
            return None
        with self._lock:
            self._pids[execution_id] = pid
            self._replaying.add(execution_id)
        tail = self._make_tail(paths["events"])
        tail.reset()  # from the top on purpose -- see the docstring
        logger.info(
            "adopted graph execution %s, still running as pid %s",
            execution_id, pid,
        )
        threading.Thread(
            target=self._watch,
            args=(execution_id, pid, tail, True),
            name=f"backend-graph-adopt-{execution_id}",
            daemon=True,
        ).start()
        return pid

    def _escalate(self, execution_id: ExecutionId, pid: int) -> None:
        """Hard-kill a run that ignored a cooperative stop."""
        threading.Event().wait(self._stop_grace)
        if not self._gateway.is_alive(pid):
            return
        logger.warning(
            "graph execution %s (pid %s) ignored the stop for %.0fs; killing",
            execution_id, pid, self._stop_grace,
        )
        self._gateway.kill(pid)

    # ------------------------------------------------------------------
    # The watcher
    # ------------------------------------------------------------------

    def _claim(self, execution_id: ExecutionId) -> bool:
        """``queued -> running``, or lose the race and do nothing.

        Same CAS the old in-process thread performed, at the same point in
        the sequence, so the ordering the stop use case documents still
        holds: it signals before it CASes, and if it wins the row from
        ``queued`` the child is killed here rather than left building a
        graph nobody is waiting for.
        """
        execution = self._executions.get(execution_id)
        if execution is None or execution.status is not GraphStatus.QUEUED:
            return False  # stopped/reconciled before the claim; that writer won
        execution.mark_running(at=self._clock.now())
        return self._writer.commit(execution, expected=GraphStatus.QUEUED)

    def _watch(
        self,
        execution_id: ExecutionId,
        pid: int,
        tail,
        catch_up: bool = False,
    ) -> None:
        """Tail one run's event file until it ends, then finalise.

        The mirror image of the in-process callback loop this replaced,
        with the same two properties: re-reads the row rather than
        trusting its own memory (the authoritative status lives there),
        and contains its own exceptions (losing progress reporting must
        never lose the run -- the final write carries the results anyway).

        ``catch_up`` is set only by ``adopt``: the file is being read from
        the top over a span that already happened, and node records in
        that span must not be recorded again.
        """
        outcomes: list[dict] = []
        try:
            if not catch_up and not self._claim(execution_id):
                # A stop won the row while the child was still starting.
                # It never wrote anything, and nothing is going to read
                # its event file, so it goes now.
                self._gateway.kill(pid)
                return
            while True:
                outcomes += self._apply(execution_id, tail.poll(), catch_up)
                if catch_up and tail.caught_up:
                    # Exactly caught up, not "a poll came back empty": a
                    # poll is also empty while the child is mid-line, and
                    # treating that as caught-up would start recording node
                    # results the row already has.
                    catch_up = False
                    with self._lock:
                        self._replaying.discard(execution_id)
                if not self._gateway.is_alive(pid):
                    # A final drain, and this is a correctness fix rather
                    # than tidiness. The child writes its outcome record
                    # and *then* exits, so everything it ever said is on
                    # disk by the time liveness says no. A watcher that
                    # was mid-batch when that happened -- persisting one
                    # node result per CAS is slow enough on a long graph
                    # that a fast child finishes underneath it -- would
                    # otherwise break without ever reading the outcome,
                    # and report a successful run as
                    # "exited without reporting an outcome". Observed live
                    # on a 3000-node graph: the event file ended with a
                    # clean outcome, and the row said the process crashed.
                    outcomes += self._apply(execution_id, tail.poll(), catch_up)
                    break
                execution = self._executions.get(execution_id)
                if execution is None or execution.status.is_terminal:
                    # The stop path owns the outcome now; its writer said
                    # how this run ended, and re-deciding would overwrite
                    # a terminal row with a second, different ending.
                    break
                threading.Event().wait(self._poll_interval)
        except Exception:  # noqa: BLE001 -- the thread must not die silently
            logger.exception("watcher for graph execution %s crashed", execution_id)
        finally:
            self._finish(
                execution_id,
                pid,
                bool(outcomes),
                outcomes[-1].get("error") if outcomes else None,
            )

    def _apply(
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
                    self._record_node(execution_id, event.payload)
            elif kind is EventKind.MONITOR:
                self._republish_monitor(event.payload)
            else:  # OUTCOME
                outcomes.append(event.payload)
        return outcomes

    def _record_node(self, execution_id: ExecutionId, payload: dict) -> None:
        """Persist a finished node's result and announce its progress.

        Re-fetches the row every time (the authoritative status lives
        there, not in this thread's memory) and never raises into the
        watcher.
        """
        node_id = str(payload.get("node_id", ""))
        duration_ms = float(payload.get("duration_ms") or 0.0)
        try:
            execution = self._executions.get(execution_id)
            if execution is None or execution.status is not GraphStatus.RUNNING:
                return  # stopped/deleted mid-run; drop this sample
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
            self._events.emit(
                GraphExecutionProgressed(
                    execution_id=execution_id,
                    node_id=node_id,
                    ok=bool(payload.get("ok")),
                    duration_ms=duration_ms,
                    occurred_at=self._clock.now(),
                )
            )
        except Exception:  # noqa: BLE001 -- progress must not abort the run
            logger.exception(
                "recording result for node %r (execution %s) failed",
                node_id, execution_id,
            )

    def _republish_monitor(self, payload: dict) -> None:
        """Put a child's monitor report onto this server's own bus.

        The child's bus is unreachable -- a per-process deque of asyncio
        queues -- so this is the bridge that makes a dashboard opened
        mid-run work, including its history, because the file the watcher
        replays *is* the history.
        """
        if self._monitor_bus is None:
            return
        try:
            self._monitor_bus.report(
                str(payload.get("monitor_id", "")), dict(payload.get("data") or {})
            )
        except Exception:  # noqa: BLE001 -- telemetry must not abort the run
            logger.exception("republishing a monitor report failed")

    def _finish(
        self,
        execution_id: ExecutionId,
        pid: int,
        saw_outcome: bool,
        outcome_error: str | None,
    ) -> None:
        """Terminal write for one watched run.

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
            with self._lock:
                stopped = execution_id in self._stop_requested
            if not saw_outcome:
                execution.mark_failed(
                    at=at,
                    error="execution process exited without reporting an "
                          "outcome (crashed, or a device fault killed it) -- "
                          "see the execution log",
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
        finally:
            self._release(execution_id, pid)

    def _release(self, execution_id: ExecutionId, pid: int) -> None:
        """Forget a finished run: its pid and its stop marker.

        The in-process gateway keeps a thread and a cancel event per
        execution, so it needs telling; the subprocess one keeps a Popen
        so a later ``is_alive`` reaps rather than falling through to
        /proc. The scratch files are left on disk -- they are the run's
        log and event history, which is what makes a failed run
        diagnosable after the fact.
        """
        with self._lock:
            self._pids.pop(execution_id, None)
            self._stop_requested.discard(execution_id)
            self._replaying.discard(execution_id)
        reap = getattr(self._gateway, "reap", None)
        if reap is not None:
            reap(pid)

    def _replayed(self, execution_id: ExecutionId) -> bool:
        """Has this adopted run finished catching up on its history?

        Exists because "the replay window closed" is the one fact the
        adoption tests need to observe and there is no way to observe it
        from outside -- the watcher catches up inside its own loop, and
        without a signal the only honest thing a test can do is sleep and
        hope, which asserts the race rather than the behaviour.
        """
        with self._lock:
            return execution_id not in self._replaying

    def _fail_leftover(self, execution_id: ExecutionId) -> None:
        """Last-ditch row repair: never leave the row active.

        A row stuck in ``queued`` blocks the single-active check, so every
        next start fails until a restart reconciles it.
        """
        try:
            execution = self._executions.get(execution_id)
            if execution is None:
                return
            self._writer.fail_if_unfinished(
                execution,
                error="execution could not be started (see the server log)",
            )
        except Exception:  # noqa: BLE001 -- already in the failure path
            logger.exception(
                "could not fail leftover execution %s after a launch failure",
                execution_id,
            )
