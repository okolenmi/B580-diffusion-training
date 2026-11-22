"""GraphExecutionSupervisor -- watches one graph run, wherever it runs.

The run itself happens in a child (``SubprocessGraphTaskGateway``) or on a thread here (``InProcessGraphTaskGateway``);
the supervisor's job is the same for both:

1. write the graph where the child can read it, spawn, and record the pid **before** starting the watcher, so a stop
   arriving while the row is still ``queued`` reaches a process that exists;
2. watch the run by *tailing its event file*: node records are persisted and announced, monitor records go onto this
   server's own bus (which is what lets a dashboard opened mid-run see the history);
3. finalise from the child's own ``outcome`` record, so "the graph failed" is distinguishable from "the child died";
4. escalate a stop the child ignored to a hard kill after the grace period, then fail the row rather than leaving it
   ``running`` (which blocks the single-active check until the next restart).

The record handling, the process stopping and the terminal write are collaborators (``ExecutionRecorder``,
``ChildStopper``, ``ExecutionFinisher``); this class owns the pid table and the watcher loop. Everything durable goes
through the repository CAS, so races resolve deterministically: a stop that wins the row first means the watcher finds
it terminal and publishes nothing. The admission claim goes back exactly when the child actually stops, because every
watcher path funnels through ``_finish`` and ``_release``.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..domain.graph import GraphDefinition, NodeResult
from ..domain.value_objects import ExecutionId, GraphStatus
from .memory_admission import LedgerSource
from .ports.graph_task_stream import EventKind
from .ports.peak_store import PeakStore
from .child_stopper import ChildStopper
from .event_publisher import EventPublisher
from .execution_finisher import ExecutionFinisher
from .execution_recorder import ExecutionRecorder
from .lifecycle_writer import ExecutionLifecycleWriter
from .ports.clock import Clock
from .ports.execution_launcher import ExecutionLauncher, RecordedOutcome
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
        memory_ledger: LedgerSource | None = None,
        peak_store: PeakStore | None = None,
        monitor_bus=None,
        scratch_dir: Path,
        poll_interval: float = 0.1,
        stop_grace: float = 15.0,
        make_tail=None,
        max_poll_failures: int = 5,
        max_record_attempts: int = 3,
    ) -> None:
        self._executions = executions
        self._writer = writer
        self._gateway = gateway
        self._clock = clock
        self._scratch_dir = scratch_dir
        self._poll_interval = poll_interval
        # How many consecutive poll failures the watcher rides out before giving up on a run it is otherwise watching.
        # One transient error (a locked database) must not cost the run its supervisor; a persistent one must not
        # cost the server a thread that retries forever.
        self._max_poll_failures = max_poll_failures
        # How to open a reader over a run's event file. Injected because "a file" is infrastructure and this is
        # application, which must not import infrastructure.
        self._make_tail = make_tail
        self._recorder = ExecutionRecorder(
            executions=executions, writer=writer, events=events, clock=clock, monitor_bus=monitor_bus,
            peak_store=peak_store, max_record_attempts=max_record_attempts,
        )
        self._stopper = ChildStopper(gateway=gateway, grace=stop_grace)
        self._finisher = ExecutionFinisher(
            executions=executions, writer=writer, clock=clock, memory_ledger=memory_ledger,
            log_path_for=lambda execution_id: self._paths_for(execution_id)["log"],
        )
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
        # The event file starts empty every time, before the spawn. Records are appended, so a new run at a path that
        # still holds an old run's records would inherit them: row ids restart after a database reset, and the watcher
        # would then read more node records than the graph has nodes, which the domain refuses to load. Adoption never
        # comes through here, and is the one path that needs the old records kept.
        paths["events"].write_text("", encoding="utf-8")
        self._recorder.clear_streams(graph)
        launch = self._launch_request(execution_id, paths)
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

    def _launch_request(self, execution_id: ExecutionId, paths: dict[str, Path]) -> GraphTaskLaunch:
        """What the child is told, including its memory picture, read off the row (the only place it exists).

        Unknown is a named state, not a zero: the launch still proceeds, and the child warns instead of checking
        against nothing.
        """
        execution = self._executions.get(execution_id)
        effective = execution.memory if execution is not None else None
        memory_grant_mb = execution.reserved_mb if execution is not None else None
        memory_budget_mb = effective.demand_mb if effective is not None else None
        # "auto" with nothing remembered about the graph is a *ceiling* (all the ledger can give), routinely above what
        # a shared card has free: the child adapts such a run to the real free memory, while a stated or remembered
        # demand is checked exactly.
        memory_adaptive = bool(
            effective is not None and effective.demand_source == "unknown" and effective.vram_max_mb == "auto"
        )
        vram_min_mb = float(effective.vram_min_mb) if effective is not None else 0.0
        ram_ceiling = None if effective is None else effective.ram_max_mb
        is_number = isinstance(ram_ceiling, (int, float)) and not isinstance(ram_ceiling, bool)
        missing = [
            name
            for name, value in (("--memory-grant-mb", memory_grant_mb), ("--memory-budget-mb", memory_budget_mb))
            if value is None
        ]
        if missing:
            logger.warning(
                "execution %s: launching without %s; the child will treat them as unknown rather than check "
                "against nothing", execution_id, " and ".join(missing),
            )
        return GraphTaskLaunch(
            execution_id=execution_id,
            graph_path=paths["graph"],
            event_path=paths["events"],
            log_path=paths["log"],
            command_dir=paths.get("commands"),
            memory_budget_mb=memory_budget_mb,
            memory_grant_mb=memory_grant_mb,
            memory_adaptive=memory_adaptive,
            vram_min_mb=vram_min_mb,
            ram_max_mb=float(ram_ceiling) if is_number else None,
        )

    def _paths_for(self, execution_id: ExecutionId) -> dict[str, Path]:
        base = self._scratch_dir / f"execution_{execution_id}"
        return {
            "graph": base.with_suffix(".graph.json"),
            "events": base.with_suffix(".events.jsonl"),
            "log": base.with_suffix(".log"),
            "commands": base.with_suffix(".commands"),
        }

    def recorded_outcome(self, execution_id: ExecutionId) -> RecordedOutcome | None:
        """Replay this run's event file and report what it said happened.

        Read from the top on purpose. This is not the watcher's
        incremental tail -- nothing has been read from this file yet in
        this server's lifetime -- and the offset starts at zero precisely
        so the history is rebuilt.

        Returns ``None`` unless the file carries an ``outcome`` record,
        which is the run saying "this is how I ended". Everything before it
        is node results, and they are returned *with* the verdict rather
        than discarded: a run that completed while no server was watching
        did real work, and reporting it as crashed throws that away.

        A partially-written trailing line is not an error. The writer
        appends one record per ``write(2)`` so a kill can tear a line but
        not merge two, and a torn tail means the run was killed -- which
        is the ``None`` case, decided by the missing outcome rather than by
        the torn line.
        """
        tail = self._make_tail(self._paths_for(execution_id)["events"])
        results: list[NodeResult] = []
        error: str | None = None
        said_how_it_ended = False
        while True:
            for event in tail.poll():
                if event.kind is EventKind.NODE:
                    payload = event.payload
                    results.append(
                        NodeResult(
                            node_id=str(payload.get("node_id", "")),
                            ok=bool(payload.get("ok")),
                            outputs=dict(payload.get("outputs") or {}),
                            error=payload.get("error"),
                            duration_ms=float(payload.get("duration_ms") or 0.0),
                        )
                    )
                elif event.kind is EventKind.MEMORY:
                    # A run that finished while no server was watching
                    # still reported its peak; the reconcile drain files
                    # it under the row's key, same as the steady watcher.
                    self._recorder.record_peak(execution_id, event.payload)
                elif event.kind is EventKind.OUTCOME:
                    said_how_it_ended = True
                    raw = event.payload.get("error")
                    error = None if raw is None else str(raw)
            if tail.caught_up:
                break
        if not said_how_it_ended:
            return None
        return RecordedOutcome(results=tuple(results), error=error)

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
            target=self._stopper.escalate,
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

    def has_running_child(self, execution_id: ExecutionId) -> bool:
        """Whether anything of this execution is alive but unwatchable.

        Deliberately **not** answered by killing it. The supervisor is not
        the owner of a process it did not start, and a user can still find
        one in ``ps`` -- a position the tests take deliberately
        (``test_a_run_with_no_event_file_is_not_adopted``). What it must
        not do is let "I will not adopt it" be read as "it is gone": a
        caller that concludes the row is debris and fails it releases the
        single-active check while the child still holds the card, which is
        the two-trainers-on-one-card outcome this class spends its
        comments on avoiding.

        So the answer is a fact, and the caller decides. The reconciler
        uses it to leave the row running and say so; the row then settles
        on a later sweep, once the process is actually gone.
        """
        pids = self._gateway.find_running_all(execution_id)
        return bool(pids)

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
        abandoned = False
        consecutive_failures = 0
        try:
            if not catch_up and not self._claim(execution_id):
                # A stop won the row while the child was still starting.
                # It never wrote anything, and nothing is going to read
                # its event file, so it goes now.
                self._gateway.kill(pid)
                return
            while True:
                # The body is guarded per iteration rather than around the
                # whole loop. A single failed poll used to end the
                # supervision of a run that was perfectly healthy, and the
                # `finally` then wrote it off as a dead process while the
                # child carried on: no stop, no kill, a terminal row that
                # released the single-active check, and two trainers on one
                # card. Most poll failures are transient -- a locked
                # database during someone else's transaction -- so they are
                # ridden out; only a run of them means the watcher really
                # cannot continue.
                try:
                    # `tail.poll()` advances its offset as it hands a batch
                    # over, so from that moment the batch lives only in this
                    # frame and a retry cannot re-read it. Nothing here
                    # depends on being able to: every path inside `_apply`
                    # is guarded per record -- `_record_node` retries its own
                    # write, `_republish_monitor` swallows its failure -- so a
                    # cycle that fails loses no records, and the row read
                    # below happens after the batch has been applied rather
                    # than before it, which makes no difference to that.
                    outcomes += self._recorder.apply(execution_id, tail.poll(), catch_up)
                    execution = self._executions.get(execution_id)
                    if catch_up and tail.caught_up:
                        # Exactly caught up, not "a poll came back empty": a
                        # poll is also empty while the child is mid-line, and
                        # treating that as caught-up would start recording
                        # node results the row already has.
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
                        # and report a successful run as "exited without
                        # reporting an outcome". Observed live on a
                        # 3000-node graph: the event file ended with a clean
                        # outcome, and the row said the process crashed.
                        outcomes += self._recorder.apply(execution_id, tail.poll(), catch_up)
                        break
                    if execution is None or execution.status.is_terminal:
                        # The stop path owns the outcome now; its writer said
                        # how this run ended, and re-deciding would overwrite
                        # a terminal row with a second, different ending.
                        break
                    consecutive_failures = 0
                except Exception:  # noqa: BLE001 -- one bad poll, not the run
                    consecutive_failures += 1
                    logger.warning(
                        "polling graph execution %s failed (%d in a row)",
                        execution_id, consecutive_failures, exc_info=True,
                    )
                    if consecutive_failures >= self._max_poll_failures:
                        raise
                threading.Event().wait(self._poll_interval)
        except Exception:  # noqa: BLE001 -- the thread must not die silently
            logger.exception(
                "watcher for graph execution %s can no longer continue",
                execution_id,
            )
            abandoned = self._stopper.abandon(execution_id, pid)
            if abandoned:
                # The child may well have finished as it was being stopped,
                # and its own record is better evidence than our failure to
                # watch it -- the same reasoning as the drain above.
                try:
                    outcomes += self._recorder.apply(execution_id, tail.poll(), catch_up)
                except Exception:  # noqa: BLE001 -- already giving up
                    logger.exception(
                        "final drain after abandoning execution %s failed",
                        execution_id,
                    )
        finally:
            self._finish(
                execution_id,
                pid,
                bool(outcomes),
                outcomes[-1].get("error") if outcomes else None,
                abandoned=abandoned,
            )

    def _finish(
        self,
        execution_id: ExecutionId,
        pid: int,
        saw_outcome: bool,
        outcome_error: str | None,
        abandoned: bool = False,
    ) -> None:
        """Terminal write for one watched run, then forget it."""
        try:
            with self._lock:
                stopped = execution_id in self._stop_requested
            self._finisher.finish(execution_id, saw_outcome, outcome_error, abandoned=abandoned, stopped=stopped)
        finally:
            self._release(execution_id, pid)

    def _release(self, execution_id: ExecutionId, pid: int) -> None:
        """Forget a finished run: its pid, its stop marker, its claim,
        its peak bookkeeping.

        The in-process gateway keeps a thread and a cancel event per
        execution, so it needs telling; the subprocess one keeps a Popen
        so a later ``is_alive`` reaps rather than falling through to
        /proc. The scratch files are left on disk -- they are the run's
        log and event history, which is what makes a failed run
        diagnosable after the fact.

        The ledger claim goes back here too: every watcher path funnels
        through ``_finish``'s ``finally``, and the release is
        idempotent, so a row finalised twice (watcher racing a stop)
        still releases exactly the one claim.
        """
        self._finisher.release_claim(execution_id)
        self._recorder.forget(execution_id)
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
