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

Allocator memory is not this class's business -- ``run_execution``
releases it in its own ``finally``, in whichever process ran the graph
(docs 08 S-05). The *server-side* admission claim (MEM-03's ledger,
``reserved_mb``) is: every watcher path funnels through ``_finish`` and
``_release``, so the claim goes back exactly when the child actually
stops, under whatever owner it was renamed to. The same watcher is also
where a child's reported memory peaks get filed (MEM-04 #2): a ``memory``
record's ``peak_mb`` goes into the peak store under the fingerprint key
admission stored on the row -- one writer, monotonic, only on a rise.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from ..domain.events import GraphExecutionProgressed
from ..domain.graph import GraphDefinition, NodeResult
from ..domain.value_objects import ExecutionId, GraphStatus
from .memory_admission import LedgerSource, graph_owner, release
from .ports.graph_task_stream import EventKind
from .ports.peak_store import PeakStore
from .event_publisher import EventPublisher
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
        # CAS-then-announce lives in the writer; ``events`` is for
        # telemetry only.
        self._writer = writer
        self._gateway = gateway
        self._events = events
        self._clock = clock
        # Where claims are handed back (``_release``). None only where
        # the container has no ledger at all -- and then no claim can
        # exist either, because a start without a ledger refuses before
        # it reserves. Both composition roots always pass one.
        self._memory_ledger = memory_ledger
        # Where a child's reported peaks are filed (MEM-04 #2, see
        # ``_record_peak``). None only in tests that do not exercise
        # peaks -- a run whose peaks are not recorded still runs; both
        # composition roots pass the store.
        self._peak_store = peak_store
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
        # How many consecutive poll failures the watcher rides out before it
        # gives up on a run it is otherwise watching. One transient error --
        # "database is locked" during another writer's transaction -- must
        # not cost the run its supervisor; a persistent one must not cost
        # the server a thread that retries forever.
        self._max_poll_failures = max_poll_failures
        # How many times one node's result is persisted before it is
        # given up on. Separate from the poll bound because the two
        # failures are not recoverable in the same way: a failed poll can
        # simply be repeated, while a failed *record* cannot -- see
        # `_record_node`.
        self._max_record_attempts = max_record_attempts
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
        # The peak this run has already had filed, so frames that repeat
        # it cost no statement (see ``_record_peak``). Popped in
        # ``_release`` with everything else the watcher forgets.
        self._peak_seen: dict[ExecutionId, float] = {}

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
                    self._record_peak(execution_id, event.payload)
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
                    outcomes += self._apply(execution_id, tail.poll(), catch_up)
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
                        outcomes += self._apply(execution_id, tail.poll(), catch_up)
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
            abandoned = self._abandon(execution_id, pid)
            if abandoned:
                # The child may well have finished as it was being stopped,
                # and its own record is better evidence than our failure to
                # watch it -- the same reasoning as the drain above.
                try:
                    outcomes += self._apply(execution_id, tail.poll(), catch_up)
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

    def _abandon(self, execution_id: ExecutionId, pid: int) -> bool:
        """Stop a run we have lost the ability to watch. True if it was live.

        This exists for one invariant: **a row is never terminal while its
        child is alive.** A terminal row releases the single-active check,
        so writing one for a run that is still holding the card is how two
        trainers end up on one 12 GB device -- and an unwatched run is also
        a run the API can no longer stop, because ``cancel`` needs the pid
        this watcher was holding.

        So when the watcher gives up, it does not simply write the row off.
        It asks the child to stop, escalates to a kill if that is ignored,
        and only then lets the row be finalised -- with a note that says
        supervision failed rather than claiming the process died.

        Returns ``False`` when the child had already gone, which is the
        ordinary case: a watcher that breaks because liveness said no has
        nothing to abandon.
        """
        try:
            if not self._gateway.is_alive(pid):
                return False
        except Exception:  # noqa: BLE001 -- liveness is what failed
            # Assume the worst. The alternative is leaving a process that
            # holds the card unaccounted for, on the strength of a check
            # that did not answer.
            logger.warning(
                "could not determine whether pid %s is alive; treating it as "
                "alive and stopping it", pid,
            )

        logger.error(
            "graph execution %s can no longer be supervised; stopping its "
            "child (pid %s) rather than leaving it running unobserved",
            execution_id, pid,
        )
        try:
            self._gateway.request_stop(pid)
        except Exception:  # noqa: BLE001 -- escalate regardless
            logger.exception("asking pid %s to stop failed", pid)
        threading.Event().wait(self._stop_grace)
        try:
            still_running = self._gateway.is_alive(pid)
        except Exception:  # noqa: BLE001 -- liveness is what failed
            # Liveness has already failed once on this run, so a second
            # failure here is expected rather than surprising. It must not
            # be read as "so it probably stopped": not knowing is the one
            # state in which the card is still held and the run is still
            # unstoppable through the API. Unknown means alive.
            logger.warning(
                "cannot tell whether pid %s stopped; killing it rather than "
                "leaving it unaccounted for", pid,
            )
            still_running = True
        if still_running:
            logger.warning(
                "pid %s did not stop within %.0fs of supervision failing; "
                "killing", pid, self._stop_grace,
            )
            try:
                self._gateway.kill(pid)
            except Exception:  # noqa: BLE001 -- nothing further to try
                logger.exception("killing pid %s failed", pid)
        return True

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
            elif kind is EventKind.MEMORY:
                self._record_peak(execution_id, event.payload)
            else:  # OUTCOME
                outcomes.append(event.payload)
        return outcomes

    def _record_node(self, execution_id: ExecutionId, payload: dict) -> None:
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

        Measured on a run writing 40 node records, one injected failure at
        each position of the poll cycle: 0 results lost when it landed
        outside the write, 1 lost when it landed inside -- and no field of
        the row recorded the loss. Losing one step of a long run to a locked
        database is not a rounding error, so the write is retried.

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
                    # Already stored. `commit` is a compare-and-swap *and*
                    # then a publish, so it can fail after the row was
                    # already updated -- and the retry then re-read the row,
                    # finds this node's result on it, and appends a second
                    # copy. Measured against the real SQLite repository with
                    # one injected publish failure: 12 nodes produced 35
                    # stored results, 12 distinct, because every retry
                    # after the CAS succeeded added another.
                    #
                    # A node runs once per execution, so a duplicate
                    # node_id is never legitimate: the entity already
                    # refuses a row carrying more results than the graph has
                    # nodes. Skipping on sight is both the fix and the
                    # invariant.
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

    def _record_peak(self, execution_id: ExecutionId, payload: dict) -> None:
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

    def _finish(
        self,
        execution_id: ExecutionId,
        pid: int,
        saw_outcome: bool,
        outcome_error: str | None,
        abandoned: bool = False,
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
            # Hand the claim back *before* the row can read terminal, so
            # the invariant an observer sees is strict: this run's row
            # only reaches a terminal status once the capacity is free.
            # Not at the top of `_finish`: `_release` clears
            # `_stop_requested`, and `stopped` must be read first. The
            # finally below releases again (idempotent) and covers the
            # early-return path, where a stop or reconcile already
            # wrote the row before this watcher got here.
            release(self._memory_ledger, graph_owner(execution_id))
            if not saw_outcome:
                # Which of the two "no outcome" stories is true depends on
                # whether the child was still there. Saying "crashed, or a
                # device fault killed it" about a process we ourselves
                # stopped -- and that was running perfectly well -- is the
                # wrong answer twice over: it sends the reader to the
                # execution log for a crash that never happened.
                #
                # The log's tail goes INTO the row. That sentence used to
                # end "see the execution log", and the startup sweep then
                # deleted that log -- so the only place a user could look
                # for the traceback of a crashed run was destroyed by the
                # next server start, and the row said nothing more than
                # "it died". The row is what survives; the evidence belongs
                # in it.
                execution.mark_failed(
                    at=at,
                    error=self._with_log_tail(execution_id, (
                        "supervision of this run failed, so it was stopped; "
                        "it reported no outcome -- see the server log and "
                        "the execution log"
                        if abandoned else
                        "execution process exited without reporting an "
                        "outcome (crashed, or a device fault killed it) -- "
                        "see the execution log"
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
        finally:
            self._release(execution_id, pid)

    #: How much of a crashed child's log is copied into its row. Four
    #: kilobytes is a traceback and a bit; a full training log is megabytes
    #: of progress lines, and a row is not a log file.
    LOG_TAIL_BYTES = 4096

    def _with_log_tail(self, execution_id: ExecutionId, error: str) -> str:
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
            log_path = self._paths_for(execution_id)["log"]
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
        release(self._memory_ledger, graph_owner(execution_id))
        with self._lock:
            self._pids.pop(execution_id, None)
            self._stop_requested.discard(execution_id)
            self._replaying.discard(execution_id)
            self._peak_seen.pop(execution_id, None)
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
