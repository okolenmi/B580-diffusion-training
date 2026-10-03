"""Adoption: a graph run that outlived the server that started it.

WP-22 moved execution into a child process, and a child in its own
session does not die with its parent. That is the entire reason
isolation is worth having -- a device fault or an OOM kill takes the run
down instead of the server -- but it has a second consequence that has
to be handled rather than discovered: a server restart is no longer the
end of a run that is still going.

These tests are the restart. They are built around the real pieces rather
than a mock of them, because the thing being tested is the *handover*:

* the gateway's pid discovery reads live ``/proc``, so the child found
  is the one actually running, not a recorded number;
* the supervisor's replay reads the real event file the child is
  appending to, so a torn line or a mid-write offset is exercised for
  real rather than simulated.

What is faked is the database: the interesting assertions are about which
records the watcher chose to apply, and an in-memory recorder shows that
directly where a row would only show its consequences. The row's own CAS
is already covered by test_graph_execution.py.

Run directly: python backend/tests/test_graph_adoption.py
"""

from __future__ import annotations

import datetime
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.event_publisher import EventPublisher
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.ports.execution_launcher import ExecutionLauncher
from backend.application.ports.graph_task_gateway import (
    GraphTaskGateway,
    GraphTaskLaunch,
)
from backend.domain.graph import GraphDefinition, GraphNodeSpec
from backend.domain.value_objects import ExecutionId, GraphStatus
from backend.infrastructure.graph_event_stream import (
    EventKind,
    ExecutionEventTail,
    ExecutionEventWriter,
)
from backend.infrastructure.graph_task_gateway import SubprocessGraphTaskGateway
from backend.infrastructure.workspace import WorkspaceLayout
from backend.tests.support import (
    FakeClock,
    RecordingEventBus,
    check,
    run_tests_concurrently,
    wait_until,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix="backend-graph-adopt-"))

#: Generous, because a child costs ~2s of startup and these tests are
#: about a handover rather than about latency.
WAIT = 90.0


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------

class RecordingGateway(GraphTaskGateway):
    """A gateway whose children are dicts of records the test wrote.

    Lets the replay semantics be asserted directly -- which record kinds
    the watcher applied, in which order -- instead of inferred from a
    row's final state. Also the only way to produce a file the test
    controls the timing of, which is what makes "the replay window closed
    before the live records arrived" testable at all.
    """

    def __init__(self, event_path: Path, *, alive: bool = True) -> None:
        self.event_path = event_path
        self.alive = alive
        self.stopped: list[int] = []
        self.killed: list[int] = []
        self.found: int | None = None

    def spawn(self, launch: GraphTaskLaunch) -> int:
        raise AssertionError("adoption tests never spawn")

    def request_stop(self, pid: int) -> None:
        self.stopped.append(pid)

    def kill(self, pid: int) -> None:
        self.killed.append(pid)

    def is_alive(self, pid: int) -> bool:
        return self.alive

    def find_running(self, execution_id: ExecutionId) -> int | None:
        return self.found

    def write(self, *records: dict) -> None:
        """Append records as the fake child would.

        Goes through the real writer, so the bytes on disk are the bytes a
        child would have written -- sanitisation, allow_nan and all. A
        double that hand-rolled its JSON would be testing a format nothing
        produces.
        """
        writer = ExecutionEventWriter(self.event_path)
        for record in records:
            writer.node(record) if record["kind"] == "node" else (
                writer.monitor(record["monitor_id"], record["data"])
                if record["kind"] == "monitor"
                else writer.outcome(error=record["error"],
                                    results_count=record["results_count"])
            )
        writer.close()


class RecordingWriter:
    """A lifecycle writer that records commits instead of writing rows.

    Records the *node* each commit carries, not just the execution id: the
    assertion this exists for is about which node results reached the row,
    and an id alone cannot tell "recorded node c" from "recorded c again".
    """

    def __init__(self) -> None:
        self.committed: list[str] = []
        self.failed: list[tuple] = []

    def commit(self, execution, expected=None) -> bool:
        recorded = getattr(execution, "results", ())
        newest = recorded[-1].node_id if recorded else None
        if newest is not None:
            self.committed.append(newest)
        return True

    def fail_if_unfinished(self, execution, error: str) -> bool:
        self.failed.append((execution.id, error))
        return True


class StubExecutions:
    """The one row an adoption watcher reads.

    The watcher re-reads the row on every poll because the row's status is
    the authority on whether the run is still live -- so a double that
    returned nothing would make the watcher conclude the row had gone
    terminal and exit, and every assertion here would be about a watcher
    that had already stopped watching. One real ``GraphExecution`` in
    ``running``, and the re-read is honest.
    """

    def __init__(self, execution_id: int = 1, *, graph: GraphDefinition | None = None) -> None:
        from backend.domain.entities.graph_execution import GraphExecution

        self._execution = GraphExecution.create(graph=graph or GraphDefinition(), created_at=_NOW)
        self._execution.assign_id(ExecutionId(execution_id))
        self._execution.mark_running(at=_NOW)

    def get(self, execution_id):
        return self._execution if execution_id == self._execution.id else None


_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


class RunningExecutions(ExecutionLauncher):
    """Launcher that keeps runs alive forever, so the watcher is the only
    thing that ends them. The watcher exits when the row goes terminal,
    which is what stops the test from hanging."""

    def __init__(self) -> None:
        self._terminal: set = set()

    def launch(self, execution_id, graph) -> None:
        raise AssertionError("adoption tests never launch")

    def cancel(self, execution_id) -> None:
        raise AssertionError("adoption tests never cancel")

    def adopt(self, execution_id):
        return None

    def recorded_outcome(self, execution_id):
        return None

    def go_terminal(self, execution_id) -> None:
        self._terminal.add(execution_id)


def _supervisor(gateway, monitor_bus=None, poll: float = 0.02,
                writer=None, executions=None) -> GraphExecutionSupervisor:
    return GraphExecutionSupervisor(
        executions=executions if executions is not None else StubExecutions(),
        writer=writer if writer is not None else RecordingWriter(),
        gateway=gateway,
        events=EventPublisher(events=RecordingEventBus()),
        clock=FakeClock(),
        monitor_bus=monitor_bus,
        scratch_dir=TMP,
        make_tail=ExecutionEventTail,
        poll_interval=poll,
        stop_grace=0.5,
    )


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class CollectingBus:
    """Monitor bus that keeps every report, so history can be asserted."""

    def __init__(self) -> None:
        self.reports: list[tuple[str, dict]] = []
        self.cleared: list[str] = []
        self._lock = threading.Lock()

    def report(self, monitor_id: str, data: dict) -> None:
        with self._lock:
            self.reports.append((monitor_id, data))

    def clear(self, monitor_id: str) -> None:
        self.cleared.append(monitor_id)

    def ids(self) -> list[str]:
        with self._lock:
            return [monitor_id for monitor_id, _ in self.reports]


# --------------------------------------------------------------------------
# The replay boundary
# --------------------------------------------------------------------------

def test_replay_skips_node_results_but_keeps_monitor_history() -> None:
    print("\n== adoption: node results are not recorded twice, history is kept ==")
    bus = CollectingBus()
    # The writer and repository are handed in rather than swapped in
    # afterwards: the assertion is about which records the watcher chose
    # to apply *across the whole run*, so a double installed after
    # adoption would only ever see the live half -- and the double-count
    # bug this test exists to catch is entirely in the replayed half.
    writer = RecordingWriter()
    executions = StubExecutions()
    supervisor = _supervisor(RecordingGateway(TMP / "unused.jsonl"),
                             monitor_bus=bus, writer=writer, executions=executions)
    supervisor._gateway.found = 4242
    gateway = supervisor._gateway

    # The gateway writes where the supervisor *looks*, not where the test
    # picked: adoption resolves the event file from the execution id, so a
    # test that invented its own path would exercise a lookup that finds
    # nothing -- silently, and with the same result as adoption working.
    path = supervisor._paths_for(ExecutionId(1))["events"]
    gateway.event_path = path

    # Two nodes and two reports already written -- the span the new
    # server missed.
    gateway.write(
        {"kind": "node", "node_id": "a", "ok": True, "outputs": {}, "error": None,
         "duration_ms": 1.0},
        {"kind": "monitor", "monitor_id": "m1", "data": {"step": 1}},
        {"kind": "node", "node_id": "b", "ok": True, "outputs": {}, "error": None,
         "duration_ms": 1.0},
        {"kind": "monitor", "monitor_id": "m1", "data": {"step": 2}},
    )
    check(supervisor.adopt(ExecutionId(1)) == 4242,
          "the run is adopted by the pid the gateway found")

    check(_wait_for(lambda: len(bus.reports) == 2),
          f"both missed reports are republished (got {bus.reports})")
    check(bus.ids() == ["m1", "m1"], "under the monitor id the child used")

    # The replay is over once the *watcher's own* tail has consumed
    # everything. Watching a second tail for it would be a different
    # reader: its offset starts at zero, so it is never caught up on a
    # file that has records, and the wait would spin until it timed out
    # rather than reporting the bug it was meant to detect.
    check(_wait_for(lambda: supervisor._replayed(ExecutionId(1))),
          "the replay window closed on a file the child is done with")
    gateway.write(
        {"kind": "node", "node_id": "c", "ok": True, "outputs": {}, "error": None,
         "duration_ms": 1.0},
        {"kind": "outcome", "error": None, "results_count": 3},
    )
    # Waited on the *live* record specifically, not on "some commit":
    # waiting on the first commit would be satisfied by the very bug this
    # is checking for, since a replay that wrongly records the two
    # historical nodes commits them first and immediately.
    check(_wait_for(lambda: "c" in writer.committed),
          f"the live node result reached the row (got {writer.committed})")

    check(
        writer.committed == ["c"],
        f"and only the live one: the two replayed results were not recorded "
        f"again, which would leave the row with three results for a graph "
        f"the domain says has two (got {writer.committed})",
    )
    check(len(bus.reports) == 2,
          f"while the monitor history the run missed is still there "
          f"(got {bus.reports})")
    check(gateway.killed == [],
          f"nothing was killed: the run was adopted, not restarted "
          f"(got {gateway.killed})")


def test_a_live_child_is_found_by_argv_not_by_a_stored_pid() -> None:
    print("\n== adoption: the child is found by what its argv says it is ==")
    # Spawned for real. The point is that discovery reads the live system:
    # a pid kept in a row is a claim about the past, and read after a
    # reboot it names whatever process the number was recycled to.
    graph = TMP / "adopt_graph.json"
    graph.write_text(
        '{"format": 1, "nodes": [{"id": "a", "class_name": "FloatConstantNode",'
        ' "params": {"value": 1.0}}], "edges": []}',
        encoding="utf-8",
    )
    gateway = SubprocessGraphTaskGateway(WorkspaceLayout(REPO_ROOT))
    pid = gateway.spawn(
        GraphTaskLaunch(
            execution_id=ExecutionId(77),
            graph_path=graph,
            event_path=TMP / "adopt_real.events.jsonl",
            log_path=TMP / "adopt_real.log",
        )
    )
    try:
        check(gateway.find_running(ExecutionId(77)) == pid,
              f"the running child is found for its execution id (got {pid})")
        check(gateway.find_running(ExecutionId(78)) is None,
              "and a different id finds nothing, rather than the nearest "
              "match -- the substring trap that would adopt another run's "
              "history into this row")
    finally:
        gateway.kill(pid)
        wait_until(lambda: not gateway.is_alive(pid), timeout=30.0)

    check(gateway.find_running(ExecutionId(77)) is None,
          "once it is gone there is nothing to adopt")


def test_a_run_with_no_event_file_is_not_adopted() -> None:
    print("\n== adoption: a live child whose output goes nowhere is debris ==")
    gateway = RecordingGateway(TMP / "never-written.events.jsonl")
    gateway.found = 555
    supervisor = _supervisor(gateway)
    check(supervisor.adopt(ExecutionId(2)) is None,
          "adoption refuses a run the new server could not observe")
    check(gateway.killed == [],
          f"and does not kill it either: the supervisor is not the owner of "
          f"a process it did not start, and a user can still find it in "
          f"`ps` (got {gateway.killed})")


def test_an_in_process_run_is_never_adoptable() -> None:
    print("\n== adoption: a thread run cannot outlive the server, and says so ==")
    from backend.infrastructure.graph_task_gateway import InProcessGraphTaskGateway

    check(InProcessGraphTaskGateway().find_running(ExecutionId(3)) is None,
          "the in-process gateway reports nothing to adopt, which is the "
          "honest answer rather than a limitation")


def test_adoption_counts_are_reported_separately() -> None:
    print("\n== reconcile: adopted and failed are counted apart ==")
    from backend.application.dto import ReconcileResult

    # One type for both reconcilers, and the dataset one adopts nothing --
    # so the field has to default or that call site changes for no reason.
    check(ReconcileResult(cleaned=3).adopted == 0,
          "the dataset reconciler's single-argument construction still "
          "works, and reports nothing adopted")
    check(ReconcileResult(cleaned=2, adopted=1) == ReconcileResult(cleaned=2, adopted=1),
          "and a graph reconcile can report both counts at once")
    check(ReconcileResult(cleaned=0, adopted=5).cleaned == 0,
          "counting them together would hide how much was thrown away, so "
          "they are separate fields rather than one")


def test_the_default_is_the_child() -> None:
    print("\n== the default is isolation, and stays that way ==")
    from backend.config import (
        DEFAULT_GRAPH_EXECUTION_MODE,
        GRAPH_EXECUTION_CHILD,
        Settings,
    )

    # Pinned because the default was deliberately left on the old path
    # while the new one was being built, and flipping it was the last
    # step rather than the first. A later edit that moved it back would
    # otherwise pass every test in the suite: the tests build their own
    # gateway explicitly, so the default is the one thing they never
    # exercise.
    check(DEFAULT_GRAPH_EXECUTION_MODE == GRAPH_EXECUTION_CHILD,
          f"the named default is the child gateway (got {DEFAULT_GRAPH_EXECUTION_MODE!r})")
    check(Settings.load({}).graph_execution_mode == GRAPH_EXECUTION_CHILD,
          "and an unset environment gets it")
    check(Settings(project_root=Path(".")).graph_execution_mode == GRAPH_EXECUTION_CHILD,
          "and so does a constructed Settings, whose field default and the "
          "env default are separate paths that could disagree")

    # The rollback has to keep working, or "flip it back" is not a plan.
    check(
        Settings.load({"BACKEND_GRAPH_EXECUTION": "inprocess"}).graph_execution_mode
        == "inprocess",
        "and the rollback still overrides it",
    )


def test_a_run_that_finishes_while_the_watcher_is_busy_is_not_a_crash() -> None:
    print("\n== the outcome arrives after the last poll, not before it ==")
    # Found live, not thought of: a 3000-node graph whose child finished
    # and exited while the watcher was still persisting the batch it had
    # already read. The event file ended with a clean outcome; the row
    # said "exited without reporting an outcome (crashed, or a device
    # fault killed it)".
    #
    # The ordering is reproduced rather than raced for: the fake child
    # writes the rest of its records at the moment the watcher asks
    # whether it is still alive, which is precisely when a real child
    # that is faster than the watcher does it -- it finishes writing and
    # exits between the watcher's poll and its liveness check.
    gateway = _LateFinishingGateway(TMP / "unused.jsonl")
    executions = StubExecutions(9, graph=_graph_of(800))
    supervisor = _supervisor(gateway, executions=executions)
    gateway.found = 7777
    # Where the supervisor looks, not where the test picked: adoption
    # resolves the event file from the execution id, so a fake writing to
    # a path of its own choosing would find nothing and fail for a reason
    # that has nothing to do with the bug.
    gateway.event_path = supervisor._paths_for(ExecutionId(9))["events"]

    # The part the watcher reads first: half the graph, no outcome.
    gateway.write(
        *(
            {"kind": "node", "node_id": f"n{i}", "ok": True, "outputs": {},
             "error": None, "duration_ms": 1.0}
            for i in range(400)
        )
    )

    check(supervisor.adopt(ExecutionId(9)) == 7777, "the run is adopted")
    check(
        _wait_for(lambda: gateway.finished),
        f"the child finished and exited (got finished={gateway.finished})",
    )
    check(
        _wait_for(lambda: executions.get(ExecutionId(9)).status.is_terminal),
        f"the watcher finalised the row (got "
        f"{executions.get(ExecutionId(9)).status})",
    )
    final = executions.get(ExecutionId(9))
    check(
        final.status is GraphStatus.FINISHED,
        "as **finished**, read from the outcome record the child wrote -- "
        "before this was fixed the row said the process crashed, which is "
        f"how a successful run gets reported as a hardware fault "
        f"(got {final.status}: {final.error})",
    )
    check(
        final.results[-1].node_id == "n799",
        "and every node result is on the row, including the 400 the "
        f"watcher had not read when liveness said no (got "
        f"{len(final.results)} results, last {final.results[-1].node_id})",
    )


def _graph_of(node_count: int) -> GraphDefinition:
    """A graph with ``node_count`` distinct nodes, for the domain's own rule.

    The entity refuses to load a row carrying more results than the graph
    has nodes, so a stub whose graph was empty would be asserting against
    a row that cannot exist.
    """
    return GraphDefinition(
        nodes=tuple(
            GraphNodeSpec(id=f"n{i}", class_name="FloatConstantNode", params={})
            for i in range(node_count)
        )
    )


class _LateFinishingGateway(RecordingGateway):
    """Writes the rest of its records when asked whether it is alive."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.finished = False
        self._rest = [
            {"kind": "node", "node_id": f"n{i}", "ok": True, "outputs": {},
             "error": None, "duration_ms": 1.0}
            for i in range(400, 800)
        ] + [{"kind": "outcome", "error": None, "results_count": 800}]

    def is_alive(self, pid: int) -> bool:
        if self.finished:
            return False
        self.write(*self._rest)
        self.finished = True
        return False  # it exited as it finished writing


def test_a_new_run_does_not_inherit_the_last_one_s_records() -> None:
    print("\n== execution 1 today is not execution 1 yesterday's file ==")
    # Observed live: deleting backend.db restarts row ids at 1, so a new
    # execution 1 lands on the previous execution 1's event file. Records
    # are appended, so the new run inherited 3000 stale node records; the
    # watcher read 6000 results for a 3000-node graph, which the domain
    # refuses to load, and the row was stuck running forever with every
    # read and write of it raising.
    scratch = TMP / "reused-id"
    scratch.mkdir(parents=True, exist_ok=True)
    events = scratch / "execution_1.events.jsonl"

    stale = ExecutionEventWriter(events)
    stale.node({"node_id": "n0", "ok": True, "outputs": {}, "error": None,
                "duration_ms": 1.0})
    stale.outcome(error="a previous run's ending", results_count=1)
    stale.close()
    check(events.stat().st_size > 0, "the previous run left records behind")

    written: list[int] = []

    class _Recording(RecordingGateway):
        """Writes one record at spawn, the way a child would."""

        def spawn(self, launch):
            writer = ExecutionEventWriter(self.event_path)
            writer.node({"node_id": "fresh", "ok": True, "outputs": {},
                         "error": None, "duration_ms": 1.0})
            writer.outcome(error=None, results_count=1)
            writer.close()
            written.append(1)
            return 1

    gateway = _Recording(events)
    supervisor = _supervisor(gateway)
    supervisor._paths_for = lambda execution_id: {
        "graph": scratch / "execution_1.graph.json",
        "events": events,
        "log": scratch / "execution_1.log",
    }
    # Written where launch says it will be, so the assertion is about the
    # real path rather than about a path the test substituted afterwards.
    scratch.joinpath("execution_1.graph.json").write_text("{}", encoding="utf-8")

    written_now: list[str] = []
    original_spawn = gateway.spawn

    def spawn_capturing(launch):
        pid = original_spawn(launch)
        written_now.append(launch.event_path.name)
        return pid

    gateway.spawn = spawn_capturing  # type: ignore[method-assign]
    supervisor.launch(ExecutionId(1), _graph_of(1))

    check(written_now == ["execution_1.events.jsonl"],
          f"the child was told where to write (got {written_now})")
    check(
        [e.payload.get("node_id")
         for e in ExecutionEventTail(events).poll()
         if e.kind is EventKind.NODE] == ["fresh"],
        "and the only node record in the file is this run's -- not the "
        "previous run's, which would be read as this graph's own results",
    )


def main() -> None:
    tests = [
        test_replay_skips_node_results_but_keeps_monitor_history,
        test_a_live_child_is_found_by_argv_not_by_a_stored_pid,
        test_a_run_with_no_event_file_is_not_adopted,
        test_an_in_process_run_is_never_adoptable,
        test_adoption_counts_are_reported_separately,
        test_the_default_is_the_child,
        test_a_run_that_finishes_while_the_watcher_is_busy_is_not_a_crash,
        test_a_new_run_does_not_inherit_the_last_one_s_records,
    ]
    run_tests_concurrently(tests)


if __name__ == "__main__":
    main()
