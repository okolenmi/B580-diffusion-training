"""Graph execution tests -- entity, repository, supervisor, use cases (M4).

Four layers of the execution lifecycle (docs/design/backend/
05-graph-runtime.md sections 4-5): the state machine's invariants, the
SQLite CAS, the supervisor's thread + per-node persistence, and the use
cases around them (single-active start, stop, reconcile, history,
library).

The graph runtime is real (fixture nodes, no-op memory releaser); only
the threads are untimed policy. No torch, no GPU.

Run directly: python backend/tests/test_graph_execution.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.event_publisher import EventPublisher
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.lifecycle_writer import ExecutionLifecycleWriter
from backend.application.errors import (
    GraphExecutionActiveError,
    GraphExecutionNotFoundError,
    GraphExecutionNotActiveError,
    GraphInvalidError,
    GraphNotFoundError,
    InvalidQueryError,
)
from backend.application.use_cases import (
    ReconcileGraphExecutions,
    SweepExecutionScratch,
)
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.exceptions import DomainError, InvalidTransitionError
from backend.domain.graph import GraphDefinition, GraphEdgeSpec, GraphNodeSpec, NodeResult
from backend.application.ports.execution_launcher import (
    ExecutionLauncher,
    RecordedOutcome,
)
from backend.domain.value_objects import ExecutionId, GraphStatus
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.tests.support import (
    FakeClock,
    RecordingEventBus,
    build_services,
    check,
    finish,
    fixture_graph_registry,
    wait_until,
)

NOW = datetime(2026, 3, 1, 9, 0, 0, tzinfo=UTC)


def node(node_id: str, class_name: str, **params) -> GraphNodeSpec:
    return GraphNodeSpec(id=node_id, class_name=class_name, params=params)


def edge(from_node, from_port, to_node, to_port) -> GraphEdgeSpec:
    return GraphEdgeSpec(
        from_node=from_node, from_port=from_port,
        to_node=to_node, to_port=to_port,
    )


def make(*nodes, edges=()) -> GraphDefinition:
    return GraphDefinition(nodes=tuple(nodes), edges=tuple(edges))


VALID = make(
    node("v", "ScaleNode", value=3.0, factor=5.0),
    node("s", "SumNode", b=1.0),
    edges=(edge("v", "scaled", "s", "a"),),
)


def codes(events) -> list[str]:
    return [event.event_type for event in events]


class NothingToAdopt(ExecutionLauncher):
    """A launcher with no runs of its own to find.

    The default for the reconcile tests: they are about what happens to
    rows nobody is watching, and the case where something *is* still
    running is the adoption tests in test_graph_task_gateway.py, which
    needs a real child to be true.
    """

    def __init__(self, recorded=None) -> None:
        self.recorded = recorded or {}

    def launch(self, execution_id, graph) -> None:
        raise AssertionError("the reconcile tests never launch")

    def cancel(self, execution_id) -> None:
        raise AssertionError("the reconcile tests never cancel")

    def adopt(self, execution_id):
        return None

    def recorded_outcome(self, execution_id):
        return self.recorded.get(execution_id)


# ==========================================================================
# Section A: the entity owns its lifecycle
# ==========================================================================
print("-- entity --")

execution = GraphExecution.create(graph=VALID, created_at=NOW)
check(
    execution.status is GraphStatus.QUEUED and execution.id is None,
    "create() starts queued with no id",
)
queued_events = execution.collect_events()
check(queued_events == [], "nothing buffered before an id exists")
execution.assign_id(ExecutionId(7))
check(codes(execution.collect_events()) == ["graph_execution_queued"],
      "assign_id buffers GraphExecutionQueued")
check(execution.collect_events() == [], "event buffer drains")
try:
    execution.assign_id(ExecutionId(8))
    check(False, "re-binding an id is refused")
except DomainError:
    check(True, "re-binding an id is refused")

execution.mark_running(at=NOW)
check(
    execution.status is GraphStatus.RUNNING and execution.started_at == NOW,
    "queued -> running stamps started_at",
)
execution.record_result(
    NodeResult(node_id="v", ok=True, outputs={"scaled": 15.0}, duration_ms=1.5),
    at=NOW,
)
check(len(execution.results) == 1, "results append while running")
execution.mark_finished(at=NOW)
check(
    execution.status is GraphStatus.FINISHED
    and execution.finished_at == NOW
    and execution.updated_at == NOW,
    "running -> finished stamps both terminal timestamps",
)
try:
    execution.record_result(
        NodeResult(node_id="s", ok=True), at=NOW
    )
    check(False, "no results after a terminal state")
except InvalidTransitionError:
    check(True, "no results after a terminal state")
try:
    execution.mark_running(at=NOW)
    check(False, "terminal states are final")
except InvalidTransitionError:
    check(True, "terminal states are final")

queued = GraphExecution.create(graph=VALID, created_at=NOW)
queued.assign_id(ExecutionId(9))
queued.stop(at=NOW, reason="stop requested")
check(
    queued.status is GraphStatus.STOPPED and queued.error == "stop requested",
    "queued -> stopped is allowed (never ran, note recorded)",
)
failed = GraphExecution.create(graph=VALID, created_at=NOW)
failed.assign_id(ExecutionId(10))
failed.collect_events()  # drain the queued event from assign_id
failed.mark_failed(at=NOW, error="node exploded")
check(
    failed.status is GraphStatus.ERROR and failed.error == "node exploded",
    "queued -> error with the reason (reconcile's path)",
)
check(
    codes(failed.collect_events()) == ["graph_execution_failed"],
    "terminal transition buffers its event",
)

# ==========================================================================
# Section B: the SQLite repository (real CAS, real JSON columns)
# ==========================================================================
print("-- repository --")

db = SqliteDatabase(Path(tempfile.mkdtemp(prefix="backend-graph-repo-")) / "g.db")
db.initialize()
repo = SqliteGraphExecutionRepository(db)

row = GraphExecution.create(graph=VALID, created_at=NOW)
check(row.id is None, "unpersisted execution has no id")
repo.add(row)
check(row.id is not None and row.id >= 1, "add binds the id")

fetched = repo.get(row.id)
check(fetched.graph == VALID, "graph snapshot round-trips through JSON")
check(fetched.created_at == NOW and fetched.status is GraphStatus.QUEUED,
      "timestamps and status round-trip")

fetched.mark_running(at=NOW)
fetched.record_result(
    NodeResult(node_id="v", ok=True, outputs={"scaled": 15.0}, duration_ms=2.0),
    at=NOW,
)
check(repo.update_if_status(fetched, expected=GraphStatus.QUEUED) is True,
      "CAS succeeds on the stored status")
check(repo.update_if_status(fetched, expected=GraphStatus.QUEUED) is False,
      "CAS fails once the status moved (second writer loses)")
again = repo.get(row.id)
check(
    again.status is GraphStatus.RUNNING and len(again.results) == 1
    and again.results[0].outputs == {"scaled": 15.0},
    "results persist node by node (partial results survive a crash)",
)

other = GraphExecution.create(graph=make(node("o", "SumNode", a=1.0, b=1.0)),
                              created_at=NOW)
repo.add(other)
check(repo.find_active().id in (row.id, other.id),
      "find_active returns a queued/running row")
check(
    [item.id for item in repo.list_executions(limit=10)] == [other.id, row.id],
    "list is newest-first",
)
check(
    {item.id for item in repo.list_unfinished()} == {row.id, other.id},
    "list_unfinished feeds reconcile",
)

other.mark_running(at=NOW)
check(repo.update_if_status(other, expected=GraphStatus.QUEUED), "claim queued->running")
fetched.mark_finished(at=NOW)
check(
    repo.update_if_status(fetched, expected=GraphStatus.RUNNING) is True,
    "running -> finished persists",
)
check(repo.find_active().id == other.id, "terminal rows drop out of find_active")
other.mark_failed(at=NOW, error="x")
repo.update_if_status(other, expected=GraphStatus.RUNNING)
check(repo.find_active() is None, "no active row when everything is terminal")

unpersisted = GraphExecution.create(graph=VALID, created_at=NOW)
try:
    repo.update(unpersisted)
    check(False, "updating an unpersisted row is refused")
except DomainError:
    check(True, "updating an unpersisted row is refused")

third = GraphExecution.create(graph=VALID, created_at=NOW)
repo.add(third)
check(repo.delete_all() == 3, "delete_all wipes every row (counted)")
check(repo.get(row.id) is None, "rows are gone after delete_all")

# ==========================================================================
# Section C: use cases + supervisor end to end
# ==========================================================================
print("-- use cases / supervisor --")

releases = {"n": 0}
events = RecordingEventBus()
runtime = ReflectedGraphRuntime(fixture_graph_registry(),
                                memory_releaser=lambda: releases.__setitem__(
                                    "n", releases["n"] + 1
                                ))
services = build_services(events=events, graph_runtime=runtime)
graphs = services.graphs

# Validation (same authority the run endpoint uses).
valid_result = graphs.validate.execute(VALID)
check(valid_result.ok and valid_result.issues == (), "validate accepts the valid graph")
invalid_result = graphs.validate.execute(make(node("s", "NoSuchNode")))
check(
    not invalid_result.ok
    and [i.code for i in invalid_result.issues] == ["unknown_class"],
    "validate reports the issue list with ok=false",
)

# Start -> run -> finish.
summary = graphs.start_execution.execute(VALID)
check(
    summary.status is GraphStatus.QUEUED and summary.execution_id >= 1,
    "start returns the queued summary (row written before launch)",
)
check(
    "graph_execution_queued" in codes(events.published),
    "queued event published by the start use case",
)
finished = wait_until(
    lambda: (graphs.get_execution.execute(summary.execution_id).status.is_terminal),
    timeout=5.0,
)
check(finished, "worker reaches a terminal state")
check(
    "graph_execution_started" in codes(events.published),
    "started event published when the worker claimed the row",
)
detail = graphs.get_execution.execute(summary.execution_id)
check(detail.status is GraphStatus.FINISHED, "clean graph ends finished")
check(
    [r.node_id for r in detail.results] == ["v", "s"] and all(r.ok for r in detail.results),
    "per-node results persisted in execution order",
)
check(detail.graph == VALID.as_dict(), "stored snapshot is the submitted graph")
check("graph_execution_finished" in codes(events.published), "finished event published")
progressed = [t for t in codes(events.published) if t == "graph_execution_progressed"]
check(len(progressed) == 2, "one progress event per node")
check(releases["n"] >= 1, "memory released after the run")

# Invalid start: full issue list, no row, no launch.
rows_before = graphs.list_executions.execute(limit=500).count
try:
    graphs.start_execution.execute(make(node("s", "NoSuchNode")))
    check(False, "invalid graph refused with graph_invalid")
except GraphInvalidError as exc:
    check(
        exc.code == "graph_invalid"
        and exc.details
        and exc.details[0]["code"] == "unknown_class",
        "graph_invalid carries every issue as details",
    )
check(
    graphs.list_executions.execute(limit=500).count == rows_before,
    "refused start persists nothing",
)

# Node failure ends as error (divergence: legacy reported finished).
boom = graphs.start_execution.execute(make(node("b", "BoomNode")))
check(
    wait_until(
        lambda: graphs.get_execution.execute(boom.execution_id).status.is_terminal,
        timeout=5.0,
    ),
    "failing run reaches a terminal state",
)
boom_detail = graphs.get_execution.execute(boom.execution_id)
check(
    boom_detail.status is GraphStatus.ERROR
    and boom_detail.error is not None
    and "boom" in boom_detail.error,
    "node failure -> status error with the reason",
)
check(
    len(boom_detail.results) == 1 and not boom_detail.results[0].ok,
    "failed node reported in results",
)

# Single-active: a second start is refused while one runs.
slow = graphs.start_execution.execute(make(node("slow", "SlowNode", seconds=0.4)))
try:
    graphs.start_execution.execute(VALID)
    check(False, "second start refused while an execution is active")
except GraphExecutionActiveError as exc:
    check(
        exc.code == "graph_execution_active"
        and exc.details["execution_id"] == slow.execution_id,
        "second start refused while an execution is active",
    )

# Stop: signal, then CAS. Returns the stopped row.
#
# The cancel event is set *before* the CAS on purpose, so there are two
# correct winners: this endpoint, or the worker thread noticing the event
# it was just given and stopping the row itself. Which one wins is a
# genuine race -- asserting a specific winner made this check fail about
# one run in ten. The invariant is the outcome, not the path: the
# execution ends stopped, and the loser reports that honestly.
try:
    stopped_dto = graphs.stop_execution.execute(slow.execution_id)
    check(
        stopped_dto.status is GraphStatus.STOPPED
        and stopped_dto.error == "stop requested",
        "stop claims queued/running -> stopped",
    )
except GraphExecutionNotActiveError as exc:
    check(
        exc.details["status"] == "stopped",
        f"the worker won the stop race and said so (got {exc.details})",
    )
check(
    graphs.get_execution.execute(slow.execution_id).status is GraphStatus.STOPPED,
    "the execution ends stopped whoever wrote it",
)
check("graph_execution_stopped" in codes(events.published), "stopped event published")
try:
    graphs.stop_execution.execute(slow.execution_id)
    check(False, "stopping a terminal execution is 409")
except GraphExecutionNotActiveError as exc:
    check(
        exc.code == "graph_execution_not_active"
        and exc.details["status"] == "stopped",
        "stopping a terminal execution is 409 with the winner's status",
    )
try:
    graphs.stop_execution.execute(999_999)
    check(False, "stopping an unknown execution is 404")
except GraphExecutionNotFoundError:
    check(True, "stopping an unknown execution is 404")
try:
    graphs.get_execution.execute(999_999)
    check(False, "get of an unknown execution is 404")
except GraphExecutionNotFoundError:
    check(True, "get of an unknown execution is 404")

time.sleep(0.5)  # let the stopped worker finish its quiet exit

# History bounds mirror runs (1..500).
for bad_limit in (0, -1, 501):
    try:
        graphs.list_executions.execute(limit=bad_limit)
        check(False, f"limit {bad_limit} refused")
    except InvalidQueryError:
        check(True, f"limit {bad_limit} refused")
check(
    graphs.list_executions.execute(limit=500).count >= 3,
    "limit=500 accepted; history listed newest-first",
)

# Reconcile: both non-terminal shapes are dead-process debris.
reconcile_db = SqliteDatabase(
    Path(tempfile.mkdtemp(prefix="backend-graph-rec-")) / "r.db"
)
reconcile_db.initialize()
reconcile_repo = SqliteGraphExecutionRepository(reconcile_db)
left_queued = GraphExecution.create(graph=VALID, created_at=NOW)
reconcile_repo.add(left_queued)
left_running = GraphExecution.create(graph=VALID, created_at=NOW)
reconcile_repo.add(left_running)
left_running.mark_running(at=NOW)
check(reconcile_repo.update_if_status(left_running, expected=GraphStatus.QUEUED),
      "fixture: one row left running")

reconcile_events = RecordingEventBus()
reconcile_clock = FakeClock()
sweep = ReconcileGraphExecutions(
    executions=reconcile_repo,
    writer=ExecutionLifecycleWriter(
        clock=reconcile_clock,
        repository=reconcile_repo,
        events=EventPublisher(events=reconcile_events),
    ),
    launcher=NothingToAdopt(),
    clock=reconcile_clock,
)
result = sweep.execute()
check(result.cleaned == 2 and result.adopted == 0,
      f"reconcile fails every unfinished row (got cleaned={result.cleaned} "
      f"adopted={result.adopted})")
check(
    reconcile_repo.get(left_queued.id).status is GraphStatus.ERROR
    and reconcile_repo.get(left_queued.id).error
    == "server stopped before the execution started",
    "queued debris: never started",
)
check(
    reconcile_repo.get(left_running.id).error
    == "server restarted while the execution was in flight",
    "running debris: died mid-flight (partial results kept)",
)
check(
    codes(reconcile_events.published).count("graph_execution_failed") == 2,
    "one failed event per swept row",
)
check(sweep.execute().cleaned == 0, "second sweep finds nothing")

# Reconcile: a run that finished while nothing was watching it.
#
# Found live, not by reading: a 3000-node run was left going while the
# server was SIGKILLed. It completed all 3000 nodes and wrote a clean
# outcome record, and the startup sweep reported the row as `error` with
# zero results -- because "there is no process" was the only thing it
# checked. This is round-2 finding N-03 arriving again in newer code; it
# was recorded as moot when the run route was removed, which it was, for
# that route.
print("-- reconcile: the run's own record wins over its absence --")

recovered = [
    # One per node of VALID, which has two: the entity refuses a result
    # count the graph cannot have, and it should.
    NodeResult(node_id=node_id, ok=True, outputs={"value": float(i)},
               error=None, duration_ms=1.0)
    for i, node_id in enumerate(("v", "s"))
]
recorded_db = SqliteDatabase(
    Path(tempfile.mkdtemp(prefix="backend-graph-rec2-")) / "r.db"
)
recorded_db.initialize()
recorded_repo = SqliteGraphExecutionRepository(recorded_db)
finished_while_down = GraphExecution.create(graph=VALID, created_at=NOW)
recorded_repo.add(finished_while_down)
finished_while_down.mark_running(at=NOW)
check(recorded_repo.update_if_status(finished_while_down, expected=GraphStatus.QUEUED),
      "fixture: one row left running")

also_failed = GraphExecution.create(graph=VALID, created_at=NOW)
recorded_repo.add(also_failed)
also_failed.mark_running(at=NOW)
recorded_repo.update_if_status(also_failed, expected=GraphStatus.QUEUED)

recorded_events = RecordingEventBus()
sweep_recorded = ReconcileGraphExecutions(
    executions=recorded_repo,
    writer=ExecutionLifecycleWriter(
        clock=reconcile_clock,
        repository=recorded_repo,
        events=EventPublisher(events=recorded_events),
    ),
    launcher=NothingToAdopt(recorded={
        finished_while_down.id: RecordedOutcome(results=tuple(recovered), error=None),
        also_failed.id: RecordedOutcome(results=(), error="node n2 raised ValueError"),
    }),
    clock=reconcile_clock,
)
result = sweep_recorded.execute()

row = recorded_repo.get(finished_while_down.id)
check(row.status is GraphStatus.FINISHED,
      f"a run that finished while the server was down is finished, not "
      f"failed (got {row.status.value}: {row.error})")
check(row.error is None, "with no error recorded against it")
check(len(row.results) == 2, f"and its node results recovered ({len(row.results)})")
check([r.node_id for r in row.results] == ["v", "s"],
      "in the order the run recorded them")
check(row.results[0].outputs == {"value": 0.0},
      f"with their outputs intact ({row.results[0].outputs})")

row = recorded_repo.get(also_failed.id)
check(row.status is GraphStatus.ERROR,
      f"a run that recorded its own failure is failed (got {row.status.value})")
check(row.error == "node n2 raised ValueError",
      f"carrying the run's own error, not the sweep's wording ({row.error})")

check(result.cleaned == 0,
      f"neither row counted as cleaned debris (got {result.cleaned})")
check(result.adopted == 0, "and neither was adopted")
check(
    codes(recorded_events.published).count("graph_execution_finished") == 1
    and codes(recorded_events.published).count("graph_execution_failed") == 1,
    f"announced as finished and failed respectively, not two failures "
    f"(got {codes(recorded_events.published)})",
)
check(sweep_recorded.execute().cleaned == 0, "and a second sweep finds nothing")

# And a row with no record at all is still debris -- the fix must not turn
# absence into success.
debris_db = SqliteDatabase(
    Path(tempfile.mkdtemp(prefix="backend-graph-rec3-")) / "r.db"
)
debris_db.initialize()
debris_repo = SqliteGraphExecutionRepository(debris_db)
killed = GraphExecution.create(graph=VALID, created_at=NOW)
debris_repo.add(killed)
killed.mark_running(at=NOW)
debris_repo.update_if_status(killed, expected=GraphStatus.QUEUED)
debris_sweep = ReconcileGraphExecutions(
    executions=debris_repo,
    writer=ExecutionLifecycleWriter(
        clock=reconcile_clock,
        repository=debris_repo,
        events=EventPublisher(events=RecordingEventBus()),
    ),
    launcher=NothingToAdopt(),
    clock=reconcile_clock,
)
check(debris_sweep.execute().cleaned == 1
      and debris_repo.get(killed.id).status is GraphStatus.ERROR,
      "a killed run with no outcome record is still failed as debris")

# ==========================================================================
# Section D: library use cases (server-side saved graphs)
# ==========================================================================
print("-- library --")

payload = {
    "nodes": [{"id": "n1", "class_name": "NoSuchNodeYet", "params": {}}],
    "edges": [],
    "palette_note": "keep me",
}
saved = graphs.save_graph.execute("  my graph  ", payload, description="demo")
check(saved.created is True, "first save answers 201")
check(saved.graph.name == "my graph", "name trimmed")
check(
    saved.graph.graph["format"] == 1
    and saved.graph.graph["palette_note"] == "keep me",
    "format stamped; unknown keys preserved verbatim",
)
check(
    graphs.get_graph.execute("my graph").graph == saved.graph.graph,
    "get returns exactly what was stored",
)
again_save = graphs.save_graph.execute("my graph", payload, description="demo2")
check(again_save.created is False, "replace answers 200")
check(graphs.list_graphs.execute().count == 1, "list counts saved graphs")
check(graphs.list_graphs.execute().graphs[0].description == "demo2",
      "replace updates the description")

# Saving never validates class names -- validation happens at run time.
check(
    graphs.validate.execute(
        GraphDefinition.from_dict(saved.graph.graph)
    ).ok is False,
    "a saved graph referencing a missing class still loads, and fails at run",
)

try:
    graphs.get_graph.execute("missing")
    check(False, "unknown saved graph is 404")
except GraphNotFoundError:
    check(True, "unknown saved graph is 404")
try:
    graphs.save_graph.execute("", payload)
    check(False, "empty name refused")
except InvalidQueryError:
    check(True, "empty name refused")
try:
    graphs.save_graph.execute("x" * 121, payload)
    check(False, "over-long name refused")
except InvalidQueryError:
    check(True, "over-long name refused")

check(graphs.delete_graph.execute("my graph").deleted is True, "delete removes it")
try:
    graphs.delete_graph.execute("my graph")
    check(False, "deleting twice is 404")
except GraphNotFoundError:
    check(True, "deleting twice is 404")

# ==========================================================================
# Section E: execution history wipe
# ==========================================================================
print("-- delete history --")

deleted = graphs.delete_executions.execute()
check(deleted.deleted >= 3, f"delete_executions counts rows ({deleted.deleted})")
check(graphs.list_executions.execute(limit=500).count == 0, "history empty after wipe")
check(
    "graph_executions_deleted" in codes(events.published),
    "deletion published its event",
)

# ==========================================================================
# Section G: encapsulation, rehydration and derived terminal-ness
# ==========================================================================
print("-- encapsulation --")

guard = GraphExecution.create(graph=make(node("s", "ScaleNode")), created_at=NOW)
guard.assign_id(ExecutionId(11))
guard.mark_running(at=NOW)
for field, value, label in (
    ("status", GraphStatus.FINISHED, "status"),
    ("results", (), "results"),
    ("finished_at", None, "finished_at"),
    ("graph", make(node("z", "ZeroNode")), "graph"),
    ("error", "boom", "error"),
):
    try:
        setattr(guard, field, value)
        check(False, f"{label} must not be writable from outside")
    except AttributeError:
        check(True, f"{label} has no setter")
check(guard.status is GraphStatus.RUNNING, "the refused writes changed nothing")

print("-- terminal-ness is derived --")
from backend.domain.value_objects import GRAPH_TRANSITIONS  # noqa: E402

check(
    GraphStatus.FINISHED.is_terminal
    and GraphStatus.ERROR.is_terminal
    and GraphStatus.STOPPED.is_terminal,
    "finished/error/stopped are terminal",
)
check(
    not GraphStatus.QUEUED.is_terminal and not GraphStatus.RUNNING.is_terminal,
    "queued/running are not",
)
check(
    all(
        status.is_terminal == (not GRAPH_TRANSITIONS[status])
        for status in GraphStatus
    ),
    "terminal == 'this state has no way out', for every state",
)
check(
    len(GRAPH_TRANSITIONS) == len(list(GraphStatus)),
    "the table names every state",
)

print("-- restore() checks cross-field rules --")
_graph = make(node("s", "ScaleNode"), node("v", "ZeroNode"))


def _row(**overrides):
    base = dict(
        id=ExecutionId(12),
        status=GraphStatus.RUNNING,
        graph=_graph,
        created_at=NOW,
        updated_at=NOW,
        started_at=NOW,
    )
    base.update(overrides)
    return base


check(
    GraphExecution.restore(**_row()).status is GraphStatus.RUNNING,
    "a sane row loads",
)
for overrides, fragment, label in (
    (dict(status=GraphStatus.RUNNING, started_at=None), "started_at is null",
     "running without started_at"),
    (dict(status=GraphStatus.STOPPED, finished_at=None), "finished_at is null",
     "terminal without finished_at"),
    (
        dict(
            status=GraphStatus.ERROR,
            finished_at=NOW,
            results=(NodeResult(node_id="a", ok=True),
                     NodeResult(node_id="b", ok=True),
                     NodeResult(node_id="c", ok=True)),
        ),
        "results for a",
        "more results than the graph has nodes",
    ),
):
    try:
        GraphExecution.restore(**_row(**overrides))
        check(False, f"{label} must be rejected on load")
    except DomainError as exc:
        check(fragment in str(exc), f"{label} refused (got {exc})")

# ==========================================================================
# Section A2: from_dict's tolerance is load-bearing, so it is pinned
# ==========================================================================
print("-- from_dict --")

# Found by scripts/mutation_report.py graph.py: all 29 survivors there
# were the same shape -- the default in a str(raw.get(key, "")) mutated to
# None, "" or junk -- for every field of from_dict. Nothing supplied a
# payload missing those keys, so the tolerance its own docstring promises
# ("missing params defaults to {}, unknown keys are ignored") was a claim
# about code no test had exercised.
g = GraphDefinition.from_dict({"format": 1})
check(g.nodes == () and g.edges == (),
      f"an empty payload decodes to an empty graph (got {g.as_dict()})")

g = GraphDefinition.from_dict({"nodes": [{"id": "a"}], "edges": [{}]})
check(len(g.nodes) == 1 and g.nodes[0].id == "a",
      "a node with no class_name still decodes")
check(g.nodes[0].class_name == "",
      f"and the class name defaults to empty, not None or junk "
      f"(got {g.nodes[0].class_name!r})")
check(g.nodes[0].params == {},
      f"and params defaults to an empty dict (got {g.nodes[0].params!r})")
check(len(g.edges) == 1 and g.edges[0].from_node == "",
      "an edge with every field absent decodes to empty strings")
check(all(getattr(g.edges[0], f) == "" for f in
          ("from_node", "from_port", "to_node", "to_port")),
      "all four edge fields, not just the first")

# A node with no id at all is a different case from a node with no
# class_name, and it is the one that reaches the `str()` call: a payload
# with no `nodes` key never enters the comprehension at all, so an
# empty-payload test does not cover it. `validate()` is what refuses an
# empty id -- the decoder's job is only to decline inventing one.
g = GraphDefinition.from_dict({"nodes": [{"class_name": "SumNode"}]})
check(len(g.nodes) == 1 and g.nodes[0].id == "",
      f"a node with no id decodes to an empty id, not None or junk "
      f"(got {g.nodes[0].id!r})")

g = GraphDefinition.from_dict({
    "format": 1,
    "nodes": [{"id": "a", "class_name": "SumNode", "params": {"b": 2.0},
               "unknown_key": "ignored"}],
    "unknown_top_level": [1, 2, 3],
})
check(g.nodes[0].params == {"b": 2.0},
      f"unknown keys inside a node are ignored, not folded into params "
      f"(got {g.nodes[0].params})")
check(len(g.nodes) == 1,
      "and an unknown top-level key does not become a node")

check(GraphDefinition.from_dict(VALID.as_dict()) == VALID,
      "a round trip through as_dict/from_dict is lossless, which is the "
      "contract that actually matters -- it is how saved graphs load")

# ==========================================================================
# Section: a retried write is neither lost nor doubled
# ==========================================================================
print("-- a failed write is retried, and the retry is idempotent --")

# The supervisor retries a node result whose write failed, because a record
# that fails to persist is gone rather than deferred: `tail.poll()` advances
# its offset as it hands a batch over, so the record exists only in that
# frame and the watcher cannot re-read it. Measured before the retry, one
# transient "database is locked" cost the run that step's result silently,
# and the row still finished normally.
#
# Retrying a write is only safe if a failed write changed nothing -- and
# "the write failed" does not mean that here. `ExecutionLifecycleWriter.
# commit` compare-and-swaps the row and *then* publishes, so a failure in
# the second half leaves the first half done. A blind retry then re-reads
# the row, finds the result already there, and appends a second copy.
# Measured against this repository with one injected post-swap failure:
# 12 nodes, 35 stored results, 12 distinct.
#
# Real SQLite and the real lifecycle writer, because the safety of the
# retry depends on `get` being uncached -- a property of the repository,
# not of the supervisor, and a double that cached would hide it.

def _retry_case(where: str) -> list[str]:
    """Store N results with one commit failure before or after the swap."""
    case_db = SqliteDatabase(
        Path(tempfile.mkdtemp(prefix=f"backend-retry-{where}-")) / "g.db"
    )
    case_db.initialize()
    case_repo = SqliteGraphExecutionRepository(case_db)
    clock = FakeClock()
    # A real publisher: the supervisor announces each node's progress after
    # storing it, and a None there would be logged as an exception on every
    # single call -- burying the output this section exists to produce.
    from backend.application.event_publisher import EventPublisher
    from backend.infrastructure.events.callback_event_bus import CallbackEventBus

    writer = ExecutionLifecycleWriter(
        clock=clock, repository=case_repo,
        events=EventPublisher(events=CallbackEventBus()),
    )

    graph = GraphDefinition(nodes=tuple(
        GraphNodeSpec(id=f"n{i}", class_name="FloatConstantNode", params={})
        for i in range(N_RETRY_NODES)
    ))
    row = GraphExecution.create(graph=graph, created_at=NOW)
    case_repo.add(row)
    row.mark_running(at=NOW)
    case_repo.update_if_status(row, expected=GraphStatus.QUEUED)

    real_commit = writer.commit
    injected = {"done": False}

    def commit(aggregate, *, expected, prior=()):
        if where == "before" and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("database is locked")
        swapped = real_commit(aggregate, expected=expected, prior=prior)
        if where == "after" and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("publish failed after the row was written")
        return swapped

    writer.commit = commit
    supervisor = GraphExecutionSupervisor(
        executions=case_repo, writer=writer, gateway=None,
        events=EventPublisher(events=CallbackEventBus()),
        clock=clock,
        scratch_dir=Path(tempfile.mkdtemp(prefix="backend-retry-scratch-")),
    )
    for i in range(N_RETRY_NODES):
        supervisor._record_node(
            row.id,
            {"node_id": f"n{i}", "ok": True, "outputs": {}, "error": None,
             "duration_ms": 1.0},
        )

    stored = case_repo.get(row.id)
    return [r.node_id for r in stored.results]


N_RETRY_NODES = 12
for _where, _what in (
    ("before", "the swap never happened"),
    ("after", "the swap happened and publishing is what failed"),
):
    _ids = _retry_case(_where)
    check(len(_ids) == N_RETRY_NODES,
          f"failure {_where} the swap: all {N_RETRY_NODES} results stored "
          f"({len(_ids)} stored)")
    check(len(set(_ids)) == len(_ids),
          f"failure {_where} the swap: none stored twice "
          f"({len(_ids) - len(set(_ids))} duplicate(s))")


# ==========================================================================
# Section: the scratch sweep keeps what is live and takes what is not
# ==========================================================================
print("-- scratch sweep: terminal or orphaned, and nothing else --")

# The safety half of round-3 N3-07, and the half that is easy to get
# backwards. Deleting the scratch of a run that is *going* would leave a
# watcher with nothing to drain and a restarted server with nothing to
# adopt from, which is the failure the event file exists to prevent. The
# API-level test covers the deletion; this covers the refusal.

sweep_db = SqliteDatabase(
    Path(tempfile.mkdtemp(prefix="backend-sweep-")) / "g.db"
)
sweep_db.initialize()
sweep_repo = SqliteGraphExecutionRepository(sweep_db)
sweep_dir = Path(tempfile.mkdtemp(prefix="backend-sweep-scratch-"))


def _write_scratch(run_id: int) -> list[Path]:
    """The three files the supervisor writes for one execution."""
    out = []
    for suffix, body in (
        (".graph.json", b'{"format": 1, "nodes": [], "edges": []}'),
        (".events.jsonl", b'{"kind": "node", "node_id": "a", "ok": true}\n'),
        (".log", b"child log\n"),
    ):
        path = sweep_dir / f"execution_{run_id}{suffix}"
        path.write_bytes(body)
        out.append(path)
    return out


# A running row keeps its scratch, even when the sweep is asked directly.
live = GraphExecution.create(graph=VALID, created_at=NOW)
sweep_repo.add(live)
live.mark_running(at=NOW)
sweep_repo.update_if_status(live, expected=GraphStatus.QUEUED)
live_files = _write_scratch(1)

kept_result = SweepExecutionScratch(sweep_repo, sweep_dir).execute()
check(all(path.exists() for path in live_files),
      f"a running execution keeps all three of its files "
      f"(missing {[p.name for p in live_files if not p.exists()]})")
check(kept_result.files == 0 and kept_result.runs == 0 and kept_result.kept == 1,
      f"and the sweep reports having kept it rather than removed it "
      f"({kept_result})")

# Finished: the results and the outcome are in the row, so the files are
# read by nobody.
done = sweep_repo.get(live.id)
done.mark_finished(at=NOW)
sweep_repo.update_if_status(done, expected=GraphStatus.RUNNING)

freed = SweepExecutionScratch(sweep_repo, sweep_dir).execute()
check(all(not path.exists() for path in live_files),
      f"a finished execution's files are all removed "
      f"(left {[p.name for p in live_files if p.exists()]})")
check(freed.runs == 1 and freed.files == 3 and freed.bytes > 0,
      f"and the sweep says what it freed, in files and bytes ({freed})")

# Orphaned: no row at all, which is what a replaced database leaves. These
# have no way to be recognised as anything else, ever, by anyone.
orphan_files = _write_scratch(2)
orphan = SweepExecutionScratch(sweep_repo, sweep_dir).execute()
check(all(not path.exists() for path in orphan_files),
      "a file whose row no longer exists is removed too -- it is the case "
      "that otherwise survives forever")
check(orphan.files == 3, f"and counted ({orphan.files})")

# Not ours. The sweep matches three names it writes itself; anything else in
# the directory is somebody's, and a name that merely looks close is not a
# reason to delete a file.
stranger = sweep_dir / "execution_1.events.jsonl.bak"
stranger.write_bytes(b"not ours\n")
also_not_ours = sweep_dir / "notes.txt"
also_not_ours.write_bytes(b"not ours\n")
SweepExecutionScratch(sweep_repo, sweep_dir).execute()
check(stranger.exists() and also_not_ours.exists(),
      "files whose names are not one of the three the supervisor writes are "
      "left alone")

# And a directory that has never run anything is not an error, and does not
# get created just to be swept.
absent = Path(tempfile.mkdtemp(prefix="backend-sweep-none-")) / "never"
never = SweepExecutionScratch(sweep_repo, absent).execute()
check(never.files == 0 and not absent.exists(),
      f"a scratch directory that does not exist is reported as nothing to "
      f"do, and is not conjured into being ({never})")



# ==========================================================================
print("\n-- R4-02: a crashed run's evidence must survive the sweep --")
# The first version of the sweep deleted all three files for every terminal
# run. For a child that died without writing an outcome, the row said
# "exited without reporting an outcome (crashed, or a device fault killed
# it) -- see the execution log", and the next server start deleted the
# execution log. The one place a user could look for the traceback was
# removed by the code tidying up.

crash_dir = Path(tempfile.mkdtemp(prefix="backend-sweep-crash-"))
crash_db = SqliteDatabase(crash_dir / "g.db")
crash_db.initialize()
crash_repo = SqliteGraphExecutionRepository(crash_db)


def _write_scratch_into(directory: Path, run_id: int,
                        log_body: bytes = b"child log\n") -> list[Path]:
    out = []
    for suffix, body in (
        (".graph.json", b'{"format": 1, "nodes": [], "edges": []}'),
        (".events.jsonl", b'{"kind": "node", "node_id": "a", "ok": true}\n'),
        (".log", log_body),
    ):
        path = directory / f"execution_{run_id}{suffix}"
        path.write_bytes(body)
        out.append(path)
    return out


def _crash_scratch(run_id: int, log_body: bytes = b"child log\n") -> list[Path]:
    out = []
    for suffix, body in (
        (".graph.json", b'{"format": 1, "nodes": [], "edges": []}'),
        (".events.jsonl", b'{"kind": "node", "node_id": "a", "ok": true}\n'),
        (".log", log_body),
    ):
        path = crash_dir / f"execution_{run_id}{suffix}"
        path.write_bytes(body)
        out.append(path)
    return out


crashed = GraphExecution.create(graph=VALID, created_at=NOW)
crash_repo.add(crashed)
crashed.mark_running(at=NOW)
crash_repo.update_if_status(crashed, expected=GraphStatus.QUEUED)
failed_files = _crash_scratch(1)
failed_row = crash_repo.get(crashed.id)
failed_row.mark_failed(at=NOW, error="exited without reporting an outcome")
crash_repo.update_if_status(failed_row, expected=GraphStatus.RUNNING)

sweep = SweepExecutionScratch(crash_repo, crash_dir).execute()
graph_path, events_path, log_path = failed_files
check(not graph_path.exists(),
      "a failed run's graph.json is removed -- 38% of its bytes and read "
      "by nobody once the row is terminal")
check(not events_path.exists(),
      "and so is its events.jsonl, which is the other 62%")
check(log_path.exists(),
      f"but its log survives a sweep ({log_path.name}) -- that log is the "
      f"only place the traceback of a crash was")

# The headline case: 25 failed runs, only the newest 20 logs remain. The
# count is on purpose -- these logs are tens of bytes, so the bound that
# matters is the number of directories, not the bytes.
bulk_dir = Path(tempfile.mkdtemp(prefix="backend-sweep-bulk-"))
bulk_db = SqliteDatabase(bulk_dir / "g.db")
bulk_db.initialize()
bulk_repo = SqliteGraphExecutionRepository(bulk_db)
bulk_logs = []
for run_id in range(1, 26):
    row = GraphExecution.create(graph=VALID, created_at=NOW)
    bulk_repo.add(row)
    row.mark_running(at=NOW)
    bulk_repo.update_if_status(row, expected=GraphStatus.QUEUED)
    row = bulk_repo.get(row.id)
    row.mark_failed(at=NOW, error="boom")
    bulk_repo.update_if_status(row, expected=GraphStatus.RUNNING)
    bulk_logs.append(_write_scratch_into(bulk_dir, run_id))

kept_bulk = SweepExecutionScratch(bulk_repo, bulk_dir, failed_log_keep=20).execute()
surviving = [p for p in bulk_logs if p[2].exists()]
check(len(surviving) == 20,
      f"25 failed runs keep the newest 20 logs ({len(surviving)})")
kept_ids = sorted(
    int(p[2].stem.split("_")[1]) for p in surviving
)
check(kept_ids == list(range(6, 26)),
      f"and they are runs 6..25 -- the newest twenty, not whichever sorted "
      f"first ({kept_ids[:4]}...{kept_ids[-2:]})")
check(not any(p[0].exists() or p[1].exists() for p in bulk_logs),
      "every graph.json and events.jsonl is gone regardless")

# A zero retention is honoured rather than treated as unlimited.
all_gone = SweepExecutionScratch(
    bulk_repo, bulk_dir, failed_log_keep=0).execute()
check(not any(p[2].exists() for p in bulk_logs),
      f"retention of 0 keeps nothing, rather than keeping all "
      f"({sum(1 for p in bulk_logs if p[2].exists())} left)")

# A log with invalid UTF-8 must not raise anywhere in this path.
binary_dir = Path(tempfile.mkdtemp(prefix="backend-sweep-binary-"))
binary_db = SqliteDatabase(binary_dir / "g.db")
binary_db.initialize()
binary_repo = SqliteGraphExecutionRepository(binary_db)
binary_row = GraphExecution.create(graph=VALID, created_at=NOW)
binary_repo.add(binary_row)
binary_row.mark_running(at=NOW)
binary_repo.update_if_status(binary_row, expected=GraphStatus.QUEUED)
binary_row = binary_repo.get(binary_row.id)
binary_row.mark_failed(at=NOW, error="boom")
binary_repo.update_if_status(binary_row, expected=GraphStatus.RUNNING)
binary_log = binary_dir / "execution_1.log"
binary_log.write_bytes(b"\xff\xfe not utf-8 at all \x80\x81\n")
for suffix in (".graph.json", ".events.jsonl"):
    (binary_dir / f"execution_1{suffix}").write_bytes(b"x\n")
binary_sweep = SweepExecutionScratch(binary_repo, binary_dir).execute()
check(binary_sweep.files == 2 and binary_log.exists(),
      f"a log of invalid UTF-8 is kept and does not raise "
      f"(removed {binary_sweep.files}, log kept {binary_log.exists()})")

# A finished run still loses everything, immediately: it succeeded, so the
# log says nothing the row does not.
ok_dir = Path(tempfile.mkdtemp(prefix="backend-sweep-ok-"))
ok_db = SqliteDatabase(ok_dir / "g.db")
ok_db.initialize()
ok_repo = SqliteGraphExecutionRepository(ok_db)
ok_row = GraphExecution.create(graph=VALID, created_at=NOW)
ok_repo.add(ok_row)
ok_row.mark_running(at=NOW)
ok_repo.update_if_status(ok_row, expected=GraphStatus.QUEUED)
ok_row = ok_repo.get(ok_row.id)
ok_row.mark_finished(at=NOW)
ok_repo.update_if_status(ok_row, expected=GraphStatus.RUNNING)
ok_files = []
for suffix, body in ((".graph.json", b"{}"), (".events.jsonl", b"\n"),
                     (".log", b"all fine\n")):
    path = ok_dir / f"execution_1{suffix}"
    path.write_bytes(body)
    ok_files.append(path)
SweepExecutionScratch(ok_repo, ok_dir).execute()
check(not any(p.exists() for p in ok_files),
      f"a successful run loses all three, log included -- keeping it would "
      f"be keeping nothing ({[p.name for p in ok_files if p.exists()]})")
# ==========================================================================
print("\n-- R4-02: a crashed run's log tail goes into the row --")
# The sweep now keeps a failed run's log, but the row has to stand on its
# own too: the log is a bounded set of the newest failures, and a row whose
# log has aged out still has to say something useful.

tail_dir = Path(tempfile.mkdtemp(prefix="backend-tail-"))
tail_db = SqliteDatabase(tail_dir / "g.db")
tail_db.initialize()
tail_repo = SqliteGraphExecutionRepository(tail_db)
from backend.infrastructure.events.callback_event_bus import (  # noqa: E402
    CallbackEventBus,
)
tail_supervisor = GraphExecutionSupervisor(
    executions=tail_repo, writer=tail_repo, gateway=None,
    events=EventPublisher(events=CallbackEventBus()), clock=FakeClock(),
    scratch_dir=tail_dir,
)

tail_row = GraphExecution.create(graph=VALID, created_at=NOW)
tail_repo.add(tail_row)
tail_row.mark_running(at=NOW)
tail_repo.update_if_status(tail_row, expected=GraphStatus.QUEUED)
tail_row = tail_repo.get(tail_row.id)
tail_paths = tail_supervisor._paths_for(tail_row.id)
tail_paths["log"].write_text(
    "starting\n" + "noise line\n" * 5000
    + "Traceback (most recent call last):\nRuntimeError: the card fell over\n",
    encoding="utf-8",
)

tail = tail_supervisor._with_log_tail(tail_row.id, "exited without an outcome")
check("the card fell over" in tail,
      f"the traceback reaches the row ({tail[-60:]!r})")
check("starting" not in tail,
      f"the *beginning* of the log is dropped -- a row is not a log file, "
      f"and the bound is what stops it becoming megabytes (tail is "
      f"{len(tail)} chars from a "
      f"{tail_paths['log'].stat().st_size if tail_paths['log'].exists() else 0}"
      f"-byte log)")
check("bytes of the execution log" in tail,
      "and it says which log and how much of it, so a reader knows the row "
      "has been truncated rather than complete")

# Read from the end: a crash's traceback is at the bottom.
check(tail.rstrip().endswith("RuntimeError: the card fell over"),
      "the tail is the *end* of the log, not the beginning")

# Missing and unreadable degrade to the plain message. This runs in the
# `finally` of a run that has already failed, so a reader that raised would
# replace a real error with an internal one.
tail_paths["log"].unlink()
check(tail_supervisor._with_log_tail(tail_row.id, "plain") == "plain",
      "a log that is not there degrades to the plain error rather than raising")

tail_paths["log"].write_bytes(b"\xff\xfe\x80 not utf-8 \x81\n")
binary_tail = tail_supervisor._with_log_tail(tail_row.id, "plain")
check(binary_tail != "plain",
      "a log of invalid UTF-8 does not raise either -- a traceback from a "
      "dying process can contain a partial line")

# And the size bound is a real bound.
tail_paths["log"].write_text("x" * (GraphExecutionSupervisor.LOG_TAIL_BYTES * 3),
                             encoding="utf-8")
big = tail_supervisor._with_log_tail(tail_row.id, "plain")
check(len(big) <= GraphExecutionSupervisor.LOG_TAIL_BYTES + 400,
      f"a large log is cut to the bound ({len(big)} chars for a "
      f"{GraphExecutionSupervisor.LOG_TAIL_BYTES * 3}-byte log)")

finish()
