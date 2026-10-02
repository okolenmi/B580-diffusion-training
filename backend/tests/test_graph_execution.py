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
from backend.application.lifecycle_writer import ExecutionLifecycleWriter
from backend.application.errors import (
    GraphExecutionActiveError,
    GraphExecutionNotFoundError,
    GraphExecutionNotActiveError,
    GraphInvalidError,
    GraphNotFoundError,
    InvalidQueryError,
)
from backend.application.use_cases import ReconcileGraphExecutions
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.exceptions import DomainError, InvalidTransitionError
from backend.domain.graph import GraphDefinition, GraphEdgeSpec, GraphNodeSpec, NodeResult
from backend.application.ports.execution_launcher import ExecutionLauncher
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

    def launch(self, execution_id, graph) -> None:
        raise AssertionError("the reconcile tests never launch")

    def cancel(self, execution_id) -> None:
        raise AssertionError("the reconcile tests never cancel")

    def adopt(self, execution_id):
        return None


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

finish()
