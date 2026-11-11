"""Parallel and planned runs: memory admission is the only gate, `mode` says what a start does when there is no room,
and planned rows start in order when room appears. Run as `python backend/tests/test_parallel_runs.py`."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import GraphInvalidError, MemoryUnavailableError
from backend.domain.value_objects import GraphStatus
from backend.infrastructure.graph.runtime import ContractGraphRuntime
from backend.infrastructure.persistence.graph_execution_repository import SqliteGraphExecutionRepository
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.planned_runs_pump import PlannedRunsPump
from backend.tests.support import build_services, check, finish, fixture_tree_registry, gdef, gnode, wait_until

BUDGET = {"vram_max_mb": 4096.0, "strict": False}  # two fit beside each other on the 11,192 MB test ledger, three do not


def slow(seconds: float):
    return gdef([gnode("s", "SlowNode", seconds=seconds)])


db = SqliteDatabase(Path(tempfile.mkdtemp(prefix="backend-parallel-")) / "p.db")
db.initialize()
repo = SqliteGraphExecutionRepository(db)
services = build_services(graph_runtime=ContractGraphRuntime(fixture_tree_registry()), graph_executions=repo)
start = services.graphs.start_execution
stop = services.graphs.stop_execution


def status_of(execution_id: int) -> GraphStatus:
    return repo.get(execution_id).status


print("-- two runs side by side --")
a = start.execute(slow(30), memory_overrides=BUDGET, mode="parallel")
b = start.execute(slow(30), memory_overrides=BUDGET, mode="parallel")
check(a.status is GraphStatus.QUEUED and b.status is GraphStatus.QUEUED, "a second run is admitted beside the first")
check(repo.get(a.execution_id).reserved_mb and repo.get(b.execution_id).reserved_mb, "each holds its own claim")

print("-- no room: what each mode does --")
try:
    start.execute(slow(30), memory_overrides=BUDGET, mode="parallel")
    check(False, "parallel refuses a third run")
except MemoryUnavailableError:
    check(True, "parallel refuses a third run with the breakdown")
check(not repo.list_planned(), "a refusal leaves no row behind")
c = start.execute(slow(0.2), memory_overrides=BUDGET, mode="auto")
d = start.execute(slow(0.2), memory_overrides=BUDGET, mode="queue")
check(c.status is GraphStatus.PLANNED and d.status is GraphStatus.PLANNED, "auto and queue plan a run that does not fit")
check(repo.get(c.execution_id).reserved_mb is None and repo.get(c.execution_id).memory.vram_max_mb == 4096.0,
      "a planned row holds nothing but remembers its budget")
check([e.id for e in repo.list_planned()] == [c.execution_id, d.execution_id], "planned rows wait in order of creation")
check(start.execute(slow(0.2), memory_overrides={"vram_max_mb": 512.0, "strict": False}, mode="auto").status is GraphStatus.PLANNED,
      "a small run does not overtake the line")
try:
    start.execute(gdef([gnode("x", "NoSuchNode")]), mode="auto")
    check(False, "an invalid graph is refused before anything is planned")
except GraphInvalidError:
    check(len(repo.list_planned()) == 3, "an invalid graph is refused before anything is planned")

print("-- cancel a planned run --")
cancelled = stop.execute(d.execution_id)
check(cancelled.status is GraphStatus.STOPPED and not [e for e in repo.list_planned() if e.id == d.execution_id],
      "a planned run can be cancelled and leaves the line")

print("-- planned rows start when room appears --")
pump = PlannedRunsPump(start.promote_planned)
check(pump.tick() == 0, "nothing starts while both holders run")
stop.execute(a.execution_id)
check(wait_until(lambda: pump.tick() > 0 or status_of(c.execution_id) is not GraphStatus.PLANNED, timeout=10.0),
      "the oldest planned run starts once room is freed")
check(repo.get(c.execution_id).reserved_mb is not None, "its claim is stored when it starts")
check(wait_until(lambda: (pump.tick(), not repo.list_planned())[1], timeout=10.0), "the small run behind it starts too")
stop.execute(b.execution_id)

print("-- queue mode --")
check(wait_until(lambda: repo.find_active() is None, timeout=15.0), "fixture: everything has ended")
q = start.execute(slow(0.1), memory_overrides=BUDGET, mode="queue")
check(q.status is GraphStatus.QUEUED, "queue mode starts at once when nothing else lives or waits")

finish()
