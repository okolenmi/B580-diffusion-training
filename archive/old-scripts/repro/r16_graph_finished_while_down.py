"""N3-03: a graph run whose child FINISHED while the server was down is recorded failed 'server restarted...'
even though its event file holds the remaining node results and a clean outcome record."""
import os, sys, tempfile, importlib.util, datetime, json
from pathlib import Path
sys.path.insert(0, os.environ.get("REPO","."))
os.environ.setdefault("COMFY_DIR", "/tmp/fakecomfy")
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.persistence.graph_execution_repository import SqliteGraphExecutionRepository
from backend.application.lifecycle_writer import ExecutionLifecycleWriter
from backend.application.event_publisher import EventPublisher
from backend.application.use_cases.reconcile_graph_executions import ReconcileGraphExecutions
from backend.infrastructure.graph_event_stream import ExecutionEventWriter, ExecutionEventTail
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.graph import GraphDefinition
from backend.tests.support import FakeClock, RecordingEventBus
spec = importlib.util.spec_from_file_location("T", "backend/tests/test_graph_adoption.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp); db = SqliteDatabase(tmp/"b.db"); db.initialize()
    repo = SqliteGraphExecutionRepository(db); clock = FakeClock(); pub = EventPublisher(events=RecordingEventBus())
    writer = ExecutionLifecycleWriter(clock=clock, repository=repo, events=pub)
    g = GraphDefinition.from_dict({"nodes":[{"id":n,"class_name":"X"} for n in "abc"],"edges":[]})
    row = GraphExecution.create(graph=g, created_at=clock.now()); repo.add(row); row = repo.get(row.id)
    row.mark_running(at=clock.now()); repo.update_if_status(row, expected=row.__class__.__mro__ and __import__("backend.domain.value_objects",fromlist=["GraphStatus"]).GraphStatus.QUEUED)
    # what the child wrote before it exited 0 -- while nobody was watching:
    scratch = tmp/"scratch"; scratch.mkdir(); ev = scratch/f"execution_{row.id}.events.jsonl"
    w = ExecutionEventWriter(ev)
    for n in "abc": w.node({"node_id": n, "ok": True, "outputs": {}, "error": None, "duration_ms": 5})
    w.outcome(error=None, results_count=3)
    gw = T.RecordingGateway(ev, alive=False); gw.found = None                 # no live child: it already exited
    sup = T.GraphExecutionSupervisor(executions=repo, writer=writer, gateway=gw, events=pub, clock=clock,
                                     monitor_bus=None, scratch_dir=scratch, make_tail=ExecutionEventTail)
    res = ReconcileGraphExecutions(executions=repo, writer=writer, launcher=sup, clock=clock).execute()
    r = repo.get(row.id)
    print(f"event file ends with: node a,b,c + outcome(error=None)   ->   row: status={r.status.value} results={len(r.results)}/3 error={r.error!r:.70}")
    print(f"reconcile result: adopted={res.adopted} cleaned={res.cleaned}")
