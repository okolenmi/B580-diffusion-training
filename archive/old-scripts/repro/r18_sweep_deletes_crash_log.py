"""N4-02: the startup sweep deletes the child's log of a run that CRASHED -- the only place its traceback lives."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.environ.get("REPO","."))
os.environ.setdefault("COMFY_DIR","/tmp/fakecomfy")
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.persistence.graph_execution_repository import SqliteGraphExecutionRepository
from backend.application.use_cases.sweep_execution_scratch import SweepExecutionScratch
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.graph import GraphDefinition
from backend.domain.value_objects import GraphStatus
from backend.tests.support import FakeClock
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); db=SqliteDatabase(tmp/"b.db"); db.initialize(); repo=SqliteGraphExecutionRepository(db); clock=FakeClock()
    g=GraphDefinition.from_dict({"nodes":[{"id":"a","class_name":"X"}],"edges":[]})
    row=GraphExecution.create(graph=g, created_at=clock.now()); repo.add(row); row=repo.get(row.id)
    row.mark_running(at=clock.now()); repo.update_if_status(row, expected=GraphStatus.QUEUED)
    row.mark_failed(at=clock.now(), error="execution process exited without reporting an outcome (crashed, or a device fault killed it)")
    repo.update_if_status(row, expected=GraphStatus.RUNNING)
    scratch=tmp/"scratch"; scratch.mkdir()
    log=scratch/f"execution_{row.id}.log"; log.write_text("Traceback (most recent call last):\n ... RuntimeError: UR_RESULT_ERROR_DEVICE_LOST\n")
    (scratch/f"execution_{row.id}.events.jsonl").write_text("")
    print("before sweep: log exists =", log.exists(), "| row.status =", repo.get(row.id).status.value, "| row.error =", repo.get(row.id).error[:60])
    SweepExecutionScratch(repo, scratch).execute()
    print("after  sweep: log exists =", log.exists(), "<- the only record of WHY it crashed" if not log.exists() else "")
