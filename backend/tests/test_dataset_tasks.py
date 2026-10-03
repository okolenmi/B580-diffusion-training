"""Dataset task lifecycle tests -- SqliteDatasetTasks CAS, start/stop,
reconcile, and the lazy dead-row sweep (fake gateway, real SQLite).

Run directly: python backend/tests/test_dataset_tasks.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.dataset_task_sweeper import DatasetTaskSweeper
from backend.application.dto import StartDatasetTaskCommand
from backend.application.errors import (
    DatasetNotFoundError,
    DatasetTaskActiveError,
    DatasetTaskLaunchError,
    DatasetTaskNotActiveError,
    DatasetTaskNotFoundError,
    InvalidQueryError,
)
from backend.application.ports.dataset_tasks import TaskKind, TaskStatus
from backend.application.use_cases import (
    ListDatasetTasks,
    ReconcileDatasetTasks,
    StartDatasetTask,
    StopDatasetTask,
)
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.dataset_tasks import SqliteDatasetTasks
from backend.infrastructure.persistence.cas import compare_and_swap_status
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.workspace import WorkspaceLayout
from backend.tests.support import (
    FakeClock,
    FakeDatasetTaskGateway,
    check,
    finish,
    make_v2_dataset,
)

root = Path(tempfile.mkdtemp(prefix="backend-dataset-tasks-"))
layout = WorkspaceLayout(root)
library = SqliteDatasetLibrary(layout)
database = SqliteDatabase(root / "backend.db")
database.initialize()
clock = FakeClock()
repo = SqliteDatasetTasks(database, clock)
gateway = FakeDatasetTaskGateway()


def expect(exc_type, fn, label):
    try:
        fn()
    except exc_type:
        check(True, label)
    else:
        check(False, f"{label} (expected {exc_type.__name__})")


# -- repository: add / CAS transitions --------------------------------------

task = repo.add(dataset="d", kind=TaskKind.INGEST_LORA, total=10, params={"seed": 42})
check(task.status == "pending" and task.pid is None, "add creates a pending row")
check(task.total == 10 and task.params == {"seed": 42}, "payload round-trips")
check(repo.find_active("d").id == task.id, "find_active sees it")
check(len(repo.list_unfinished()) == 1, "list_unfinished sees it")

check(repo.update_progress(task.id, 3, pid=99) is True, "progress CAS wins")
running = repo.get(task.id)
check(running.status == "running" and running.current == 3 and running.pid == 99,
      "progress writes status/count/pid")
check(repo.finalize_if_active(task.id, TaskStatus.FINISHED) is True,
      "finish CAS wins")
check(repo.finalize_if_active(task.id, TaskStatus.FINISHED) is False,
      "second finish loses")
check(repo.finalize_if_active(task.id, TaskStatus.FAILED, error="late") is False,
      "fail after finish loses")
check(repo.finalize_if_active(task.id, TaskStatus.KILLED) is False,
      "kill after finish loses")
check(repo.find_active("d") is None, "terminal row is not active")
check(len(repo.list_for("d", active_only=True)) == 0, "active_only hides it")
check(len(repo.list_for("d")) == 1, "history kept")

t2 = repo.add(dataset="d", kind=TaskKind.INGEST_LORA, total=5, params={})
check(repo.finalize_if_active(t2.id, TaskStatus.KILLED) is True,
      "kill wins on pending")
check(repo.update_progress(t2.id, 1, pid=7) is False, "progress cannot resurrect")
check(repo.get(t2.id).status == "killed", "row stays killed")

# -- StartDatasetTask --------------------------------------------------------

ckpt = root / "ckpt"
ckpt.mkdir()
(ckpt / "model.safetensors").write_bytes(b"st")
imgs = root / "images"
imgs.mkdir()
for i in range(3):
    (imgs / f"{i}.png").write_bytes(b"png")
make_v2_dataset(root, "work")

start = StartDatasetTask(
    library=library, tasks=repo, gateway=gateway,
    checkpoints_dir=lambda: ckpt
)
stop = StopDatasetTask(tasks=repo, gateway=gateway)
command = StartDatasetTaskCommand(
    dataset="work", image_dir=str(imgs), model="model.safetensors"
)

first = start.execute(command)
check(first.status == "running" and first.pid == 7777,
      "row records pid immediately after spawn")
check(first.total == 3, "total counts images by the legacy rule")
check(len(gateway.spawned) == 1, "gateway spawned once")
launch = gateway.spawned[0]
check(launch.kind == TaskKind.INGEST_LORA, "launch carries kind")
check(launch.params["image_dir"] == str(imgs), "launch carries image_dir")
check(launch.params["model"] == str(ckpt / "model.safetensors"),
      "model resolved to an absolute path inside checkpoints")
check(launch.dataset_root.name == "work", "launch carries dataset root")

expect(DatasetTaskActiveError, lambda: start.execute(command),
       "second start blocked while active")

ended = stop.execute(first.id)
check(ended.status == "killed", "stop flips row to killed")
check(gateway.killed == [first.pid], "stop signals the recorded pid")

expect(DatasetTaskNotActiveError, lambda: stop.execute(first.id),
       "stop of a terminal task refused")
expect(DatasetTaskNotFoundError, lambda: stop.execute(424242),
       "stop of an unknown id refused")

# spawn failure: row created, failed, nothing left active; the error
# still surfaces to the caller (500 in the API, assertion here).
gateway.spawn_error = DatasetTaskLaunchError("boom: no venv")
expect(DatasetTaskLaunchError, lambda: start.execute(command),
       "launch failure re-raises to the caller")
second = repo.list_for("work")[0]  # newest row
check(second.status == "failed" and "boom" in (second.error or ""),
      "launch failure lands in the row")
check(repo.find_active("work") is None, "launch failure leaves nothing active")
gateway.spawn_error = None

# input validation (nothing spawned, no rows)
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", kind="teacher", image_dir=str(imgs),
           model="model.safetensors")),
       "unknown kind refused")
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", image_dir="relative/path",
           model="model.safetensors")),
       "relative image_dir refused")
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", image_dir=str(root / "missing"),
           model="model.safetensors")),
       "missing image_dir refused")
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", image_dir=str(imgs),
           model="../escape.safetensors")),
       "model traversal refused")
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", image_dir=str(imgs), model="missing.safetensors")),
       "missing model refused")
empty = root / "empty-images"
empty.mkdir()
expect(InvalidQueryError,
       lambda: start.execute(StartDatasetTaskCommand(
           dataset="work", image_dir=str(empty),
           model="model.safetensors")),
       "zero images refused")
check(len(gateway.spawned) == 1, "no spawns from rejected commands")

# -- ReconcileDatasetTasks ---------------------------------------------------

dead = repo.add(dataset="rec", kind=TaskKind.INGEST_LORA, total=9, params={})
repo.update_progress(dead.id, 4, pid=8888)  # gateway does not know 8888
zombie = repo.add(dataset="rec", kind=TaskKind.INGEST_LORA, total=9, params={})
repo.update_progress(zombie.id, 1, pid=gateway.next_pid)
gateway.alive.add(gateway.next_pid)
gateway.next_pid += 1
never = repo.add(dataset="rec", kind=TaskKind.INGEST_LORA, total=9, params={})  # pending, no pid

reconciled = ReconcileDatasetTasks(
    sweeper=DatasetTaskSweeper(tasks=repo, gateway=gateway, clock=clock)
).execute()
check(reconciled.cleaned == 2, "reconcile fails dead-pid and never-started rows")
check(repo.get(dead.id).status == "failed", "dead running row failed")
check(repo.get(never.id).status == "failed", "pending row failed")
check(repo.get(zombie.id).status == "running", "genuinely alive row kept")
check("reconciled" in (repo.get(dead.id).error or ""), "reconcile note recorded")

# -- ListDatasetTasks sweep + filters ----------------------------------------

make_v2_dataset(root, "sweep")
listing = ListDatasetTasks(library=library, tasks=repo)
sweeper = DatasetTaskSweeper(tasks=repo, gateway=gateway, clock=clock)

gone = repo.add(dataset="sweep", kind=TaskKind.INGEST_LORA, total=1, params={})
repo.update_progress(gone.id, 1, pid=6666)  # dies without reporting
fresh = repo.add(dataset="sweep", kind=TaskKind.INGEST_LORA, total=1, params={})  # just spawned

# A read is a read: listing rows must not rewrite any row, least of all
# one belonging to another dataset (docs 08 S-03).
result = listing.execute("sweep")
check([t.id for t in result.tasks] == [fresh.id, gone.id],
      "both active rows are listed, newest first")
check(repo.get(gone.id).status == "running", "listing swept nothing")
check(repo.get(fresh.id).status == "pending", "nor the pending row")

# The sweeper is the one definition of dead, and start/reconcile call it.
check(sweeper.sweep() == 1, "the sweeper fails only the dead-pid row")
check(repo.get(gone.id).status == "failed", "dead running row swept")
check("not running" in (repo.get(gone.id).error or ""), "death reason recorded")
check(repo.get(fresh.id).status == "pending", "young pending row untouched")

clock.advance(120)
check(sweeper.sweep() == 1, "the stale pending row is swept once it ages out")
check("never reported" in (repo.get(fresh.id).error or ""),
      "age-out reason recorded")
history = listing.execute("sweep", active_only=False)
check([t.id for t in history.tasks] == [fresh.id, gone.id],
      "history returns newest first")
expect(DatasetNotFoundError, lambda: listing.execute("no-such"),
       "listing an unknown dataset refused")

# ---------------------------------------------------------------------------
# The shared compare-and-swap, on its own
# ---------------------------------------------------------------------------
#
# Every terminal transition in the backend is this one statement -- graph
# executions and dataset tasks both go through it (2026-10-02), so its
# two defining properties are pinned here rather than inferred from two
# adapters that each used to have their own copy.

_cas_db = SqliteDatabase(
    Path(tempfile.mkdtemp(prefix="backend-cas-")) / "cas.db"
)
_cas_db.initialize()
with _cas_db.connection() as _conn:
    _conn.execute(
        "CREATE TABLE things (id INTEGER PRIMARY KEY, status TEXT, note TEXT)"
    )
    for _id in (1, 2, 3):
        _conn.execute(
            "INSERT INTO things (id, status) VALUES (?, ?)", (_id, "running")
        )

with _cas_db.connection() as _conn:
    check(
        compare_and_swap_status(
            _conn, table="things", row_id=1,
            from_statuses=("running",), to_status="finished",
        ) is True,
        "a row in an expected status is swapped",
    )
    check(
        compare_and_swap_status(
            _conn, table="things", row_id=1,
            from_statuses=("running",), to_status="failed",
        ) is False,
        "and the second swap from the same expected status loses",
    )
    row = _conn.execute(
        "SELECT status, note FROM things WHERE id = 1"
    ).fetchone()
    check(row[0] == "finished" and row[1] is None,
          f"the winner's status stands and no column it did not set moved "
          f"(got {tuple(row)})")

    check(
        compare_and_swap_status(
            _conn, table="things", row_id=2,
            from_statuses=("pending", "running"), to_status="killed",
            set_columns={"note": "stopped"},
        ) is True,
        "several accepted statuses: any of them wins",
    )
    note = _conn.execute(
        "SELECT note FROM things WHERE id = 2"
    ).fetchone()[0]
    check(note == "stopped", f"and the extra column was written (got {note!r})")

    check(
        compare_and_swap_status(
            _conn, table="things", row_id=3,
            from_statuses=("running",), to_status="finished",
        ) is True,
        "a different row is independent",
    )

    try:
        compare_and_swap_status(
            _conn, table="things", row_id=3, from_statuses=(), to_status="x"
        )
        check(False, "an unguarded swap must be refused")
    except ValueError:
        check(True,
              "an empty expected-status set is refused rather than turning "
              "into an UPDATE that matches everything -- that is a "
              "different operation and almost never what the caller meant")

finish()
