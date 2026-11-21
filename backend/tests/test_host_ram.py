"""Host memory: ``memory.ram_max_mb`` (a run's stated RAM ceiling) is read, checked at admission and enforced by the child.

  A  reading the machine: /proc text for the machine's RAM and for one process, what is left out and what is unknown
  B  admission: a stated ceiling is refused or planned by what the machine can give; ``auto`` is never checked
  C  through the start use case: a refusal writes no row; live runs' ceilings count as promised
  D  the child: the guard asks for a stop once, an unreadable process is not over, the stop becomes the outcome's error
  E  the launch: the ceiling reaches the worker's command line

Run directly: python backend/tests/test_host_ram.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import MemoryUnavailableError
from backend.application.execution_admission import ExecutionAdmission
from backend.application.host_ram_admission import HOST_RESERVE_MB, HostRamAdmission
from backend.application.ports.graph_task_gateway import GraphTaskLaunch
from backend.application.ports.host_memory import HostMemory, HostRam
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.memory_settings import EffectiveMemory
from backend.infrastructure import graph_task_worker as worker
from backend.infrastructure.graph_task_gateway import SubprocessGraphTaskGateway
from backend.infrastructure.host_memory import parse_held_mb, parse_meminfo
from backend.infrastructure.host_ram_guard import HostRamGuard
from backend.tests.support import build_services, gdef, gedge, gnode, wait_until

VALID = gdef(
    (gnode("v", "ScaleNode", value=3.0, factor=5.0), gnode("s", "SumNode", b=1.0)),
    (gedge("v", "scaled", "s", "a"),),
)

PASSED = FAILED = 0


def check(condition: bool, label: str) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL: {label}")


def raises(exc, fn, label: str):
    try:
        fn()
    except exc as err:
        check(True, label)
        return err
    check(False, f"{label} (nothing was raised)")
    return None


class FakeHost(HostMemory):
    def __init__(self, total_mb: float | None, available_mb: float = 0.0) -> None:
        self.ram = None if total_mb is None else HostRam(total_mb, available_mb)
        self.reads = 0

    def snapshot(self):
        self.reads += 1
        return self.ram


# ---- A ---------------------------------------------------------------------
print("-- A: reading --")
MEMINFO = "MemTotal:       32768000 kB\nMemFree:         1000000 kB\nMemAvailable:   16384000 kB\nBuffers: 5 kB\n"
ram = parse_meminfo(MEMINFO)
check(ram is not None and round(ram.total_mb) == 32000 and round(ram.available_mb) == 16000, "the machine's total and available RAM come from /proc/meminfo")
check(parse_meminfo("MemTotal: 100 kB\n") is None and parse_meminfo("") is None, "a reading without MemAvailable is unknown, not zero")
STATUS = "Name:\tpython\nVmRSS:\t 9000000 kB\nRssAnon:\t 3072000 kB\nRssFile:\t 5000000 kB\nRssShmem:\t 1024000 kB\n"
check(round(parse_held_mb(STATUS)) == 4000, "a process holds its anonymous and shared memory; mapped files (reclaimable) are left out")
check(round(parse_held_mb("VmRSS:\t 2048000 kB\n")) == 2000, "a kernel that does not split the resident set gives the whole of it")
check(parse_held_mb("Name: x\n") is None, "nothing readable is unknown")

# ---- B ---------------------------------------------------------------------
print("-- B: admission --")
none_promised = lambda: {}
HostRamAdmission(FakeHost(32000, 20000), none_promised).check(16000)
check(True, "a ceiling the machine can give passes")
err = raises(MemoryUnavailableError, lambda: HostRamAdmission(FakeHost(32000, 8000), none_promised).check(16000), "a ceiling above what is available now is refused")
check(err is not None and err.details["reason"] == "host_memory" and err.details["holders"] == {} and "8000" in str(err), "the refusal names the reason and the numbers, and nobody holds anything to wait for")
err = raises(MemoryUnavailableError, lambda: HostRamAdmission(FakeHost(32000, 30000), lambda: {"graph:1": 20000.0}).check(16000), "the total less the reserve less what running graphs were promised also limits")
check(err is not None and err.details["holders"] == {"graph:1": 20000.0} and err.details["promised_mb"] == 20000.0, "with promises held the refusal is worth waiting for and lists who holds them")
err = raises(MemoryUnavailableError, lambda: HostRamAdmission(FakeHost(16000, 15000), lambda: {"graph:1": 2000.0}).check(16000), "a ceiling above the machine's whole total can never be given")
check(err is not None and err.details["holders"] == {}, "so waiting for the running graphs would not help")
HostRamAdmission(FakeHost(32000, 20000), none_promised).check(32000 - HOST_RESERVE_MB - 12000 + 0.0)
HostRamAdmission(FakeHost(None), none_promised).check(10**9)
check(True, "a machine that cannot report its RAM is not checked")

ledgerless = ExecutionAdmission(ledger=lambda: None, peak_source=lambda g: None, host_ram=HostRamAdmission(host := FakeHost(32000, 1000), none_promised))
settings = dict(vram_min_mb=0.0, vram_max_mb="auto", strict=True, policy="demand_driven", demand_mb=None, demand_source="unknown", fingerprint_key=None)
raises(MemoryUnavailableError, lambda: ledgerless.claim(EffectiveMemory(ram_max_mb=16000.0, **settings)), "claim refuses a stated ceiling the machine cannot give")
before = host.reads
try:
    ledgerless.claim(EffectiveMemory(ram_max_mb="auto", **settings))
except MemoryUnavailableError as refusal:
    check(refusal.details.get("reason") != "host_memory", "auto goes on to the card's admission")
check(host.reads == before, "auto never reads the host: it has no ceiling of its own")

# ---- C ---------------------------------------------------------------------
print("-- C: the start use case --")
from backend.infrastructure.persistence.graph_execution_repository import SqliteGraphExecutionRepository
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from datetime import datetime, UTC

root = Path(tempfile.mkdtemp(prefix="host-ram-"))
db = SqliteDatabase(root / "graphs.db")
db.initialize()
repo = SqliteGraphExecutionRepository(db)
host = FakeHost(32000, 4000)
graphs = build_services(project_root=root, graph_executions=repo, host_memory=host).graphs
raises(MemoryUnavailableError, lambda: graphs.start_execution.execute(VALID, memory_overrides={"ram_max_mb": 16000}, mode="parallel"),
       "a run whose ceiling the machine cannot give is refused at the start")
raises(MemoryUnavailableError, lambda: graphs.start_execution.execute(VALID, memory_overrides={"ram_max_mb": 16000}, mode="auto"),
       "auto mode does not plan it either: nothing running will give the memory back")
check(repo.list_executions() == [], "a refusal writes no row")

host.ram = HostRam(32000, 30000)
ok = graphs.start_execution.execute(VALID, memory_overrides={"ram_max_mb": 16000})
check(ok.status != "planned" and ok.execution_id, "the same run starts when the machine can give it")
check(wait_until(lambda: repo.get(ok.execution_id).status.is_terminal, timeout=15.0), "the run ends")
check(graphs.start_execution._promised_ram() == {}, "an ended run promises nothing")

# a live run that was promised 20000 MB leaves the machine 32000 - 1024 - 20000 for the next one
live = GraphExecution.plan(graph=VALID, created_at=datetime(2026, 3, 1, tzinfo=UTC),
                           memory=EffectiveMemory(ram_max_mb=20000.0, **settings), name=None, origin=None)
live = repo.add(live)
live.promote(at=datetime(2026, 3, 1, tzinfo=UTC), memory=live.memory, reserved_mb=0.0)
repo.update(live)
promised = graphs.start_execution._promised_ram()
check(promised == {f"graph:{live.id}": 20000.0}, f"a live run's ceiling is promised under its claim owner ({promised})")
waiting = graphs.start_execution.execute(VALID, memory_overrides={"ram_max_mb": 16000}, mode="auto")
check(waiting.status == "planned", f"with the promise held, a run that would not fit beside it waits (got {waiting.status})")
check(graphs.start_execution.execute(VALID, memory_overrides={"ram_max_mb": 8000}, mode="parallel").status != "planned", "one that fits beside it starts")

# ---- D ---------------------------------------------------------------------
print("-- D: the child --")
cancel = threading.Event()
readings = iter([1000.0, 2500.0, 9000.0])
guard = HostRamGuard(2000.0, cancel, read=lambda: next(readings))
guard.check()
check(guard.breach is None and not cancel.is_set(), "below the ceiling nothing happens")
guard.check()
check(guard.breach is not None and cancel.is_set() and "2500" in guard.breach and "2000" in guard.breach, "over it: the run is asked to stop and the reason names both numbers")
first = guard.breach
guard.check()
check(guard.breach == first, "the first breach is the reason that stays")
silent_cancel = threading.Event()
unreadable = HostRamGuard(1.0, silent_cancel, read=lambda: None)
unreadable.check()
check(unreadable.breach is None and not silent_cancel.is_set(), "a process that cannot be read is not over its ceiling")
raises(ValueError, lambda: HostRamGuard(0.0, threading.Event()), "a ceiling of zero is not a ceiling")


class FakeDevice:
    def memory_stats(self):
        return None  # CPU: the device has no frames; the host check must not depend on them


class Recorder:
    def __init__(self):
        self.outcomes = []

    def memory(self, **_):
        pass

    def node(self, _):
        pass

    def outcome(self, **kw):
        self.outcomes.append(kw)


cancel = threading.Event()
guard = HostRamGuard(500.0, cancel, read=lambda: 900.0)
with worker.MemoryTelemetry(FakeDevice(), Recorder(), SimpleNamespace(budget_mb=None), "unavailable", cancel, interval_s=0.02, guard=guard):
    check(cancel.wait(2.0), "the telemetry thread stops a run over its ceiling even when the device reports nothing")

class StoppedRuntime:
    def execute(self, graph, cancel_event, on_node_done):
        return SimpleNamespace(error=None, results=[])

recorder = Recorder()
worker.run_execution(None, recorder, threading.Event(), StoppedRuntime(), guard)
check(len(recorder.outcomes) == 1 and recorder.outcomes[0]["error"] == guard.breach, "the stop is written as the outcome's error, not as a quiet stop")
recorder = Recorder()
worker.run_execution(None, recorder, threading.Event(), StoppedRuntime(), None)
check(recorder.outcomes[0]["error"] is None, "without a breach the outcome is unchanged")
check(worker._build_parser().parse_args(["--execution", "1", "--graph", "g", "--events", "e", "--ram-max-mb", "4096"]).ram_max_mb == 4096.0, "the worker takes the ceiling")
check(worker._build_parser().parse_args(["--execution", "1", "--graph", "g", "--events", "e"]).ram_max_mb is None, "and none without it")

# ---- E ---------------------------------------------------------------------
print("-- E: the launch --")
launch = lambda **kw: GraphTaskLaunch(execution_id=1, graph_path=Path("g"), event_path=Path("e"), log_path=Path("l"), **kw)
gateway = SubprocessGraphTaskGateway(SimpleNamespace(venv_python="python", project_root=Path(".")))
command = gateway._build_command(launch(ram_max_mb=2048.0))
check(command[command.index("--ram-max-mb") + 1] == "2048.0", "a stated ceiling is on the worker's command line")
check("--ram-max-mb" not in gateway._build_command(launch()), "and absent when there is none")

print(f"\n{PASSED} checks passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
