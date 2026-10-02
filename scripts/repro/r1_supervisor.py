"""Repros against the REAL RunSupervisor + JsonlProgressSource (+fakes for gateway/repo)."""
import sys, json, tempfile, time, logging
from pathlib import Path
sys.path.insert(0, __import__("os").environ.get("REPO","."))
from backend.application.dto import StartTrainingCommand
from backend.application.errors import RunAlreadyActiveError
from backend.tests.support import (FakeClock, FakeConfigInspector, FakeTrainingGateway,
    InMemoryRunRepository, RecordingEventBus, build_services, wait_until)

def env(tmp, **kw):
    project = Path(tmp)/"project"; (project/"configs").mkdir(parents=True)
    (project/"configs"/"t.toml").write_text("[common]\nsteps = 100\n")
    repo = InMemoryRunRepository(); gw = FakeTrainingGateway()
    svc = build_services(runs=repo, events=RecordingEventBus(), gateway=gw,
        inspector=FakeConfigInspector(), clock=FakeClock(), project_root=project,
        runs_dir=Path(tmp)/"runs", poll_interval=0.05, **kw)
    return svc, repo, gw, Path(tmp)/"runs"

def line(**d): return json.dumps(d)+"\n"

print("=== R1: last progress lines written just before exit are never read ===")
with tempfile.TemporaryDirectory() as tmp:
    svc, repo, gw, runs = env(tmp)
    # hold the supervisor on its sleep: it ticks every 50ms; we write final lines and
    # 'exit' the process inside the same tick window (trainer writes final step then exits)
    dto = svc.start_training.execute(StartTrainingCommand(config_path="configs/t.toml"))
    prog = runs/"run_1"/"log.progress.jsonl"
    time.sleep(0.12)
    with open(prog,"a") as f:
        f.write(line(phase="training_start", total_steps=100))
        for s in range(1,101): f.write(line(phase="step", step=s, total=100, loss=0.1, avg=0.1, lr=1e-4))
    gw.alive.discard(4242)       # trainer exits immediately after its last write
    wait_until(lambda: repo.get(1).status.value!="running", timeout=2)
    r = repo.get(1)
    print(f"  final status={r.status.value}  done_steps={r.done_steps}/{r.total_steps}")
    print("  BUG CONFIRMED" if (r.status.value=='completed' and r.done_steps<100) else "  not reproduced")

print("\n=== R2: one malformed-but-valid-JSON progress line silently kills the supervisor ===")
logging.basicConfig(level=logging.ERROR, format="  [log] %(message)s")
with tempfile.TemporaryDirectory() as tmp:
    svc, repo, gw, runs = env(tmp)
    svc.start_training.execute(StartTrainingCommand(config_path="configs/t.toml"))
    prog = runs/"run_1"/"log.progress.jsonl"
    time.sleep(0.1)
    with open(prog,"a") as f:
        f.write(line(phase="step", step="n/a", total=100, loss=0.5))   # wrong type, valid JSON
    time.sleep(0.3)
    with open(prog,"a") as f:
        f.write(line(phase="step", step=50, total=100, loss=0.4))
    time.sleep(0.3)
    r = repo.get(1)
    print(f"  after a good line (step 50): done_steps={r.done_steps}  status={r.status.value}")
    gw.alive.discard(4242); time.sleep(0.4)
    print(f"  process exited 0 -> status still '{repo.get(1).status.value}' (supervisor is gone, nothing finalises it)")
    try:
        svc.start_training.execute(StartTrainingCommand(config_path="configs/t.toml"))
    except RunAlreadyActiveError as e:
        print("  next start blocked:", e, "<- until backend restart")

print("\n=== R3: artifacts.prepare() failure strands a 'created' row and blocks all starts ===")
with tempfile.TemporaryDirectory() as tmp:
    svc, repo, gw, runs = env(tmp)
    real_prepare = svc.start_training._artifacts.prepare
    def boom(run_id): raise PermissionError("[Errno 13] runs dir not writable")
    svc.start_training._artifacts.prepare = boom
    try: svc.start_training.execute(StartTrainingCommand(config_path="configs/t.toml"))
    except Exception as e: print("  first start ->", type(e).__name__, e)
    svc.start_training._artifacts.prepare = real_prepare      # operator fixes permissions
    try: svc.start_training.execute(StartTrainingCommand(config_path="configs/t.toml"))
    except Exception as e: print("  retry after fixing ->", type(e).__name__, e)
    print("  rows:", [(r.id, r.status.value) for r in repo.list_runs(limit=10)] if hasattr(repo,'list_runs') else "n/a")
