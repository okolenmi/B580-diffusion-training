import sys, os, tempfile
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
os.environ["VENV_PYTHON"] = "/bin/true"          # harmless: exits 0 immediately
from backend.infrastructure.workspace import WorkspaceLayout
from backend.infrastructure.subprocess_gateway import SubprocessTrainingGateway
from backend.application.ports.training_gateway import TrainingLaunch
from core.config_io import write_config
from core.config_model import TrainingConfig

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); runs=tmp/"runs"; (runs/"run_1").mkdir(parents=True)
    legacy_log = runs/"run_1"/"log.txt"
    legacy_log.write_text("LEGACY server run #1 -- 3 days of training history\n"*100)
    before = legacy_log.stat().st_size
    cfg = tmp/"c.toml"; write_config(cfg, TrainingConfig())
    layout = WorkspaceLayout(tmp, runs_dir=runs, settings_kv=lambda k,d: str(tmp) if k=="comfy_dir" else d)
    gw = SubprocessTrainingGateway(layout)
    # a *new* backend DB hands out id=1 again:
    launch = TrainingLaunch(run_id=1, config_path=cfg, mode="lora", total_steps=10, start_from="teacher",
        reset_optimizer=False, log_path=layout.log_path(1), progress_path=layout.progress_path(1))
    gw.spawn(launch)
    print(f"legacy run_1/log.txt: {before} bytes  ->  {legacy_log.stat().st_size} bytes after the new backend's first run")
