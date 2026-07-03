"""Training run control routes — start, stop, status, log."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from . import db
from .config import settings
from core.config_io import read_config, write_config
from .control import build_training_command, get_control_options
from .form_utils import deep_merge, form_to_nested_overrides
from .process_manager import kill_process_by_pid
from .service import TrainingService, get_training_service
from core.config_model import TrainingConfig

router = APIRouter()


def _resolve_config_path(config_path: str) -> Path:
    p = Path(config_path)
    if not p.is_absolute():
        p = settings.project_root / config_path
    return p


@router.get("/control/options")
async def control_options(config: str = Query("")):
    """Return available start-from options based on config and filesystem."""
    if not config:
        return {"error": "Config path is required"}
    cfg_path = _resolve_config_path(config)
    try:
        cfg = read_config(cfg_path)
    except Exception:
        return {"start_from": {}, "has_unfinished_run": False, "last_finished": None}
    return get_control_options(cfg, settings.db_path)


@router.post("/run/start")
async def start_run(
    request: Request,
    service: TrainingService = Depends(get_training_service),
):
    """Start a training run.

    Accepts FormData with:
      - config: path to config file
      - start_from: teacher | student | resume | lora_checkpoint
      - reset_optimizer: true/false
      - config field overrides (flat or dotted keys)
    """
    form_data = await request.form()

    config_path = str(form_data.get("config", "")).strip()
    if not config_path:
        return JSONResponse(status_code=400, content={"error": "Config path is required"})

    cfg_path = _resolve_config_path(config_path)

    # Load existing config, deep-merge form overrides, save
    config = read_config(cfg_path)
    overrides = form_to_nested_overrides(form_data, ignore_keys={"config"})
    if overrides:
        merged = deep_merge(config.model_dump(mode="json"), overrides)
        config = TrainingConfig.model_validate(merged)
        write_config(cfg_path, config)

    start_from = str(form_data.get("start_from", "teacher"))
    reset_optimizer = str(form_data.get("reset_optimizer", "false")).lower() == "true"
    total_steps = config.common.steps

    run_id = db.create_run(settings.db_path, config_path, config.tuning.method, total_steps)

    cmd = build_training_command(
        config=config,
        config_path=str(cfg_path),
        start_from=start_from,
        reset_optimizer=reset_optimizer,
        total_steps=total_steps,
        run_id=run_id,
    )

    try:
        started_run_id = service.start_run(
            config_path=config_path,
            mode=config.tuning.method,
            total_steps=total_steps,
            cmd=cmd,
            run_id=run_id,
        )
        return {"run_id": started_run_id}
    except Exception as e:
        db.update_run_status(settings.db_path, run_id, "failed", error_msg=str(e))
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/run/stop")
async def stop_run(
    force: bool = Query(False),
    service: TrainingService = Depends(get_training_service),
):
    if service.is_running:
        service.stop_run(force=force)
        return {"ok": True, "force": force}

    active = db.get_active_run(settings.db_path)
    if active and active.get("pid"):
        pid = active["pid"]
        if kill_process_by_pid(pid):
            status = "killed" if force else "stopped"
            db.update_run_status(settings.db_path, active["id"], status,
                                 error_msg="Killed via PID (no active worker)" if force else None)
            return {"ok": True, "force": force, "method": "pid", "pid": pid}

    return {"error": "No run in progress"}


@router.post("/run/reset")
async def reset_worker(service: TrainingService = Depends(get_training_service)):
    service.reset()
    return {"ok": True}


@router.get("/run/status")
async def run_status(service: TrainingService = Depends(get_training_service)):
    active = db.get_active_run(settings.db_path)
    if active:
        return active
    return {"status": "idle"}


@router.get("/run/log")
async def get_run_log(
    run_id: Optional[int] = Query(None),
    lines: int = Query(100, ge=1, le=500),
    service: TrainingService = Depends(get_training_service),
):
    target_id = run_id or service.active_run_id
    if not target_id:
        return {"log": []}
    return {"log": service.get_log_tail(target_id, lines)}
