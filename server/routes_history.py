"""Run history and events routes."""

from typing import Optional
from fastapi import APIRouter, Query, Depends

from . import db
from .config import settings
from .service import TrainingService, get_training_service

router = APIRouter(prefix="/runs")


@router.get("")
async def list_runs(limit: int = Query(20, ge=1, le=100)):
    return db.list_runs(settings.db_path, limit)


@router.get("/{run_id}")
async def get_run(run_id: int):
    run = db.get_run(settings.db_path, run_id)
    if not run:
        return {"error": "Run not found"}
    return run


@router.get("/{run_id}/events")
async def get_events(run_id: int, limit: int = Query(100, ge=1, le=500)):
    events = db.get_recent_events(settings.db_path, run_id, limit)
    return list(reversed(events))


@router.get("/{run_id}/log")
async def get_run_log(
    run_id: int, 
    lines: int = Query(200, ge=1, le=500),
    service: TrainingService = Depends(get_training_service)
):
    """Get log output for a specific run."""
    log = service.get_log_tail(run_id, lines)
    if not log:
        return {"log": "", "error": f"Log file not found for run #{run_id}"}
    return {"log": log}


@router.get("/{run_id}/previews")
async def get_run_previews(run_id: int):
    """Return the mid-training preview manifest for a run, if any.

    Reads runs/run_{id}/previews/manifest.json (written incrementally by
    core.preview_sampler.PreviewGenerator during training) and rewrites each
    filename into a full URL under the /runs static mount, so the frontend
    can drop these straight into <img src=...> without knowing the
    filesystem layout.
    """
    import json

    manifest_path = settings.run_dir(run_id) / "previews" / "manifest.json"
    if not manifest_path.exists():
        return {"enabled": False, "steps": []}

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return {"enabled": True, "steps": []}

    steps = []
    for entry in manifest:
        step = entry.get("step")
        files = entry.get("files", [])
        if step is None or not files:
            continue
        steps.append({
            "step": step,
            "urls": [f"/runs/run_{run_id}/previews/step_{step:07d}/{fname}" for fname in files],
        })

    return {"enabled": True, "steps": steps}


@router.post("/logs/clear")
async def clear_logs():
    """Delete all run directories and database entries, except active run."""
    import shutil
    from .config import settings
    from . import db
    from .service import get_training_service
    
    runs_dir = settings.runs_dir
    service = get_training_service()
    active_run_id = service.active_run_id
    count = 0
    
    if runs_dir.exists():
        active_dir_name = f"run_{active_run_id}" if active_run_id is not None else None
        for d in runs_dir.glob("run_*"):
            if d.is_dir() and d.name != active_dir_name:
                try:
                    shutil.rmtree(d)
                    count += 1
                except Exception:
                    pass
    
    db.delete_all_runs(settings.db_path, exclude_id=active_run_id)
    return {"ok": True, "count": count, "active_run_id": active_run_id}
