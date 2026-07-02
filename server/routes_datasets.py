"""Dataset management routes — unified API for the refactored lifecycle."""

import os
import signal
import multiprocessing
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pathlib import Path
from typing import List, Optional, Dict
import json

from .config import settings
from manager.dataset import ManagedDatasetLibrary, ManagedDataset
from manager.builder import DataTaskRunner
from manager.db import (
    create_task,
    update_task_status
)

router = APIRouter(prefix="/datasets")

# --- Task Process Manager ---

class DataTaskManager:
    def __init__(self):
        self._processes: Dict[int, multiprocessing.Process] = {}
        self._task_to_root: Dict[int, Path] = {}

    def start_task(self, dataset_root: Path, task_id: int, func, *args, **kwargs):
        kwargs['task_id'] = task_id
        
        p = multiprocessing.Process(
            target=func,
            args=args,
            kwargs=kwargs,
            daemon=True
        )
        p.start()
        self._processes[task_id] = p
        self._task_to_root[task_id] = dataset_root
        return p.pid

    def kill_task(self, db_path: Path, task_id: int) -> bool:
        if task_id in self._processes:
            p = self._processes[task_id]
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
            del self._processes[task_id]
        
        if task_id in self._task_to_root:
            del self._task_to_root[task_id]
        
        update_task_status(db_path, task_id, 'killed')
        return True

    def kill_all_for_dataset(self, dataset_root: Path):
        to_kill = [tid for tid, root in self._task_to_root.items() 
                   if root.resolve() == dataset_root.resolve()]
        for tid in to_kill:
            self.kill_task(dataset_root / "metadata.db", tid)

_task_manager = DataTaskManager()

# --- Dependencies ---

_library: Optional[ManagedDatasetLibrary] = None
_runner: Optional[DataTaskRunner] = None

def get_library():
    global _library
    if _library is None:
        _library = ManagedDatasetLibrary(settings.project_root / "datasets")
    return _library

def get_data_runner():
    global _runner
    if _runner is None:
        _runner = DataTaskRunner(device="xpu")
    return _runner

# --- Routes ---

@router.get("")
async def list_library(lib: ManagedDatasetLibrary = Depends(get_library)):
    """List all portable datasets in the library."""
    return lib.list_datasets()


from .schemas import DatasetCreateRequest

@router.post("/create")
async def create_new_dataset(
    req: DatasetCreateRequest,
    lib: ManagedDatasetLibrary = Depends(get_library)
):
    """Create a new dataset directory."""
    try:
        ds = lib.create_dataset(req.name, req.description)
        return {"ok": True, "name": ds.name, "path": str(ds.root)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/delete")
async def api_delete_dataset(
    name: str,
    lib: ManagedDatasetLibrary = Depends(get_library)
):
    """Physically remove a dataset folder."""
    try:
        ds = lib.get_dataset(name)
        _task_manager.kill_all_for_dataset(ds.root)
        lib.delete_dataset(name)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/checkpoints")
async def get_checkpoints():
    """Scan ComfyUI checkpoints folder."""
    base = settings.comfy_dir / "models" / "checkpoints"
    if not base.exists(): return []
    return sorted([str(p.relative_to(base)) for p in base.glob("**/*") 
                   if p.suffix.lower() in (".safetensors", ".ckpt")])


@router.post("/tasks/start")
async def api_start_task(
    dataset_name: str = Body(...),
    type: str = Body(...),
    model_name: str = Body(...),
    # Teacher params
    prompt_mode: str = Body("list"),
    prompts: str = Body(None),
    keywords: str = Body(None),
    keywords_file: str = Body(None),
    template: str = Body(None),
    
    neg_mode: str = Body("list"),
    negative_prompt: str = Body(""),
    neg_keywords: str = Body(None),
    neg_keywords_file: str = Body(None),
    neg_template: str = Body(None),
    neg_min_keywords: int = Body(3),
    neg_max_keywords: int = Body(10),
    
    cfg_min: float = Body(3.0),
    cfg_max: float = Body(9.0),
    steps_min: int = Body(20),
    steps_max: int = Body(30),
    t_mode: str = Body("uniform"),
    t_low: int = Body(20),
    t_high: int = Body(999),
    batch_size: int = Body(1),
    seed: int = Body(42),
    n_conditions: int = Body(10),
    n_samples_per_cond: int = Body(1),
    min_keywords: int = Body(3),
    max_keywords: int = Body(10),
    # Real params
    image_dir: str = Body(None),
    recursive: bool = Body(True),
    auto_caption: bool = Body(True),
    resize_mode: str = Body("center_crop"),
    ingest_latent_size: int = Body(64),
    lib: ManagedDatasetLibrary = Depends(get_library),
    runner: DataTaskRunner = Depends(get_data_runner)
):
    """Start a background generation task."""
    try:
        ds = lib.get_dataset(dataset_name)
        model_path = settings.comfy_dir / "models" / "checkpoints" / model_name
        
        if type == "teacher":
            if prompt_mode == "list":
                pos_cfg = [p.strip() for p in (prompts or "").split("\n") if p.strip()]
            else:
                pos_cfg = {
                    "keywords": [k.strip() for k in (keywords or "").split("\n") if k.strip()] if keywords else None,
                    "keywords_file": keywords_file,
                    "template": template,
                    "min": min_keywords, "max": max_keywords
                }
            
            if neg_mode == "list":
                neg_cfg = negative_prompt
            else:
                neg_cfg = {
                    "keywords": [k.strip() for k in (neg_keywords or "").split("\n") if k.strip()] if neg_keywords else None,
                    "keywords_file": neg_keywords_file,
                    "template": neg_template,
                    "min": neg_min_keywords, "max": neg_max_keywords
                }
            
            if not pos_cfg: raise ValueError("Positive prompts not configured.")

            total = n_conditions * n_samples_per_cond
            task_id = create_task(ds.db_path, 'teacher', total)
            
            _task_manager.start_task(
                ds.root, task_id, runner.run_teacher_task,
                ds.root, model_path, pos_cfg, 
                neg_cfg=neg_cfg,
                n_conditions=n_conditions, 
                n_samples_per_cond=n_samples_per_cond,
                steps_range=(steps_min, steps_max), 
                cfg_range=(cfg_min, cfg_max),
                batch_size=batch_size,
                seed=seed,
                t_mode=t_mode, t_low=t_low, t_high=t_high
            )
        
        elif type == "real":
            if not image_dir: raise ValueError("Image directory not specified")
            
            p_dir = Path(image_dir)
            img_exts = {".png", ".jpg", ".jpeg", ".webp"}
            if recursive: files = [p for p in p_dir.glob("**/*") if p.is_file() and p.suffix.lower() in img_exts]
            else: files = [p for p in p_dir.glob("*") if p.is_file() and p.suffix.lower() in img_exts]
            
            task_id = create_task(ds.db_path, 'real', len(files))
            
            _task_manager.start_task(
                ds.root, task_id, runner.run_ingestion_task,
                ds.root, model_path, p_dir,
                recursive=recursive, resize_mode=resize_mode,
                latent_size=ingest_latent_size,
                t_mode=t_mode, t_low=t_low, t_high=t_high
            )
            
        return {"ok": True, "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}/tasks/active")
async def api_get_active_tasks(name: str, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Check for running tasks in this dataset."""
    try:
        ds = lib.get_dataset(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds.get_active_tasks()


@router.post("/{name}/tasks/{task_id}/kill")
async def api_kill_task(name: str, task_id: int, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Force-terminate a running task."""
    try:
        ds = lib.get_dataset(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        _task_manager.kill_task(ds.db_path, task_id)
    return {"ok": True}


@router.get("/{name}/trajectories/pending")
async def get_pending(name: str, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Get trajectories awaiting review (Staging)."""
    try:
        ds = lib.get_dataset(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds.get_staging_trajectories()


@router.get("/{name}/trajectories/archived")
async def get_archived(name: str, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Get trajectories that have been committed (Archive)."""
    try:
        ds = lib.get_dataset(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds.get_archived_trajectories()


@router.post("/{name}/trajectories/{traj_id}/toggle-type")
async def api_toggle_type(name: str, traj_id: int, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Toggle trajectory type between 'good' and 'bad'."""
    try:
        ds = lib.get_dataset(name)
        ds.toggle_trajectory_type(traj_id)
        return {"ok": True}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Trajectory {traj_id} not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/trajectories/{traj_id}/edit")
async def api_edit_trajectory(
    name: str, 
    traj_id: int, 
    prompt: str = Body(None),
    neg_prompt: str = Body(None),
    cfg: float = Body(None),
    lib: ManagedDatasetLibrary = Depends(get_library)
):
    """Update positive, negative prompts and/or CFG for a trajectory."""
    try:
        ds = lib.get_dataset(name)
        ds.update_trajectory(traj_id, prompt=prompt, neg_prompt=neg_prompt, cfg=cfg)
        return {"ok": True}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Trajectory {traj_id} not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/trajectories/{traj_id}/reject")
async def api_reject_trajectory(name: str, traj_id: int, lib: ManagedDatasetLibrary = Depends(get_library)):
    """Physically delete a single trajectory."""
    try:
        ds = lib.get_dataset(name)
        ds.discard_trajectories([traj_id])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/training-sets/create")
async def api_commit_to_set(
    name: str, 
    set_name: str = Body(...),
    traj_ids: List[int] = Body(...),
    lib: ManagedDatasetLibrary = Depends(get_library)
):
    """Commit trajectories from Staging to Archive and create a training set."""
    try:
        ds = lib.get_dataset(name)
        set_id = ds.commit_to_set(traj_ids, set_name)
        return {"ok": True, "set_id": set_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
