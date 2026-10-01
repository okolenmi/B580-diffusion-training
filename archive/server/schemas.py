from typing import Any, Optional
from pydantic import BaseModel, Field


class RunStartRequest(BaseModel):
    """Request to start a new training run."""
    config_path: str
    mode: str = "distillation"
    total_steps: int = 10000


class RunStatusResponse(BaseModel):
    """Status of a training run."""
    run_id: Optional[int] = None
    status: str
    pid: Optional[int] = None
    step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    error: Optional[str] = None


class LogTailResponse(BaseModel):
    """Tail of a run log."""
    run_id: int
    lines: list[str]
    total_lines: int


class ConfigFile(BaseModel):
    """Configuration file information."""
    name: str
    path: str
    last_modified: float


class ServerInfo(BaseModel):
    """General server information."""
    comfy_dir: str
    db_path: str
    is_running: bool
    active_run_id: Optional[int] = None


class DatasetCreateRequest(BaseModel):
    """Request to create a new dataset."""
    name: str
    description: Optional[str] = None
