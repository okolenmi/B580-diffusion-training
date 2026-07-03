"""Configuration file management — unified CRUD for TOML configs."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request

from .config import settings
from .form_utils import coerce_dict, deep_merge, dotted_to_nested, form_to_nested_overrides
from core.config_io import (
    read_config,
    write_config,
    config_to_toml_string,
    config_from_toml_string,
)
from core.config_model import TrainingConfig

router = APIRouter(prefix="/config")

# start_from / reset_optimizer are per-launch choices, not config file
# contents -- core/cli.py already has its own correct mechanism for
# applying (and persisting, when appropriate) them via --fresh /
# --reset-optimizer CLI flags. Merging the raw launch-form value directly
# into the saved TOML here would be redundant with that, and dangerous:
# TrainingConfig.start_from is a validated Literal["teacher","student",
# "resume"], but the launch form can also send "lora_checkpoint" (a valid
# launch choice, not a valid persisted value) -- letting that reach
# model_validate() would raise instead of launching the run.
SYNTHETIC_KEYS = {"config", "start_from", "reset_optimizer"}


def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = settings.project_root / path
    return p


@router.get("")
async def get_config(path: str = ""):
    """Read and validate a TOML config file, return as nested JSON."""
    if not path:
        return {"error": "Config path is required"}
    cfg_path = _resolve_config_path(path)
    if not cfg_path.exists():
        return {"error": f"Config file not found: {path}"}
    try:
        config = read_config(cfg_path)
        return config.model_dump(mode="json")
    except Exception as e:
        return {"error": str(e)}


@router.get("/raw")
async def get_config_raw(path: str = ""):
    """Read a TOML config file and return its raw text."""
    if not path:
        return {"error": "Config path is required"}
    cfg_path = _resolve_config_path(path)
    if not cfg_path.exists():
        return {"error": f"Config file not found: {path}"}
    try:
        with open(cfg_path) as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}


@router.put("")
async def put_config(request: Request):
    """Save configuration — accepts JSON body or FormData.

    FormData supports both old flat keys (mode, steps, lr) and
    new section-based dotted keys (tuning.method, common.steps).
    Synthetic keys (config, start_from, reset_optimizer) are ignored.
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        config_path = body.pop("config", "convert-cfg.toml")
        for key in SYNTHETIC_KEYS:
            body.pop(key, None)
        # Convert dotted keys to nested dict (same as form-data path)
        dotted = {k: v for k, v in body.items() if "." in k}
        nested = {k: v for k, v in body.items() if "." not in k}
        nested.update(dotted_to_nested(dotted))
        overrides = coerce_dict(nested)
    else:
        form_data = await request.form()
        config_path = form_data.get("config", "convert-cfg.toml")
        overrides = form_to_nested_overrides(form_data, ignore_keys=SYNTHETIC_KEYS)

    cfg_path = _resolve_config_path(str(config_path))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Read existing config if available, then deep-merge overrides
        if cfg_path.exists():
            config = read_config(cfg_path)
            merged = deep_merge(config.model_dump(mode="json"), overrides)
            config = TrainingConfig.model_validate(merged)
        else:
            config = TrainingConfig.model_validate(overrides)
        write_config(cfg_path, config)
        return config.model_dump(mode="json")
    except Exception as e:
        return {"error": str(e)}


@router.put("/raw")
async def put_config_raw(
    path: Annotated[str, Form()],
    content: Annotated[str, Form()],
):
    """Save raw TOML content — validates before writing."""
    cfg_path = _resolve_config_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config = config_from_toml_string(content)
        write_config(cfg_path, config)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}

