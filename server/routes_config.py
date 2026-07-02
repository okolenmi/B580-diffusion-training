"""Configuration file management — unified CRUD for TOML configs."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request

from .config import settings
from core.config_io import (
    read_config,
    write_config,
    config_to_toml_string,
    config_from_toml_string,
    FLAT_TO_SECTION_MAP,
)
from core.config_model import TrainingConfig

router = APIRouter(prefix="/config")

SYNTHETIC_KEYS = {"config"}


def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = settings.project_root / path
    return p


def _dotted_to_nested(dotted: dict[str, str]) -> dict:
    """Convert dotted-key form data to nested dict."""
    result: dict = {}
    for key, value in dotted.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _coerce_form_value(value: str):
    """Best-effort type coercion for form string values."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def _coerce_dict(obj):
    """Recursively coerce string values in a nested dict."""
    if isinstance(obj, dict):
        return {k: _coerce_dict(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return _coerce_form_value(obj)
    return obj


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides dict into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _migrate_flat_keys(data: dict[str, str]) -> dict[str, str]:
    """Rename old flat keys to section-based dotted keys."""
    result = {}
    for key, value in data.items():
        if key in FLAT_TO_SECTION_MAP:
            result[FLAT_TO_SECTION_MAP[key]] = value
        else:
            result[key] = value
    return result


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
        nested.update(_dotted_to_nested(dotted))
        overrides = _coerce_dict(nested)
    else:
        form_data = await request.form()
        config_path = form_data.get("config", "convert-cfg.toml")
        raw: dict[str, str] = {}
        for key, value in form_data.items():
            if key in SYNTHETIC_KEYS:
                continue
            raw[key] = str(value)
        raw = _migrate_flat_keys(raw)
        overrides = _coerce_dict(_dotted_to_nested(raw))

    cfg_path = _resolve_config_path(str(config_path))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Read existing config if available, then deep-merge overrides
        if cfg_path.exists():
            config = read_config(cfg_path)
            merged = _deep_merge(config.model_dump(mode="json"), overrides)
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

