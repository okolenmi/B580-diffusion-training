"""Settings and options routes."""

from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Form, Query
from fastapi.responses import JSONResponse

from . import db
from .config import settings
from .options import build_option_tree

router = APIRouter()


@router.get("/options/tree")
async def options_tree(config: str = Query("")):
    """Return the full option schema as a JSON tree."""
    cfg_path = config if config else None
    tree = build_option_tree(cfg_path)
    return JSONResponse(content=tree)


@router.get("/settings")
async def get_settings_endpoint():
    """Get server settings."""
    try:
        comfy_dir = str(settings.comfy_dir)
    except RuntimeError as e:
        # This is exactly the situation this endpoint needs to survive:
        # the whole point of exposing comfy_dir/venv_python here is so the
        # UI can show *why* it's unresolved and let you fix it -- a hard
        # crash here would make that impossible to ever get out of.
        comfy_dir = f"<not resolved: {e}>"

    return {
        "default_config": db.get_setting(settings.db_path, "default_config", ""),
        "comfy_dir": comfy_dir,
        "comfy_dir_override": db.get_setting(settings.db_path, "comfy_dir", ""),
        "venv_python": settings.venv_python,
        "venv_python_override": db.get_setting(settings.db_path, "venv_python", ""),
    }


@router.post("/settings")
async def update_settings_endpoint(
    default_config: Annotated[str, Form()] = "",
    comfy_dir: Annotated[str, Form()] = "",
    venv_python: Annotated[str, Form()] = "",
):
    """Update server settings.

    comfy_dir / venv_python were previously accepted by the UI but silently
    dropped here -- only default_config was ever actually persisted. Both
    are validated against the filesystem before being saved so a typo can't
    silently break every subsequent run; an empty value clears the override
    and falls back to auto-detection (COMFY_DIR/VENV_PYTHON env vars, then
    sibling-folder heuristics).
    """
    errors = {}

    if comfy_dir and not Path(comfy_dir).is_dir():
        errors["comfy_dir"] = f"'{comfy_dir}' is not a directory"
    else:
        db.set_setting(settings.db_path, "comfy_dir", comfy_dir)

    if venv_python and not Path(venv_python).is_file():
        errors["venv_python"] = f"'{venv_python}' is not a file"
    else:
        db.set_setting(settings.db_path, "venv_python", venv_python)

    db.set_setting(settings.db_path, "default_config", default_config)

    if errors:
        return JSONResponse(status_code=400, content={"ok": False, "errors": errors})
    return {"ok": True}
