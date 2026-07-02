"""Settings and options routes."""

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
    return {
        "default_config": db.get_setting(settings.db_path, "default_config", ""),
        "comfy_dir": str(settings.comfy_dir),
    }


@router.post("/settings")
async def update_settings_endpoint(
    default_config: Annotated[str, Form()] = "",
):
    """Update server settings."""
    db.set_setting(settings.db_path, "default_config", default_config)
    return {"ok": True}
