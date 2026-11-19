"""Artifact endpoint -- the pictures a running graph leaves for its page.

``GET /api/v1/artifacts/{path}``: the PNG a node reported by path (``Preview Image``); 404 ``artifact_not_found`` for anything else,
including a path that leaves the folder. A file may be written again under the same name (the name follows the recipe), so a client
revalidates instead of trusting its copy.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ...application.errors import ArtifactNotFoundError
from ...application.services import ApplicationServices
from ..deps import get_services

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


@router.get("/{path:path}", responses={404: {"description": "artifact_not_found"}})
def get_artifact(path: str, services: ApplicationServices = Depends(get_services)) -> FileResponse:
    found = services.artifacts.resolve(path) if services.artifacts is not None else None
    if found is None:
        raise ArtifactNotFoundError(f"no picture at {path!r}")
    return FileResponse(found, media_type="image/png", headers={"Cache-Control": "no-cache"})
