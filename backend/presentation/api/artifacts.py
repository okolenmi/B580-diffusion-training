"""Artifact endpoint -- the pictures a running graph leaves for its page.

``GET /api/v1/artifacts/{path}``: the PNG a node reported by path (``Preview Image``), or the ``live.jpg`` a sampler's live preview keeps writing
over; 404 ``artifact_not_found`` for anything else, including a path that leaves the folder. A picture's name follows what made it, but ``live.jpg``
is written again at every look, so a client revalidates instead of trusting its copy (and asks for it with the frame's ``?v=`` to be sure).
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
    return FileResponse(found, media_type="image/jpeg" if found.suffix.lower() == ".jpg" else "image/png", headers={"Cache-Control": "no-cache"})
