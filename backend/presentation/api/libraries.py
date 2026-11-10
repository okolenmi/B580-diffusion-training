"""The Datasets page's API: image libraries (images, captions, sets, caches) and folder imports."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, Field

from ...application.services import ApplicationServices
from ..deps import get_services

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


class CreateIn(BaseModel):
    name: str


class CaptionEditIn(BaseModel):
    """Change one caption channel. ``ids`` null means everything the set and query select."""

    ids: list[str] | None = None
    set: str = "all"
    query: str = ""
    channel: str = "text"
    mode: str = Field(description="set | prepend | append | replace | clear")
    text: str = ""
    find: str = ""


class IdsIn(BaseModel):
    ids: list[str]
    set: str | None = None


class SetIn(BaseModel):
    name: str
    ids: list[str]


class ImportIn(BaseModel):
    folder: str
    recursive: bool = True
    set: str = "all"


@router.get("")
def list_datasets(services: ApplicationServices = Depends(get_services)) -> dict:
    return {"datasets": services.libraries.list.execute()}


@router.post("", status_code=201)
def create_dataset(body: CreateIn, services: ApplicationServices = Depends(get_services)) -> dict:
    return services.libraries.create.execute(body.name)


@router.delete("/{name}", status_code=204)
def delete_dataset(name: str, services: ApplicationServices = Depends(get_services)) -> Response:
    services.libraries.delete.execute(name)
    return Response(status_code=204)


@router.get("/{name}/images")
def list_images(name: str, set: str = "all", q: str = "", offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=300),
                services: ApplicationServices = Depends(get_services)) -> dict:
    return services.libraries.images.execute(name, set_name=set, query=q, offset=offset, limit=limit)


@router.get("/{name}/images/{image_id}/file", response_class=Response)
def read_image(name: str, image_id: str, thumb: int | None = Query(None, ge=16, le=1024),
               services: ApplicationServices = Depends(get_services)) -> Response:
    content, media_type = services.libraries.read_image.execute(name, image_id, thumb)
    return Response(content=content, media_type=media_type, headers={
        "Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"})


@router.post("/{name}/captions")
def edit_captions(name: str, body: CaptionEditIn, services: ApplicationServices = Depends(get_services)) -> dict:
    changed = services.libraries.edit_captions.execute(
        name, ids=body.ids, set_name=body.set, query=body.query, channel=body.channel, mode=body.mode, text=body.text, find=body.find)
    return {"changed": changed}


@router.post("/{name}/images/remove")
def remove_images(name: str, body: IdsIn, services: ApplicationServices = Depends(get_services)) -> dict:
    """Out of one set (``set``), or out of the library when ``set`` is null; the files stay on disk."""
    return {"removed": services.libraries.remove_images.execute(name, body.ids, body.set)}


@router.post("/{name}/sets")
def add_to_set(name: str, body: SetIn, services: ApplicationServices = Depends(get_services)) -> dict:
    return {"name": body.name, "members": services.libraries.add_to_set.execute(name, body.name, body.ids)}


@router.delete("/{name}/sets/{set_name}", status_code=204)
def delete_set(name: str, set_name: str, services: ApplicationServices = Depends(get_services)) -> Response:
    services.libraries.delete_set.execute(name, set_name)
    return Response(status_code=204)


@router.delete("/{name}/caches/{vae}", status_code=204)
def delete_cache(name: str, vae: str, services: ApplicationServices = Depends(get_services)) -> Response:
    services.libraries.delete_cache.execute(name, vae)
    return Response(status_code=204)


@router.post("/{name}/imports", status_code=202)
def start_import(name: str, body: ImportIn, services: ApplicationServices = Depends(get_services)) -> dict:
    return services.libraries.start_import.execute(name, body.folder, recursive=body.recursive, set_name=body.set)


@router.get("/{name}/imports")
def list_imports(name: str, services: ApplicationServices = Depends(get_services)) -> dict:
    return {"imports": services.libraries.imports.execute(name)}
