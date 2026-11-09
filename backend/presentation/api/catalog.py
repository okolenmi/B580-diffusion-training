"""Node catalog endpoints -- what the editor learns about node classes (rework/contracts/catalog.md, resolve.md).

* ``GET /api/v1/catalog`` -- nodes, settings, handle types, families, load errors; ``refresh=true`` scans the node packages again;
* ``GET /api/v1/starters`` and ``POST /api/v1/starters/{name}`` -- ready-made graph documents (``starters.md``);
* ``POST /api/v1/graphs/nodes/{class_name}/resolve`` -- one node's shape for the params being edited (404 unknown class; bad
  params are an ``error`` state in a 200 answer).

The bodies are the contract's documents, passed through as they are.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...application.services import ApplicationServices
from ..deps import get_services

router = APIRouter(prefix="/api/v1", tags=["catalog"])


class ResolveIn(BaseModel):
    params: dict = Field(default_factory=dict)
    settings: dict = Field(default_factory=dict)


@router.get("/catalog")
def get_catalog(refresh: bool = Query(False), services: ApplicationServices = Depends(get_services)) -> dict:
    return services.graphs.node_catalog.execute(refresh=refresh)


@router.get("/catalog/models/{scope}")
def list_model_files(scope: str, services: ApplicationServices = Depends(get_services)) -> dict:
    """The files a model node's picker offers for one scope (``checkpoints``, ``loras``, ...)."""
    return {"scope": scope, "files": services.graphs.model_files.execute(scope)}


@router.get("/catalog/libraries")
def list_image_libraries(services: ApplicationServices = Depends(get_services)) -> dict:
    """The datasets a node can open (image libraries), for the dataset picker."""
    return {"names": services.graphs.image_libraries.execute()}


class StarterIn(BaseModel):
    options: dict = Field(default_factory=dict)


@router.get("/starters")
def list_starters(services: ApplicationServices = Depends(get_services)) -> dict:
    return {"starters": services.graphs.list_starters.execute()}


@router.post("/starters/{name}", responses={404: {"description": "unknown starter"}, 422: {"description": "missing or refused option"}})
def build_starter(name: str, body: StarterIn, services: ApplicationServices = Depends(get_services)) -> dict:
    """The format-3 document of the starter for these options; the editor loads it as it is."""
    return services.graphs.build_starter.execute(name, body.options)


@router.post("/graphs/nodes/{class_name}/resolve", responses={404: {"description": "unknown node class"}})
def resolve_node(class_name: str, body: ResolveIn, services: ApplicationServices = Depends(get_services)) -> dict:
    return services.graphs.resolve_node.execute(class_name, body.params, body.settings)
