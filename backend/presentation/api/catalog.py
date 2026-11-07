"""Node catalog endpoints -- what the editor learns about node classes (rework/contracts/catalog.md, resolve.md).

* ``GET /api/v1/catalog`` -- nodes, settings, handle types, families, load errors; ``refresh=true`` scans the node packages again;
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


@router.post("/graphs/nodes/{class_name}/resolve", responses={404: {"description": "unknown node class"}})
def resolve_node(class_name: str, body: ResolveIn, services: ApplicationServices = Depends(get_services)) -> dict:
    return services.graphs.resolve_node.execute(class_name, body.params, body.settings)
