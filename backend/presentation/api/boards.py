"""Board endpoints -- what a board (a graph view or an app) can bind to.

* ``POST /api/v1/boards/modules`` -- what a graph offers a board: per node instance its editable fields and live displays, plus the
  graph's own modules. Pure read, always 200 (the "add a card" tray);
* ``POST /api/v1/boards/resolve`` -- a *draft* board in the body (the editor's unsaved state), answered card by card: does each still
  bind to the graph (``ok`` / ``wired`` / ``missing_*``), plus the graph's validation issues. Always 200 for a well-formed document
  (422 ``board_invalid`` lists every shape problem).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...application.errors import AppsUnavailableError
from ...application.services import ApplicationServices, AppServices
from ..deps import get_services
from ..schemas import GraphModulesOut, ModulesIn, ResolveIn, ResolveOut, graph_modules_out, resolve_out

router = APIRouter(prefix="/api/v1/boards", tags=["boards"])


def apps_of(services: ApplicationServices = Depends(get_services)) -> AppServices:
    if services.apps is None:
        raise AppsUnavailableError("this server was started without app support")
    return services.apps


@router.post("/modules", response_model=GraphModulesOut)
def describe_modules(body: ModulesIn, apps: AppServices = Depends(apps_of)):
    """Fields and live displays a graph offers a board."""
    return graph_modules_out(apps.describe.execute(body.to_definition()))


@router.post("/resolve", response_model=ResolveOut, responses={422: {"description": "board_invalid: the document's shape"}})
def resolve_board(body: ResolveIn, apps: AppServices = Depends(apps_of)):
    """Card-by-card binding status of a draft board."""
    return resolve_out(apps.resolve.execute(body.doc, graph=body.graph))
