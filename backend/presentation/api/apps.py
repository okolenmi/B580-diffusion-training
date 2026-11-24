"""App endpoints -- published, frozen pages over a graph.

Contract over ``services.apps`` (``docs/contracts/apps.md``):

* ``GET/PUT/DELETE /api/v1/apps[/{name}]`` -- the library (PUT: 201 first save, 200 replace; 422 ``app_invalid`` lists every problem).
  A save whose graph snapshot changed makes the next ``version``; a layout change alone does not;
* ``PUT /api/v1/apps/{name}/graph`` -- replace the graph the app owns (the graph editor's Save in app mode); page, style and description
  stay, and the answer lists the cards whose target the new graph no longer has;
* ``POST /api/v1/apps/{name}/run`` -- run the app's graph with the values typed into it. Only fields the app exposes may be set; the
  rest is the same ``StartGraphExecution`` as the editor's Run (422 ``graph_invalid``, 409 ``graph_execution_active``,
  409 ``memory_unavailable``), and the run records its origin.

What a board can bind to and whether it still does are ``/api/v1/boards/*`` (``boards.py``).

Every error leaves as the one envelope; see ``presentation/errors.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from ...application.errors import AppsUnavailableError
from ...application.services import ApplicationServices, AppServices
from ..deps import get_services
from ..schemas import (
    AppGraphIn,
    AppGraphOut,
    AppIn,
    AppRunIn,
    DeleteAppOut,
    ExecutionSummaryOut,
    SavedAppListOut,
    SavedAppOut,
    execution_summary_out,
    saved_app_list_out,
    saved_app_out,
)

router = APIRouter(prefix="/api/v1/apps", tags=["apps"])

_ERROR_404 = {"description": "app not found"}


def apps_of(services: ApplicationServices = Depends(get_services)) -> AppServices:
    if services.apps is None:
        raise AppsUnavailableError("this server was started without app support")
    return services.apps


@router.get("", response_model=SavedAppListOut)
def list_apps(apps: AppServices = Depends(apps_of)):
    return saved_app_list_out(apps.list.execute())


@router.put("/{name}", response_model=SavedAppOut, responses={422: {"description": "app_invalid / invalid_query"}})
def save_app(name: str, body: AppIn, response: Response, apps: AppServices = Depends(apps_of)):
    """Upsert: 201 first save, 200 replace."""
    result = apps.save.execute(name, body.doc, description=body.description)
    response.status_code = 201 if result.created else 200
    return saved_app_out(result.app)


@router.get("/{name}", response_model=SavedAppOut, responses={404: _ERROR_404})
def get_app(name: str, apps: AppServices = Depends(apps_of)):
    return saved_app_out(apps.get.execute(name))


@router.delete("/{name}", response_model=DeleteAppOut, responses={404: _ERROR_404})
def delete_app(name: str, apps: AppServices = Depends(apps_of)):
    return DeleteAppOut(deleted=apps.delete.execute(name).deleted)


@router.put("/{name}/graph", response_model=AppGraphOut, responses={404: _ERROR_404, 422: {"description": "invalid_query"}})
def save_app_graph(name: str, body: AppGraphIn, apps: AppServices = Depends(apps_of)):
    """The app's own graph, replaced; the cards that lost their target are listed."""
    result = apps.save_graph.execute(name, body.graph)
    return AppGraphOut(app=saved_app_out(result.app), orphaned_cards=list(result.orphaned_cards))


@router.post(
    "/{name}/run", response_model=ExecutionSummaryOut, status_code=201,
    responses={404: _ERROR_404,
               409: {"description": "an execution is active / memory unavailable"},
               422: {"description": "app_invalid (a value the app does not expose) / graph_invalid"}},
)
def run_app(name: str, body: AppRunIn, apps: AppServices = Depends(apps_of)):
    """Run the app's graph with these values."""
    return execution_summary_out(
        apps.run.execute(
            name, body.values,
            memory_overrides=body.memory_overrides.as_overrides() if body.memory_overrides else None,
            mode=body.mode,
        )
    )
