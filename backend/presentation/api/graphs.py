"""Node-graph endpoints -- catalog, validation, executions, library (M4).

Resource-oriented contract over ``services.graphs``
(docs/design/backend/05-graph-runtime.md section 6):

* ``GET /api/v1/graphs/nodes`` -- the palette (auto-discovered classes
  grouped by domain; ``refresh=true`` re-walks ``nodes/``); ``POST
  /nodes/{class_name}/diagnostics`` -- live per-input hints;
* ``POST /validate`` -- complete issue list, always 200 (``ok=false``
  = the run endpoint would reject);
* ``POST /run`` -- 201 + summary; 422 ``graph_invalid`` carrying *all*
  issues, 409 ``graph_execution_active`` while another runs (single
  B580, deliberate);
* ``GET/DELETE /executions`` -- newest-first history / wipe; ``GET
  /executions/{id}``; ``POST /executions/{id}/stop`` (409 while
  terminal, 404 unknown);
* ``GET/PUT/DELETE /library[/{name}]`` -- saved graphs (PUT: 201 first
  save, 200 replace; names are trimmed, 1..120 chars).

Every error leaves as the one envelope; see ``presentation/errors.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from ...application.limits import DEFAULT_EXECUTION_PAGE_SIZE
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    DeleteExecutionsOut,
    DeleteGraphOut,
    DiagnosticsIn,
    DiagnosticsOut,
    ExecutionListOut,
    ExecutionOut,
    GraphCatalogOut,
    GraphRunIn,
    ExecutionSummaryOut,
    LibraryGraphIn,
    SavedGraphListOut,
    SavedGraphOut,
    ValidateOut,
    execution_list_out,
    execution_out,
    execution_summary_out,
    graph_catalog_out,
    graph_issue_out,
    saved_graph_list_out,
    saved_graph_out,
)

router = APIRouter(prefix="/api/v1/graphs", tags=["graphs"])

_ERROR_404 = {"description": "execution or saved graph not found"}
_ERROR_409 = {"description": "an execution is already active / already terminal"}
_ERROR_422 = {"description": "graph_invalid: every issue, or invalid limit/name"}


@router.get("/nodes", response_model=GraphCatalogOut)
def list_node_catalog(
    refresh: bool = Query(False),
    services: ApplicationServices = Depends(get_services),
):
    """Full palette: real classes, declared ports, presets, load errors."""
    return graph_catalog_out(services.graphs.catalog.execute(refresh=refresh))


@router.post(
    "/nodes/{class_name}/diagnostics",
    response_model=DiagnosticsOut,
    responses={400: {"description": "the node's diagnostics() rejected these params"},
               404: {"description": "unknown node class"}},
)
def node_diagnostics(
    class_name: str,
    body: DiagnosticsIn,
    services: ApplicationServices = Depends(get_services),
):
    """Per-input hint lines for one node under mid-edit params."""
    return DiagnosticsOut(
        messages=services.graphs.diagnostics.execute(class_name, dict(body.params))
    )


@router.post("/validate", response_model=ValidateOut)
def validate_graph(
    body: GraphRunIn, services: ApplicationServices = Depends(get_services)
):
    """Complete issue list (errors block a run; warnings never do)."""
    result = services.graphs.validate.execute(body.to_definition())
    return ValidateOut(
        ok=result.ok, issues=[graph_issue_out(issue) for issue in result.issues]
    )


@router.post(
    "/run",
    response_model=ExecutionSummaryOut,
    status_code=201,
    responses={409: _ERROR_409, 422: _ERROR_422},
)
def run_graph(
    body: GraphRunIn, services: ApplicationServices = Depends(get_services)
):
    """Validate (422 with all issues if bad), refuse while one runs
    (409), then queue and start the execution on a worker thread."""
    return execution_summary_out(
        services.graphs.start_execution.execute(body.to_definition())
    )


@router.get("/executions", response_model=ExecutionListOut)
def list_executions(
    limit: int = Query(DEFAULT_EXECUTION_PAGE_SIZE),
    services: ApplicationServices = Depends(get_services),
):
    """Newest-first page; ``limit`` must be 1..500 (else 422)."""
    return execution_list_out(services.graphs.list_executions.execute(limit=limit))


@router.delete("/executions", response_model=DeleteExecutionsOut)
def delete_executions(services: ApplicationServices = Depends(get_services)):
    """Wipe execution history (active rows lose their CAS and stop)."""
    return DeleteExecutionsOut(
        deleted=services.graphs.delete_executions.execute().deleted
    )


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionOut,
    responses={404: _ERROR_404},
)
def get_execution(
    execution_id: int,
    services: ApplicationServices = Depends(get_services),
):
    """Full row: lifecycle, per-node results, submission snapshot."""
    return execution_out(services.graphs.get_execution.execute(execution_id))


@router.post(
    "/executions/{execution_id}/stop",
    response_model=ExecutionOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def stop_execution(
    execution_id: int,
    services: ApplicationServices = Depends(get_services),
):
    """Request cancel, then CAS to ``stopped`` (409 if it already
    reached a terminal state, whose status rides in the envelope)."""
    return execution_out(services.graphs.stop_execution.execute(execution_id))


@router.get("/library", response_model=SavedGraphListOut)
def list_library(services: ApplicationServices = Depends(get_services)):
    """Saved graphs, most recently updated first."""
    return saved_graph_list_out(services.graphs.list_graphs.execute())


@router.put(
    "/library/{name}",
    response_model=SavedGraphOut,
    responses={422: {"description": "invalid_query: empty or >120-char name"}},
)
def save_library_graph(
    name: str,
    body: LibraryGraphIn,
    response: Response,
    services: ApplicationServices = Depends(get_services),
):
    """Upsert: 201 first save, 200 replace; payload stored verbatim
    (format stamped), no class validation until a run."""
    result = services.graphs.save_graph.execute(
        name, body.to_payload(), description=body.description
    )
    response.status_code = 201 if result.created else 200
    return saved_graph_out(result.graph)


@router.get(
    "/library/{name}", response_model=SavedGraphOut, responses={404: _ERROR_404}
)
def get_library_graph(
    name: str, services: ApplicationServices = Depends(get_services)
):
    return saved_graph_out(services.graphs.get_graph.execute(name))


@router.delete(
    "/library/{name}", response_model=DeleteGraphOut, responses={404: _ERROR_404}
)
def delete_library_graph(
    name: str, services: ApplicationServices = Depends(get_services)
):
    return DeleteGraphOut(deleted=services.graphs.delete_graph.execute(name).deleted)
