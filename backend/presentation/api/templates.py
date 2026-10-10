"""Template endpoints -- saved pages and monitor windows built on a graph.

Contract over ``services.templates`` (docs/design/15):

* ``POST /api/v1/templates/modules`` -- what a graph offers a page: per
  node instance its editable fields and live views, plus the graph's own
  modules. Pure read, always 200 (the "add a block" tray);
* ``POST /api/v1/templates/resolve`` -- a *draft* document in the body
  (the editor's unsaved state), answered block by block: does each still
  bind to the graph (``ok`` / ``wired`` / ``missing_*``), plus the graph's
  validation issues. Always 200 for a well-formed document;
* ``GET/PUT/DELETE /api/v1/templates[/{name}]`` -- the library (PUT: 201
  first save, 200 replace; 422 ``template_invalid`` lists every problem);
* ``POST /api/v1/templates/{name}/run`` -- run the page's graph with the
  values typed into it. Only fields the page exposes may be set; the rest
  is the same ``StartGraphExecution`` as the editor's Run (422
  ``graph_invalid``, 409 ``graph_execution_active``, 409
  ``memory_unavailable``).

Every error leaves as the one envelope; see ``presentation/errors.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from ...application.errors import TemplatesUnavailableError
from ...application.services import ApplicationServices, TemplateServices
from ..deps import get_services
from ..schemas import (
    DeleteTemplateOut,
    ExecutionSummaryOut,
    GraphModulesOut,
    ModulesIn,
    ResolveIn,
    ResolveOut,
    SavedTemplateListOut,
    SavedTemplateOut,
    TemplateIn,
    TemplateRunIn,
    execution_summary_out,
    graph_modules_out,
    resolve_out,
    saved_template_list_out,
    saved_template_out,
)

router = APIRouter(prefix="/api/v1/templates", tags=["templates"])

_ERROR_404 = {"description": "template not found"}


def templates_of(services: ApplicationServices = Depends(get_services)) -> TemplateServices:
    if services.templates is None:
        raise TemplatesUnavailableError("this server was started without template support")
    return services.templates


@router.post("/modules", response_model=GraphModulesOut)
def describe_modules(body: ModulesIn, templates: TemplateServices = Depends(templates_of)):
    """Fields and live views a graph offers a page."""
    return graph_modules_out(templates.describe.execute(body.to_definition()))


@router.post(
    "/resolve", response_model=ResolveOut,
    responses={422: {"description": "template_invalid: the document's shape"}},
)
def resolve_template(body: ResolveIn, templates: TemplateServices = Depends(templates_of)):
    """Block-by-block binding status of a draft document."""
    return resolve_out(templates.resolve.execute(body.doc, graph=body.graph))


@router.get("", response_model=SavedTemplateListOut)
def list_templates(
    kind: str | None = Query(None, description="'template' or 'monitor'"),
    templates: TemplateServices = Depends(templates_of),
):
    return saved_template_list_out(templates.list.execute(kind))


@router.put(
    "/{name}", response_model=SavedTemplateOut,
    responses={422: {"description": "template_invalid / invalid_query"}},
)
def save_template(
    name: str, body: TemplateIn, response: Response,
    templates: TemplateServices = Depends(templates_of),
):
    """Upsert: 201 first save, 200 replace."""
    result = templates.save.execute(name, body.doc, description=body.description)
    response.status_code = 201 if result.created else 200
    return saved_template_out(result.template)


@router.get("/{name}", response_model=SavedTemplateOut, responses={404: _ERROR_404})
def get_template(name: str, templates: TemplateServices = Depends(templates_of)):
    return saved_template_out(templates.get.execute(name))


@router.delete("/{name}", response_model=DeleteTemplateOut, responses={404: _ERROR_404})
def delete_template(name: str, templates: TemplateServices = Depends(templates_of)):
    return DeleteTemplateOut(deleted=templates.delete.execute(name).deleted)


@router.post(
    "/{name}/run", response_model=ExecutionSummaryOut, status_code=201,
    responses={404: _ERROR_404,
               409: {"description": "an execution is active / memory unavailable"},
               422: {"description": "template_invalid (a value the page does not expose) / graph_invalid"}},
)
def run_template(
    name: str, body: TemplateRunIn,
    templates: TemplateServices = Depends(templates_of),
):
    """Run the page's graph with these values."""
    return execution_summary_out(
        templates.run.execute(
            name, body.values,
            memory_overrides=(
                body.memory_overrides.as_overrides() if body.memory_overrides else None
            ),
        )
    )
