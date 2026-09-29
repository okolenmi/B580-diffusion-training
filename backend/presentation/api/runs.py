"""Runs endpoints -- read-side queries plus history deletion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...application.dto import ListRunsQuery
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import DeleteRunsOut, ListRunsOut, RunOut, run_out

router = APIRouter(tags=["runs"])


@router.get("/api/v1/runs", response_model=ListRunsOut)
def list_runs(
    limit: int = Query(50),
    status: str | None = Query(None),
    services: ApplicationServices = Depends(get_services),
) -> ListRunsOut:
    """Newest-first page; ``limit`` range and ``status`` values are
    validated by the use case (single source of truth)."""
    result = services.list_runs.execute(ListRunsQuery(limit=limit, status=status))
    return ListRunsOut(runs=[run_out(dto) for dto in result.runs], count=result.count)


@router.get("/api/v1/runs/{run_id}", response_model=RunOut)
def get_run(
    run_id: int,
    services: ApplicationServices = Depends(get_services),
) -> RunOut:
    return run_out(services.get_run.execute(run_id))


@router.delete("/api/v1/runs", response_model=DeleteRunsOut)
def delete_runs(
    services: ApplicationServices = Depends(get_services),
) -> DeleteRunsOut:
    result = services.delete_runs.execute()
    return DeleteRunsOut(deleted=result.deleted)
