"""Runs endpoints -- launch/stop/status/log plus read-side queries."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query

from ...application.dto import (
    ListRunsQuery,
    StartTrainingCommand,
)
from ...application.limits import DEFAULT_LOG_LINES, DEFAULT_RUN_PAGE_SIZE
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    DeleteRunsOut,
    ListRunsOut,
    RunLogOut,
    RunOut,
    StartRunIn,
    StopRunIn,
    run_out,
)

router = APIRouter(tags=["runs"])

_ERROR_404 = {"description": "run/config not found"}
_ERROR_409 = {"description": "run state conflict"}
_ERROR_422 = {"description": "invalid parameter"}


@router.get("/api/v1/runs", response_model=ListRunsOut)
def list_runs(
    limit: int = Query(DEFAULT_RUN_PAGE_SIZE),
    status: str | None = Query(None),
    services: ApplicationServices = Depends(get_services),
) -> ListRunsOut:
    """Newest-first page; ``limit`` range and ``status`` values are
    validated by the use case (single source of truth)."""
    result = services.list_runs.execute(ListRunsQuery(limit=limit, status=status))
    return ListRunsOut(runs=[run_out(dto) for dto in result.runs], count=result.count)


# Registered before /{run_id}: path params without a converter would
# otherwise swallow "active" and answer 422 instead of this route.
@router.get(
    "/api/v1/runs/active",
    response_model=RunOut,
    responses={404: _ERROR_404},
)
def active_run(
    services: ApplicationServices = Depends(get_services),
) -> RunOut:
    return run_out(services.get_active_run.execute())


@router.get(
    "/api/v1/runs/{run_id}",
    response_model=RunOut,
    responses={404: _ERROR_404},
)
def get_run(
    run_id: int,
    services: ApplicationServices = Depends(get_services),
) -> RunOut:
    return run_out(services.get_run.execute(run_id))


@router.post(
    "/api/v1/runs",
    response_model=RunOut,
    status_code=201,
    responses={404: _ERROR_404, 409: _ERROR_409, 422: _ERROR_422},
)
def start_run(
    body: StartRunIn,
    services: ApplicationServices = Depends(get_services),
) -> RunOut:
    """Validate config, create the run row, spawn the trainer, and
    start the supervisor. 409 when another run is already active."""
    dto = services.start_training.execute(
        StartTrainingCommand(
            config_path=body.config_path,
            start_from=body.start_from,
            reset_optimizer=body.reset_optimizer,
        )
    )
    return run_out(dto)


@router.post(
    "/api/v1/runs/{run_id}/stop",
    response_model=RunOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def stop_run(
    run_id: int,
    body: StopRunIn | None = Body(default=None),
    services: ApplicationServices = Depends(get_services),
) -> RunOut:
    force = body.force if body is not None else False
    return run_out(services.stop_training.execute(run_id, force=force))


@router.get(
    "/api/v1/runs/{run_id}/log",
    response_model=RunLogOut,
    responses={404: _ERROR_404, 422: _ERROR_422},
)
def run_log(
    run_id: int,
    lines: int = Query(DEFAULT_LOG_LINES),
    services: ApplicationServices = Depends(get_services),
) -> RunLogOut:
    """Tail of the run's log (empty until the trainer creates it)."""
    result = services.get_run_log.execute(run_id, lines=lines)
    return RunLogOut(log=result.log)


@router.delete("/api/v1/runs", response_model=DeleteRunsOut)
def delete_runs(
    services: ApplicationServices = Depends(get_services),
) -> DeleteRunsOut:
    result = services.delete_runs.execute()
    return DeleteRunsOut(deleted=result.deleted)
