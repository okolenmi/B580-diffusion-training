"""Error mapping -- every failure leaves this API as one envelope.

Design (clean-break, replaces the old server's mix of ``detail`` JSON,
bare dicts, and the global handler):

    {"error": {"code": "run_not_found", "message": "...", "details": [...]}}

* ``ApplicationError`` -> status from the code table below.
* FastAPI/pydantic request validation -> 422 ``validation_error``.
* Anything else (including domain errors escaping a use case, which is
  always a bug) -> 500 ``internal_error``, full traceback logged.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..application.errors import ApplicationError

logger = logging.getLogger(__name__)

_STATUS_BY_CODE: dict[str, int] = {
    "run_not_found": 404,
    "invalid_query": 422,
    "config_not_found": 404,
    "config_invalid": 422,
    "run_already_active": 409,
    "run_not_running": 409,
    "no_active_run": 404,
    "training_launch_failed": 500,
}


def error_body(code: str, message: str, details: list | None = None) -> dict:
    body: dict = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def _application_error(request: Request, exc: ApplicationError) -> JSONResponse:
        status = _STATUS_BY_CODE.get(exc.code, 400)
        return JSONResponse(status_code=status, content=error_body(exc.code, str(exc)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep only JSON-safe keys -- pydantic's raw errors can embed
        # exception objects inside "ctx".
        details = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", "request validation failed", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Unknown routes / method-not-allowed use the same envelope so
        # clients never see a bare {"detail": ...} anywhere.
        message = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(f"http_{exc.status_code}", message),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", "unexpected server error"),
        )
