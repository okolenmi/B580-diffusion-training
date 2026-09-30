"""Config endpoints -- document read/write, schema, launch picker.

Contract notes (clean break from the legacy API's quirks):

* Errors are always the shared envelope (never ``200 {"error": ...}``).
* ``PATCH`` deep-merges a *nested* partial object; dotted/flat keys
  and string-value coercion don't exist here -- JSON types are typed.
* Saving never mutates launch state and launching never mutates the
  config: write with ``PATCH``, launch with ``POST /runs``.
* ``GET .../options`` returns the field *schema* only (pure function
  of the config model); fetch values with ``GET .../``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from ...application.dto import StartOptionsResult
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    ConfigOptionsOut,
    ConfigPatchIn,
    ConfigRawIn,
    ConfigRawOut,
    ConfigSavedOut,
    StartOptionsOut,
    start_options_out,
)

router = APIRouter(prefix="/api/v1/config", tags=["config"])

_ERROR_404 = {"description": "config not found"}
_ERROR_422 = {"description": "invalid parameter or invalid config document"}


@router.get("", response_model=dict[str, Any], responses={404: _ERROR_404, 422: _ERROR_422})
def get_config(
    path: str = Query(""),
    services: ApplicationServices = Depends(get_services),
) -> dict[str, Any]:
    """Validated config as nested JSON (dynamic shape: mirrors
    TrainingConfig)."""
    return services.config.read.execute(path)


@router.patch("", response_model=dict[str, Any], responses={404: _ERROR_404, 422: _ERROR_422})
def patch_config(
    body: ConfigPatchIn,
    services: ApplicationServices = Depends(get_services),
) -> dict[str, Any]:
    """Deep-merge ``overrides`` into an existing config and return the
    result. The file is untouched unless the merged config validates."""
    return services.config.update.execute(body.path, body.overrides)


@router.get("/raw", response_model=ConfigRawOut, responses={404: _ERROR_404, 422: _ERROR_422})
def get_config_raw(
    path: str = Query(""),
    services: ApplicationServices = Depends(get_services),
) -> ConfigRawOut:
    return ConfigRawOut(content=services.config.read_raw.execute(path).content)


@router.put("/raw", response_model=ConfigSavedOut, responses={422: _ERROR_422})
def put_config_raw(
    body: ConfigRawIn,
    services: ApplicationServices = Depends(get_services),
) -> ConfigSavedOut:
    """Create-or-replace: the document is validated before anything
    is written, so a rejected save never truncates the file."""
    services.config.write_raw.execute(body.path, body.content)
    return ConfigSavedOut()


@router.get("/options", response_model=ConfigOptionsOut)
def get_config_options(
    services: ApplicationServices = Depends(get_services),
) -> ConfigOptionsOut:
    """Field schema for the config editor (fetched once, reused for
    every config; values come from ``GET /api/v1/config``)."""
    return ConfigOptionsOut(options=services.config.options.execute())


@router.get(
    "/start-options",
    response_model=StartOptionsOut,
    responses={404: _ERROR_404, 422: _ERROR_422},
)
def get_start_options(
    path: str = Query(""),
    services: ApplicationServices = Depends(get_services),
) -> StartOptionsOut:
    """Continue-from picker: per-option availability, whether a run is
    active, and the most recent finished run (or null)."""
    result: StartOptionsResult = services.config.start_options.execute(path)
    return start_options_out(result)
