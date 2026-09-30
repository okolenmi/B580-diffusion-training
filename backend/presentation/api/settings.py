"""Settings endpoints -- stored values, resolved paths, updates.

Updates are atomic: every provided value validates first; a rejected
request reports the full per-key map under
``{"error": {"code": "settings_invalid", "details": {...}}}`` and
persists nothing. The response is always the fresh view, so a client
never has to re-GET to learn what actually took effect.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import SettingsIn, SettingsViewOut, settings_out

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

_ERROR_400 = {"description": "one or more values invalid (details: {key: message})"}


@router.get("", response_model=SettingsViewOut)
def get_settings(
    services: ApplicationServices = Depends(get_services),
) -> SettingsViewOut:
    """``stored`` is exactly what is persisted ("" = unset);
    ``resolved`` is what each path setting points at right now
    (``null`` when nothing can resolve it)."""
    return settings_out(services.settings.read.execute())


@router.post("", response_model=SettingsViewOut, responses={400: _ERROR_400})
def update_settings(
    body: SettingsIn,
    services: ApplicationServices = Depends(get_services),
) -> SettingsViewOut:
    """Partial update: absent key = untouched, "" = clear override."""
    return settings_out(services.settings.update.execute(body.to_changes()))
