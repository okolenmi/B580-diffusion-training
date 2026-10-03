"""Installer endpoints -- readiness, first-run state, and one-time apply.

Three reads and one write, on purpose. The design's Phase C/D is "opt-in
download" and Phase E is the wizard; neither is built yet, and this router
is the part that is: the answers the wizard's first two screens need,
served as data.

**This surface writes filesystem paths.** That makes it the one place in
this project where a browser request can change where model files are read
from, which is why it is shaped the way it is:

* It sits under the same Host/Origin guard as every other state-changing
  route (ADR 0001), because there is no authentication and the threat is a
  page the user happens to have open.
* `POST /apply` is refused once the installation is configured
  (`installer_not_allowed`). The wizard is a *first-run* surface: after
  the paths are chosen, the thing to change them is Settings, which shows
  what each one resolves to. A wizard left open in a tab on a working
  machine is not a thing that gets to re-point the model directories.
* Nothing here runs pip. The install step is a separate, explicitly
  triggered operation and it does not exist yet; what exists is the report
  that says what would need installing, which is the part that can be
  useful without being able to break anything.

Every response is the same shape as the rest of the API: JSON out, the
error envelope on failure, no `200 {"error": ...}`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    InstallerApplyIn,
    InstallerApplyOut,
    InstallerReadinessOut,
    InstallerStateOut,
    readiness_out,
    state_out,
)

router = APIRouter(prefix="/api/v1/installer", tags=["installer"])

_ERROR_400 = {"description": "one or more paths invalid (details: {key: message})"}
_ERROR_409 = {"description": "already configured, or an install is running"}


@router.get("/readiness", response_model=InstallerReadinessOut)
def get_readiness(
    services: ApplicationServices = Depends(get_services),
) -> InstallerReadinessOut:
    """Can this machine start a training run right now?

    Answers per package (with versions), the device, and one verdict.
    Costs about a second: the device probe imports torch in a subprocess,
    because the server must be able to report a *missing* torch without
    having loaded one.
    """
    return readiness_out(services.installer.check.execute())


@router.get("/state", response_model=InstallerStateOut)
def get_installer_state(
    services: ApplicationServices = Depends(get_services),
) -> InstallerStateOut:
    """Whether the first-run wizard still has anything to do.

    Cheap -- settings reads and no subprocess -- so a page can ask on every
    load and use the answer to decide whether to *offer* the wizard.
    """
    return state_out(services.installer.apply.execute())


@router.post(
    "/apply",
    response_model=InstallerApplyOut,
    responses={400: _ERROR_400, 409: _ERROR_409},
)
def apply_installation(
    body: InstallerApplyIn,
    services: ApplicationServices = Depends(get_services),
) -> InstallerApplyOut:
    """Persist the paths the wizard collected. First run only.

    Returns the fresh settings view *and* the new installation state, so a
    client learns whether it is now configured without a second round-trip
    -- and learns it from the server's own answer rather than inferring
    completion from the absence of an error.
    """
    view = services.installer.apply.apply(body.to_changes())
    return InstallerApplyOut(
        settings={
            "stored": dict(view.stored),
            "resolved": dict(view.resolved),
        },
        state=state_out(services.installer.apply.execute()).model_dump(),
    )


@router.get("/manifest", response_model=dict[str, Any])
def get_manifest(
    services: ApplicationServices = Depends(get_services),
) -> dict[str, Any]:
    """What this project needs, and which list each item belongs to.

    Data rather than prose, because `requirements.txt` names four packages
    and the trainer imports four more that appear in no requirements file
    at all -- the gap this whole surface exists to make visible.
    """
    return services.installer.manifest.execute()