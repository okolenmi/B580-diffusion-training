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
    InstallerConflictsOut,
    InstallerDeviceOut,
    InstallerDevicesOut,
    InstallerReadinessOut,
    InstallerStateOut,
    conflicts_out,
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


@router.get("/devices", response_model=InstallerDevicesOut)
def get_devices(
    services: ApplicationServices = Depends(get_services),
) -> InstallerDevicesOut:
    """Every accelerator on this machine, for screen 2's choice.

    **Costs about 1.8 s** -- it imports torch, in a subprocess, because the
    question is which card a multi-gigabyte wheel should be built for and
    that cannot be answered without torch. So the wizard calls this when
    screen 2 is *shown*, not on every readiness render, and not before the
    user has been told what the choice is for.

    `enumerated` distinguishes "looked and found none" from "could not
    look". Both send an empty list, and a wizard that renders both as "no
    graphics card" would be asserting a fact it does not have.
    """
    probe = services.installer.device_probe
    rows, reason = probe.devices_with_reason()
    return InstallerDevicesOut(
        enumerated=bool(probe.enumerate_all) and reason is None,
        backend=getattr(probe, "backend", "xpu"),
        devices=[
            InstallerDeviceOut(
                index=index,
                present=row.present,
                name=row.name,
                total_memory_mb=row.total_memory_mb,
                reason=row.reason,
            )
            for index, row in enumerate(rows)
        ],
        reason=reason,
    )


@router.get("/conflicts", response_model=InstallerConflictsOut)
def get_conflicts(
    comfy_dir: str | None = None,
    services: ApplicationServices = Depends(get_services),
) -> InstallerConflictsOut:
    """May the four server packages go into ComfyUI's own virtualenv?

    **A read, and it is allowed to be expensive.** It runs a subprocess in
    ComfyUI's interpreter -- measured at 0.16 s on this machine for 185
    packages -- so the wizard calls it when the user *selects* the reuse
    option rather than on every page load. The alternative, checking on
    every load, would mean a 185-line answer rendered before anyone has
    decided whether they want to read it.

    Both paths come from the settings store rather than from the request.
    `comfy_dir` can be overridden per call so the wizard can check a
    directory the user has just typed before committing it.

    **`stored` venv_python, not `resolved`.** This took a browser check to
    find. `resolved` ends in a bare `"python"` when nothing is configured,
    and that string is about *this project's* children -- which interpreter
    a training subprocess should run under. Passing it here meant a
    first-run machine checked its own 87-package interpreter instead of
    ComfyUI's 185-package one, and reported the answer as though it were
    about ComfyUI. The two questions share a key and not a meaning; this
    endpoint is the second one, so it reads the second source and lets the
    port derive a venv from the checkout when there is nothing to read.

    Three outcomes, not two: an installed package that ComfyUI's file does
    not mention is *unknown*, and unknown is pinned rather than trusted.
    See the design doc §3.
    """
    view = services.settings.read.execute()
    report = services.installer.conflicts.execute(
        comfy_dir=comfy_dir or view.resolved.get("comfy_dir") or "",
        # Stored, not resolved -- see the docstring. An unset venv_python
        # is the normal state of a first-run machine, and the port finds
        # ComfyUI's venv from its checkout instead of falling back to
        # whatever this server happens to be running under.
        venv_python=view.stored.get("venv_python") or None,
    )
    return conflicts_out(report)


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