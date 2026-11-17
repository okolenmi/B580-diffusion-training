"""Installer endpoints: first-run state, the environment to train in, installing into it, and the model folder.

The setup page walks them in order: ``/devices`` (which card), ``/environment`` (what an environment holds), ``/install`` (fill it),
``/tokenizers`` (prompt vocabulary files), ``/apply`` (store the choices). ``/apply`` and ``/install`` are refused once the
installation is configured (``installer_not_allowed``): after setup, Settings changes paths.

State-changing routes sit under the Host/Origin guard (ADR 0001). The two that run a chosen interpreter (``/environment``,
``/environment/verify``) are POST for that reason: a GET is outside the guard and any open web page could trigger it.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from ...application.memory_admission import admit
from ...application.ports.environment import CachedDeviceProbe
from ...application.services import ApplicationServices
from ...application.use_cases.install_packages import JobNotFound, InstallJobNotFound
from ..deps import get_services
from ..schemas import (
    InstallerApplyIn,
    InstallerApplyOut,
    InstallerDeviceOut,
    InstallerDevicesOut,
    InstallerEnvironmentOut,
    InstallerInstallIn,
    InstallerInstallOut,
    InstallerStateOut,
    InstallerVerifyOut,
    environment_out,
    state_out,
)

router = APIRouter(prefix="/api/v1/installer", tags=["installer"])

_ERROR_400 = {"description": "one or more values invalid (details: {key: message})"}
_ERROR_409 = {"description": "already configured, or an install is running"}


@router.get("/state", response_model=InstallerStateOut)
def get_installer_state(services: ApplicationServices = Depends(get_services)) -> InstallerStateOut:
    """Whether setup still has anything to do. Cheap: settings reads, no subprocess."""
    job = services.installer.install.current()
    return state_out(services.installer.apply.execute(), job.snapshot() if job else None)


@router.post("/apply", response_model=InstallerApplyOut, responses={400: _ERROR_400, 409: _ERROR_409})
def apply_installation(body: InstallerApplyIn, services: ApplicationServices = Depends(get_services)) -> InstallerApplyOut:
    """Persist the models folder and the environment's interpreter. First run only."""
    return apply_out(services, services.installer.apply.apply(body.to_changes(), restart=body.restart))


@router.post("/restart", response_model=InstallerApplyOut)
def restart_server(services: ApplicationServices = Depends(get_services)) -> InstallerApplyOut:
    """Restart the server into the stored environment. First run only."""
    return apply_out(services, services.installer.apply.restart())


def apply_out(services: ApplicationServices, result) -> InstallerApplyOut:
    view = result.view
    return InstallerApplyOut(
        restarting=result.restarting,
        restart_note=result.restart_note,
        settings={"stored": dict(view.stored), "resolved": dict(view.resolved)},
        state=state_out(services.installer.apply.execute()).model_dump(),
    )


@router.get("/devices", response_model=InstallerDevicesOut)
def get_devices(
    refresh: bool = False,
    services: ApplicationServices = Depends(get_services),
) -> InstallerDevicesOut:
    """Every accelerator on this machine, for the first step.

    **Costs about 1.8 s** -- it imports torch, in a subprocess, because the
    question is which card a multi-gigabyte wheel should be built for and
    that cannot be answered without torch. So the wizard calls this when
    screen 2 is *shown*, not on every render, and not before the
    user has been told what the choice is for.

    Before torch is installed the subprocess cannot import it, so the rows are the system's own list of display adapters, each with a
    `note` saying torch checks it after the install (`SystemCards`).

    `enumerated` distinguishes "looked and found none" from "could not
    look". Both send an empty list, and a wizard that renders both as "no
    graphics card" would be asserting a fact it does not have.

    Cached and single-flight: a plain GET is outside the Origin guard, so an uncached probe would be a torch import per request.
    """
    probe = services.installer.device_probe
    if refresh and isinstance(probe, CachedDeviceProbe):
        probe.invalidate()
    ledger = services.memory_ledger()
    if ledger is None:
        rows, reason = probe.devices_with_reason()
    else:
        # Enumerating the card runs the same subprocess the readiness
        # probe does, so it is admitted on the same terms (MEM-03:
        # "and the probe"). A refusal is a 409 `memory_unavailable`
        # carrying the breakdown -- an API caller can be told exactly
        # who holds the device, where the wizard degrades to
        # "not checked" instead.
        owner = f"probe:{uuid4().hex}"
        try:
            admit(
                ledger,
                owner,
                ledger.process_overhead_mb,
                exploratory=False,
                what="device enumeration",
            )
            rows, reason = probe.devices_with_reason()
        finally:
            ledger.release(owner)
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
                note=row.detail,
            )
            for index, row in enumerate(rows)
        ],
        reason=reason,
    )


@router.post("/environment", response_model=InstallerEnvironmentOut)
def inspect_environment(body: InstallerInstallIn, services: ApplicationServices = Depends(get_services)) -> InstallerEnvironmentOut:
    """What the chosen environment holds: per package installed or missing, problems that would stop a run, download size.

    ``new`` is the project's own venv folder (nothing installed yet unless one is already there); ``existing`` needs ``venv_dir``.
    Reads package metadata through that environment's own interpreter; never writes.
    """
    return environment_out(services.installer.inspect.for_target(body.target, body.venv_dir))


@router.post("/environment/verify", response_model=InstallerVerifyOut)
def verify_environment(body: InstallerInstallIn, services: ApplicationServices = Depends(get_services)) -> InstallerVerifyOut:
    """Whether torch in that environment sees the Intel GPU. Imports torch there, so it takes seconds."""
    interpreter, device = services.installer.inspect.verify(body.target, body.venv_dir)
    return InstallerVerifyOut(present=device.present, name=device.name, total_memory_mb=device.total_memory_mb,
                              torch_version=device.torch_version, reason=device.reason,
                              server_in_environment=services.installer.apply.server_runs_in(interpreter))


@router.post("/install", response_model=InstallerInstallOut, responses={400: _ERROR_400, 409: _ERROR_409})
def start_install(body: InstallerInstallIn, services: ApplicationServices = Depends(get_services)) -> InstallerInstallOut:
    """Install what the chosen environment is missing. Returns a job id at once; the client polls ``GET /install/{id}``.

    The request names only the target. The server inspects the environment, pins everything already installed there, and installs the
    rest (torch from the Intel XPU index). Refused once the installation is configured and its environment works.
    """
    services.installer.apply.require_open()
    job = services.installer.install.execute(target=body.target, venv_dir=body.venv_dir, torch=body.torch)
    return InstallerInstallOut(**job.snapshot())


@router.get("/install/{job_id}", response_model=InstallerInstallOut)
def get_install(job_id: str, services: ApplicationServices = Depends(get_services)) -> InstallerInstallOut:
    """One install's state. Cheap, safe to poll. Unknown id (never existed, or lost in a restart) is a 404, never a fabricated success."""
    try:
        return InstallerInstallOut(**services.installer.install_status.execute(job_id))
    except JobNotFound as exc:
        raise InstallJobNotFound(job_id) from exc


@router.get("/tokenizers", response_model=list[dict[str, Any]])
def get_tokenizers(services: ApplicationServices = Depends(get_services)) -> list[dict[str, Any]]:
    """The prompt vocabulary file sets and which are already in the project's assets folder."""
    return services.installer.tokenizers.status()


@router.post("/tokenizers/{key}", response_model=dict[str, Any])
def download_tokenizer(key: str, services: ApplicationServices = Depends(get_services)) -> dict[str, Any]:
    """Download the missing files of one set (``clip``, ``qwen``, ``t5``) from the model hub."""
    return services.installer.tokenizers.download(key)


@router.get("/manifest", response_model=dict[str, Any])
def get_manifest(services: ApplicationServices = Depends(get_services)) -> dict[str, Any]:
    """What this project needs, by tier."""
    return services.installer.manifest.execute()
