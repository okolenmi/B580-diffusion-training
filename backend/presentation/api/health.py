"""Health endpoint -- liveness + version + the memory snapshot."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ...application.memory_admission import holders_without_ledger
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthOut)
def health(
    services: ApplicationServices = Depends(get_services),
) -> HealthOut:
    """Liveness, version, and who currently holds the device (MEM-03).

    The snapshot is what the ledger knows *now* -- capacity, every
    holder and its size, what is free -- so "the card looks busy" is
    answerable from the one endpoint a watchdog already polls. While no
    ledger exists (device total unknown) there is no sized snapshot to
    give, so `total_mb` is null -- an explicit unknown, never a
    fabricated zero -- with `holders` naming what the unfinished rows
    still claim (MEM-03H-03): after a restart an adopted child's row
    often *is* the answer to "why can't the probe size the card".
    """
    ledger = services.memory_ledger()
    if ledger is not None:
        memory = ledger.snapshot()
    else:
        memory = {
            "total_mb": None,
            "holders": holders_without_ledger(services.memory_ledger),
        }
    return HealthOut(
        status="ok",
        version=__version__,
        memory=memory,
    )
