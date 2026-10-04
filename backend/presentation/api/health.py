"""Health endpoint -- liveness + version + the memory snapshot."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
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
    answerable from the one endpoint a watchdog already polls. None
    while no ledger exists (device total unknown), never a fabricated
    zero.
    """
    ledger = services.memory_ledger()
    return HealthOut(
        status="ok",
        version=__version__,
        memory=ledger.snapshot() if ledger is not None else None,
    )
