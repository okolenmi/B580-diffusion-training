"""Host-RAM admission -- may a run with a stated ``memory.ram_max_mb`` start now?

Host memory is not ledgered like the card's: every run holds its own copy of the weights, and only a *stated* ceiling says how
much a run may take. The check therefore only applies to a stated ceiling (``auto`` has no limit of its own and is not
checked) and answers from two views, taking the smaller:

* what the machine reports as available now (sees every other program), and
* the machine's total, less a reserve kept for the system, less the ceilings already promised to the runs alive.

A refusal while other runs hold promises is worth waiting for (the planned-run pump retries it); one with nobody holding
anything is final.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .errors import MemoryUnavailableError
from .ports.host_memory import HostMemory

logger = logging.getLogger(__name__)

#: RAM kept for the system and everything that is not a run, below the total.
HOST_RESERVE_MB = 1024.0


class HostRamAdmission:
    def __init__(self, host: HostMemory, promised: Callable[[], dict[str, float]]) -> None:
        self._host = host
        self._promised = promised

    def check(self, ceiling_mb: float, *, what: str = "this graph execution") -> None:
        snapshot = self._host.snapshot()
        if snapshot is None:
            logger.info("host memory cannot be read here; the ceiling of %.0f MB is not checked at admission", ceiling_mb)
            return
        holders = self._promised()
        promised_mb = sum(holders.values())
        by_total = snapshot.total_mb - HOST_RESERVE_MB - promised_mb
        can_give = min(snapshot.available_mb, by_total)
        if ceiling_mb <= can_give:
            return
        # Waiting helps only when running graphs hold the difference.
        waitable = bool(holders) and ceiling_mb <= snapshot.total_mb - HOST_RESERVE_MB
        raise MemoryUnavailableError(
            f"{what} cannot be admitted: host memory: its ceiling is {ceiling_mb:.0f} MB, the machine can give {max(can_give, 0):.0f} MB "
            f"({snapshot.available_mb:.0f} MB available now; {snapshot.total_mb:.0f} MB in total, {HOST_RESERVE_MB:.0f} MB kept for "
            f"the system, {promised_mb:.0f} MB promised to running graphs)",
            details={
                "reason": "host_memory",
                "requested_mb": ceiling_mb,
                "available_mb": snapshot.available_mb,
                "total_mb": snapshot.total_mb,
                "reserve_mb": HOST_RESERVE_MB,
                "promised_mb": promised_mb,
                "holders": holders if waitable else {},
            },
        )
