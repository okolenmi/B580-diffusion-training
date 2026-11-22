"""One health probe response."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel



class HealthOut(BaseModel):
    status: str
    version: str
    #: The admission ledger's snapshot (MEM-03): capacity, holders and
    #: their sizes, what is free. None while the device total is
    #: unknown and no ledger exists.
    memory: dict | None = None
    #: Differs for every server process, so a page that restarted the server can tell the new one from the old one.
    boot_id: str = ""
    #: When this server process started, and which interpreter it really runs in (the Settings page compares it with the chosen one).
    started_at: datetime | None = None
    interpreter: str = ""
    python: str = ""


__all__ = ["HealthOut"]
