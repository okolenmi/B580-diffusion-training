"""Journal endpoints -- what the console of every page shows.

* ``GET /api/v1/journal?after=N&limit=M`` -- the entries numbered above ``N`` (the newest ``M`` of them) and ``last``, the newest number there is.
  A page keeps the highest number it has and asks again from there; ``after=0`` is the history.
* ``POST /api/v1/journal`` -- a line a page wants kept, ``{level, text}``; it is filed as coming from ``page``.

The server's warnings and errors, how each run began and ended, and the warnings nodes raised during runs are written by the server itself
(``application/journal_listener.py``, ``infrastructure/journal_log_handler.py``). 503 ``journal_unavailable`` when the container has none.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...application.errors import JournalUnavailableError
from ...application.services import ApplicationServices, JournalServices
from ..deps import get_services
from ..schemas import JournalLineIn, JournalOut, journal_out

router = APIRouter(prefix="/api/v1/journal", tags=["journal"])


def journal_of(services: ApplicationServices = Depends(get_services)) -> JournalServices:
    if services.journal is None:
        raise JournalUnavailableError("this server was started without a journal")
    return services.journal


@router.get("", response_model=JournalOut)
def read_journal(after: int = Query(0, ge=0), limit: int = Query(300, ge=1, le=1000), journal: JournalServices = Depends(journal_of)):
    entries, last = journal.read.execute(after, limit)
    return journal_out(entries, last)


@router.post("", response_model=JournalOut, status_code=201)
def write_journal(body: JournalLineIn, journal: JournalServices = Depends(journal_of)):
    entry = journal.write.execute(body.level, body.text)
    return journal_out((entry,), entry.seq)
