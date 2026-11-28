"""A logging handler that files the server's warnings and errors in the journal, so they are not only in a terminal that may be parked or closed."""

from __future__ import annotations

import logging

from ..application.ports.journal import Journal

_LEVELS = {logging.WARNING: "warn", logging.ERROR: "error", logging.CRITICAL: "error"}


class JournalLogHandler(logging.Handler):
    def __init__(self, journal: Journal) -> None:
        super().__init__(level=logging.WARNING)
        self._journal = journal

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.exc_info and record.exc_info[1] is not None:
                text += f" ({type(record.exc_info[1]).__name__}: {record.exc_info[1]})"
            self._journal.add(_LEVELS.get(record.levelno, "error" if record.levelno > logging.WARNING else "warn"), "server", f"{record.name.rsplit('.', 1)[-1]}: {text}")
        except Exception:  # noqa: BLE001 -- a handler must not raise into the code that logged
            pass


def install(journal: Journal, loggers: tuple[str, ...] = ("", "uvicorn.error")) -> None:
    """Attach to the given loggers; uvicorn's own logger does not propagate to the root one, so it is named, and this is called after its configuration ran."""
    handler = JournalLogHandler(journal)
    for name in loggers:
        logging.getLogger(name).addHandler(handler)
