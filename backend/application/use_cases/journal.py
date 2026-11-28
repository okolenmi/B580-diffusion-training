"""Read the journal, and let a page add a line to it."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.journal import LEVELS, Journal, JournalEntry

MAX_READ = 1000
MAX_PAGE_LINE = 2000


class ReadJournal:
    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def execute(self, after: int = 0, limit: int = 300) -> tuple[tuple[JournalEntry, ...], int]:
        """``(entries, last)``: the newest ``limit`` entries numbered above ``after`` and the number of the very last one, so a page can tell when nothing new came."""
        return self._journal.after(max(after, 0), max(1, min(limit, MAX_READ))), self._journal.last_seq()


class WriteJournal:
    """A line a page wants kept ("Saved ...", "Could not ..."); the source is always ``page``, so a page cannot pass itself off as a run or the server."""

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def execute(self, level: str, text: str) -> JournalEntry:
        if level not in LEVELS:
            raise InvalidQueryError(f"level must be one of {list(LEVELS)}")
        if not text.strip():
            raise InvalidQueryError("text must not be empty")
        return self._journal.add(level, "page", text[:MAX_PAGE_LINE])
