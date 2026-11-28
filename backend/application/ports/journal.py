"""Journal port -- what the server and its pages tell the person, kept.

The console of the pages used to be a strip of lines each page wrote for itself and lost when it was reloaded. The journal is the one list
behind every page's console: the server's own warnings and errors, how every run began and ended (with the reason), what a node
warned about during a run, and the lines the pages themselves write ("Saved ...", "Could not ..."). Entries are numbered, so a page asks for
"what came after the last one I have", and they survive a restart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

LEVELS = ("info", "success", "warn", "error")


@dataclass(frozen=True, slots=True)
class JournalEntry:
    seq: int
    #: Seconds since the epoch.
    at: float
    level: str
    #: Who said it: ``server``, ``page``, or ``run #N``.
    source: str
    text: str


class Journal(ABC):
    @abstractmethod
    def add(self, level: str, source: str, text: str) -> JournalEntry:
        """Append one entry; a journal that cannot write keeps the entry in memory rather than raising into what it describes."""
        raise NotImplementedError

    @abstractmethod
    def after(self, seq: int, limit: int) -> tuple[JournalEntry, ...]:
        """The entries numbered above ``seq``, oldest first, at most ``limit`` of the newest."""
        raise NotImplementedError

    @abstractmethod
    def last_seq(self) -> int:
        raise NotImplementedError
