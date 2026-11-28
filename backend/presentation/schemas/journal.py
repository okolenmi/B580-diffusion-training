"""Journal wire shapes (``GET/POST /api/v1/journal``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class JournalEntryOut(BaseModel):
    seq: int
    at: float
    level: Literal["info", "success", "warn", "error"]
    source: str
    text: str


class JournalOut(BaseModel):
    entries: list[JournalEntryOut]
    #: The number of the newest entry there is; equal to the page's cursor when nothing new came.
    last: int


class JournalLineIn(BaseModel):
    level: Literal["info", "success", "warn", "error"] = "info"
    text: str


def journal_out(entries, last: int) -> JournalOut:
    return JournalOut(entries=[JournalEntryOut(seq=e.seq, at=e.at, level=e.level, source=e.source, text=e.text) for e in entries], last=last)


__all__ = ["JournalEntryOut", "JournalOut", "JournalLineIn", "journal_out"]
