"""FileJournal -- the Journal port as a bounded list in memory with a JSON-lines file behind it.

The newest ``keep`` entries are held; the file is appended to and cut back to ``keep`` lines when it has grown to twice that, so it
neither grows forever nor is rewritten on every line. A restart reads it back and numbers on from the last entry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

from ..application.ports.journal import LEVELS, Journal, JournalEntry

logger = logging.getLogger(__name__)

MAX_TEXT = 4000


class FileJournal(Journal):
    def __init__(self, path: Path, *, keep: int = 2000) -> None:
        self._path = path
        self._keep = keep
        self._lock = threading.Lock()
        self._entries: deque[JournalEntry] = deque(maxlen=keep)
        self._lines_on_disk = 0
        self._seq = 0
        self._load()

    def _load(self) -> None:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        self._lines_on_disk = len(lines)
        for line in lines:
            try:
                raw = json.loads(line)
                self._entries.append(JournalEntry(int(raw["seq"]), float(raw["at"]), str(raw["level"]), str(raw["source"]), str(raw["text"])))
            except (ValueError, KeyError, TypeError):
                continue  # a torn last line from a power cut costs that line only
        if self._entries:
            self._seq = self._entries[-1].seq

    def add(self, level: str, source: str, text: str) -> JournalEntry:
        text = text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + " ..."
        with self._lock:
            self._seq += 1
            entry = JournalEntry(self._seq, time.time(), level if level in LEVELS else "info", source[:40], text)
            self._entries.append(entry)
            self._append(entry)
        return entry

    def _append(self, entry: JournalEntry) -> None:
        record = json.dumps({"seq": entry.seq, "at": entry.at, "level": entry.level, "source": entry.source, "text": entry.text})
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._lines_on_disk >= 2 * self._keep:
                self._path.write_text("".join(self._record(e) + "\n" for e in self._entries), encoding="utf-8")
                self._lines_on_disk = len(self._entries)
                return
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(record + "\n")
            self._lines_on_disk += 1
        except OSError as error:
            # Not logged through ``logging``: the journal listens to it, and a failing disk would then write about itself without end.
            self._lines_on_disk = 0
            print(f"journal: could not write {self._path}: {error}")

    @staticmethod
    def _record(entry: JournalEntry) -> str:
        return json.dumps({"seq": entry.seq, "at": entry.at, "level": entry.level, "source": entry.source, "text": entry.text})

    def after(self, seq: int, limit: int) -> tuple[JournalEntry, ...]:
        with self._lock:
            fresh = [e for e in self._entries if e.seq > seq]
        return tuple(fresh[-limit:])

    def last_seq(self) -> int:
        with self._lock:
            return self._seq
