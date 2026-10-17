"""Text encodings kept on disk between runs, for prompts worth remembering.

One SQLite file per identity (encoders, precision, kit format), so a changed encoder never reads another's entries.
The store is bounded: past ``max_bytes`` it stops writing and says so, and the run carries on with the memory cache.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

FORMAT = 1


class ConditioningStore:
    def __init__(self, directory: Path, identity: str, kit: Any, *, max_bytes: int = 4 << 30):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{identity}-f{FORMAT}.sqlite"
        self.kit, self.max_bytes = kit, max_bytes
        self.full = False
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
        self._db.commit()
        self._bytes = self._db.execute("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM entries").fetchone()[0]

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def get(self, prompt: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM entries WHERE key = ?", (self._key(prompt),)).fetchone()
        return None if row is None else self.kit.load_text(load_tensors(row[0]))

    def put_many(self, items: list[tuple[str, Any]]) -> int:
        """Write encodings that are not stored yet; returns how many were written."""
        written = 0
        with self._lock:
            for prompt, text in items:
                data = save_tensors({k: v.contiguous() for k, v in self.kit.dump_text(text).items()})
                if self._bytes + len(data) > self.max_bytes:
                    self.full = True
                    break
                cursor = self._db.execute("INSERT OR IGNORE INTO entries (key, data) VALUES (?, ?)", (self._key(prompt), data))
                if cursor.rowcount:
                    self._bytes += len(data)
                    written += 1
            self._db.commit()
        return written

    def close(self) -> None:
        with self._lock:
            self._db.close()
