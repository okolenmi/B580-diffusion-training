"""SqlitePeakStore -- one writer for peaks, replacing the lost-update file.

The `PeakRecord` JSON file this replaces (removed in MEM-04 #3) wrote
atomically (temp file + rename) but lost updates across processes: six
processes recording distinct peaks at the same instant ended with a
stored peak lower than the highest recorded in 69 of 150 trials.

This module is the fix: a SQLite table with `INSERT ... ON CONFLICT DO
UPDATE SET peak = MAX(peak, excluded.peak)`. One writer (the server)
applies each child's reported peak with MAX semantics. Reads return None
for unknown.

The table is created on first use, and a file SQLite could not have
written is discarded and recreated (corruption reads as unknown, and a
re-record rewrites cleanly -- the contract the removed JSON
implementation's checks now live under here). The store is thread-safe
within a process (SQLite handles concurrent access).

Implements the application's ``PeakStore`` port: admission reads through
``peak_mb``, the supervisor's watcher writes through ``record``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ..application.ports.peak_store import PeakStore

logger = logging.getLogger(__name__)

#: The only header a SQLite database can start with.
_SQLITE_HEADER = b"SQLite format 3\x00"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_peaks (
    fingerprint TEXT PRIMARY KEY,
    peak_mb REAL NOT NULL,
    samples INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class SqlitePeakStore(PeakStore):
    """Remembered peaks, per configuration, in SQLite.

    One writer (the server) applies each child's reported peak with MAX
    semantics. Reads return None for unknown. The store is thread-safe
    within a process (SQLite handles concurrent access).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        """A connection to a usable store file.

        A file SQLite could not have written (something outside it
        overwrote or truncated the cache) is discarded first and its
        table recreated, so corruption reads as unknown and the next
        record starts a clean file -- a damaged peak cache costs a
        re-measure, never correctness.

        The *header* is checked instead of catching
        `sqlite3.DatabaseError` around statements: a healthy but busy
        database raises those too (``database is locked`` is an
        ``OperationalError``), and discarding one of those because a
        peer holds it would throw away real measurements. The header is
        only wrong when this file never came from SQLite.
        """
        if self._discard_if_not_a_database():
            with sqlite3.connect(str(self._db_path)) as fresh:
                fresh.execute(_CREATE_TABLE)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _discard_if_not_a_database(self) -> bool:
        """Discard a file SQLite could not have written. True when discarded."""
        try:
            with open(self._db_path, "rb") as fh:
                header = fh.read(16)
        except FileNotFoundError:
            return False
        except OSError as exc:
            # Unreadable is not the same as damaged: leave the file be
            # and let SQLite fail with the real reason.
            logger.warning(
                "peak store %s unreadable before connect: %s", self._db_path, exc
            )
            return False
        if header == _SQLITE_HEADER:
            return False
        try:
            self._db_path.unlink()
        except OSError as exc:
            # Not removed: the next statement surfaces the same truth
            # instead of a silent fake-miss.
            logger.warning(
                "peak store %s is not a database and could not be discarded: %s",
                self._db_path, exc,
            )
            return False
        logger.warning(
            "peak store %s did not have a SQLite header; discarded "
            "-- a damaged cache costs a re-measure, not correctness",
            self._db_path,
        )
        return True

    def _ensure_table(self) -> None:
        """Create the table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def record(self, fingerprint: str, peak_mb: float) -> float:
        """Remember a peak just measured. Monotonic. Returns the stored value.

        Monotonic for the same reason `DeviceReservations.observe` is: a peak
        is a high-water mark, and a later smaller number is a step that
        happened not to be the worst one, not evidence that less is needed.
        Lowering it here would let the *next* run be admitted on a number this
        run already exceeded.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_peaks (fingerprint, peak_mb, samples, updated_at)
                VALUES (?, ?, 1, datetime('now'))
                ON CONFLICT(fingerprint) DO UPDATE SET
                    peak_mb = MAX(peak_mb, excluded.peak_mb),
                    samples = samples + 1,
                    updated_at = datetime('now')
                """,
                (fingerprint, float(peak_mb)),
            )
            row = conn.execute(
                "SELECT peak_mb FROM memory_peaks WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            return float(row["peak_mb"]) if row else float(peak_mb)

    def peak_mb(self, fingerprint: str) -> float | None:
        """The remembered peak for one configuration.

        None when nothing has been measured for it. **Not zero** -- zero
        is a claim that a run needs nothing, and admitting on that is how
        two runs end up believing the card is theirs (the port's rule,
        for every implementation).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT peak_mb FROM memory_peaks WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            return None if row is None else float(row["peak_mb"])

    def reservation_mb(self, fingerprint: str, *, pillow_mb: float = 150.0) -> float | None:
        """What to reserve for this configuration before starting it.

        None when nothing has been measured for it. **Not zero** -- zero is a
        claim that a run needs nothing, and admitting on that is how two runs
        end up believing the card is theirs.
        """
        peak = self.peak_mb(fingerprint)
        return None if peak is None else peak + pillow_mb

    def known(self) -> dict[str, float]:
        """Every remembered peak, for a person to read."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fingerprint, peak_mb FROM memory_peaks"
            ).fetchall()
            return {row["fingerprint"]: float(row["peak_mb"]) for row in rows}

    def forget(self, fingerprint: str) -> None:
        """Drop one configuration's measurement, so the next run re-measures."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memory_peaks WHERE fingerprint = ?",
                (fingerprint,),
            )
