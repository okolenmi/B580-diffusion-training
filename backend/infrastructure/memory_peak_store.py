"""SqlitePeakStore -- one writer for peaks, replacing the lost-update file.

The existing `PeakRecord` (nodes/memory/peak_record.py) writes atomically
(temp file + rename) but loses updates across processes: six processes
recording distinct peaks at the same instant ended with a stored peak
lower than the highest recorded in 69 of 150 trials.

This module is the fix: a SQLite table with `INSERT ... ON CONFLICT DO
UPDATE SET peak = MAX(peak, excluded.peak)`. One writer (the server)
applies each child's reported peak with MAX semantics. Reads return None
for unknown.

The table is created on first use. The store is thread-safe within a
process (SQLite handles concurrent access).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqlitePeakStore:
    """Remembered peaks, per configuration, in SQLite.

    One writer (the server) applies each child's reported peak with MAX
    semantics. Reads return None for unknown. The store is thread-safe
    within a process (SQLite handles concurrent access).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create the table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_peaks (
                    fingerprint TEXT PRIMARY KEY,
                    peak_mb REAL NOT NULL,
                    samples INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

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

    def reservation_mb(self, fingerprint: str, *, pillow_mb: float = 150.0) -> float | None:
        """What to reserve for this configuration before starting it.

        None when nothing has been measured for it. **Not zero** -- zero is a
        claim that a run needs nothing, and admitting on that is how two runs
        end up believing the card is theirs.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT peak_mb FROM memory_peaks WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is None:
                return None
            return float(row["peak_mb"]) + pillow_mb

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
