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
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from ..application.ports.peak_store import PeakStore

logger = logging.getLogger(__name__)

#: The only header a SQLite database can start with.
_SQLITE_HEADER = b"SQLite format 3\x00"

#: How long a statement waits for another process's write lock. The default
#: (5 s) is short for six processes recording at once on a slow disk.
_BUSY_TIMEOUT_S = 30.0

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_peaks (
    fingerprint TEXT PRIMARY KEY,
    peak_mb REAL NOT NULL,
    samples INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _identity(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


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
        self._discard_if_not_a_database()
        conn = sqlite3.connect(str(self._db_path), timeout=_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        try:
            # On every connect, not just the first: a file that is empty
            # (new, or truncated to nothing) has no table, and a read of it
            # must answer "unknown", not raise "no such table". When the
            # table exists this is a schema read and takes no write lock.
            with conn:
                conn.execute(_CREATE_TABLE)
        except BaseException:
            conn.close()
            raise
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One connection, one transaction, and the connection is *closed*.

        ``with sqlite3.connect(...) as conn`` commits but leaves the
        connection open until it is garbage-collected; every call here used
        to leak a file handle (and, in a process that records often, a
        lock holder) that way.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _discard_if_not_a_database(self) -> bool:
        """Discard a file SQLite could not have written. True when discarded.

        **An empty file is not damage.** ``sqlite3.connect`` creates the
        file at zero length and writes the header at the first commit, so a
        peer that is *creating* the store right now looks exactly like a
        file with no header. Treating that as corruption deleted the file
        from under the peer's open connection: measured on 60 trials of six
        processes constructing the store at the same instant, 177 ``disk I/O
        error``s, 10 ``no such table``s and 27 trials that ended with the
        highest recorded peak missing. SQLite itself treats a zero-length
        file as a new database, so there is nothing to recover from.

        A file with *some other* content really is not ours and is
        discarded -- but only if it is still the very file that was
        inspected. Two processes that both found the same bad file would
        otherwise each unlink, and the second would delete the clean store
        the first had just written.
        """
        try:
            with open(self._db_path, "rb") as fh:
                header = fh.read(16)
                inspected = os.fstat(fh.fileno())
        except FileNotFoundError:
            return False
        except OSError as exc:
            # Unreadable is not the same as damaged: leave the file be
            # and let SQLite fail with the real reason.
            logger.warning(
                "peak store %s unreadable before connect: %s", self._db_path, exc
            )
            return False
        if header == _SQLITE_HEADER or header == b"":
            return False
        try:
            now = os.stat(self._db_path)
            # Identity is more than the inode number: a filesystem hands a
            # freed inode straight to the next file created, so a peer's
            # replacement can carry the same number. Size and modification
            # time tell the two files apart.
            if _identity(now) != _identity(inspected):
                return False  # a peer already replaced it; theirs is the clean one
            self._db_path.unlink()
        except FileNotFoundError:
            return False  # a peer discarded it first
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
        with self._tx() as conn:
            conn.execute(_CREATE_TABLE)

    def record(self, fingerprint: str, peak_mb: float) -> float:
        """Remember a peak just measured. Monotonic. Returns the stored value.

        Monotonic for the same reason `DeviceReservations.observe` is: a peak
        is a high-water mark, and a later smaller number is a step that
        happened not to be the worst one, not evidence that less is needed.
        Lowering it here would let the *next* run be admitted on a number this
        run already exceeded.
        """
        with self._tx() as conn:
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
        with self._tx() as conn:
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
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT fingerprint, peak_mb FROM memory_peaks"
            ).fetchall()
            return {row["fingerprint"]: float(row["peak_mb"]) for row in rows}

    def forget(self, fingerprint: str) -> None:
        """Drop one configuration's measurement, so the next run re-measures."""
        with self._tx() as conn:
            conn.execute(
                "DELETE FROM memory_peaks WHERE fingerprint = ?",
                (fingerprint,),
            )
