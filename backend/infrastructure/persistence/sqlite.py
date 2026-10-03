"""SqliteDatabase -- one connection per thread, WAL, versioned migrations.

Why this shape:

* **Thread-local connections.** SQLite connections are not safe to
  share across threads; the API layer runs handlers in a thread pool
  while SSE/monitoring runs elsewhere, so each thread gets its own
  connection, cached in ``threading.local``.
* **WAL + busy_timeout.** The old ``server/`` opened a fresh
  connection per call with default journal mode; two processes on the
  same file could collide with "database is locked". WAL lets this
  backend share the data directory with another process (or the old
  server) as long as writers take turns, and ``busy_timeout`` turns a
  collision into a short wait instead of an error.
* **Versioned migrations.** ``migrations/*.sql`` apply in filename
  order once; applied names are recorded in ``schema_migrations``.

``connection()`` is a context manager that commits on success and
rolls back on error. It is **not reentrant** -- never nest two
``with connection():`` blocks on the same database in one thread.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

#: SQLite's INTEGER holds a signed 64-bit value, and the driver raises
#: ``OverflowError`` rather than truncating when handed a larger one. An id
#: taken from a URL path segment is an arbitrary-length integer, so "it is an
#: int" does not imply "it can be bound to a query" -- and an integer this
#: large cannot name a row that exists, so it is *not found* rather than a
#: server fault.
#:
#: Defined here rather than in each repository because the limit belongs to
#: the database, not to any one table, and two repositories had the same
#: missing guard.
SQLITE_MIN_INT = -(2 ** 63)
SQLITE_MAX_INT = 2 ** 63 - 1


def fits_in_sqlite_int(value: int) -> bool:
    """Whether ``value`` is bindable to an INTEGER column."""
    return SQLITE_MIN_INT <= value <= SQLITE_MAX_INT


class SqliteDatabase:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._local = threading.local()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, migrations_dir: Path | None = None) -> None:
        """Create the directory and apply pending migrations (idempotent).

        Each migration is applied **atomically**: its statements and the
        row recording it as applied commit together or not at all
        (docs 07 F-16). ``executescript`` commits any open transaction
        before it runs and does not wrap what it runs, so a migration
        that failed halfway used to leave its earlier statements applied
        *and* unrecorded -- the next start then died on "table already
        exists". Wrapping the text in ``BEGIN``/``COMMIT`` (with the
        bookkeeping INSERT inside) is what makes "all or nothing" true
        for DDL as well as DML in SQLite.

        ``migrations_dir`` exists for tests that need a failing
        migration; production always uses the packaged directory.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "  version TEXT PRIMARY KEY,"
                "  applied_at TEXT NOT NULL"
                ");"
            )
        with self.connection() as conn:
            applied = {
                row["version"]
                for row in conn.execute("SELECT version FROM schema_migrations")
            }
        if migrations_dir is None:
            migrations_dir = Path(__file__).resolve().parent / "migrations"
        for migration in sorted(migrations_dir.glob("*.sql")):
            if migration.stem in applied:
                continue
            body = migration.read_text(encoding="utf-8")
            stamp = datetime.now(UTC).isoformat()
            with self.connection() as conn:
                try:
                    conn.executescript(
                        "BEGIN;\n"
                        f"{body}\n;\n"
                        "INSERT INTO schema_migrations (version, applied_at) "
                        f"VALUES ('{migration.stem}', '{stamp}');\n"
                        "COMMIT;"
                    )
                except sqlite3.Error:
                    # Undo whatever the failed script managed to apply;
                    # the version stays unrecorded, so the next start
                    # retries this migration from a clean slate.
                    conn.rollback()
                    raise

    def close(self) -> None:
        """Close this thread's cached connection (no-op if none)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the thread's connection; commit on exit, rollback on error."""
        conn = self._acquire()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def _acquire(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn
