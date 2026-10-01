"""SQLite job store — persists training runs, status, and settings."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path):
    """Create tables if they don't exist, and run migrations."""
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                config_path TEXT    NOT NULL,
                mode        TEXT    NOT NULL DEFAULT 'distillation',
                status      TEXT    NOT NULL DEFAULT 'pending',
                phase       TEXT    NOT NULL DEFAULT 'cache',  -- cache | training
                total_steps INTEGER NOT NULL DEFAULT 0,
                done_steps  INTEGER NOT NULL DEFAULT 0,
                cache_done  INTEGER NOT NULL DEFAULT 0,
                cache_total INTEGER NOT NULL DEFAULT 0,
                current_loss REAL,
                avg_loss    REAL,
                started_at  REAL,
                finished_at REAL,
                log_path    TEXT,
                pid         INTEGER,
                error_msg   TEXT
            );

            CREATE TABLE IF NOT EXISTS run_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                event_type  TEXT    NOT NULL,
                step        INTEGER NOT NULL DEFAULT 0,
                loss        REAL,
                avg_loss    REAL,
                lr          REAL,
                message     TEXT,
                created_at  REAL    NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idxEvents_run ON run_events(run_id);
        """)


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(db_path: Path, config_path: str, mode: str,
               total_steps: int) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO runs (config_path, mode, total_steps, status, started_at) "
            "VALUES (?, ?, ?, 'running', ?)",
            (config_path, mode, total_steps, time.time()),
        )
        conn.commit()
        return cur.lastrowid


def update_run_status(db_path: Path, run_id: int, status: str,
                      error_msg: str | None = None, pid: int | None = None):
    with _connect(db_path) as conn:
        if status in ("finished", "failed", "stopped", "killed"):
            conn.execute(
                "UPDATE runs SET status=?, finished_at=?, error_msg=? WHERE id=?",
                (status, time.time(), error_msg, run_id),
            )
        elif pid is not None:
            conn.execute(
                "UPDATE runs SET status=?, pid=? WHERE id=?",
                (status, pid, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status=? WHERE id=?",
                (status, run_id),
            )
        conn.commit()


def update_run_progress(db_path: Path, run_id: int, done_steps: int,
                        current_loss: float | None = None,
                        avg_loss: float | None = None,
                        total_steps: int | None = None):
    with _connect(db_path) as conn:
        if total_steps is not None:
            conn.execute(
                "UPDATE runs SET done_steps=?, current_loss=?, avg_loss=?, total_steps=? WHERE id=?",
                (done_steps, current_loss, avg_loss, total_steps, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET done_steps=?, current_loss=?, avg_loss=? WHERE id=?",
                (done_steps, current_loss, avg_loss, run_id),
            )
        conn.commit()


def update_run_phase(db_path: Path, run_id: int, phase: str,
                     cache_done: int = 0, cache_total: int = 0):
    """Update the run's phase (cache | training) and cache progress."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET phase=?, cache_done=?, cache_total=? WHERE id=?",
            (phase, cache_done, cache_total, run_id),
        )
        conn.commit()


def get_run(db_path: Path, run_id: int) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_active_run(db_path: Path) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE status='running' ORDER BY id DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None


def list_runs(db_path: Path, limit: int = 20) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_all_runs(db_path: Path, exclude_id: int | None = None):
    """Delete all runs and their events from the database."""
    with _connect(db_path) as conn:
        if exclude_id is not None:
            conn.execute("DELETE FROM run_events WHERE run_id != ?", (exclude_id,))
            conn.execute("DELETE FROM runs WHERE id != ?", (exclude_id,))
        else:
            conn.execute("DELETE FROM run_events")
            conn.execute("DELETE FROM runs")
        conn.commit()


def log_event(db_path: Path, run_id: int, event_type: str,
              step: int = 0, loss: float | None = None,
              avg_loss: float | None = None, lr: float | None = None,
              message: str | None = None):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO run_events (run_id, event_type, step, loss, avg_loss, lr, message, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, event_type, step, loss, avg_loss, lr, message, time.time()),
        )
        conn.commit()


def get_recent_events(db_path: Path, run_id: int, limit: int = 100) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM run_events WHERE run_id=? ORDER BY id DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_setting(db_path: Path, key: str, default: str = "") -> str:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,),
        ).fetchone()
        return row["value"] if row else default


def set_setting(db_path: Path, key: str, value: str):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
