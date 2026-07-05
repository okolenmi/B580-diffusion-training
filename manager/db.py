"""SQLite dataset store — tracks multi-source training data within a local dataset folder."""

import sqlite3
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_local_db(db_path: Path):
    """Initialize a local metadata.db inside a dataset folder."""
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS info (
                name        TEXT PRIMARY KEY,
                description TEXT,
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                type        TEXT    NOT NULL, -- 'teacher' | 'real'
                model_path  TEXT,
                config      TEXT,             -- JSON blob
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS shards (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT    NOT NULL UNIQUE, -- Relative to dataset_root
                sample_count INTEGER NOT NULL DEFAULT 0,
                size_bytes  INTEGER NOT NULL DEFAULT 0,
                is_temporary INTEGER NOT NULL DEFAULT 0,
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trajectories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                shard_id    INTEGER NOT NULL REFERENCES shards(id) ON DELETE CASCADE,
                shard_index INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                seed        INTEGER,
                prompt      TEXT,
                preview_path TEXT, -- Relative to dataset_root
                metadata    TEXT
            );

            CREATE TABLE IF NOT EXISTS training_sets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                description TEXT,
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS set_members (
                set_id      INTEGER NOT NULL REFERENCES training_sets(id) ON DELETE CASCADE,
                trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
                PRIMARY KEY (set_id, trajectory_id)
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending', -- 'running' | 'finished' | 'failed' | 'killed'
                current_val INTEGER NOT NULL DEFAULT 0,
                total_val   INTEGER NOT NULL DEFAULT 0,
                pid         INTEGER,
                error       TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_traj_source ON trajectories(source_id);
            CREATE INDEX IF NOT EXISTS idx_traj_shard ON trajectories(shard_id);
        """)


def set_dataset_info(db_path: Path, name: str, description: str = None):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO info (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, time.time())
        )
        conn.commit()


def add_source(db_path: Path, name: str, source_type: str,
               model_path: str = None, config: dict = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO sources (name, type, model_path, config, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source_type, model_path, json.dumps(config) if config else None, time.time())
        )
        conn.commit()
        return cur.lastrowid


def add_shard(db_path: Path, rel_file_path: str, count: int, size: int, is_temp: bool = False) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO shards (file_path, sample_count, size_bytes, is_temporary, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel_file_path, count, size, 1 if is_temp else 0, time.time())
        )
        conn.commit()
        return cur.lastrowid


def add_trajectory(db_path: Path, source_id: int, shard_id: int, shard_index: int,
                   sample_count: int, seed: int, prompt: str, preview_path: str = None,
                   metadata: dict = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, preview_path, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, shard_id, shard_index, sample_count, seed, prompt, preview_path, json.dumps(metadata) if metadata else None)
        )
        conn.commit()
        return cur.lastrowid


def get_trajectories(db_path: Path, source_id: int = None, is_temp: bool = None) -> list[dict]:
    """List trajectories for curation or training."""
    query = "SELECT t.*, s.file_path as shard_path, s.is_temporary FROM trajectories t JOIN shards s ON t.shard_id = s.id"
    conditions = []
    params = []

    if source_id is not None:
        conditions.append("t.source_id = ?")
        params.append(source_id)
    if is_temp is not None:
        conditions.append("s.is_temporary = ?")
        params.append(1 if is_temp else 0)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def delete_trajectory(db_path: Path, trajectory_id: int):
    """Permanently delete a trajectory from the local DB."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM trajectories WHERE id = ?", (trajectory_id,))
        conn.execute("DELETE FROM set_members WHERE trajectory_id = ?", (trajectory_id,))
        conn.commit()


def get_shards(db_path: Path, is_temp: bool = None) -> list[dict]:
    """List all data shards in the dataset."""
    query = "SELECT * FROM shards"
    params = []
    if is_temp is not None:
        query += " WHERE is_temporary = ?"
        params.append(1 if is_temp else 0)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def delete_shard(db_path: Path, shard_id: int):
    """Delete a shard record and its physical file if no trajectories remain."""
    with _connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trajectories WHERE shard_id = ?", (shard_id,)).fetchone()[0]
        if count > 0:
            return False

        shard = conn.execute("SELECT file_path FROM shards WHERE id = ?", (shard_id,)).fetchone()
        if shard:
            file_path = db_path.parent / shard["file_path"]
            if file_path.exists():
                file_path.unlink()
            conn.execute("DELETE FROM shards WHERE id = ?", (shard_id,))
            conn.commit()
            return True
    return False


def create_training_set(db_path: Path, name: str, description: str = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO training_sets (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, time.time())
        )
        conn.commit()
        return cur.lastrowid


def add_to_set(db_path: Path, set_id: int, trajectory_id: int):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO set_members (set_id, trajectory_id) VALUES (?, ?)",
            (set_id, trajectory_id)
        )
        conn.commit()


def get_training_sets(db_path: Path):
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM training_sets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_training_set_by_name(db_path: Path, name: str) -> Optional[int]:
    """Look up a training set by name, return its ID or None."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id FROM training_sets WHERE name = ?", (name,)).fetchone()
        return row["id"] if row else None


def get_training_set_trajectories(db_path: Path, set_id: int):
    """Get the physical map for all trajectories in a set."""
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT t.id, s.file_path, t.shard_index, t.sample_count, t.prompt, t.seed, t.metadata
            FROM set_members sm
            JOIN trajectories t ON sm.trajectory_id = t.id
            JOIN shards s ON t.shard_id = s.id
            WHERE sm.set_id = ?
        """, (set_id,)).fetchall()
        return [dict(r) for r in rows]


def create_task(db_path: Path, task_type: str, total: int) -> int:
    now = time.time()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO tasks (type, status, total_val, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (task_type, 'pending', total, now, now)
        )
        conn.commit()
        return cur.lastrowid


def update_task_progress(db_path: Path, task_id: int, current: int, status: str = 'running', pid: int = None):
    with _connect(db_path) as conn:
        if pid:
            conn.execute(
                "UPDATE tasks SET current_val = ?, status = ?, pid = ?, updated_at = ? WHERE id = ?",
                (current, status, pid, time.time(), task_id)
            )
        else:
            conn.execute(
                "UPDATE tasks SET current_val = ?, status = ?, updated_at = ? WHERE id = ?",
                (current, status, time.time(), task_id)
            )
        conn.commit()


def update_task_status(db_path: Path, task_id: int, status: str, error: str = None):
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, time.time(), task_id)
        )
        conn.commit()


def get_active_tasks(db_path: Path):
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM tasks WHERE status = 'running' OR status = 'pending'").fetchall()
        return [dict(r) for r in rows]
