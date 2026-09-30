"""SQLite dataset store — tracks multi-source training data within a local dataset folder.

Format v2 (PRAGMA user_version = 2): queryable columns instead of a metadata
JSON blob, shard-level `layout` instead of `is_temporary`, membership-only
training sets, no `tasks` table (task lifecycle belongs to the server, see
docs/design/backend/04-dataset-format.md). v1 databases are still readable by
this module's legacy-facing helpers, but writers/readers that require v2 must
call ensure_v2() first.
"""

import sqlite3
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

FORMAT_VERSION = 2

# Legacy task helpers below still operate on the v1 `tasks` table so the old
# server can boot and keep managing v1 datasets during the transition; v2
# datasets have no tasks table (they fail loudly if called), and the new
# backend tracks dataset tasks in backend.db instead. Removed at M5 archive.


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # WAL + busy_timeout: the server writing while a trainer child reads is
    # the documented "database is locked" scenario. foreign_keys makes the
    # ON DELETE CASCADE clauses in the schema real.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def get_format_version(db_path: Path) -> int:
    """PRAGMA user_version of a dataset DB (0 for a missing/new file)."""
    if not Path(db_path).exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def ensure_v2(db_path: Path):
    """Raise a migration-guidance error unless the dataset is format v2.

    Called at the v2-only entry points (loader, ingestion builder, and the
    backend's dataset port). Legacy curation helpers deliberately do NOT call
    this so the old server keeps working on v1 datasets until migrated.
    """
    v = get_format_version(db_path)
    if v != FORMAT_VERSION:
        raise ValueError(
            f"Dataset at '{Path(db_path).parent}' is in legacy format "
            f"(user_version={v}, expected {FORMAT_VERSION}). Run "
            f"scripts/migrate_dataset_format.py on it (or regenerate the "
            f"dataset), then retry."
        )


def init_local_db(db_path: Path):
    """Initialize a local metadata.db (format v2) inside a dataset folder."""
    with _connect(db_path) as conn:
        # Never stamp v2 onto a pre-existing v1 table (CREATE IF NOT EXISTS
        # would leave the old columns while user_version lied about them).
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(trajectories)")}
        if existing and "neg_prompt" not in existing:
            raise ValueError(
                f"Dataset DB '{db_path}' exists in legacy format. Run "
                f"scripts/migrate_dataset_format.py on it instead of re-initializing."
            )
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
                layout      TEXT    NOT NULL DEFAULT 'single_latent',
                                -- 'single_latent' | 'compressed_traj' (other values =
                                -- legacy/unknown; readers skip, they never guess)
                sample_count INTEGER NOT NULL DEFAULT 0,
                size_bytes  INTEGER NOT NULL DEFAULT 0,
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
                neg_prompt  TEXT NOT NULL DEFAULT '',
                model_type  TEXT NOT NULL DEFAULT 'eps', -- 'eps' | 'vpred'
                type        TEXT NOT NULL DEFAULT 'good', -- 'good' | 'bad'
                cfg         REAL,                         -- teacher-only
                source_path TEXT, -- Origin image, relative to ingestion image_dir
                latent_h    INTEGER NOT NULL DEFAULT 0,
                latent_w    INTEGER NOT NULL DEFAULT 0,
                preview_path TEXT, -- Relative to dataset_root
                extra       TEXT  -- Residual JSON (crop_idx, batch_idx, ...)
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

            CREATE INDEX IF NOT EXISTS idx_traj_source ON trajectories(source_id);
            CREATE INDEX IF NOT EXISTS idx_traj_shard ON trajectories(shard_id);
        """)
        conn.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
        conn.commit()


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


def add_shard(db_path: Path, rel_file_path: str, count: int, size: int,
              layout: str = "single_latent") -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO shards (file_path, layout, sample_count, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel_file_path, layout, count, size, time.time())
        )
        conn.commit()
        return cur.lastrowid


def add_trajectory(db_path: Path, source_id: int, shard_id: int, shard_index: int,
                   sample_count: int, seed: int, prompt: str, preview_path: str = None,
                   neg_prompt: str = "", model_type: str = "eps",
                   type: str = "good", cfg: float = None,
                   source_path: str = None, latent_h: int = 0, latent_w: int = 0,
                   extra: dict = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, "
            "prompt, preview_path, neg_prompt, model_type, type, cfg, source_path, "
            "latent_h, latent_w, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, shard_id, shard_index, sample_count, seed, prompt, preview_path,
             neg_prompt, model_type, type, cfg, source_path, latent_h, latent_w,
             json.dumps(extra) if extra else None)
        )
        conn.commit()
        return cur.lastrowid


def _project_row(r: sqlite3.Row) -> dict:
    """Row dict + a synthesized `metadata` JSON *string*.

    v1 stored that string on the row; v2 keeps columns and projects it back so
    legacy consumers (server UI parsing traj.metadata) keep working. New code
    reads the columns, not this projection.
    """
    d = dict(r)
    extra = {}
    if d.get("extra"):
        try:
            parsed = json.loads(d["extra"])
            if isinstance(parsed, dict):
                extra = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    meta = dict(extra)
    meta.update({"neg": d.get("neg_prompt") or "", "model_type": d.get("model_type") or "eps",
                 "type": d.get("type") or "good"})
    if d.get("cfg") is not None:
        meta["cfg"] = d["cfg"]
    d["metadata"] = json.dumps(meta)
    return d


def get_trajectories(db_path: Path, source_id: int = None,
                     committed: bool = None) -> list[dict]:
    """List trajectories for curation or training.

    committed=None -> all; True -> members of at least one training set;
    False -> not yet in any set (v2's replacement for the old staging view).
    """
    query = ("SELECT t.*, s.file_path AS file_path, s.layout AS layout "
             "FROM trajectories t JOIN shards s ON t.shard_id = s.id")
    conditions = []
    params = []

    if source_id is not None:
        conditions.append("t.source_id = ?")
        params.append(source_id)
    if committed is True:
        conditions.append("EXISTS (SELECT 1 FROM set_members sm WHERE sm.trajectory_id = t.id)")
    elif committed is False:
        conditions.append("NOT EXISTS (SELECT 1 FROM set_members sm WHERE sm.trajectory_id = t.id)")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [_project_row(r) for r in rows]


def delete_trajectory(db_path: Path, trajectory_id: int):
    """Permanently delete a trajectory from the local DB."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM trajectories WHERE id = ?", (trajectory_id,))
        conn.execute("DELETE FROM set_members WHERE trajectory_id = ?", (trajectory_id,))
        conn.commit()


def get_shards(db_path: Path, layout: str = None) -> list[dict]:
    """List all data shards in the dataset."""
    query = "SELECT * FROM shards"
    params = []
    if layout is not None:
        query += " WHERE layout = ?"
        params.append(layout)

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
    """Get the physical map (columns + shard file/layout) for all trajectories in a set."""
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT t.*, s.file_path AS file_path, s.layout AS layout
            FROM set_members sm
            JOIN trajectories t ON sm.trajectory_id = t.id
            JOIN shards s ON t.shard_id = s.id
            WHERE sm.set_id = ?
        """, (set_id,)).fetchall()
        return [dict(r) for r in rows]


# --- Legacy task helpers (v1 `tasks` table only; see module docstring) ---


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
