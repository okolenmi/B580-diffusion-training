-- 001_initial: run history for the new backend.
--
-- Timestamps are stored as ISO-8601 UTC strings (what
-- `datetime.isoformat()` emits, parseable by `fromisoformat`).
-- Status values are the RunStatus enum: created | running |
-- completed | failed | cancelled.

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    status        TEXT    NOT NULL,
    config_path   TEXT    NOT NULL,
    mode          TEXT    NOT NULL,
    phase         TEXT,
    total_steps   INTEGER NOT NULL DEFAULT 0,
    done_steps    INTEGER NOT NULL DEFAULT 0,
    current_loss  REAL,
    avg_loss      REAL,
    pid           INTEGER,
    exit_code     INTEGER,
    error         TEXT,
    log_path      TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
