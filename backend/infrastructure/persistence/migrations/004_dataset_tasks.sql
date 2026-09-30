-- 004_dataset_tasks: background ingestion task lifecycle (M3b).
--
-- Task state is server state: it lives here, not in a dataset's
-- metadata.db (format v2 removed the dataset-side `tasks` table --
-- docs/design/backend/04-dataset-format.md). Timestamps are ISO-8601
-- UTC strings, same convention as `runs`. Status values:
-- pending | running | finished | failed | killed.
--
-- `params` is the JSON launch payload (image_dir, model, flags...) so
-- the UI can show what a task was doing without re-deriving it.

CREATE TABLE IF NOT EXISTS dataset_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset     TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    pid         INTEGER,
    current_val INTEGER NOT NULL DEFAULT 0,
    total_val   INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    params      TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dataset_tasks_dataset ON dataset_tasks (dataset);
CREATE INDEX IF NOT EXISTS idx_dataset_tasks_status ON dataset_tasks (status);
