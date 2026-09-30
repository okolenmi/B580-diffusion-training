-- 005_graphs: node-graph executions + saved graph library (M4).
--
-- `graph_executions` replaces the legacy memory-only execution dict
-- (docs/design/backend/05-graph-runtime.md): rows survive restarts,
-- carry the submission snapshot in `graph` (JSON: {"format":1,
-- "nodes":[...],"edges":[...]}) and per-node results/timings in
-- `results`, and their status column is what every CAS transition
-- swaps. Status values: queued | running | finished | error | stopped.
-- Timestamps are ISO-8601 UTC strings, same convention as `runs`.
--
-- `saved_graphs` replaces the editor's browser-only persistence
-- (localStorage `ng_graph_v1`). `graph` is stored verbatim (format
-- stamped, unknown keys preserved) -- saving never validates class
-- names; validation happens at run time.

CREATE TABLE IF NOT EXISTS graph_executions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    status      TEXT    NOT NULL,
    graph       TEXT    NOT NULL,
    results     TEXT    NOT NULL DEFAULT '[]',
    error       TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_graph_executions_status
    ON graph_executions(status);

CREATE TABLE IF NOT EXISTS saved_graphs (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    graph       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
