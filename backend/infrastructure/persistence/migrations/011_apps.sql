-- Apps replace templates (saved templates are dropped, not migrated). An app is one JSON document (a snapshot of the graph and a tree of
-- cards, stored verbatim like saved_graphs.graph) plus its version, which grows only when the graph's hash changes.
DROP TABLE IF EXISTS templates;
CREATE TABLE apps (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    version     INTEGER NOT NULL,
    graph_hash  TEXT NOT NULL,
    doc         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX idx_apps_updated ON apps (updated_at DESC);

-- origin: what started the run when it was not the editor's Run button: {"app": name, "version": n, "values": {...}} (NULL otherwise).
ALTER TABLE graph_executions ADD COLUMN origin_json TEXT;
