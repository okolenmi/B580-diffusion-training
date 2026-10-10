-- Templates: saved pages (a board of blocks + a snapshot of the graph they
-- belong to), and monitor windows (a board of live views). The whole
-- document is one JSON column, stored verbatim like saved_graphs.graph:
-- the layout vocabulary is the frontend's to grow, and a column per block
-- field would make every addition a migration. `kind` is a real column
-- only so a list can be filtered without decoding every document.
CREATE TABLE IF NOT EXISTS templates (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'template',
    doc         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_updated ON templates (updated_at DESC);
