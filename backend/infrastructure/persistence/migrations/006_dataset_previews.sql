-- 006_dataset_previews: dataset card preview pointer (M8f).
--
-- Which item image fronts a dataset on the cards list is server view
-- state -- same decision as task rows (004): it lives here, not in a
-- dataset's metadata.db, so the dataset directory stays byte-for-byte
-- what the trainer wrote (docs/design/backend/04-dataset-format.md).
--
-- `preview_path` is a dataset-relative path exactly as stored in
-- `trajectories.preview_path`. Readers validate the file still exists
-- at resolve time and otherwise fall back to the first non-bad item's
-- preview, so a stale pointer degrades to the fallback, never to a
-- broken image.

CREATE TABLE IF NOT EXISTS dataset_previews (
    dataset      TEXT PRIMARY KEY,
    preview_path TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
