-- name: what the owner calls this run in the list (NULL: unnamed, shown as its number). Renamed at any time,
-- so it is written by its own statement and is not part of the row's lifecycle columns.
ALTER TABLE graph_executions ADD COLUMN name TEXT;
