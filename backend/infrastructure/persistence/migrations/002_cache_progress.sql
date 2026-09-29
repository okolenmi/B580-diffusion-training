-- 002_cache_progress: cache-phase telemetry columns.
--
-- The trainer's progress file reports cache-building progress
-- (items done / est. trajs total) alongside step telemetry; the run
-- row mirrors it so the API can serve cache phase without re-reading
-- the file. NULL = not in cache phase / never reported.

ALTER TABLE runs ADD COLUMN cache_done INTEGER;
ALTER TABLE runs ADD COLUMN cache_total INTEGER;
