-- memory_json: the effective memory settings computed when the
-- execution was admitted (MEM-02, ADR 0005). Written once at INSERT,
-- never updated -- the row is what a restart re-derives the held total
-- from, so the admitted values must not drift with later edits to the
-- graph or to the peak record. NULL for rows written before this
-- column and for executions admitted without a capacity reading (the
-- entity then restores memory=None).
ALTER TABLE graph_executions ADD COLUMN memory_json TEXT;
