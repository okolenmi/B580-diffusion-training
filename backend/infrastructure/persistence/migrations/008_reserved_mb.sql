-- reserved_mb: the device-MB claim the admission ledger held when this
-- row was admitted (MEM-03, ADR 0005). Written at INSERT, while the
-- start lock is held and the claim is live; read back on startup to
-- rebuild the ledger from the rows of unfinished children. NULL for
-- rows written before this column -- and note what is NOT here: a
-- refused start writes no row at all, so there is no "refused" value
-- to encode.
ALTER TABLE graph_executions ADD COLUMN reserved_mb REAL;
ALTER TABLE dataset_tasks ADD COLUMN reserved_mb REAL;
