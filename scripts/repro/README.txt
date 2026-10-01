Reproductions for docs/design/backend/07-review-2026-10-01.md
Run from the repo root (or set REPO=/path/to/repo):  python3 scripts/repro/rN_*.py
Deps: the repo's requirements (torch, pydantic, fastapi, httpx, tomli_w).

r1_supervisor.py     F-07 (final samples lost), F-01 (supervisor dies, run stuck), F-02 (stranded 'created' row)
r2_write_chain.py    F-06 (settings + upload write outside managed dirs), F-15 (rejected update still mkdirs)
r3_raw_toml.py       F-08 (raw TOML save drops comments / unknown keys)
r4_torn.py           F-07 (half-written line lost for good), NaN passes the reader
r5_runid_collision.py F-04 (new run truncates legacy runs/run_1/log.txt; uses /bin/true as interpreter)
r6_migration.py      F-16 (non-atomic migration runner; synthetic failing migration)
r7_discard.py        F-05 (discard deletes a shard file, then DB rolls back)
r8_nan.py            F-03 (NaN serialized as invalid JSON)
Each prints what it observed; none modify the repo (they use temp dirs).
