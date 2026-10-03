Reproductions for docs/design/backend/07-review-2026-10-01.md (r1..r8) and
for the third external review, archive/review-r3/ (r14..r16).
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

r14..r16 come from the third external review (archive/review-r3/). They need
COMFY_DIR set, because they spawn a real graph execution child, and r14/r16
exit 0 when their finding no longer reproduces.
r14_watcher_crash_live_child.py  N3-01, FIXED. A transient read error marked
    the row failed while the child kept running, with no stop and no kill --
    so the single-active check then allowed a second trainer on the same card.
    Prints "not reproduced" now: the row is finalised with a note that
    supervision failed, and the child is stopped and killed.
r15_stale_event_id.py             N3-04, OPEN. A Last-Event-ID from a previous
    server process is accepted as valid once the new process has got that far.
r16_graph_finished_while_down.py  N3-02, FIXED before this review was even read
    (commit 13563d6). Prints status=finished results=3/3; before that fix it
    printed status=error results=0/3.
