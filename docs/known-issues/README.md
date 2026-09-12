# Known issues / suspicious findings / deferred work

*New here? Start at the root [`README.md`](../../README.md). This is
the bug/quirk tracker for this project -- check here before assuming
something odd you've hit is new. See
[`docs/review_notes.md`](../review_notes.md) for a note on how current
this collection is likely to be relative to the most recent work.*

This folder used to be one file, `docs/suspicious_findings.md` (~525
lines), organized into four status sections. It's been split into one
file per status so each can be scanned on its own -- content is
unchanged from the original, only reorganized. Entries within each
file are newest-first, as before.

**Note (2026-08): this collection predates and sits outside the
`nodes/` design-doc effort** (see [`../design/`](../design/README.md)).
Unless a `nodes/` path is named explicitly, an entry describes the
legacy `core/` production pipeline, not the design doc's rewrite. Treat
this as an informal, unaudited collection, not a spec -- entries can be
stale, already fixed elsewhere, or (as two entries here used to be)
about a feature `nodes/` never had in the first place: `nodes/`
currently implements plain supervised LoRA training only, no
distillation and no chain-mixing of any kind, so a DAgger/chain-mixing
finding about `core/trainer.py` isn't a `nodes/` backlog item and was
removed rather than carried forward. Two dangling pointers to
`docs/optimizer_execution_redesign_plan.md` and
`docs/nodes_package_design.md` (both deleted) were also cleaned up here
previously -- the substantive content they pointed from is kept, just
not the broken link. (Other dangling pointers to those same two deleted
files still exist elsewhere in the codebase, outside this doc --
flagged in [`docs/review_notes.md`](../review_notes.md) item 6.)

## The four categories

| File | What's in it |
|---|---|
| [`open.md`](open.md) | Confirmed or suspected issues with no fix landed yet. Check here first if you've hit something odd. |
| [`resolved.md`](resolved.md) | Fixed, with the real root cause and the fix location -- kept as a record so the same investigation doesn't happen twice. |
| [`deferred.md`](deferred.md) | Confirmed but not urgent -- real findings, intentionally not acted on yet. |
| [`pending-testing.md`](pending-testing.md) | Code-level fixes that exist but haven't been confirmed against real hardware/training runs yet. |
