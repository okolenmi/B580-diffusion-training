# Review notes -- one open judgment call

About documentation and project hygiene, not about the training code
(those live in [`docs/known-issues/`](known-issues/README.md)).

**This file should shrink, not grow.** Fix an item and delete it rather
than marking it done. Three batches of previously-listed items have been
resolved and removed rather than kept as a record — the 2026-10-01
documentation cleanup was one of them, and it deleted three documents
whose content had moved somewhere better. `git log -- docs/` has the
history of what used to be here.

## Still open

1. **Cross-document section-number references are structurally fragile —
   a standing trade-off, not a task with an end state.** Docs *and*
   source comments reference `docs/design/` files by section number
   ("design 3.1", "section 11.3"). Nothing checks that those numbers
   still point at the same section, so a renumbering, insertion or move
   silently invalidates every citation of the old number — across the
   design docs themselves and dozens of source files.

   Partly mitigated on 2026-10-01: `scripts/check_doc_links.py` (in
   `full_gate.sh`) now resolves every markdown link, heading anchor and
   `docs/**.md` path cited from source. That catches a citation whose
   *file* is gone, and an anchor that does not exist — but a *number*
   inside prose ("see 3.1") is invisible to it.

   The structural fix, if anyone wants it: stable per-section anchor
   IDs, so citations point at an ID rather than a position. Nothing
   depends on the answer yet.