# Review notes -- flagged, not yet resolved

A running list of things noticed while working in this repo that don't
have a clear owner or an obvious fix, kept separate from
`docs/known-issues/` because these are about the *documentation and
project hygiene*, not the training code itself. Two kinds of entries:
things that are **confusing or internally inconsistent** (worth fixing
for clarity, not necessarily bugs), and things that are **judgment
calls for a maintainer**, not something checkable from the repo alone.

This file should shrink over time, not grow -- once an item below is
checked and resolved, delete it rather than marking it "done" and
leaving it here. (Two large batches of previously-listed items were
resolved and removed rather than kept as a record: first a docs
restructuring pass, a progress-file resync, and several
dangling-reference cleanups on 2026-09-15; then, on 2026-09-28, the
real-hardware confirmation of the five `pending-testing.md` fixes, the
`convert-cfg.toml` handling decision (now gitignored, with a committed
`convert-cfg.example.toml`), the B580-hardware wording (confirmed
directly against the actual device), the phantom `2c1f0ff` commit
trailer (annotated), and the single test command (now
`run_tests.py`). Check `git log -- docs/review_notes.md` if the
history of what used to be here matters.)

## Still open

1. **Cross-document section-number references are structurally
   fragile -- a real, ongoing risk, not a one-time cleanup item.** Both
   docs and source-code comments reference `docs/design/`'s files by
   section number (e.g. "design 3.1," "section 11.3"). Nothing
   automates that correctness. If a section ever gets renumbered,
   inserted, or moved to a different file, every citation of its old
   number -- across `docs/design/` itself and dozens of source files --
   silently goes stale with no mechanism to catch it. Flagged as a
   standing design trade-off (stable per-section anchor IDs would fix
   this structurally), not a task with an end state.
