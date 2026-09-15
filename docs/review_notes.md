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
leaving it here. (A large batch of previously-listed items -- a docs
restructuring pass, a `progress.md` resync, several dangling-reference
cleanups -- was resolved and removed from this file on 2026-09-15,
rather than kept as a record. Check `git log -- docs/review_notes.md`
if the history of what used to be here matters.)

## Still open

1. **`docs/known-issues/pending-testing.md`'s one entry (the non-square-
   dataset VRAM ratchet fix, `max_aspect_ratio` in
   `manager/builder.py`'s `run_lora_ingestion_task`) may have been
   confirmed on real hardware since without the doc being updated** --
   not checkable from the repo alone, needs whoever has the actual B580
   hardware to re-ingest a non-square dataset and confirm
   `vram_reserved` stays bounded, then move the entry to `resolved.md`
   if so.

2. **`convert-cfg.toml` (repo root) contains what looks like one
   specific person's real setup, not an anonymized template**: a
   literal filesystem path (`comfy_dir = "/home/okolenmi/comfy/ComfyUI/"`),
   a dataset named `"test"`, and what looks like a real trigger word
   (`"ttw001"`) in the preview prompts. If this is meant as a
   copy-and-edit example for new users, it'd be worth genericizing the
   path and calling that out in a comment; if it's someone's actual
   working config that ended up committed, it's worth confirming
   whether it should be gitignored instead. Not changed here since it's
   a judgment call for whoever owns that file, not a docs question.

3. **This project's Intel Arc B580 / XPU focus is stated in the docs
   (root `README.md`, `docs/setup.md`) as an inference, not a
   confirmed fact from an authoritative source.** It's a reasonable
   inference (the repo's own name, `device = "xpu"` throughout example
   configs, and a `docs/known-issues/` entry that references "the same
   B580 hardware" when comparing against another project's discussion)
   -- but nothing in the repo states this as a design constraint
   directly. Worth a maintainer confirming the wording in those two
   docs is accurate, and correcting it if there's more nuance (e.g.
   whether other hardware is also actively supported/tested).

4. **`docs/design/resources-controller/08-consolidation.md`'s "Last
   synced" trailer references commit `2c1f0ff`, which doesn't resolve
   to a real commit object in this repository's history at all** --
   not just stale, actually unfindable, likely from a rewritten/rebased
   history predating whatever was originally cloned. Left as-is
   (annotated with this finding rather than a guessed replacement) --
   that file is part of the Resources Controller design docs, out of
   scope for anyone working outside that route. Worth a maintainer who
   does own that route clarifying what the trailer is actually supposed
   to mean and re-syncing it properly.

5. **No single command runs this project's entire test suite.**
   `nodes/smoke_tests/run_all.py` is a convenience runner, but it only
   covers `nodes/smoke_tests/` -- `server/smoke_tests/` and
   `manager/smoke_tests/` each need to be run individually, as noted in
   `docs/setup.md`. Small enough it's probably not worth much design
   thought, but worth a one-line top-level runner if this becomes
   annoying enough to notice again.

6. **Cross-document section-number references are structurally
   fragile -- a real, ongoing risk, not a one-time cleanup item.** Both
   docs and source-code comments reference `docs/design/`'s files by
   section number (e.g. "design 3.1," "section 11.3"). Nothing
   automates that correctness. If a section ever gets renumbered,
   inserted, or moved to a different file, every citation of its old
   number -- across `docs/design/` itself and dozens of source files --
   silently goes stale with no mechanism to catch it. Flagged as a
   standing design trade-off (stable per-section anchor IDs would fix
   this structurally), not a task with an end state.
