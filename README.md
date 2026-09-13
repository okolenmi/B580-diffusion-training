# B580 Diffusion Training

A LoRA training pipeline for SDXL, built as a set of ComfyUI-adjacent
tools, developed and tuned against Intel Arc B580 (XPU) hardware.

This file is the map. It stays short on purpose -- every topic that
needs real depth lives in its own doc under `docs/`, linked below.
Read this file first, then jump to whichever doc matches what you're
actually trying to do.

## What this project actually is

Two parallel systems live in this one repository, at different levels
of maturity:

- **The legacy pipeline (`core/` + `manager/`)** -- a config-driven
  (TOML) command-line trainer. This is the current *production* path:
  what real training runs actually use today. Entry point: `convert.py`.
- **The node-graph rewrite (`nodes/` + `server/`)** -- a from-scratch,
  strict-OOP redesign of the same training pipeline, exposed through a
  browser-based visual node editor. This is where new design work
  lands; it wraps and reuses legacy code rather than duplicating it,
  and is not yet a full production replacement for the legacy path.
  Entry point: `run_server.sh`.

`core/`/`manager/` are treated as reference material by the `nodes/`
rewrite -- correct, working code that gets wrapped, not rewritten, per
the project's own stated rule (see
[`docs/architecture.md`](docs/architecture.md)).

## Goals

Stated in full in
[`docs/design/01-design-goals-and-constraints.md`](docs/design/01-design-goals-and-constraints.md);
summarized here:

1. **VRAM first, speed second** -- every VRAM-saving choice either
   costs nothing in speed or has a named, estimable speed cost.
2. **Strict OOP** -- behavior lives on objects implementing declared
   interfaces, not on flag-driven functions or `hasattr()`/`isinstance()`
   sniffing.
3. **No singletons** -- shared state is passed in explicitly, never
   read from a module global.
4. **Composition over inheritance, and over rewriting** -- new
   capability is a new small class implementing an existing interface;
   verified old code gets wrapped, not re-derived.
5. **One reviewed place for device-memory lifecycle** -- every reusable
   device buffer goes through `nodes/memory/manager.py`'s `MemoryManager`.
6. **Don't overcomplicate** -- every abstraction exists because a
   concrete, named problem needs it.
7. **Modern techniques earn a place only with real evidence** -- a
   specific paper/source, plus an honest calibration of how far to
   trust it.

## Quick start

Full setup detail (folder layout, `.env`, running either pipeline,
running the test suite) is in [`docs/setup.md`](docs/setup.md). The
short version:

```bash
# Legacy CLI trainer -- run from ComfyUI's own root directory
cd /path/to/ComfyUI
python /path/to/this-project/convert.py --config my_run.toml

# Node-graph web UI -- run from this project's own directory
./run_server.sh   # serves on http://0.0.0.0:8765 by default
```

## Where to find things

```
docs/
├── setup.md                 Environment setup, running either pipeline, running tests
├── architecture.md          Codebase map: core/manager/server/nodes and how they relate
├── review_notes.md          New-reader audit: confusing/stale things flagged for follow-up
├── status/
│   └── progress.md          What's implemented in nodes/ today (STALE -- see banner in the file)
├── known-issues/            Bug/quirk tracker, split by status
│   ├── open.md
│   ├── resolved.md
│   ├── deferred.md
│   └── pending-testing.md
└── design/                  The full nodes/ rewrite design, one file per topic
    ├── 01-design-goals-and-constraints.md
    ├── 02-foundational-ontology.md
    ├── 03-training-step-orchestration.md
    ├── 04-lora-adapter-mechanics-and-loss-weighting.md
    ├── 05-coordination-registry-observability.md
    ├── 06-composition-walkthrough.md
    ├── 07-deferred-or-rejected.md
    ├── 08-validation-and-implementation-status.md   <- most reliable "what's real" doc
    ├── 09-prioritized-backlog.md
    ├── 10-node-surface-and-precision-control.md
    └── resources-controller/    Active, in-progress redesign -- most recently updated
        ├── 01-context-and-ground-truth.md
        ├── 02-phase-1-and-2.md
        ├── 03-phase-3-interactive-node-support.md
        ├── 04-phase-4-resource-preset-abstraction.md
        ├── 05-phase-5-resources-controller-node.md
        ├── 06-phase-6-lora-training-config.md
        ├── 07-post-phase-6-bugfixes.md
        └── 08-consolidation.md
```

Each folder with more than one file has its own `README.md` index with
a "read this when..." table -- go there rather than guessing which
numbered file you need. The two most useful entry points if you don't
know where to start:

- [`docs/design/README.md`](docs/design/README.md) -- full rationale
  for the `nodes/` rewrite; its own index points at "what's actually
  real" (file 08) vs "what's still open" (file 09).
- [`docs/design/resources-controller/README.md`](docs/design/resources-controller/README.md)
  -- the most recently active work in the repo; its status banner is
  the single most current summary of anything resource-policy or
  precision related.

## Current status, in one paragraph

The `nodes/` rewrite's original 12-item backlog is complete and
equivalence-tested (see
[`docs/design/08-validation-and-implementation-status.md`](docs/design/08-validation-and-implementation-status.md)).
Since then, work has moved to a Resources Controller / precision
redesign
([`docs/design/resources-controller/`](docs/design/resources-controller/README.md))
that is itself now well past its "Phase 6" milestone, plus a live
per-step VRAM budget enforcer and 8-bit optimizer-state quantization.
**[`docs/status/progress.md`](docs/status/progress.md) predates all of
that** -- see [`docs/review_notes.md`](docs/review_notes.md) for
specifics. Treat
[`docs/design/resources-controller/README.md`](docs/design/resources-controller/README.md)'s
own status banner (top of that file) as the most current single source
of truth for what's actually landed recently.

## A note on how these docs are meant to be maintained

`docs/design/` and `docs/design/resources-controller/` are living
planning documents, not archives -- they're written to be edited in
place as work lands (status banners at the top, sections marked
"done"/"still open" inline) rather than superseded by a new file each
time. Keep doing that: it's why they're still trustworthy despite their
size. `docs/status/progress.md` is supposed to work the same way but
has drifted (see [`docs/review_notes.md`](docs/review_notes.md)) -- a
reminder that this pattern only works if it's actually kept up.

This folder structure is new as of a recent docs-restructuring pass --
the content that used to live in four large files (`PROGRESS.md`,
`docs/training_pipeline_design.md`,
`docs/resources_controller_redesign_plan.md`,
`docs/suspicious_findings.md`, all at the repo/`docs/` top level) has
been split by topic into the folders above, with the content itself
preserved (verified line-for-line during the split) and only
reorganized. A follow-up pass then updated every source-code comment
and docstring that referenced the old flat paths (52 files across
`nodes/`, `server/`, `manager/`) to point at the correct split file --
most citations named a specific section/phase number, which had to be
looked up against the section-number-to-file mapping in
`docs/design/README.md` and `docs/design/resources-controller/README.md`
rather than mechanically renamed, since one old path now maps to up to
eleven different files. A handful of citations pointed at content from
two now-deleted docs (`docs/nodes_package_design.md`,
`docs/optimizer_execution_redesign_plan.md`) that predated this
restructuring entirely; where the cited content still exists somewhere
current it's now pointed there, and where it doesn't survive anywhere,
the comment was reworded to state the fact directly rather than cite a
source that isn't there. See
[`docs/review_notes.md`](docs/review_notes.md) for the full accounting
of what was found and fixed.
