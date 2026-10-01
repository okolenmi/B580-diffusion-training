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

- **The TOML trainer (`core/` + `manager/`)** -- a config-driven
  command-line trainer. This is the current *production* path: what real
  training runs actually use today. Entry point: `python -m core.cli`.
- **The node-graph pipeline (`nodes/` + `backend/`)** -- a strict-OOP
  redesign of the same pipeline, exposed through a browser-based visual
  node editor served by `backend/` (REST API under `/api/v1`, layered
  application design, own docs under
  [`docs/design/backend/`](docs/design/backend/README.md)). This is
  where new design work lands. Entry point: `run_server.sh`. (The
  first-cut web layer `server/` was retired to `archive/server/` at M9;
  `backend/` is its clean-room replacement.)

These are one trainer with two front ends, not an old system and a new
one. The web UI's Start button launches `python -m core.cli` as a
supervised subprocess, so both entry points run the same math.

The project does **not** rewrite working code to reimplement it. Each
domain is adopted where a verified-equivalent `nodes/` version exists
and the old wrapper is retired; wrapping is the fallback for a domain
nobody has rewritten yet, not a destination. As of 2026-10-02 the
optimizer domain, text encoding and LoRA/UNet injection have all been
adopted, and `nodes/` imports nothing from `core/` --
[`docs/core-inventory.md`](docs/core-inventory.md) maps what still does
and why (see also
[`docs/architecture.md`](docs/architecture.md)).

## Goals

Stated once here, referenced rather than repeated throughout:

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
# TOML trainer -- run from this project's own root directory
python -m core.cli --config config.toml

# Node-graph web UI -- also from this project's root
./run_server.sh   # serves on http://0.0.0.0:8766 by default
```

## Where to find things

The rule the docs now follow: **a document earns its place by saying
something the code cannot.** Design rationale, rejected alternatives
with their reasoning, hard-won constraints, hardware measurements and
deferred-with-a-reason all stay. File trees, endpoint tables,
implementation-status inventories and phase-by-phase plan narratives
were removed in the 2026-10-01 cleanup -- open the module instead, or
ask the server, which serves its own OpenAPI schema.

```
docs/
├── setup.md                 Environment setup, running either pipeline, running tests
├── architecture.md          Codebase map: core/manager/backend/nodes and how they relate
├── core-inventory.md        What `core/` is (the training engine) and the eight things only it has
├── training-diagnostics.md  Fixed-probe / gradient-alignment tools: is a LoRA damaging a t region?
├── review_notes.md          The one open documentation-hygiene judgment call
├── known-issues/            Bug/quirk tracker, split by status. Cited from source
│   ├── open.md              Issues with no fix landed -- check here before assuming something is new
│   ├── resolved.md          Measured results kept as prose (runs/ is gitignored, so this IS the record)
│   ├── deferred.md          Confirmed, intentionally not acted on
│   └── pending-testing.md   Fixes that exist but were never confirmed against real hardware
└── design/
    ├── README.md            Index of the nodes/ rationale docs -- start here
    ├── 02..10               One file per topic: rationale and evidence, not reference
    ├── resources-controller/  The Resources Controller redesign, incl. the hardware results
    └── backend/                 The web backend and its frontend: architecture,
                                  API contract, dataset format, graph runtime,
                                  what is deferred and why
```

Two entry points if you do not know where to start:

- [`docs/design/README.md`](docs/design/README.md) -- the index for
  the `nodes/` rationale, including **07-deferred-or-rejected.md, which
  you should read before proposing anything**.
- [`docs/design/resources-controller/README.md`](docs/design/resources-controller/README.md)
  -- the most recently active work, and the most current hardware
  numbers.

For the backend specifically, `docs/design/backend/README.md` indexes
those docs, and `01-architecture.md` holds the layering rules a change
has to keep.

## Current status, in one paragraph

The `nodes/` rewrite's original 12-item backlog is complete and
equivalence-tested; `docs/design/08-validation-and-implementation-status.md`
section 9.2 is the honest answer to "what is built but *not* yet
validated", and `09-prioritized-backlog.md` says what is left. Since
then the work has moved to the Resources Controller and precision
redesign, and the web layer has been rewritten behind `backend/`
(M1-M9 shipped, `server/` archived). The five previously
hardware-unconfirmed fixes have all been run and confirmed on the real
B580, with numbers, recorded in
[`docs/known-issues/resolved.md`](docs/known-issues/resolved.md); the
reusable harness for that is `scripts/hw_validate.py`.

## A note on how these docs are meant to be maintained

Delete a document when its content has moved somewhere better, not when
it gets old: if the code now says it, the code says it. When something
is deferred or rejected, keep the *reason* -- that is the part nobody
can reconstruct -- and let the rest go.

`scripts/check_doc_links.py` runs in `scripts/full_gate.sh`. It
resolves every markdown link, every heading anchor, and every
`docs/**.md` path cited from a source comment, because nothing renders
these docs and nothing else would notice a citation rotting.

`docs/review_notes.md` holds the one open documentation-hygiene judgment
call. It is meant to shrink: fix an item and delete it rather than
marking it done. The 2026-10-01 cleanup removed three documents this
way, and `git log` has them.
