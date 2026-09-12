# Architecture overview

A map of the codebase, not a design rationale -- for *why* things are
shaped this way, see `docs/design/` (the "why" lives there in depth,
see that folder's own `README.md` index; this doc just orients a new
reader fast).

## The two pipelines

| | Legacy pipeline | Node-graph rewrite |
|---|---|---|
| Packages | `core/`, `manager/` | `nodes/`, `server/` |
| Entry point | `convert.py` + a TOML config | `run_server.sh` (browser UI) |
| Status | Current production path | Active development; wraps and reuses the legacy pipeline rather than replacing it outright |
| Config style | One big TOML file, many flat fields | A visual graph of typed `Node`s wired together |

**The legacy pipeline is treated as reference material by the rewrite**
-- correct, working code that gets *wrapped*, not re-derived or
copy-edited. This is a stated project rule, not an accident: where old
math is verified correct (UNet forward, LoRA layers, noise scheduling),
`nodes/` calls into it rather than reimplementing it. Concretely:
`core/`, `manager/` are not modified by the `nodes/` rewrite's own work
(section 9.3 of `docs/design/08-validation-and-implementation-status.md`
says this explicitly); bugs found *in* legacy code while building the rewrite get
fixed in place, but the rewrite doesn't restructure the legacy modules
themselves.

## Top-level layout

```
convert.py            Legacy CLI entry point (see docs/setup.md)
convert-cfg.toml       An example/real TOML config for convert.py
paths.py               Single source of truth for path resolution
                        (COMFY_DIR, VENV_PYTHON, dataset/model paths).
                        Both pipelines read this.
run_server.sh          Node-graph web UI entry point
server_cli.py          Thin CLI wrapper run_server.sh actually invokes

core/                  Legacy: trainer, optimizers, noise schedule,
                        LoRA math, UNet wrapper, VAE decode, caching.
manager/               Legacy: dataset ingestion/storage, a small
                        sqlite-backed dataset loader, preview generation.

nodes/                 The rewrite: typed Node/Port graph-construction
                        primitives (core.py) plus domain subpackages:
  ├─ components/         Rewritten, non-legacy versions of core/-level
  │                      concerns as they get migrated (see that dir's
  │                      own README.md for exactly what's landed).
  ├─ dataset/            Batch sourcing, prefetching, timestep modes.
  ├─ memory/             DeviceResident ABC, MemoryManager, resource
  │                      coordination/offload orchestration.
  ├─ model/              UNet/LoRA/DoRA construction, adapter
  │                      strategies, frozen-weight storage (incl. NF4),
  │                      the Resources Controller (see
  │                      docs/design/resources-controller/README.md).
  ├─ optimizer/          Algorithm x ExecutionStrategy x Handle
  │                      composition -- this subpackage is the rewrite's
  │                      own reference implementation of its house
  │                      style, cited throughout the design docs.
  ├─ primitive/           Small standalone value/utility nodes.
  ├─ train/              The training step pipeline itself
  │                      (TrainingStepPipeline/StepPhase), the trainer
  │                      node.
  └─ smoke_tests/        CPU-only tests for everything above; see
                         docs/setup.md for how to run them.

server/                Web server + graph executor for the nodes/ UI:
                        topological execution, port-compatibility
                        checking, the browser-side editor
                        (server/static/), REST routes per concern
                        (datasets, training, config, monitoring).

docs/                  This folder. See the root README.md's map for
                        what's where.
```

## Design principles, in short

Full statement and reasoning:
`docs/design/01-design-goals-and-constraints.md`, and the root `README.md`'s
"Goals" section for the condensed list. The one worth internalizing
before touching `nodes/` code: **a `Builder` (construction-time,
config-in/runtime-object-out) is a different kind of thing from a
runtime object (real state, called every training step)** -- collapsing
the two is the specific anti-pattern this whole rewrite exists to move
away from.

## Where the interesting complexity actually lives

If you're trying to understand *why* the codebase looks the way it
does rather than just *where things are*, these are the sections worth
reading in full rather than skimming:

- `docs/design/08-validation-and-implementation-status.md`, section 9
  (`Implementation status`) -- the single most reliable "what's
  actually real" table in this repo, more current than
  `docs/status/progress.md`.
- `docs/design/07-deferred-or-rejected.md`, section 7 (`Deliberately
  deferred or rejected`) -- saves you from re-proposing something
  already considered and rejected with real reasoning (GaLore, flow
  matching, automatic VRAM-pressure eviction, etc.).
- `docs/design/resources-controller/` -- the most recently active
  work, and the current best example of how a multi-session redesign
  gets tracked in this project (status banner at the top, edited in
  place as phases land).
