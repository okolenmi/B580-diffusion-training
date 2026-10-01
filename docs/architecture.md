# Architecture overview

A map of the codebase, not a design rationale -- for *why* things are
shaped this way, see `docs/design/` (the "why" lives there in depth,
see that folder's own `README.md` index; this doc just orients a new
reader fast).

## The two pipelines

| | TOML trainer | Node-graph rewrite |
|---|---|---|
| Packages | `core/`, `manager/` | `nodes/`, `backend/` |
| Entry point | `python -m core.cli` + a TOML config | `run_server.sh` (browser UI) |
| Status | Current production path | Active development; reuses the legacy pipeline where nothing better exists yet, replaces it domain by domain where it does |
| Config style | One big TOML file, many flat fields | A visual graph of typed `Node`s wired together |

`core/` in particular is not dead code -- it is the training engine
every path runs through, including the web UI's. See
[`core-inventory.md`](core-inventory.md) for the dependency map and the
capabilities that exist only there.

**The legacy pipeline (`core/`, `manager/`) is not modified by this
project** (section 9.3 of `docs/design/08-validation-and-implementation-status.md`
says this explicitly) -- it's the current production path, and bugs
found in it while building the rewrite get fixed in place, not
restructured. What *has* changed: `core/`/`manager/` used to be treated
as permanent reference material that `nodes/` would always wrap rather
than reimplement. That's no longer the rule. Where `nodes/` has since
built its own independent, verified-equivalent version of something
`core/` does (the `optimizer/` domain's `Algorithm`+`ExecutionStrategy`
split; `components/diffusion.py`'s noise-schedule/parameterization
objects), that version is canonical and the old `core/`-wrapping `Node`
gets retired. The `optimizer/` domain is fully unified this way as of
2026-10-02: the last holdout, `AdafactorOptimizerNode`, is deleted, and
`nodes/optimizer/` now imports nothing from `core.optimizers` (see
`docs/known-issues/open.md` for the one unmeasured performance trade that
retirement accepted). Text encoding was unwired the same way on 2026-10-02: `SDXLClipEncoder`
moved to `nodes/model/clip_encoder.py` (it was self-contained, so this
was a relocation, not a reimplementation), and `core/clip_encode.py` is
now a re-export shim for `core/`'s and `manager/`'s own use.

LoRA/UNet injection and dataset ingestion still wrap `core/`/`manager/`
directly (tracked in `docs/design/09-prioritized-backlog.md`), and those
are harder than the two that just went: `core.unet_wrapper.ComfyUNetWrapper`
is the model every LoRA path in the graph is built on, and `core.lora`'s
`_inject_lora` is a tree-walk that `nodes/` substitutes its own layer
classes into by patching module-level names. Wrapping is the fallback
for a domain nobody's rewritten yet, not a destination.

## Design principles, in short

Full statement and reasoning: the root `README.md`'s "Goals" section
(the seven constraints every design choice here is checked against).
The one worth internalizing
before touching `nodes/` code: **a `Builder` (construction-time,
config-in/runtime-object-out) is a different kind of thing from a
runtime object (real state, called every training step)** -- collapsing
the two is the specific anti-pattern this whole rewrite exists to move
away from.

For what's actually real versus built-but-unvalidated, read
`docs/design/08-validation-and-implementation-status.md` section 9.2;
for the backend layer's decisions and contracts,
`docs/design/backend/01-architecture.md`.
