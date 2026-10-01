# Architecture overview

A map of the codebase, not a design rationale -- for *why* things are
shaped this way, see `docs/design/` (the "why" lives there in depth,
see that folder's own `README.md` index; this doc just orients a new
reader fast).

## The two front ends

| | TOML trainer | Node-graph pipeline |
|---|---|---|
| Packages | `core/`, `manager/` | `nodes/`, `backend/` |
| Entry point | `python -m core.cli` + a TOML config | `run_server.sh` (browser UI) |
| Status | Current production path | Active development; adopts each domain as its own version becomes available |
| Config style | One big TOML file, many flat fields | A visual graph of typed `Node`s wired together |

One trainer, two front ends. The web UI's Start button launches
`python -m core.cli` as a supervised subprocess, so both entry points
run the same math.

**How domains move between them.** This project does not reimplement
working code. Where `nodes/` builds its own independent,
verified-equivalent version of something `core/` does, that version
becomes canonical and the `core/`-wrapping code is retired -- wrapping
is the fallback for a domain nobody has rewritten yet, not a
destination. Three domains have moved that way as of 2026-10-02:

* **The optimizer domain** (`Algorithm` + `ExecutionStrategy` split),
  fully: the last holdout, `AdafactorOptimizerNode`, is deleted. See
  `docs/known-issues/open.md` for the one unmeasured performance trade
  that retirement accepted.
* **Text encoding**: `SDXLClipEncoder` moved to
  `nodes/model/clip_encoder.py`. It was self-contained, so this was a
  relocation rather than a reimplementation.
* **LoRA/UNet injection**: `LoRALinear`/`LoRAConv2d`, `_inject_lora`,
  `ComfyUNetWrapper` and `derive_seed` moved to `nodes/model/` and
  `nodes/components/`. Owning the walk let `_inject_lora` take the
  adapter classes as an argument instead of having `nodes/` rebind the
  walk's module globals to change what it built -- which removed a
  concurrent-build race, deleted `lora_class_cache.py`, and fixed four
  `isinstance` gates that had been silently skipping every DoRA and NF4
  layer.

`core/` itself is not dead code: it is still the production trainer,
`backend/` still spawns it, and its own modules have capabilities no
`nodes/` version provides (mid-run previews, latent caching, the
teacher-trajectory builders). What has changed is that `nodes/` no
longer depends on any of it -- see
[`core-inventory.md`](core-inventory.md) for the dependency map.


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
