# 11. First-run installer and dependency gate

*[← design index](README.md)*

**Status: plan, not built.** Nothing in this document exists yet. It is
written down because the flow has decisions in it that are much cheaper to
make now than to debug later — particularly the one about installing into
a venv the user did not create.

## The problem

Today, starting this project is a conversation the user has to have with
the documentation. `run_server.sh` auto-detects ComfyUI and an interpreter;
if either is missing it falls through to an error from deep inside
`paths.py`. `requirements.txt` is a list nobody is required to install.
A wrong interpreter produces a wall of identical
`ModuleNotFoundError: No module named 'torch'` tracebacks (a failure mode
`run_tests.py` already works around, which is evidence of how much it
bites). A model path that does not resolve produces a settings page with
a null in it.

None of that is *hard*, but all of it is friction, and it is friction a
new user pays before seeing anything work.

## What exists already

Do not rebuild these; the installer is mostly a front end over them.

| Already there | Where |
|---|---|
| Auto-detection of ComfyUI dir, interpreter, checkpoint/LoRA dirs, runs dir | `paths.py` |
| The resolution *policy*, in one place, with the tier order documented | `backend/infrastructure/path_tiers.py` |
| A way to override every model directory in one call | `WorkspaceDirs` — `backend/infrastructure/workspace.py` |
| One settings key for the whole model tree | `models_dir` — `path_tiers.models_dir` |
| Settings that validate, resolve and report what they resolved to | `POST/GET /api/v1/settings` |
| Dependency lists | `requirements.txt`, `requirements-dev.txt` |
| A Config page | `frontend/js/views/config.js`, route `/settings` |
| Localhost-only request guard | `backend/presentation/security.py` |

## What is missing

1. **A machine-readable list of what this project needs.** Nothing states
   it: not the Python version floor, not which packages, not which of them
   are optional, not how big they are.
2. **A check that runs before the server does.** `run_server.sh` starts
   the server and lets the failure surface as a traceback.
3. **A way to fetch what is absent** — opt-in, showing what will be
   downloaded first.
4. **A first-run state.** Nothing currently distinguishes "not configured"
   from "configured and working". The settings page renders happily with
   nulls in it.
5. **The venv decision**, which is the sharpest one below.
6. **Two model locations** rather than one (see below).

## The flow

```
run_server.sh
  │
  ├─ dependency check ──── missing ──► offer to download ──► done
  │                          │
  │                          └─ declined ──► stop, with the missing list
  │
  ├─ server starts (degraded: installer only, if unconfigured)
  │
  └─ installer wizard
       1. use the ComfyUI venv to save disk space?  ── no ──► create our own
                                     │ yes
                                     ├─ install succeeded ──────────────┐
                                     └─ failed (conflict)               │
                                          ├─► create a new venv ────────┤
                                          └─► stop: show the common     │
                                               reasons, nothing else     │
                                                                          │
       2. where do model files live?  ◄────────────────────────────────────┘
            ├─ ComfyUI's own location (default)
            └─ this project's storage, for large files

       3. the rest of the configuration
```

### 1. Dependency check, before the server

A `scripts/check_requirements.py` that answers *satisfied / missing /
optional-missing / wrong-version* per item and exits non-zero on a
missing one. Report-only at first — see the phasing.

The requirement list has to be data, not prose, and it has to distinguish:

* **required** — the server does not start without it
* **required only for training** — the UI works, a run does not
* **optional** — a capability that degrades quietly
* **provided by the ComfyUI venv** — torch and the XPU stack, which this
  project does not install into its own venv and should not try to

That last category is the one that makes the ComfyUI-venv option
attractive *and* risky, and it is the next section.

### 2. The ComfyUI venv option

Reusing ComfyUI's venv saves real disk — torch plus the XPU backend is
multiple gigabytes, and a user who already has ComfyUI has it already.
That is a good reason and the user's stated one.

**The cost is that we would be modifying a venv the user did not create,
that another application depends on.** `pip install` into it may upgrade
or remove a package ComfyUI needs, and the failure mode is "ComfyUI broke
and the installer did it". Specific requirements for doing this safely:

* **Never silently.** The user is asked; the answer is recorded in the
  project config so the next run does not ask again.
* **Dry run first.** Show the planned changes (`--dry-run` diff of what
  would be installed, upgraded, downgraded or removed) and require
  confirmation. A plain version bump of `fastapi` is fine; anything that
  would *remove* a package ComfyUI imports is not, and should be refused
  outright with the reason.
* **Record a receipt.** What was installed, at what versions, so a later
  "ComfyUI stopped working" can be traced to a version. Without this the
  feature is unrecoverable in practice.
* **Never touch it during a normal start.** Only during an explicit
  install step. A server that re-resolves its venv on every boot is a
  server that can break a user's ComfyUI on a Tuesday.

### 3. When installing into the ComfyUI venv fails

A conflict is expected, not exceptional — ComfyUI pins its own stack.

* **Report which packages conflicted**, in the user's terms. "torch 2.x
  requires X, but Y is pinned to Z" rather than a pip traceback.
* **Two options, and only two:**
  1. **Create a new venv** for this project and install there. Costs
     disk, which is the thing the user was trying to avoid, so the size
     estimate has to be shown before they choose.
  2. **Stop.** The user fixes it manually and re-runs.
* **On stop, show the most common causes and nothing more.** No third
  path, no "try again with a different resolver". A wizard that offers to
  guess at a dependency conflict is worse than one that stops and says
  what usually causes it — the causes are few and they are all visible to
  the user, who is the only one who knows what else they need installed.

This is a deliberate narrowness. The alternative — resolving conflicts
automatically — is the thing that would break a user's ComfyUI.

### 4. Where model files live, and looking in two places

The user chooses:

* **ComfyUI's own location** — `<comfy>/models/{checkpoints,loras}`. The
  default, already what `paths.py` resolves, and shares models with
  ComfyUI for free.
* **This project's own storage**, for large files. Checkpoints are
  routinely gigabytes; keeping them beside ComfyUI's `models/` is a
  default that suits a laptop badly.

**And then nodes that look for a LoRA or a checkpoint must look in both
places.** That is the part with real design in it, because model
resolution today is *one root with an override*:

* `path_tiers.checkpoints_dir` / `loras_dir` return one `Path`.
* `models_dir` sets one root.
* `FileSystemAssetStore._list_model_files` globs one directory.
* The asset pickers and browse endpoints list one directory.
* `nodes/` code resolves model paths through `paths.py`, one directory.

The change is from *a root* to **an ordered list of roots, with the
first as primary**. The principle that makes it tractable:

> **Read from many, write to one.**

Listing and search union across the roots in order, with the primary
first, so the picker's default selection is the primary's file when the
same name exists in both. Uploads, and anything else that creates a file,
always target the primary — a file written to root 2 that the picker lists
from root 1 would be a genuinely confusing bug.

Consequences to design for, not discover later:

* **A name in two roots.** Not an error. The primary wins for writes, and
  the picker should show *which* root a file came from, or a user
  debugging a "wrong version" problem has no way to tell.
* **A settings surface change.** One `models_dir` becomes a primary plus
  an ordered list. The existing single-value key has to keep working
  (it is a documented, in-use setting) — treat it as the one-element
  case, not as something to migrate.
* **`WorkspaceDirs` grows from a single root to a list.** Its docstring
  already says it exists because this project depends on another
  project's layout and that dependency should be a configuration point;
  this is that assumption being tested.
* **Every consumer has to be found**, not just the obvious ones. A
  one-root resolver left unchanged in one place is a picker that quietly
  shows only half the files.

### 5. The rest of the configuration

Already supported by the settings API: ComfyUI directory, interpreter,
model directories. What the installer should add rather than invent:
choose a default config, confirm the GPU is visible, and confirm a
checkpoint and a LoRA actually resolve — those three are the checks whose
failure the user would otherwise meet at the moment they press "start
training".

## Phasing

Each step is useful alone and keeps the suite green.

| Phase | What | Why it stands alone |
|---|---|---|
| A | `scripts/check_requirements.py`, report only | Turns an unknown into a list. Nothing else depends on it existing yet. |
| B | `run_server.sh` gates on A | The traceback becomes a sentence. **The single biggest win for the least work.** |
| C | Opt-in download | Requires B to be trusted. |
| D | First-run detection; server starts degraded | Needs B and C. |
| E | Installer wizard, CLI first | Proves the flow without designing a UI state machine. |
| F | Two model roots (read many, write one) | The biggest design item; independent of the wizard and worth doing on its own merit. |
| G | Installer UI; ComfyUI-venv option | The last and most delicate step, and the one that should not be first. |

**F before E/G is arguable** and worth deciding explicitly: multi-root
search is useful on its own (someone with models in two places today has
no way to use both), while the wizard is useful only as a whole.

## Decisions this implies, and where they should be recorded

Three of these are consequential enough to deserve a
[`docs/decisions/`](../decisions/README.md) record when they are built,
not a paragraph here:

* **We may install into a venv the user did not create.** With the safety
  requirements above, and the consequences if they are skipped.
* **Model files are searched in several roots, written to one.**
* **No authentication, but the installer is a local surface** — it writes
  to the filesystem and runs package installs, so it is exactly the
  capability ADR 0001 declines to add authentication for. It must be
  behind the same Host/Origin guard and reachable only in the unconfigured
  state.

## Not decided here

* Whether the wizard is only ever a CLI, or whether the server is allowed
  to start at all without configuration. (The flow diagram assumes the
  latter; the alternative is stricter and simpler.)
* Whether the ComfyUI-venv option is offered at all on a machine where the
  ComfyUI install cannot be verified.
* Whether "this project's own storage" is a directory this project defines,
  or a filesystem the user mounts. The plan only needs *a primary root
  and an ordered list of others*.