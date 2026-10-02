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
            main:   ComfyUI's own location (default)
                 or this project's storage, for large files
            reserve: optional second root, consulted only if main misses

       3. the rest of the configuration
```

The venv step is two lists and a constraints file: `requirements.txt` for
our own venv (torch included), `requirements-comfy-additions.txt` for
theirs (torch excluded), and `-c` a freeze of what their venv already
holds. Any conflict fails; nothing already installed is touched. That is
what "soft install" has to mean for it to be safe.

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

The last category is why the lists are split (section 2a).

That last category is the one that makes the ComfyUI-venv option
attractive *and* risky, and it is the next section.

### 2. The ComfyUI venv option

Reusing ComfyUI's venv saves real disk — torch plus the XPU backend is
multiple gigabytes, and a user who already has ComfyUI has it already.
That is a good reason and the user's stated one.

**The cost is that we would be installing into a venv the user did not
create, that another application depends on.** Unconstrained, that ends as
"ComfyUI broke and the installer did it".

It does not have to end that way, and the reason is the next section:
install only the *additions*, under a constraints file that makes changing
anything already present impossible. **"ComfyUI broke" becomes unreachable
rather than merely unlikely** — which is a better property than a
carefully-reviewed upgrade path.

What still needs deciding, and cannot be engineered away:

* **Never silently.** The user is asked, and the answer is recorded so the
  next run does not ask again.
* **Never touch it during a normal start.** Only during an explicit
  install step. A server that re-resolves its venv on every boot is a
  server that can break a user's ComfyUI on a Tuesday.
* **Freeze what we installed**, into the project's config, so the next
  time the user can be told what the additions were instead of being asked
  to diff a pip log.

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

### 2a. Two requirement lists, and a soft install

The reason this can be safe is that the installer is not installing
"this project's dependencies" into ComfyUI's venv. It is installing
**the additions**, and it is built so that *nothing already installed can
change*.

| List | Contents | Used for |
|---|---|---|
| `requirements.txt` (full) | everything, **including** torch and the XPU stack | our own venv |
| `requirements-comfy-additions.txt` | only the server's own packages — fastapi, uvicorn, python-multipart, tomli_w. **No torch.** | inside ComfyUI's venv |

The split is the first half of the safety. torch and the accelerator stack
are exactly what ComfyUI already has and pins; listing them in our
additions would invite pip to "fix" a version it thinks is wrong, which is
the failure we are avoiding. Anything in the additions list must be a
package ComfyUI has no reason to pin.

The second half is a **constraints file built from the target venv's
current state**:

```
pip freeze --all > comfy-constraints.txt          # the user never sees this
pip install -r requirements-comfy-additions.txt -c comfy-constraints.txt
```

A constraints file says "these exact versions are already here and must
not move". pip then *cannot* upgrade, downgrade or remove anything already
present — it stops and reports instead. That turns "we hope it does not
break ComfyUI" into "it is not able to break ComfyUI".

So the soft install has exactly one failure mode, and it is the right one:

> **Any conflict fails the install. Nothing is overridden. Nothing already
> installed is touched.**

Which is the "if something fails, tell me and stop" path of the flow
above — reached by construction rather than by catching an error and
apologising.

Two details worth writing down:

* **A refusal is a good outcome.** "Refused: your venv has fastapi
  0.111 and we need >=0.115, and I will not change your ComfyUI's copy" is
  a successful run. The user then picks the new-venv option knowingly.
* **Freeze what we actually installed**, into the project's config, so the
  next run of the installer can tell the user what the additions were
  rather than asking them to diff a pip log.

### 4. Where model files live, and looking in two places

The user chooses a **main** and may add a **reserve**:

* **ComfyUI's own location** — `<comfy>/models/{checkpoints,loras}` — is
  the natural default for main. Already what `paths.py` resolves, and it
  shares models with ComfyUI for free.
* **This project's own storage** is the other main, for large files.
  Checkpoints are routinely gigabytes; keeping them beside ComfyUI's
  `models/` is a default that suits a laptop badly.
* A **reserve** is a second root that is consulted only when the main does
  not have the file.

That is the whole idea: main is where things go, reserve is somewhere to
*also look*. The reason it is worth having is a real failure mode — a save
that did not land where it was supposed to. The model exists, it is on
disk somewhere, and without a second root the only options are "fail" or
"go find it by hand".

#### The rule, and where it stops helping

The question worth answering precisely, because it decides the
implementation:

> **Resolution is by lookup, not by fallback-on-error.** Ask "is this name
> in main?" If yes, that is the answer. If no, ask the reserve. A name in
> both is **main's**, without checking whether main's copy is any good.

So, concretely:

| Operation | Rule |
|---|---|
| Resolve a named model | main first, then reserve; first hit wins |
| List (picker, browse) | union of both, main's entries first |
| Upload, save, anything that creates | **main only** |
| Show in the picker | which root each file came from |

Nothing is "tried" and nothing is "probed". That is what makes it cheap —
no file is opened to decide where it lives — and it means the answer is
the same every time, which is what makes it debuggable.

**The limit, stated because it is the case the reserve does *not* save.**
If a save was interrupted, the likely result is a *truncated file in main*,
not a missing one. A truncated file in main is found in main, so the
reserve is shadowed and does not help. The reserve rescues the case where
main genuinely has nothing; it does not rescue a corrupt file in main.

Two honest options for that, and the first is the default:

1. **Main is authoritative.** The picker shows which root a file came
   from, and the user deletes the bad one. One less thing to get wrong, and
   "I deleted it and it re-downloaded" beats a silent fallback.
2. **Validate on read, fall back on a bad header.** A safetensors header is
   a few hundred bytes and a truncated file is trivially detectable — but
   this means opening files to decide resolution, which is the wrong shape
   for a picker listing thousands of models. Possible as a targeted check
   when a load *fails*, not as part of resolution.

This is the honest boundary of the feature, and it is better to write it
down than to let someone discover it.

#### What has to change

Model resolution today is *one root with an override*, in five places:

* `path_tiers.checkpoints_dir` / `loras_dir` return one `Path`.
* `models_dir` sets one root.
* `FileSystemAssetStore._list_model_files` globs one directory.
* The asset pickers and browse endpoints list one directory.
* `nodes/` code resolves model paths through `paths.py`, one directory.

All five become main-plus-reserve. The `models_dir` setting has to keep
working as-is — it is documented and in use — so it becomes the *main*,
and a new optional key carries the reserve. **Not** a migration.

Consequences to design for:

* **`WorkspaceDirs` grows.** It already exists because this project depends
  on another project's layout and that dependency should be a
  configuration point; this is that assumption being tested, and it
  already has per-directory fields that can take the reserve.
* **Every consumer has to be found, not just the obvious ones.** A
  one-root resolver left unchanged in one place is a picker that quietly
  shows half the files — and the bug is invisible, because half the files
  *is* a plausible-looking answer.

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
* **Model files are searched in main-then-reserve, written to main only**,
  and main wins even when main's copy is broken — the one case the reserve
  cannot rescue.
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
  or a filesystem the user mounts. The plan only needs *a main and a
  reserve*.
* Whether the reserve is offered as a suggestion or just accepted as a
  path. Nothing here assumes it is discovered automatically.