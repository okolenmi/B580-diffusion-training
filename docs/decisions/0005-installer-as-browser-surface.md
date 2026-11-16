# 0005 — The first-run installer is a browser surface, not a shell gate

## Context

`archive/old-documentation/design/11-first-run-and-installer.md` was written before this
project had a settings page, and it proposed starting with a shell gate:
`run_server_linux.sh` checks dependencies before launching the server and
explains what is missing instead of surfacing a `ModuleNotFoundError`.

That was the right instinct and the wrong mechanism for what this now is.
The plan's own Phase D already assumed "server starts (degraded: installer
only, if unconfigured)", and by the time the work happened the answer to
"Can this machine run a training step?" had a home: the settings API, the
node catalog, the graph validator. A shell gate answers a question in a
terminal, before the user has seen anything, and answers it *separately*
from the same question the UI can answer with more — the resolved paths,
the device, the version of each package.

The concrete case that decided it. `requirements.txt` names four packages
(fastapi, uvicorn, python-multipart, tomli_w). The trainer imports four
more — torch, numpy, safetensors, PIL — and appears in **no requirements
file**, because they are expected from ComfyUI's venv. So a user who
installs `requirements.txt` into a fresh venv gets a server that starts
and a training stack that is silently absent. A shell gate would have
caught that at startup, once, as a fixed list; the readiness surface
reports it per package with versions, with the device, and with a verdict
that the wizard's first screen renders.

## Decision

**The installer is a web surface.** `GET /api/v1/installer/readiness`,
`/state` and `/manifest` report; `POST /api/v1/installer/apply` writes, and
only while unconfigured. The page is at `/setup`, and the server starts in
every state — there is no gate that refuses to boot.

Three consequences worth stating:

* **The requirement list is data, with four tiers.** `required` (the
  server), `training` (a run needs it), `optional`, and
  `comfy_provided` (torch and the accelerator stack). The tiers exist
  because the venv decision is a *disk* decision and needs to know which
  packages this project may install into a venv it does not own.
* **Presence is read without importing.** `importlib.metadata` reports
  torch's version without putting torch in `sys.modules`, so the server can
  answer "is torch installed" on a machine that does not have it. The
  device probe imports torch in a subprocess, and is skipped entirely when
  the package check has already answered the question.
* **The shell gate is *not* the fix, but there now is a preflight.**
  `run_server_linux.sh` still starts the server whatever is missing — there is no
  gate that refuses to boot. What it gained is
  `backend/first_run.py`, which installs only the *four server packages*
  into a temporary venv and re-execs. The boundary from this record still
  holds: it fixes the one failure the server cannot report about itself
  (it cannot start), and it installs nothing that a *run* needs. Everything
  else — the venv decision, the training stack, the device — is still the
  readiness screen's job, because those are questions, not imports.

## The write is gated, and that is the security decision

The installer is the one surface in this project that can change **where
model files are read from**, on a request any browser page could make. So:

* It sits under the same Host/Origin guard as everything else
  (ADR 0001 — no authentication, because the threat is a page the user has
  open, not an attacker with an account).
* `POST /apply` is refused with **409 `installer_not_allowed`** once
  `comfy_dir`, `checkpoints_dir` and `loras_dir` all resolve. A wizard left
  open in a tab on a working machine does not get to re-point the model
  directories out from under a run that is using them. The gate is checked
  *before* validation, so a refusal does not report a path error the user
  would read as the real problem.
* The gate deliberately does **not** include the device or the training
  stack. A machine with no GPU still has chosen paths, and gating on those
  would make the wizard impossible to finish on exactly the machine that
  needs it most.
* An untouched field writes no override. "Leave it empty to accept the
  default" has to mean the resolution policy still decides, or accepting
  every default would pin paths the user never chose.

## Withdrawn in part

**The install-order argument is wrong and this record is superseded on that
point** by
[`12-installer-and-comfy-decoupling.md`](../../archive/old-documentation/design/12-installer-and-comfy-decoupling.md).

The principle survives: the installer is a web surface, not a shell prompt,
and it is behind the same Host/Origin guard as everything else. The
*consequence* does not. A wizard living inside the server can only do what
the server can already do — and the first thing missing is usually the
thing stopping the server from starting. So the version that ships cannot
help the person who needs it most.

What replaces it: a stdlib-only preflight, launched by `run_server_linux.sh` only
when the server's own imports fail, which prints a URL, opens a browser and
installs four small packages into a temporary venv. Once the server runs,
this wizard takes over and never mentions installation again.

`run_server_linux.sh` stays unchanged in the shipped version, so this record's
stated cost — no preflight — is real until that lands.

## Consequences

* **A path the user never chose is no longer invented.** Fixing this
  exposed that `path_tiers.checkpoints_dir` never consulted the settings
  store, so on a fresh checkout the model directories resolved to
  `<project_root>/checkpoints` while `comfy_dir` resolved correctly — the
  settings page and the training path disagreeing about one machine.
  Resolution now raises rather than inventing a directory that exists
  nowhere and reads as configured.
* **The opt-in download is Phase C and is not built.** It needs the venv
  decision this record defers: whether we may install into a venv the user
  did not create, under a constraints file that makes breaking their
  ComfyUI unreachable rather than merely unlikely. The page says installing
  "does not exist yet" rather than offering a button.
* **Two model roots is Phase F and is untouched.** Independent of the
  wizard, and worth doing on its own merit; it is not a wizard question.
* **`run_server_linux.sh` calls one module before launching**, and a user who
  wants no preflight still has one: `DISTILLATION_NO_BROWSER=1` stops it
  opening a tab, which matters because it otherwise reaches for the user's
  browser unattended. Its presence is otherwise silent — one subprocess
  that exits 0 and prints nothing when the packages are there, which is
  every start after the first.

## Redesign (2026-11)

The first wizard asked for a ComfyUI directory it did not need, offered to "reuse ComfyUI's virtualenv" without asking for the
virtualenv, and installed into an environment before looking at what was in it. It was replaced; what holds now:

* **Order**: graphics card, environment, install, prompt files, models folder. Each step needs only what the earlier ones produced.
* **The environment is a folder**, new (`./venv`) or existing (any virtualenv folder). The server reads it through its own
  interpreter (`POST /installer/environment`), lists what is missing, and pins every installed package for the install. The request
  to `/install` names only the target; the server decides the package list, so nothing the client sends reaches pip.
* **torch comes from the Intel XPU index in its own stage**, before the rest, so the rest never replaces it with a PyPI build. An
  environment whose torch lacks `+xpu` is refused rather than piled onto.
* **ComfyUI is optional.** "Configured" means the checkpoint and LoRA folders resolve (from `models_dir`, an explicit setting, or a
  ComfyUI checkout beside the project). The `never_install` tiers, the conflict check against ComfyUI's requirements and the readiness
  report are gone.
* **Prompt vocabulary files** are downloaded into `./assets` instead of being found inside ComfyUI.
* The routes that run a chosen interpreter (`/environment`, `/environment/verify`) are POST, inside the Origin guard.
* **The server runs in the environment the user chose.** The bootstrap environment has only the server's packages, so a server left
  there cannot import the node modules (torch) and the graph editor shows almost every node red. `/apply` therefore restarts the
  server into the chosen interpreter (`ServerRestarter`, re-running the launcher with `VENV_PYTHON`), and `backend.first_run` reads the
  stored `venv_python` on every start. An interpreter older than `MIN_PYTHON` or without the server packages is never restarted into.
* **The project's own `./venv` is completed, not refused.** A torch there that is not the XPU build is replaced (`--force-reinstall`
  on the torch stage, no pins); a user's own environment is still only ever added to.
* **Setup is a place in the sidebar**, visible while unconfigured, locking the other entries (Help excepted) while an install runs.
  `/setup` redirects to Help once configured.
