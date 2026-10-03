# 0005 — The first-run installer is a browser surface, not a shell gate

## Context

`docs/design/11-first-run-and-installer.md` was written before this
project had a settings page, and it proposed starting with a shell gate:
`run_server.sh` checks dependencies before launching the server and
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
* **The shell gate is not built.** `run_server.sh` still starts the server
  whatever is missing. The honest version of the original instinct is the
  readiness screen, not a preflight, because the failure it prevents is a
  `ModuleNotFoundError` deep in a node build — and by then the user has
  been told nothing.

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
[`12-installer-and-comfy-decoupling.md`](../design/12-installer-and-comfy-decoupling.md).

The principle survives: the installer is a web surface, not a shell prompt,
and it is behind the same Host/Origin guard as everything else. The
*consequence* does not. A wizard living inside the server can only do what
the server can already do — and the first thing missing is usually the
thing stopping the server from starting. So the version that ships cannot
help the person who needs it most.

What replaces it: a stdlib-only preflight, launched by `run_server.sh` only
when the server's own imports fail, which prints a URL, opens a browser and
installs four small packages into a temporary venv. Once the server runs,
this wizard takes over and never mentions installation again.

`run_server.sh` stays unchanged in the shipped version, so this record's
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
* **`run_server.sh` is unchanged**, so a user who wants a preflight still
  has nothing. That is the accepted cost of this shape.