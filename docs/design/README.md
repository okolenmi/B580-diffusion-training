# Design docs index

The `nodes/` rewrite's design record. **These documents are rationale,
not reference**: each one keeps the reasoning behind a technique, the
evidence for it, and the alternatives that were rejected with their
reasons. Anything that merely lists what the code contains was removed
in the 2026-10-01 cleanup — open the module instead. The code's own
docstrings now carry the "what"; these files carry the "why".

| # | File | What is worth reading it for |
|---|---|---|
| 1 | [`02-foundational-ontology.md`](02-foundational-ontology.md) | Why a `Builder` is a different kind of thing from a runtime object, and the Lin et al. derivation behind the rescaled-zero-terminal SNR schedule — including the two precision claims checked by hand rather than trusted. |
| 2 | [`03-training-step-orchestration.md`](03-training-step-orchestration.md) | Why activation-checkpoint placement is worth its own machinery (the sqrt(N) and selective-recompute grounding, plus the measured sweep that closed the question), and why the `ResourcePolicy` wrapper was built and then deleted. |
| 3, 4, 5 | [`04-lora-adapter-mechanics-and-loss-weighting.md`](04-lora-adapter-mechanics-and-loss-weighting.md) | The adapter and loss-weighting reasoning: PEFT-grounded DoRA, the traps in that seam, the published case *against* assuming QLoRA transfers to a diffusion UNet, and the loss-weighting call-site bug that a paper-correct interface did not prevent. **The densest design file here — the most-cited one in the codebase.** |
| 5 | [`05-coordination-registry-observability.md`](05-coordination-registry-observability.md) | The concurrency contract that background threads must honour, and the Acyclic Domain Dependency Rule — both cited directly by code. Also why `ComponentRegistry` and `TrainingRecipe` are deliberately not built. |
| 7 | [`07-deferred-or-rejected.md`](07-deferred-or-rejected.md) | **Read this before proposing anything.** Nine things considered and left out on purpose, each with its actual reasoning. The one entry where the code later shipped anyway keeps both sides of that story. |
| 8, 9 | [`08-validation-and-implementation-status.md`](08-validation-and-implementation-status.md) | What is implemented and — more usefully — the five pieces that are **built but unvalidated**, with the specific missing evidence for each. |
| 10 | [`09-prioritized-backlog.md`](09-prioritized-backlog.md) | What is left, in order, with the reasoning for the order and the scope boundaries. |
| 11 | [`10-node-surface-and-precision-control.md`](10-node-surface-and-precision-control.md) | Node surface and precision decisions, mostly landed: the three-orthogonal-axes decomposition, what a shared structure is worth after the same bug appeared three times, and one maintainer decision still owed. |
| 13 | [`11-core-removal.md`](11-core-removal.md) | **Done 2026-10-02.** `core/` moved to `archive/core/`, and the backend's support for it went with it. The five edges it listed, and what each became. |
| 14 | [`12-training-modes.md`](12-training-modes.md) | One of the four `tuning.method` values is implemented. Which, and what each of the other three would take. Read before adding a trainer. |
| 15 | [`13-process-isolation.md`](13-process-isolation.md) | Graph execution runs in a child process behind a port, supervised by tailing a file. The channel, the stop semantics, adoption across a restart, and the measured costs. |
| 14 | [`backend/09-event-contract.md`](backend/09-event-contract.md) | **In progress.** `/events` has no replay, so a browser tab that sleeps through `run_completed` believes the run is still running; the frontend's resync-on-connect is a workaround, not a design. Sequence numbers, a bounded lifecycle replay ring, and `Last-Event-ID`. |
| 12 | [`11-first-run-and-installer.md`](11-first-run-and-installer.md) | **Partly built; the phasing is superseded by doc 15.** Getting a new user to a working install: whether to install into the ComfyUI venv (and what must be guaranteed before we do), and looking for models in two places at once. Read before touching `paths.py` or the asset pickers -- the multi-root section is the part with design in it. |
| 15 | [`12-installer-and-comfy-decoupling.md`](12-installer-and-comfy-decoupling.md) | **Design, not built.** Why the install has to start before the server does, and the measured cost of dropping ComfyUI: 9 modules, 5,081 lines, GPL-3.0. Read before changing anything about install order or the ComfyUI dependency. |
| 16 | [`14-node-attached-monitor-widgets.md`](14-node-attached-monitor-widgets.md) | **Design, not built.** Monitor as node-attached widgets the user composes into a window, instead of one flat blob per `monitor_id`: what today's shape costs, why the trainer's report and the dashboard's key list are the real coupling, and the five questions to settle before code. |

The seven design goals every choice here is checked against live in the
root [`README.md`](../../README.md)'s Goals section.

Two adjacent folders:

* [`resources-controller/`](resources-controller/README.md) — the
  Resources Controller and precision redesign, and the most recent
  hardware-measured work in the repo.
* [`backend/`](backend/README.md) — the web backend and its frontend.
  Where this folder and `resources-controller/` conflict on anything,
  `resources-controller/` wins.

Precedent worth knowing: the standalone
`01-design-goals-and-constraints.md` that used to introduce this folder
was removed on 2026-09-28 as redundant with the root README, and
`resources-controller/07-post-phase-6-bugfixes.md` was removed once its
four bugs were fixed and recorded in code and git history. Both were
low-value records kept alive by nothing but their own existence.