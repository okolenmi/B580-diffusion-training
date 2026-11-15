# Documentation

Start with [`status.md`](status.md): what exists, what does not, and what waits for hardware.

| Read | For |
|---|---|
| [`status.md`](status.md) | what is built and verified, what is missing, the next work |
| [`architecture.md`](architecture.md) | the packages, how a run flows, where each idea lives in the code |
| [`developing.md`](developing.md) | rules for writing code here: layers, nodes, comments, tests, changing a contract |
| [`setup.md`](setup.md) | install, run, environment variables |
| [`MANUAL.md`](MANUAL.md) and `manual/` | using the trainer: settings, what to run for speed, VRAM, quality, data |
| [`contracts/`](contracts/) | the stable formats: graph JSON, catalog, resolve, events, runs, settings, datasets |
| [`decisions/`](decisions/) | decision records, each with its cost and the test that pins it |
| [`internals/`](internals/README.md) | how the harder parts work: memory, kernels, capture, addons, perf and acceptance labs, parallel runs, Anima facts |
| [`experiment-results.md`](experiment-results.md), [`research-2026-11.md`](research-2026-11.md) | what was measured and what was read; code comments cite these instead of copying numbers |
| [`tester-brief.md`](tester-brief.md) | the brief for the AI that runs hardware checks; the items themselves are in `TESTING.md` at the root |

Old plans, audits, the porting ledger and the old trees are in `archive/` (`old-documentation/rework-plans` holds the plans the rework was
built from; they say "proposal" and are history, not a description of the code).

## Keeping this folder true

- A page describes what the code does now. A plan lives in the archive once it is built.
- Numbers measured on hardware go to `experiment-results.md`; code and pages cite them.
- Status lives in one place, `status.md`; other pages do not carry a "built / not built" table.
- `scripts/check_doc_links.py` and `scripts/check_doc_refs.py` (part of `scripts/full_gate.sh`) fail on a dead link, citation or repo path.
