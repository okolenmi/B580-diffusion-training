# Backend design docs

Documents the clean-room replacement for `server/` living in
`backend/` -- architecture, API contract, and migration strategy.
The old `server/` was retired to `archive/server/` at M9
(2026-10-01), untouched, for reference; `run_server.sh`, the README
and the setup/architecture docs now point at `backend/`.

| # | Doc | What it is for |
|---|-----|----------------|
| 1 | [`01-architecture.md`](01-architecture.md) | Why this exists (the `server/` autopsy and the verdict), the user-approved decisions, **the ten layering rules a change has to keep**, the deliberate bridges to `core/`/`manager`/`nodes/`, and the concurrency model including the trainer-subprocess contract. |
| 2 | [`02-api-reference.md`](02-api-reference.md) | The parts of the API contract that OpenAPI cannot state: the browser-door threat model, the error-code table (parsed by `backend/tests/test_error_contract.py`, so it cannot drift from the code), the non-finite-float contract and the client obligation it creates, the event-stream rules, and the behavioural clauses. |
| 3 | [`03-migration-strategy.md`](03-migration-strategy.md) | Why the frontend is vanilla ES modules served by the backend on one origin, why monitor-first; which legacy endpoints were dropped **and why**; the pinned monitor data-path contract; the run-id collision story and its three guards. |
| 4 | [`04-dataset-format.md`](04-dataset-format.md) | Storage format v2: directory layout, `metadata.db` schema, why each of v1's five properties was rejected, migration from v1, and the `manager` bridges that must stay byte-identical. **A contract with the trainer — do not "improve" it.** |
| 5 | [`05-graph-runtime.md`](05-graph-runtime.md) | Graph subsystem: auto-discovery and why it replaced the hand-maintained list, the validation issue-code table and the wire-safe type check, execution lifecycle and CAS, the deliberate divergences from legacy, and **where a graph run actually lives** (in the API process, and what a restart costs). |
| 6 | [`06-visual-smoke.md`](06-visual-smoke.md) | What a browser session catches that the suites cannot: the CSS/specificity findings worth not reintroducing, the open-coverage list, and the screenshot caveat. The runnable suite is `backend/tests/visual_smoke.py`. |
| 7 | [`07-review-2026-10-01.md`](07-review-2026-10-01.md) | The register of hard-won constraints: 17 findings (F-01..F-17) with severity and evidence level, **"what is good — do not fix"**, thirteen contributor quality rules, and the repro scripts. Code comments cite this as `docs 07 F-NN`. Read before changing supervisors, the progress reader, dataset deletion or the SSE/monitor path. |
| 8 | [`08-structure-audit.md`](08-structure-audit.md) | The strict-OOP pass over `backend/` itself: a one-line index of the nineteen fixed findings (each points at the docstring now carrying its reasoning) and the **seven deferred ones in full, with their reasons** — read it before adding a use case or a port. |

What is *not* here any more, deliberately: file trees, endpoint tables,
milestone logs and implementation-status inventories. They restated the
code, they could not help but drift, and the 2026-10-01 cleanup removed
them. Open the module instead; ask the server for the API, since it
serves its own OpenAPI schema.
