# Backend design docs

Documents the clean-room replacement for `server/` living in
`backend/` -- architecture, API contract, and migration strategy.
The old `server/` was retired to `archive/server/` at M9
(2026-10-01), untouched, for reference; `run_server.sh`, the README
and the setup/architecture docs now point at `backend/`.

| # | Doc | Contents |
|---|-----|----------|
| 1 | [`01-architecture.md`](01-architecture.md) | Evaluation of the old server, layering rules, file structure, API/error/event contracts, concurrency model, milestone plan. |
| 2 | [`02-api-reference.md`](02-api-reference.md) | API contract reference: conventions, error-code table, every `/api/v1` endpoint with params/body/response shapes (SSE frame types, graph issue codes, library payload rules). |
| 3 | [`03-migration-strategy.md`](03-migration-strategy.md) | Migration strategy: frontend decisions (vanilla ES modules, backend-served, monitor-first), parity audit of the 51 legacy endpoints, monitor data-path contract, data cutover, phased decommissioning of `server/`. |
| 4 | [`04-dataset-format.md`](04-dataset-format.md) | Dataset storage format v2: directory layout, `metadata.db` schema, migration from v1, the manager bridges that must stay byte-identical. |
| 5 | [`05-graph-runtime.md`](05-graph-runtime.md) | Graph subsystem (M4): `GraphCatalog`/`GraphRuntime` ports, auto-discovery, the validation issue-code table, execution lifecycle + CAS, API surface, deliberate divergences from legacy, deferrals. |
| 6 | [`06-visual-smoke.md`](06-visual-smoke.md) | Frontend visual smoke checklist for a browser-enabled session: what curl/tests cannot cover (JS actually running), per-page steps, known deferred items. Executed 2026-10-01; findings + fixes recorded in the doc. |
| 7 | [`07-review-2026-10-01.md`](07-review-2026-10-01.md) | Review of the M8a redesign: 17 findings (F-01..F-17) with severity, evidence level (reproduced vs read), locations, fix direction and regression tests, plus quality rules for contributors. Section 7 verifies every finding still stands at `c1dc136`; reproductions live in `scripts/repro/`. Read before changing supervisors, the progress reader, dataset deletion or the SSE/monitor path. |
