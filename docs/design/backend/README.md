# Backend design docs

Documents the clean-room replacement for `server/` living in
`backend/` -- architecture, API contract, and (eventually) the
migration strategy. The old `server/` stays untouched and in use
until that migration is agreed.

| # | Doc | Contents |
|---|-----|----------|
| 1 | [`01-architecture.md`](01-architecture.md) | Evaluation of the old server, layering rules, file structure, API/error/event contracts, concurrency model, milestone plan. |
| 2 | `02-*.md` | *(planned)* API contract reference, once M2-M4 domains exist. |
| 3 | `03-*.md` | *(planned)* Migration strategy: frontend decision, data cutover, decommissioning `server/`. |
