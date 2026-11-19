# Contracts

Short normative documents for the JSON and Python interfaces that more than one
region depends on. A contract gives the exact shape. When code and a contract disagree, one of them
is a bug: fix the contract first (`../developing.md`, "Changing a public contract").

| File | Governs |
|---|---|
| `graph-json.md` | the saved graph document |
| `catalog.md` | what the server tells the editor about node classes, settings and families |
| `resolve.md` | editing-time shape resolution (dynamic ports) |
| `events.md` | the event plane and its mirroring |
| `runs.md` | run lifecycle, node states, progress and results |
| `apps.md` | apps (frozen, versioned pages over a graph) and the boards they share with graph views |
| `settings.md` | graph settings: specs, resolution order, core keys |
| `errors.md` | the HTTP error envelope and every error code with its status |

Status: implemented and stable. Change one only as `../developing.md` describes (contract page first, every user in the same commit).
