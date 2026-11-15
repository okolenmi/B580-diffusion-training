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
| `settings.md` | graph settings: specs, resolution order, core keys |

Status: **frozen at the end of phase 1** (`graph` implements all of them);
until then they may change while the code is written.
