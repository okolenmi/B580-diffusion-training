# 02 -- API contract

**Where the field-level detail lives: the server.** `/openapi.json` is
served from the route decorators themselves, so every path, query
constraint, body schema and response model below is discoverable there
and cannot drift. `backend/presentation/schemas.py` is the source of
truth for shapes.

What is worth writing down is the part OpenAPI *cannot* express: the
behaviours a client has to know, and the promises the server makes to
the other processes it talks to. That is this document.

## 1. Conventions

* **Base path**: `/api/v1`. All bodies and responses are JSON (assets
  upload is raw bytes; SSE is `text/event-stream`).
* **No authentication, but the browser doors are closed** (docs 07
  F-06): this is a documented trade for a single-user tool on loopback,
  so the two holes that need no credentials are closed instead. Every
  request's `Host` must name this server (loopback names by default,
  plus anything in `BACKEND_ALLOWED_HOSTS`) — that is what stops DNS
  rebinding. A state-changing method (`POST`/`PUT`/`PATCH`/`DELETE`)
  carrying an `Origin` must match this server's own origin or one in
  `BACKEND_ALLOWED_ORIGINS`. Refusals are 403 `forbidden_host` /
  `forbidden_origin` in the ordinary envelope.

  Two deliberate non-refusals: a request with **no** `Origin` is not a
  browser cross-site request (curl, scripts, the app's own same-origin
  fetches) and passes; and `GET`/`HEAD` stay open, because a hostile
  page can only read with those and reads are already scoped to a
  loopback deployment.

  If the UI is ever served from a different origin — a LAN address with
  a separately-served frontend — `BACKEND_ALLOWED_ORIGINS` must be set
  or cross-site writes are refused by design.
* **Timestamps**: ISO 8601 with offset (`2026-09-30T12:00:00+00:00`).
* **List responses** are wrapped: `<resource>s: [...]` plus `count`.
* **Error envelope** — every non-2xx response, including unknown routes
  and method-not-allowed (no bare `{"detail": ...}` anywhere):

  ```json
  {"error": {"code": "run_not_found", "message": "run 7 not found",
             "details": { ... }}}
  ```

  `details` is present only when the use case supplies it (the full
  issue list for `graph_invalid`, `{loc, msg, type}` entries for
  `validation_error`).
* **Error codes** — every `ApplicationError` subclass declares its own
  `code` *and* HTTP status in `backend/application/errors.py`, and
  `backend/tests/test_error_contract.py` parses the table below and
  fails if the two disagree in either direction — so this table is
  that module, and adding a code is a one-file change plus one row
  here. Pydantic body rejection is 422 `validation_error`.

  | Code | Status | Domain |
  |---|---|---|
  | `invalid_query` | 422 | runs/graphs (limit, status, name range) |
  | `forbidden_host`, `forbidden_origin` | 403 | requests (Host not served / cross-site state change — see above) |
  | `run_not_found`, `no_active_run` | 404 | runs |
  | `run_already_active`, `run_not_running` | 409 | runs |
  | `run_directory_conflict` | 409 | runs (`runs/run_<id>/` already holds files — never overwritten, docs 07 F-04) |
  | `config_not_found` | 404 | config |
  | `config_invalid` | 422 | config |
  | `training_launch_failed` | 500 | runs |
  | `settings_invalid` | 400 | settings |
  | `asset_too_large` | 413 | assets (upload body over the 8 GiB cap, or a declared `Content-Length` over it — the body is streamed, so the running total is the authoritative check) |
| `asset_exists` | 409 | assets (the upload target already exists; pass `?overwrite=true` to replace it — a `PUT` never silently overwrites a checkpoint, docs 08 N-14) |
  | `dataset_not_found`, `dataset_item_not_found`, `dataset_file_not_found`, `dataset_task_not_found` | 404 | datasets |
  | `dataset_exists`, `dataset_not_migrated`, `dataset_task_active`, `dataset_task_not_active`, `dataset_directory_conflict` | 409 | datasets (`dataset_directory_conflict`: the name is taken by a directory that is not a dataset — it is never deleted, docs 07 F-10) |
  | `dataset_task_launch_failed` | 500 | datasets |
  | `graph_invalid` | 422 | graphs |
  | `graph_not_found`, `graph_execution_not_found`, `node_class_not_found` | 404 | graphs |
  | `graph_execution_active`, `graph_execution_not_active` | 409 | graphs |
  | `node_diagnostics_failed` | 400 | graphs |
  | `installer_not_allowed` | 409 | installer (a wizard write after this installation is configured) |
  | `installer_busy` | 409 | installer (an install is already running in this process) |
  | `install_refused` | 400 | installer (a deliberate refusal: a `never_install` package aimed at a venv this project does not own, an unknown ComfyUI venv, an empty package list, or pip exiting non-zero. Nothing was changed — the message says which of those it was) |
  | `install_job_not_found` | 404 | installer (a job id this process never issued, or one lost when the server restarted; the readiness report is the source of truth for whether the packages are installed) |

## 2. Two boundary contracts clients must honour

### Non-finite floats

`NaN` and `±Inf` — a diverged trainer — never reach the wire as JSON.
Every body, SSE frame and monitor frame goes through
`backend/json_safe.py`, which sends `null` for the value and adds a
sibling `nonfinite` map naming the keys it replaced with their kind
(`{"current_loss": "inf", "avg_loss": "-inf"}`). The marker is attached
per object, so a list item names its own field (`runs[i].nonfinite`).

SQLite maps `NaN` to `NULL`, so `NaN` only ever reaches the stream
frames, never a stored row.

**Client obligation:** render a marker as a loud "diverged" state, never
as "no measurement" and never as a zero. A missing key is a gap; a
marked key is a failure, and the two must not look alike.

### The event stream

`data: {json}` per line; the first frame is
`{"type": "stream_opened", "occurred_at": ...}`; idle gaps emit comment
heartbeats (`: ping`). Each domain event serialises with a `type` field
plus its payload and `occurred_at`; the emitted types are the
`DomainEvent` subclasses in `backend/domain/events.py`.

`run_progressed`: `{type, occurred_at, run_id, step, total_steps, loss,
avg_loss, lr, phase, cache_done, cache_total}` — `null`s for unknown
fields. `graph_execution_progressed`: `execution_id`, `node_id`, `ok`,
`duration_ms`.

**No replay.** A frame published while a client is disconnected is gone
for good, so a subscriber refetches the REST state on every `(re)open`
— the dashboard does this on open plus on a 30 s safety poll, and the
run page refetches when its SSE connection drops. Progress frames are
coalesced per subscriber (a queued one is superseded by the newest) and
lifecycle frames are kept; only an all-lifecycle backlog gives up its
oldest frame, which is logged and counted.

## 3. Behaviours the schemas do not say

**Config.** `PATCH /config` is nested-only: dotted/flat keys and string
coercion do not exist here. The file must already exist — `PUT
/config/raw` is the create path. Unknown override keys are *ignored*
(`TrainingConfig` is permissive by design). Saving never mutates launch
state and launching never mutates the config: launch options ride the
`POST /runs` body. `start-options` omits `lora_checkpoint` for non-LoRA
configs rather than faking it as unavailable. A broken config propagates
its error instead of degrading to an empty 200.

**Raw config editing.** `PUT /config/raw` **stores the user's own text**:
the document is parsed to validate it, then written verbatim through a
temp sibling + rename, so comments, key order, and keys the model does
not declare all survive (docs 07 F-08). `PATCH /config` (the form
editor) still rewrites through the model, by design.

### What a client-named path may point at

Three surfaces take a path from the client, and **they do not share a
base**, so "is this contained?" has three different answers. Nothing in the
schemas can say any of it, which is why it is here.

| Surface | Base | Refused |
|---|---|---|
| `/config`, `/config/raw` | the **project root** | anything resolving outside it, by any route: `../`, an absolute path elsewhere, or a symlink inside the tree pointing out. Both verbs. |
| `/datasets/{name}/files/{path}` | that dataset's directory | traversal out of it, and any file that is not a preview image (`.png`, `.jpg`, `.jpeg`, `.webp`) within `MAX_PREVIEW_BYTES` |
| `/assets/{kind}` | the ComfyUI model directory for that kind | traversal out of it. Legitimately *outside* this project — that is where models live — so this surface is not confined to the project root and must not be. |

The config routes are confined to the project because they are an editor
for *this project's* configuration; there is nothing else they could
sensibly edit. The asset routes are not, because a model directory is a
different tree by design.

A NUL byte in any of these is refused before any path arithmetic: it is
not a path but the end of the string, and every filesystem call raises
`ValueError` on one. Note the transport difference when testing this — in a
query string `%00` decodes to a NUL, while in a JSON body it is three
ordinary characters and a perfectly legal filename.

This was a live arbitrary read *and write* until it was found by sweeping
every operation for 5xx and then following the query-string dimension:
`GET /config/raw?path=../../../../etc/passwd` returned the file, and the
same value on `PUT` wrote it. `backend/tests/test_config.py` holds both
verbs against both shapes.

**Settings.** Validation is **pure** — it looks, it never acts (docs 07
F-15), so a rejected update leaves the filesystem exactly as it was. A
*configured* `venv_python` is used as-is: a stale value fails loudly at
spawn rather than silently running some other interpreter.
`venv_python` must be an existing **executable** file, because the value
is executed on every start — "it exists" is not the contract. Managed
`models_dir` is a single key for the whole model tree: set it and
`checkpoints_dir`/`loras_dir` resolve under it, unless they are
individually overridden (the specific wins over the coarse). It is how
the app is pointed somewhere other than a ComfyUI checkout, which is
the default layout; see `docs/setup.md`.

directories (`checkpoints_dir`, `loras_dir`) are created **after** the
commit, so an update that fails validation creates nothing. An absent key
leaves the value untouched; `""` clears the override.

**Assets.** The `dataset` kind is catalog-only — browse, mkdir and
upload are refused with guidance rather than half-supported. Listings
exclude `resume/` and dotfiles. `inspect` returns a fixed per-kind
shape (checkpoint: components; LoRA: dtype/rank/key_count), never a raw
header dump.

Upload policy: the name must end in **`.safetensors`** (traversal and
unknown-kind attempts are `invalid_query` too); the body is read as a
stream that stops at the 8 GiB cap, so a chunked request cannot buffer
the server past it; the bytes land in a `.part` sibling and are renamed
into place, so a rejected or failed upload leaves no directory, no file
and no partial behind.

**Datasets.** Legacy v1 datasets are *shown*, with `stats: null` rather
than fabricated counts, and every v2-only endpoint refuses them with 409
`dataset_not_migrated` — delete excepted, so removing a legacy dataset
never requires migrating it. Three racing actors (the child's reporter,
the stop endpoint, startup reconciliation) write through a status CAS,
so exactly one final outcome wins and a late progress tick is a no-op
rather than a resurrection. Item listing is **paged by default** at 500
rows: a dataset grows by ingestion rather than by user action, so
"return everything" was a response whose size nobody bounded. The page
carries `total` (rows matching the filter in the whole dataset) and
`next_offset` (`null` on the last page) so a client can tell a
truncated page from the whole set instead of presenting one as the
other.

Dataset file paths are sandboxed: a path escaping the dataset directory
is reported as **not-found, never resolved**. `PUT /datasets/{name}
preview` takes an **item id, never a path** — the server reads the path
from the dataset's own rows.

Bulk item edits keep legacy truthy gates (an empty string never clears
in bulk; use the single-item PATCH for that), and `prepend`/`append` are
idempotent so re-applying is a no-op.

**Graphs.** A structurally bad submission is not a body rejection:
`/graphs/validate` answers **always 200** with `ok` plus the complete
issue list, and `/graphs/run` refuses the same graph with 422
`graph_invalid` carrying every error-severity finding in `details`, so
the editor can localise all of them in one round trip.

Execution lifecycle is `queued -> running -> finished | error | stopped`,
single-active enforced by a DB CAS. Node failures surface as status
`error`, not as a `finished` execution carrying an `ok: false` result —
a poller could not tell those apart. Saved library payloads are stored
**verbatim** and validated at run, never at save, so an old or
hand-edited payload loads fine and fails loudly only when executed.

**Routing order.** Page routes are registered after the API and never
under `/api/`, so an unknown API path keeps the JSON error envelope
instead of falling through to the HTML shell.

## 4. The monitor stream

`GET /api/v1/monitor/{monitor_id}/stream` is SSE with a
`{"type": "connected"}` opener, then the bus's pre-rendered frames:
history replay first (so a dashboard opened mid-run restores its chart),
then live step reports, plus `{"type": "clear"}` broadcasts when a new
run claims the id and the terminal `{"type": "run_end", "step",
"cancelled"}`.

Payload keys are the trainer's report dict **verbatim** (`step`,
`total_steps`, `loss`, `lr`, `t`, `weight_t_*`, `prob_t_*`, `*_ms`,
`vram_*`, `resident_*_mb`, `vram_budget_mb`, `grad_norm`, ...). A frame
the page does not recognise is ignored, never guessed at.

This frame format is a contract with `nodes/` and the frontend at once,
and it is pinned in `03-migration-strategy.md` section 4 (including why
`monitor_id` is deliberately not the execution id).

The page routes that serve the UI (`/`, `/monitor/{id}`, `/graph`,
`/config`, `/run/{id}`, `/datasets`, `/datasets/{name}`, `/help`,
`/settings`, plus the `/ui/*` asset mount) are registered with
`include_in_schema=False`, so they do not appear in `/openapi.json` —
which is why they are listed here.