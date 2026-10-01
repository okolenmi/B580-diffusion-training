# 02 -- API contract reference

Source of truth for request/response *field* level detail is
`backend/presentation/schemas.py`; this document is the navigable
contract (endpoint, params, envelope, error codes) the frontend is
written against. Route table also mirrored in `01-architecture.md` §API.

Status: **shipped through M6** (48 endpoints under `/api/v1`, plus the
static page routes in section 10).
Section 10 (monitor stream + static pages) is the M6 slice whose
pinned frame contract lives in `03-migration-strategy.md` §4.

## 1. Conventions

* **Base path**: `/api/v1`. All bodies and responses are JSON
  (assets upload is raw bytes; SSE is `text/event-stream`).
* **No authentication, but the browser doors are closed** (docs 07 F-06):
  every request's `Host` must name this server (loopback names by
  default, plus anything in `BACKEND_ALLOWED_HOSTS`) — that is what stops
  DNS rebinding — and a state-changing method (`POST`/`PUT`/`PATCH`/
  `DELETE`) carrying an `Origin` must match this server's own origin or
  one in `BACKEND_ALLOWED_ORIGINS`. Refusals are 403 `forbidden_host` /
  `forbidden_origin` in the ordinary envelope. Requests without an
  `Origin` (curl, scripts, server-to-server) are not browser cross-site
  requests and pass; `GET`/`HEAD` stay open.
* **Timestamps**: ISO 8601 with offset (`2026-09-30T12:00:00+00:00`),
  serialised from `datetime` fields.
* **List responses** are wrapped: `<resource>s: [...]` plus `count`.
* **Error envelope** — every non-2xx response, including unknown
  routes and method-not-allowed (no bare `{"detail": ...}` anywhere):

  ```json
  {"error": {"code": "run_not_found", "message": "run 7 not found",
             "details": { ... }}}
  ```

  `details` present only when the use case supplies it (e.g. the full
  issue list for `graph_invalid`, pydantic field errors for
  `validation_error` as `{loc, msg, type}` entries).
* **Error codes** (application codes -> status; anything unmapped is
  400; pydantic body rejection is 422 `validation_error`):

  | Code | Status | Domain |
  |---|---|---|
  | `invalid_query` | 422 | runs/graphs (limit, status, name range) |
  | `forbidden_host`, `forbidden_origin` | 403 | requests (Host not served / cross-site state change — see section 1) |
  | `run_not_found`, `no_active_run` | 404 | runs |
  | `run_already_active`, `run_not_running` | 409 | runs |
  | `run_directory_conflict` | 409 | runs (`runs/run_<id>/` already holds files — never overwritten, docs 07 F-04) |
  | `config_not_found` | 404 | config |
  | `config_invalid` | 422 | config |
  | `training_launch_failed` | 500 | runs |
  | `settings_invalid` | 400 | settings |
  | `asset_too_large` | 413 | assets (upload body or declared `Content-Length` over the 8 GiB cap) |
  | `dataset_not_found`, `dataset_item_not_found`, `dataset_task_not_found` | 404 | datasets |
  | `dataset_exists`, `dataset_not_migrated`, `dataset_task_active`, `dataset_task_not_active`, `dataset_directory_conflict` | 409 | datasets (`dataset_directory_conflict`: the name is taken by a directory that is not a dataset — it is never deleted, docs 07 F-10) |
  | `dataset_task_launch_failed` | 500 | datasets |
  | `graph_invalid` | 422 | graphs |
  | `graph_not_found`, `graph_execution_not_found`, `node_class_not_found` | 404 | graphs |
  | `graph_execution_active`, `graph_execution_not_active` | 409 | graphs |
  | `node_diagnostics_failed` | 400 | graphs |

## 2. Health & events

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status": "ok", "version": "..."}` |
| GET | `/events` | SSE (below) |

**SSE frames**: `data: {json}` per line; first frame is
`{"type": "stream_opened", "occurred_at": ...}`; idle gaps emit
comment heartbeats (`: ping`). Each domain event serialises with a
`type` field plus its payload and `occurred_at`. Emitted types:

```
run_created, run_started, run_completed, run_failed, run_cancelled,
run_progressed, runs_deleted,
graph_execution_queued, graph_execution_started,
graph_execution_progressed, graph_execution_finished,
graph_execution_failed, graph_execution_stopped,
graph_executions_deleted
```

`run_progressed`: `{type, occurred_at, run_id, step, total_steps,
loss, avg_loss, lr, phase, cache_done, cache_total}` — `null`s for
unknown fields. `graph_execution_progressed`: `execution_id`, `node_id`,
`index`, `count`. **No replay**: a frame published while a client is
disconnected is gone for good, so subscribers refetch the REST state on
every (re)open. Per-subscriber buffer (bounded): progress frames are
coalesced (a queued one is superseded by the newest), lifecycle frames
are kept; only an all-lifecycle backlog gives up its oldest frame,
which is logged and counted.

**Non-finite floats** (`NaN`, `±Inf` — a diverged trainer) never reach
the wire as JSON: every body, SSE frame and monitor frame goes through
`backend/json_safe.py`, which sends `null` for the value and adds a
sibling `nonfinite` map naming the keys it replaced with their kind
(`{"current_loss": "inf", "avg_loss": "-inf"}`). The marker is attached
per object, so a list item names its own field (`runs[i].nonfinite`).
Note SQLite maps `NaN` to `NULL`, so `NaN` only ever reaches the stream
frames, never a stored row. Clients must render a marker as a loud
"diverged" state, never as "no measurement".

## 3. Runs (subprocess training)

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/runs` | `limit` (1..500, default 50), `status` (`pending/running/completed/failed/cancelled`) | `ListRunsOut` |
| GET | `/runs/active` | — | `RunOut`; 404 `no_active_run` |
| GET | `/runs/{id}` | — | `RunOut`; 404 |
| POST | `/runs` | body `StartRunIn`: `config_path`, `start_from` (default `"teacher"`), `reset_optimizer` | `RunOut` -> **201**; 409 `run_already_active`, 409 `run_directory_conflict`, 422 `config_invalid`, 500 `training_launch_failed` |
| POST | `/runs/{id}/stop` | body `StopRunIn`: `force` (default false) | `RunOut`; 409 `run_not_running` |
| GET | `/runs/{id}/log` | `lines` (1..500) | `{"log": "<tail text>"}` — the tail is read from the **end** of the file (a run's log is the one artifact that grows without bound, docs 07 F-14) |
| DELETE | `/runs` | — | `{"deleted": N}` (history wipe) |

`RunOut`: `id, status, config_path, mode, phase, total_steps,
done_steps, current_loss, avg_loss, cache_done, cache_total, pid,
exit_code, error, log_path, created_at, updated_at, started_at,
finished_at` (nullable where listed), plus the optional `nonfinite`
marker described in section 2.

## 4. Config

Config files are validated TOML at `config_path` relative to the
project config dir; `path` selects the file.

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/config` | `path` (**required** -- empty is 422 `invalid_query`) | nested JSON mirroring `TrainingConfig` |
| PATCH | `/config` | `ConfigPatchIn{path, overrides}` deep-merge | merged config JSON; file untouched unless merged config validates (422 `config_invalid`) |
| GET | `/config/raw` | `path` | `{"content": "<toml text>"}` |
| PUT | `/config/raw` | `ConfigRawIn{path, content}` | `{"ok": true}` (create-or-replace; 422 on invalid TOML). **Stores the user's own text**: the document is parsed to validate it, then written verbatim through a temp sibling + rename, so comments, key order and keys the model does not declare survive (docs 07 F-08). `PATCH /config` (the form editor) still rewrites through the model by design |
| GET | `/config/options` | — | `{"options": [{...field schema...}]}` — schema only, no config read |
| GET | `/config/start-options` | `path` | `StartOptionsOut`: `start_from: {option: {path, available, label}}`, `has_unfinished_run`, `last_finished` or `null` |

404 `config_not_found` for a missing `path`.

## 5. Settings

Tiered override store (`settings.toml`); values are strings.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/settings` | — | `{"stored": {k: v}, "resolved": {k: v\|null}}` |
| POST | `/settings` | `SettingsIn` (all optional: `default_config, comfy_dir, venv_python, checkpoints_dir, loras_dir`) | same shape; **absent key = untouched, `""` = clear override**; 400 `settings_invalid` |

Validation is **pure** -- it looks, it never acts (docs 07 F-15): a
rejected update leaves the filesystem exactly as it was. What each key
must satisfy:

| Key | Accepted when |
|---|---|
| `comfy_dir` | it is an existing directory |
| `venv_python` | it is an existing **executable** file -- the value is executed on every start, so "it exists" is not the contract (F-06) |
| `checkpoints_dir`, `loras_dir` | absolute, not an existing non-directory, and *creatable*: the nearest existing ancestor is a writable directory. The directory itself is created **after** the commit, so an update that fails validation creates nothing |

## 6. Assets

Kinds: `checkpoint`, `lora`, `dataset` (last is catalog-only: browse
and mkdir/upload unsupported — the booleans in the catalog say so).

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/assets/{kind}` | — | `AssetCatalogOut`: `kind, base_dir, files[], options[{value,label}], upload_supported, browse_supported` |
| GET | `/assets/{kind}/browse` | `path` (relative; `""` = root) | `AssetBrowseOut`: `kind, path, folders[], files[]` |
| GET | `/assets/{kind}/inspect` | `path` | header-only safetensors metadata: checkpoint `{kind, path, components}` / lora `{kind, path, dtype, rank, key_count}` |
| PUT | `/assets/{kind}/folders/{path}` | — | `AssetPathOut` -> **201** |
| PUT | `/assets/{kind}/files/{path}` | raw bytes body (streamed, bounded) | `AssetPathOut` -> **201**; **413** `asset_too_large` if the body or its declared `Content-Length` exceeds **8 GiB** |

Upload policy (contract in `application/ports/asset_store.py`,
enforced presentation + adapter): the name must end in **`.safetensors`**
(422 `invalid_query` otherwise — traversal/unknown-kind attempts are
422 too); the body is read as a stream that stops at the cap, so a
chunked request cannot buffer the server past it; the bytes land in a
`.part` sibling and are renamed into place, so a rejected or failed
upload leaves no directory, no file and no partial behind.

## 7. Datasets

Legacy v1 datasets list with `stats: null` (identity only); v2 adds
stats/sets/tasks. 409 `dataset_not_migrated` when v2 endpoints hit a v1
dataset.

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/datasets` | — | `DatasetListOut`: `datasets[{info, stats\|null}], count` |
| POST | `/datasets` | `CreateDatasetIn{name, description?}` | `DatasetSummaryOut` -> **201**; 409 `dataset_exists`, 409 `dataset_directory_conflict` |
| GET | `/datasets/{name}` | — | `DatasetDetailOut`: `info, stats, sets[], active_tasks[]` |
| DELETE | `/datasets/{name}` | — | `{"deleted": true}`; 409 while a task is active |
| GET | `/datasets/{name}/items` | `committed` (bool, optional), `limit` (1..500, **default: no limit, i.e. every row**), `offset` (>=0, default 0) | `DatasetItemsOut`: `items[], count, limit, offset`; 422 for an out-of-range `limit`/`offset`. Paging is opt-in — the curation UI still asks for everything; `limit`/`offset` let a caller page through a huge dataset instead of materialising it (docs 07 F-14) |
| PATCH | `/datasets/{name}/items` | `BulkUpdateItemsIn{item_ids, prompt?, prompt_mode("set"/"prepend"/"append"), neg_prompt?, neg_prompt_mode(same)?, cfg?, type("good"/"bad")?}` — legacy truthy gates (empty never clears in bulk); `prepend`/`append` are idempotent (re-applying is a no-op) | `{"updated": N}`; 422 `invalid_query` for unknown mode/type or an all-empty change set |
| PATCH | `/datasets/{name}/items/{id}` | `UpdateItemIn` — every `null` field untouched; `""` clears a caption; explicit `type` replaces the legacy toggle | `DatasetItemOut` |
| POST | `/datasets/{name}/items/discard` | `ItemIdsIn` | `{"deleted": N}` |
| GET | `/datasets/{name}/files/{path}` | — | file bytes (item previews); media type from the suffix; 404 `dataset_not_found` / `dataset_file_not_found` (missing **or** escaping the dataset dir — an escape is reported as not-found, never resolved) |
| PUT | `/datasets/{name}/preview` | `SetPreviewIn{item_id}` — an id, never a path (the server reads the path from the dataset's own rows) | `DatasetPreviewOut{preview_path}`; 404 `dataset_not_found` / `dataset_item_not_found`, 409 `dataset_not_migrated`, 422 `invalid_query` when the item has no preview image or its file is gone |
| GET | `/datasets/{name}/sets` | — | `DatasetSetsOut`: `sets[{id,name,description,created_at,members}], count` |
| POST | `/datasets/{name}/sets` | `CommitItemsIn{item_ids, name}` | `CommitOut{set_id, set_name, added}` -> **201** |
| GET | `/datasets/{name}/tasks` | `active_only` (bool) | `DatasetTasksOut` (sweeps dead rows first) |
| POST | `/datasets/{name}/tasks` | `StartDatasetTaskIn` — one body, discriminated by `kind`: **`ingest_lora`** `{image_dir(absolute, not sandboxed), recursive, resize_mode("fit"/"center_crop"/"pad"/"resize"), latent_size, max_aspect_ratio, neg_prompt, seed}` vs **`generate_teacher`** `{prompt_mode("list"/"keywords"), prompts, keywords, keywords_file, template, min/max_keywords, neg_mode("list"/"keywords"), negative_prompt, neg_*…, cfg_min/max, steps_min/max, t_mode("uniform"/"low"/"mid"/"high"/"logit"), t_low/t_high, batch_size, n_conditions, n_samples_per_cond}`; shared `{model(relative to checkpoints dir, sandboxed), seed, latent_size, model_type("eps"/"vpred")}`, `image_dir` defaults empty and is ignored by `generate_teacher` | `DatasetTaskOut` -> **201**; `total` = image count (ingest) or `n_conditions × n_samples_per_cond` (generate); 409 `dataset_task_active`; 422 `invalid_query` for unknown kind/mode/enum, inverted ranges, empty prompt/keyword sources (validated in `application/teacher_prompts.py` *before* the row exists) |
| POST | `/datasets/{name}/tasks/{id}/stop` | — | `DatasetTaskOut` (SIGKILL); 409 if terminal |

`DatasetItemOut`: `id, source_id, shard_id, prompt, neg_prompt,
model_type, type, cfg, seed, source_path, latent_h, latent_w,
preview_path, committed`. `DatasetTaskOut`: `id, dataset, kind, status,
pid, current, total, error, params, created_at, updated_at`.
`DatasetSummaryOut`/`DatasetDetailOut` carry `preview_path`: the
**resolved** card image (stored override when its file still exists,
else the first non-bad item's preview, else null -- never a dead or
guessed path).

## 8. Graphs

Submission body for `validate`/`run` (`GraphRunIn`):
`nodes: [{id, class_name, params}]`, `edges: [{from_node, from_port,
to_node, to_port}]`. A structurally bad submission is **not** a body
rejection — the endpoint validates and answers 422 `graph_invalid`
with the complete issue list in `details`
(`[{severity, code, message, node_id, edge_index, param}]`, so the
editor can localise every problem in one round trip.

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/graphs/nodes` | `refresh` (bool) | `GraphCatalogOut`: `count, domains: {domain: [GraphNodeOut]}, load_errors[]` |
| POST | `/graphs/nodes/{class}/diagnostics` | `DiagnosticsIn{params}` | `{"messages": {input: [lines]}}`; 404 unknown class, 400 `node_diagnostics_failed` |
| POST | `/graphs/validate` | `GraphRunIn` | `ValidateOut{ok, issues[]}` — **always 200** (`ok=false` = `/run` would refuse) |
| POST | `/graphs/run` | `GraphRunIn` | `ExecutionSummaryOut` -> **201**; 422 `graph_invalid`, 409 `graph_execution_active` (single-active) |
| GET | `/graphs/executions` | `limit` (1..500, default 50) | `ExecutionListOut` (summaries, newest first) |
| DELETE | `/graphs/executions` | — | `{"deleted": N}` (active rows lose their CAS and stop) |
| GET | `/graphs/executions/{id}` | — | `ExecutionOut`: summary + `results[{node_id, ok, outputs, error, duration_ms}]` + `graph` snapshot; 404 |
| POST | `/graphs/executions/{id}/stop` | — | `ExecutionOut`; 409 `graph_execution_not_active` if terminal |
| GET | `/graphs/library` | — | `SavedGraphListOut` (summaries, most recently updated first) |
| PUT | `/graphs/library/{name}` | `LibraryGraphIn` (graph payload + `description`; unknown keys ride along verbatim) | `SavedGraphOut` -> **201** first save / **200** replace; 422 `invalid_query` (empty or >120-char name) |
| GET | `/graphs/library/{name}` | — | `SavedGraphOut{name, description, graph, node_count, created_at, updated_at}`; 404 `graph_not_found` |
| DELETE | `/graphs/library/{name}` | — | `{"deleted": bool}`; 404 |

**Execution lifecycle** (persisted; reconciled at startup):
`queued -> running -> finished | error | stopped`. Single-active
enforced with a DB CAS. `NodeResultOut.outputs` values are the node's
returned mapping (for sources, identity values of their inputs).

**Catalog shape** (`GraphNodeOut`): `class_name, display_name, domain
(module-path derived), module, doc, bases[], inputs[]/outputs[]` of
`PortOut{name, type, required, doc, type_mro, default, default_repr,
path_kind, choices, visible_when, widget_only}`, plus `node_kind,
presets[], has_diagnostics`. Outputs carry identity defaults (no
default / no hints).

## 9. Library payload format

Saved graph payloads are stored **verbatim** as submitted (with
`format` stamped). Validation happens at run, never at save — old or
hand-edited payloads load fine and fail loudly only when executed.
Client-side legacy `localStorage` graphs (`ng_graph_v1`) import through
`PUT /graphs/library/{name}` (see `03-migration-strategy.md` §5).

## 10. Monitor stream + static pages (M6)

| Method | Path | Notes |
|---|---|---|
| GET | `/monitor/{monitor_id}/stream` | SSE (below) |
| GET | `/` | serves `frontend/index.html` (app shell) |
| GET | `/monitor/{monitor_id}` | serves `frontend/monitor.html` (id read client-side) |
| GET | `/graph` | serves `frontend/graph.html` (editor, M7) |
| GET | `/ui/*` | frontend ES modules + css |

**Monitor SSE frames**: `{"type": "connected"}` opener, then the
bus's pre-rendered `data: {json}` frames -- history replay first (a
dashboard opened mid-run restores its chart), then live step reports,
plus `{"type": "clear"}` broadcasts when a new run claims the id and
the terminal `{"type": "run_end", "step", "cancelled"}`. Payload keys
are the trainer's report dict verbatim (`step`, `total_steps`,
`loss`, `lr`, `t`, `weight_t_*`, `prob_t_*`, `*_ms`, `vram_*`,
`resident_*_mb`, ...); frames without `step` that the page doesn't
recognise are ignored, never guessed at. The frame contract and the
port/wiring facts are pinned in `03-migration-strategy.md` §4.

Page routes are registered after the API and never under `/api/`, so
unknown API routes keep the JSON error envelope (section 1).
