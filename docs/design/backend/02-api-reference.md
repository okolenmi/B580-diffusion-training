# 02 -- API contract reference

Source of truth for request/response *field* level detail is
`backend/presentation/schemas.py`; this document is the navigable
contract (endpoint, params, envelope, error codes) the frontend is
written against. Route table also mirrored in `01-architecture.md` §API.

Status: **shipped through M4** (47 endpoints under `/api/v1`).
Section 10 (monitor stream) is the M5 slice described in
`03-migration-strategy.md` §4 and is not yet served.

## 1. Conventions

* **Base path**: `/api/v1`. All bodies and responses are JSON
  (assets upload is raw bytes; SSE is `text/event-stream`).
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
  | `run_not_found`, `no_active_run` | 404 | runs |
  | `run_already_active`, `run_not_running` | 409 | runs |
  | `config_not_found` | 404 | config |
  | `config_invalid` | 422 | config |
  | `training_launch_failed` | 500 | runs |
  | `settings_invalid` | 400 | settings |
  | `dataset_not_found`, `dataset_item_not_found`, `dataset_task_not_found` | 404 | datasets |
  | `dataset_exists`, `dataset_not_migrated`, `dataset_task_active`, `dataset_task_not_active` | 409 | datasets |
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
`index`, `count`. Per-subscriber queue (bounded; overflow drops the
event with a debug log — a slow client never blocks a run).

## 3. Runs (subprocess training)

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/runs` | `limit` (1..500, default 50), `status` (`pending/running/completed/failed/cancelled`) | `ListRunsOut` |
| GET | `/runs/active` | — | `RunOut`; 404 `no_active_run` |
| GET | `/runs/{id}` | — | `RunOut`; 404 |
| POST | `/runs` | body `StartRunIn`: `config_path`, `start_from` (default `"teacher"`), `reset_optimizer` | `RunOut` -> **201**; 409 `run_already_active`, 422 `config_invalid`, 500 `training_launch_failed` |
| POST | `/runs/{id}/stop` | body `StopRunIn`: `force` (default false) | `RunOut`; 409 `run_not_running` |
| GET | `/runs/{id}/log` | `lines` (1..500) | `{"log": "<tail text>"}` |
| DELETE | `/runs` | — | `{"deleted": N}` (history wipe) |

`RunOut`: `id, status, config_path, mode, phase, total_steps,
done_steps, current_loss, avg_loss, cache_done, cache_total, pid,
exit_code, error, log_path, created_at, updated_at, started_at,
finished_at` (nullable where listed).

## 4. Config

Config files are validated TOML at `config_path` relative to the
project config dir; `path` selects the file.

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/config` | `path` (default: default config) | nested JSON mirroring `TrainingConfig` |
| PATCH | `/config` | `ConfigPatchIn{path, overrides}` deep-merge | merged config JSON; file untouched unless merged config validates (422 `config_invalid`) |
| GET | `/config/raw` | `path` | `{"content": "<toml text>"}` |
| PUT | `/config/raw` | `ConfigRawIn{path, content}` | `{"ok": true}` (create-or-replace; 422 on invalid TOML) |
| GET | `/config/options` | — | `{"options": [{...field schema...}]}` — schema only, no config read |
| GET | `/config/start-options` | `path` | `StartOptionsOut`: `start_from: {option: {path, available, label}}`, `has_unfinished_run`, `last_finished` or `null` |

404 `config_not_found` for a missing `path`.

## 5. Settings

Tiered override store (`settings.toml`); values are strings.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/settings` | — | `{"stored": {k: v}, "resolved": {k: v\|null}}` |
| POST | `/settings` | `SettingsIn` (all optional: `default_config, comfy_dir, venv_python, checkpoints_dir, loras_dir`) | same shape; **absent key = untouched, `""` = clear override**; 400 `settings_invalid` |

## 6. Assets

Kinds: `checkpoint`, `lora`, `dataset` (last is catalog-only: browse
and mkdir/upload unsupported — the booleans in the catalog say so).

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/assets/{kind}` | — | `AssetCatalogOut`: `kind, base_dir, files[], options[{value,label}], upload_supported, browse_supported` |
| GET | `/assets/{kind}/browse` | `path` (relative; `""` = root) | `AssetBrowseOut`: `kind, path, folders[], files[]` |
| GET | `/assets/{kind}/inspect` | `path` | header-only safetensors metadata: checkpoint `{kind, path, components}` / lora `{kind, path, dtype, rank, key_count}` |
| PUT | `/assets/{kind}/folders/{path}` | — | `AssetPathOut` -> **201** |
| PUT | `/assets/{kind}/files/{path}` | raw bytes body | `AssetPathOut` -> **201** |

422 `invalid_query` for traversal/unsupported-kind attempts.

## 7. Datasets

Legacy v1 datasets list with `stats: null` (identity only); v2 adds
stats/sets/tasks. 409 `dataset_not_migrated` when v2 endpoints hit a v1
dataset.

| Method | Path | Query / body | Response |
|---|---|---|---|
| GET | `/datasets` | — | `DatasetListOut`: `datasets[{info, stats\|null}], count` |
| POST | `/datasets` | `CreateDatasetIn{name, description?}` | `DatasetSummaryOut` -> **201**; 409 `dataset_exists` |
| GET | `/datasets/{name}` | — | `DatasetDetailOut`: `info, stats, sets[], active_tasks[]` |
| DELETE | `/datasets/{name}` | — | `{"deleted": true}`; 409 while a task is active |
| GET | `/datasets/{name}/items` | `committed` (bool, optional) | `DatasetItemsOut`: `items[], count` |
| PATCH | `/datasets/{name}/items` | `BulkUpdateItemsIn{item_ids, prompt?, prompt_mode("set"/"append"), neg_prompt?, cfg?}` | `{"updated": N}` |
| PATCH | `/datasets/{name}/items/{id}` | `UpdateItemIn` — every `null` field untouched; `""` clears a caption; explicit `type` replaces the legacy toggle | `DatasetItemOut` |
| POST | `/datasets/{name}/items/discard` | `ItemIdsIn` | `{"deleted": N}` |
| GET | `/datasets/{name}/sets` | — | `DatasetSetsOut`: `sets[{id,name,description,created_at,members}], count` |
| POST | `/datasets/{name}/sets` | `CommitItemsIn{item_ids, name}` | `CommitOut{set_id, set_name, added}` -> **201** |
| GET | `/datasets/{name}/tasks` | `active_only` (bool) | `DatasetTasksOut` (sweeps dead rows first) |
| POST | `/datasets/{name}/tasks` | `StartDatasetTaskIn{kind, image_dir, model, recursive, resize_mode, latent_size, neg_prompt, model_type, seed, max_aspect_ratio}` | `DatasetTaskOut` -> **201**; 409 `dataset_task_active` |
| POST | `/datasets/{name}/tasks/{id}/stop` | — | `DatasetTaskOut` (SIGKILL); 409 if terminal |

`DatasetItemOut`: `id, source_id, shard_id, prompt, neg_prompt,
model_type, type, cfg, seed, source_path, latent_h, latent_w,
preview_path, committed`. `DatasetTaskOut`: `id, dataset, kind, status,
pid, current, total, error, params, created_at, updated_at`.

## 8. Graphs

Submission body for `validate`/`run` (`GraphRunIn`):
`nodes: [{id, class_name, params}]`, `edges: [{from_node, from_port,
to_node, to_port}]`. A structurally bad submission is **not** a body
rejection — the endpoint validates and answers 422 `graph_invalid`
with the complete issue list in `details.issues`
(`{severity, code, message, node_id, edge_index, param}`), so the
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

## 10. Monitor stream (M5 slice — planned, not yet served)

`GET /api/v1/monitor/{monitor_id}/stream` — mirrors the legacy
`/api/nodegraph/monitor/{id}/stream` frame contract exactly
(`{"type": "connected"}` first, then unwrapped step-report dicts,
`{"type": "clear"}`, terminal `{"type": "run_end"}`; history replay on
subscribe). Full facts and the port/wiring plan:
`03-migration-strategy.md` §4.
