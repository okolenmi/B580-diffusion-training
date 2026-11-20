# Contract: HTTP errors

Every non-2xx response, including unknown routes and method-not-allowed, uses one envelope (no bare `{"detail": ...}`):

```json
{"error": {"code": "run_not_found", "message": "run 7 not found", "details": { ... }}}
```

`details` is present only when the use case supplies it (the full issue list for `graph_invalid`, `{loc, msg, type}` entries for
`validation_error`). A body the schema rejects is 422 `validation_error`.

Every `ApplicationError` subclass declares its own `code` and HTTP status in `backend/application/errors.py`, and
`backend/tests/test_error_contract.py` parses the table below and fails if the two disagree in either direction. Adding a code is one
class plus one row here.

The browser doors are closed without authentication: a request's `Host` must name this server (loopback names, plus `BACKEND_ALLOWED_HOSTS`),
and a state-changing method carrying an `Origin` must match this server's origin or one in `BACKEND_ALLOWED_ORIGINS`. A request with no
`Origin` passes (curl, scripts, same-origin fetches), and `GET`/`HEAD` stay open.

| Code | Status | Domain |
|---|---|---|
| `invalid_query` | 422 | runs/graphs (limit, status, name range) |
| `forbidden_host`, `forbidden_origin` | 403 | requests (Host not served / cross-site state change) |
| `run_not_found` | 404 | runs |
| `run_directory_conflict` | 409 | runs (`runs/run_<id>/` already holds files) |
| `training_launch_failed` | 500 | runs |
| `settings_invalid` | 400 | settings |
| `asset_too_large` | 413 | assets (upload body over the 8 GiB cap, or a declared `Content-Length` over it — the body is streamed, so the running total is the authoritative check) |
| `asset_exists` | 409 | assets (the upload target already exists; pass `?overwrite=true` to replace it — a `PUT` never silently overwrites a checkpoint) |
| `dataset_not_found`, `dataset_file_not_found` | 404 | datasets (image libraries) |
| `dataset_exists` | 409 | datasets (image libraries) |
| `graph_invalid` | 422 | graphs |
| `graph_not_found`, `graph_execution_not_found`, `node_class_not_found`, `starter_not_found` | 404 | graphs |
| `graph_execution_not_active` | 409 | graphs |
| `action_not_found` | 404 | graphs (the run's graph has no such node, or the node declares no such action) |
| `action_unavailable` | 409 | graphs (the node has the action but nothing in the graph listens for what it produces; the message says what to add) |
| `app_invalid` | 422 | apps (the app document's shape, or a value submitted to run one that the app does not expose / that is fed by another node — `details` lists every problem) |
| `board_invalid` | 422 | boards (the shape of a draft view or app sent to `/boards/resolve`; `details` lists every problem) |
| `app_not_found` | 404 | apps |
| `artifact_not_found` | 404 | artifacts (a picture a node left for the page is not there, or the path leaves the folder) |
| `apps_unavailable` | 503 | apps, boards (the server was started without app support) |
| `memory_unavailable` | 409 | graphs, datasets (memory admission: this start cannot fit on the device — either the device total is unknown (`details.reason = device_total_unknown`, and `details.holders` names what the unfinished rows still claim) or the ledger's refusal is in `details`: capacity, foreign reserve, every holder and its size, free, asked. A refused start writes no row and holds nothing) |
| `node_diagnostics_failed` | 400 | graphs |
| `installer_not_allowed` | 409 | installer (a wizard write after this installation is configured) |
| `install_refused` | 400 | installer (a deliberate refusal: no usable environment at the chosen folder, a torch that is not the Intel XPU build, nothing left to install, or pip exiting non-zero. Nothing was changed — the message says which) |
| `install_job_not_found` | 404 | installer (a job id this process never issued, or one lost when the server restarted; inspect the environment again to see what is installed) |
| `tokenizer_download_failed` | 502 | installer (a tokenizer file set could not be fetched from the model hub, or a downloaded file failed validation; the message says where to place the files by hand) |
