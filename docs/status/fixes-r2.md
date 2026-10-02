# Round-2 fixes: progress

Source: `docs/design/backend/08-review-2026-10-02.md` (12 findings,
N-01..N-14) and `TASK-backend-fixes.md` (22 work packages), handed over
from an external review. Every finding was re-verified against this
branch before being worked on; the reproductions named in the review
live outside the repo and were run against each commit.

Branch: `fixes/review-r2`.

## Done

| WP | Finding | Commit | What proves it |
|---|---|---|---|
| WP-01 | N-04 SSE drops per-node graph events | `60bbaff` | `test_json_safe.py::test_client_buffer` + `test_delivery_class_assignment`; the review's `r13_sse_coalescing.py` now delivers A..F instead of only F |
| WP-02 | N-05 dataset file endpoint serves any file | `10e457c` | `test_api_datasets.py` allowlist/header cases; `r11_dataset_file_endpoint.py` now 404s metadata-equivalent, a shard and an svg |
| WP-03 | N-02 upload blocks the loop and doubles RAM; N-14 silent overwrite | `ce3bb8b` | `test_assets.py::test_assets` streaming checks (0.2 MB peak for a 48 MB body; a health check served *during* an open upload), plus the store-level abort/overwrite cases |
| WP-04 | N-06 run page re-broke resync / non-finite / silent catch | `d79ba82` | `frontend/tests/events.test.mjs` (13 node:test cases, no browser) |
| WP-05 | N-08 monitor bus can raise into the training thread | `d5eccfe` | `smoke_test_monitor_bus.py`: bad frame does not raise, does not poison the replay, warns once |
| WP-06 | N-12 orphan shard files after a failed discard | `b4a5b88` | `test_dataset_library.py`: unreferenced removed, referenced untouched, a locked file does not stop the sweep |
| WP-07 | N-03 a finished run recorded failed 0/100 | `51ac7f4` | `test_start_stop.py::test_reconcile_reads_the_progress_file_of_a_dead_run`; `r10_finished_while_down.py` now prints `completed done=100/100` |

## Still open

**Phase 1 Track A is complete.**

Phase 1B: WP-08 (adopted-pid liveness by number only -- a recycled pid
keeps a run `running` and `stop()` refused), WP-09 (garbled log note),
WP-10 (hermetic tests; `test_training_adapter.py` needs `COMFY_DIR`).

Phase 2: WP-11..WP-18. **WP-11 (ruff+mypy with a committed baseline)
gates the rest** -- the code already carries `# noqa: BLE001` and
`# type: ignore` markers for tools that are not run, which is how N-11's
50 mypy errors and 92 ruff findings survived review.

Phase 3: WP-19..WP-22, explicitly blocked on the `core/` removal being
merged. That removal is now largely done on this branch (the node graph
imports no `core.*`), so those blockers need re-checking rather than
assuming.

## Notes for whoever continues

* **The review is not wrong about the rules, only about the current
  state.** Several of its findings were already fixed by the round-1
  work it was reviewing; each WP below says so where it applies, and the
  finding was re-verified rather than taken on trust.
* **`docs/design/backend/02-api-reference.md`'s error table is a build
  input.** `backend/tests/test_error_contract.py` parses it and fails if
  a code exists without a documented row. Adding an error class means
  editing that table in the same commit -- it caught WP-03's
  `asset_exists` on the first run.
* **The review's `r9_upload_blocking.py` writes into ComfyUI's real
  `models/loras`.** Its `build_services()` sets no `loras_dir`, so
  `path_tiers` falls through to the actual model directory. It left a
  600 MB `big.safetensors` there, removed on sight. Use a script that
  pins `loras_dir` to a temp directory before rerunning it.
* **`_env`-style fixtures must not write to the developer's tree.** One
  first draft of WP-02's test overwrote a dataset's real `metadata.db`,
  which broke every later case in that file through an unrelated code
  path (the catalog listing opens that sqlite file).
* **Palette counts in `test_graph_discovery.py`/`test_api_graphs.py` are
  derived, not literal** (`support.concrete_node_classes()`), so adding
  or retiring a node does not require editing tests.