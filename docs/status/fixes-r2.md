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
| WP-08 | N-07 adopted-pid liveness by number only | `a1e0358` | `test_training_adapter.py`, driven with a real `sleep 30`; verified failing against the old `is_alive` |
| WP-09 | N-09 garbled log note | `a1e0358` | `test_start_stop.py` pins both exact strings |
| WP-10 | N-01 test_training_adapter needs a ComfyUI dir | `3c3df85` | passes with `.env` hidden, which is what makes the old code fail |

## Still open

**Phases 1 and 1B are complete** -- every one of the 12 round-2
findings is now either fixed (N-01..N-09, N-12, N-14) or, for N-02/N-14,
fixed and measured. Nothing from the review's Phase 1 or 1B remains.

Phase 2: WP-11..WP-18. WP-11 is done and it now gates every subsequent
change -- see the note at the end about what it caught.

| WP | What | Commit | Evidence |
| --- | --- | --- | --- |
| WP-11 | ruff+mypy against a committed baseline | `308074d` | `scripts/quality_baseline.json`; verified failing on an injected error |
| WP-12 | no method named `list`; `require_id()` instead of `id` + ignore | `c4126ff` | mypy 39 -> 25, no `valid-type` left, 24/24 backend |
| WP-13 | repository contract, fake vs SQLite adapter | `6522c60` | found real drift in the fake; both now pass one contract |
| WP-14 | budgets in one module; dataset items paged | `27cb7c0` | 1,200 rows -> 3 pages, every id exactly once; verified live |
| WP-15 | one console, one message box, one error sentence | `3d2403a` | `frontend/tests/` 30 cases; found the suite's 2.1 GB of leaked /tmp |
| WP-16 | property tests at the untrusted boundaries; one place to point the app away from ComfyUI | `f66d47b` | found a NUL byte turning a 422 into a 500; truncation verified at every byte offset; live-verified redirect |
| WP-17 | decision records; a check for documented numbers | `e65643d` | found the migration strategy claiming 47 endpoints against an actual 50 |
| WP-21 | event contract: `seq`, lifecycle replay ring, `Last-Event-ID`, generated JSON Schema | `1295946`, `555e61d` | renamed `cache_total` and watched the check name `run.js: e.cache_total`; 70 payload/schema pairs cross-checked against real `jsonschema` |

| WP-20 | the one compare-and-swap; `ExecutionLifecycleWriter.fail_if_unfinished` | `9a1eff0`, `244ba73` | the dataset adapter's five hand-rolled guarded UPDATEs became one `finalize_if_active` |
| WP-22 | process isolation for graph execution | `bcfa939`, `903737c`, `1d77bf2`, `8ec48cd` | `docs/design/13-process-isolation.md`; a SIGKILLed child fails its row while the server keeps serving, and a run survives a server restart |

Phase 3: complete. WP-19 declined, WP-21 done, WP-20 done, WP-22 done.

* **WP-21 is done**, in the two halves it was specified as. One
  deliberate deviation: the review asks for the delivery-class table
  "next to the dataclasses", and it is in
  `application/event_delivery.py` instead, because
  `domain/events.py` opens by stating that events have "no knowledge of
  JSON, SSE, or any transport" — and a delivery class is exactly that.
  `application/` is where this project puts policy, the infrastructure
  bus can import it without inverting the layering, and one location was
  the part that mattered.
* **WP-20 stays blocked**, and now for a second reason. It touches
  `subprocess_gateway.py`, which hardcodes `cmdline_marker="core.cli"`
  and the argv `-m core.cli` — code the `core/` removal will disturb. The
  PID-reuse guard fixed in `75e3fed` lives in the same module, which is
  an argument for not churning it.
* **WP-20: mostly dissolved, one real extraction done** (`9a1eff0`).
  The review named `RunSupervisor` vs `GraphExecutionSupervisor`
  hand-rolling the same lifecycle, and a `ProcessGateway` base shared by
  the training and dataset-task gateways. Both pairs lost a member when
  `core/` went, so most of the finding was resolved by removal rather
  than by refactoring, and inventing an abstraction from what remained
  would have been solving a problem that no longer exists. Measured
  rather than assumed:

  - `GraphExecutionSupervisor` (205 lines: thread per execution, cancel
    registry, per-node progress, terminal CAS) and
    `DatasetTaskSweeper` (89 lines: a repair pass, no thread, no events)
    are **not two copies of one thing** -- a sweeper judges dead rows, a
    supervisor owns a run's life. Nothing to unify.
  - `ProcessGateway`: `SubprocessTrainingGateway` is deleted;
    `SubprocessDatasetTaskGateway` is the only one left, and the
    identity logic it would have shared already lives once in
    `process_identity.py`.

  What *was* real: the graph-execution terminal-repair rule was written
  twice in the same package -- `GraphExecutionSupervisor._fail_leftover`
  (crash repair) and `ReconcileGraphExecutions` (startup sweep) both
  read the status, marked failed with a note, and CAS'd from what they
  read. That is now `ExecutionLifecycleWriter.fail_if_unfinished`, which
  takes a clock and keeps the note caller-supplied (a crash and a
  restart are different facts). Pinned in `test_value_objects.py`,
  including the two properties that matter: a terminal row is not
  re-failed or re-announced, and a lost CAS announces nothing.

  Followed up, because leaving a duplicated rule on the grounds that it
  was tidier to leave it is not a defence: the dataset-task terminal CAS
  is now the same statement (`244ba73`). `finish_if_active` /
  `fail_if_active` / `kill_if_active` were three port methods and two
  near-identical bodies in one adapter, plus two more hand-rolled
  guarded UPDATEs in `update_progress` -- five copies of one rule in a
  single file. They are now one `finalize_if_active(task_id, status,
  error=None)` over `infrastructure/persistence/cas.py`, which the
  graph-execution repository calls too.

  What is still two things, deliberately: the statement (`cas.py`) and
  the announcement (`application/lifecycle_writer.py`). Dataset tasks
  have no events, so they call the first without the second. Giving them
  events is a feature decision, not a cleanup -- and until it is made,
  each half has one definition rather than the pair having one, which
  was the actual complaint.
* **WP-19 declined.** Every target is a test file or a schema module, and
  splitting them to hit a line count rather than to fix a comprehension
  problem adds indirection and a large reviewable diff for no measured
  gain. The review's numbers are stale as well (it gives
  `presentation/schemas.py` as 997; it is 1008).
* **WP-22 was deferred, then done.** The original judgement was that
  isolation is a large change to a path that works, bought against a
  risk not yet observed. That was right when it was written and stopped
  being right once the device-fault question was settled as a driver
  problem (`docs/known-issues/resolved.md`): the isolation is not buying
  insurance against a rare fault, it is buying back the server.

  Four commits, one per step, each green:

  | step | commit | what it established |
  | --- | --- | --- |
  | 1–2 | `bcfa939` | the event file is the only channel; both gateways call one producer |
  | 3 | `903737c` | the child path, real children, mutation-checked |
  | 4 | `1d77bf2` | adoption across a restart, pid found by argv rather than stored |
  | 5 | `8ec48cd` | default flipped to `child`, and the backend's logging made visible |

  What it bought, measured on a live server rather than argued: a
  `SIGKILL` of the run's process now fails that run's row and leaves the
  server serving, where before it killed the server too; and a run
  survives the server being killed underneath it, because a child in its
  own session outlives its parent.

  What it cost, also measured: ~1.95 s of startup per run, and one
  operator-visible change they should expect — after a restart the log
  now says `adopted N still-running graph execution(s)` instead of
  failing the row, which is the point but is also a new line to read.

  `BACKEND_GRAPH_EXECUTION=inprocess` is the rollback. Giving up
  isolation also gives up adoption, since a thread cannot outlive the
  process it is in.

`docs/design/11-core-removal.md` records the five edges `core/` had,
what each became, and what the route cost.

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
* **`env -u COMFY_DIR` does not simulate a fresh clone.** `path_tiers`
  loads the repo's gitignored `.env`, which sets `COMFY_DIR`. Hide
  `.env` as well when checking whether a test depends on the developer's
  machine (that is how N-01 was actually reproduced).