# TASK: fix the open findings in `backend/` and `frontend/`

Repository: `okolenmi/B580-diffusion-training` (single user, one Intel Arc B580,
12 GB; a local FastAPI backend + vanilla-JS frontend that supervises LoRA
training). Start from `main` at commit `5b7724f` or later (`git pull` first).
This document is self-contained: every finding you need is described here.
If the files `docs/design/backend/07-review-2026-10-01.md` and
`08-review-2026-10-02.md` exist in the repo, they hold the long versions.

You are working from a written review in which every problem was reproduced or
carefully read. Your job is to fix them, one work package (WP) at a time, with
tests. Work in the order given. You may stop after any WP; each one must leave
the repo green and committed.

---------------------------------------------------------------------------

## 0. How to work (read fully; these rules matter more than speed)

**Git.** Create a branch `fixes/review-r2`. One commit per WP, message
`WP-NN: <short summary>`, body = finding id, what changed, which tests prove it.
Run the verification commands (section 0.3) before every commit. Never commit
a red suite.

**Scope discipline.**
- Minimal diffs. Do not reformat, rename or "tidy" code outside the WP.
- Do **not** touch `core/` (another session is deleting it right now) and do
  not edit `nodes/` unless a WP says so. Never edit the `import core...` lines
  you see in `backend/`; leave them exactly as they are.
- Do not change any HTTP response shape or event payload unless the WP says so
  (the frontend depends on them). If you must, update the frontend caller, the
  schema in `backend/presentation/schemas.py`, and the tests in the same commit.
- No new runtime dependencies. Dev-only tools are allowed only where a WP says.
- Never weaken, skip, delete or loosen an existing test to get green. If an
  existing test pins behaviour that a WP deliberately changes, change that test
  and say so in the commit body.
- If the suite is red *before* your change, check with `git stash`; if it is red
  on the baseline too, note it in your final report and move on. Do not "fix"
  unrelated failures.
- When unsure between a bigger and a smaller change, take the smaller one and
  write the doubt in the final report.

**0.1 Rules that apply to every change** (each one comes from a real bug):
1. Everything from outside the process (files written by the trainer, request
   bodies, settings, file names) is hostile input: parse per record, coerce
   types, never let one bad record end a loop or thread.
2. Validate first, then act. No row, file or directory is created before every
   check has passed. DB commits come before file deletions.
3. Never emit non-JSON (`NaN`/`Infinity`). The server already has
   `backend/json_safe.py`; use it, do not bypass it.
4. No silent swallowing: a `catch {}` / `except Exception: pass` must log or
   count what it dropped.
5. `async def` route handlers must not call blocking file or DB code.
6. A comment that says "bounded", "atomic", "safe", "never" is a claim. Every
   such claim needs a test that would fail if the claim were false.
7. **Fix the pattern, not the instance.** After fixing a defect, `grep` the
   repo for siblings of the same pattern and fix or test each one.
8. Tests build their own world: no dependence on the developer's directory
   layout, environment variables, ports, or the contents of `runs/`,
   `datasets/`, `checkpoints/` (those are user data; tests must never write to
   them).
9. Event kinds have a delivery class: **state** (newest wins), **delta** (every
   one must arrive), **lifecycle** (never dropped). Decide the class before
   touching queues or coalescing.

**0.2 Known traps: things that look wrong but are intentional. Do not "fix".**
- `except Exception:  # noqa: BLE001 -- <reason>` at thread boundaries and in
  best-effort notes is deliberate. Keep them.
- `check(True, ...)` inside `except X:` in tests is correct: the failing branch
  is `check(False, ...)` inside the `try`. Do not rewrite these.
- `backend/presentation/security.py` (Host/Origin guard) was reviewed and is
  correct. Do not change its logic.
- Seeding `sqlite_sequence` in `persistence/run_repository.py` is deliberate
  (it prevents run-id collisions with directories already on disk).
- `device="xpu"` hardcoded in `dataset_task_worker.py` is deliberate (the
  project targets one B580).
- FastAPI `Depends(...)` in default arguments triggers ruff `B008`; that is the
  framework idiom. Ignore that rule (WP-11 configures it).
- `docs/known-issues/resolved.md` is only for fixes confirmed on real hardware.
  Put anything not hardware-confirmed into `pending-testing.md`.

**0.3 Environment and verification commands.** No GPU is needed; everything
here runs on CPU.
```
pip install -r requirements.txt pillow httpx uvicorn pydantic-settings tomli_w
python3 backend/tests/run_all.py          # backend suite
python3 run_tests.py                      # nodes + manager suites
```
Until WP-10 is done, one backend test (`test_training_adapter.py`) needs a
ComfyUI directory; run it as `COMFY_DIR=/tmp/fakecomfy python3 ...`
(`mkdir -p /tmp/fakecomfy`). The test helpers live in `backend/tests/support.py`
(`build_services`, `asgi_request`, `seed_run`, fakes). Tests use the project's
own `check(cond, msg)` style, not pytest; follow it. New test files must be
registered the way the others are (look at `backend/tests/run_all.py`).

**0.4 Coordination with the `core/` removal.**
The backend still depends on `core/` in these places: `core.cli` (the training
subprocess entry point), `core.config_model` / `core.config_io` (the TOML
config editor and the training command builder), `core.xpu_env`,
`core.comfy_setup`. Therefore:
- **Track A** (WP-01..06, WP-11..18) does not need `core.cli` and is safe to do
  now.
- **Track B** (WP-07..10) concerns the `core.cli` subprocess run route. Before
  starting Track B run `ls core/cli.py`. If the file is gone or the run route
  was re-pointed, **stop Track B** and write the reason in your final report;
  a human decides whether those WPs still apply.
- Phase 3 (WP-19..22) must wait until the `core/` removal is merged.

---------------------------------------------------------------------------

## 1. Phase 1: bug fixes

### WP-01 SSE buffer drops per-node graph events (N-04) [Track A]
**Problem (reproduced).** `backend/presentation/sse.py::ClientBuffer` treats
`run_progressed` and `graph_execution_progressed` as the same "progress" kind
and, on every `put`, evicts *all* queued progress frames. `run_progressed` is
state (newest wins). `graph_execution_progressed` is a **delta** (one node
finished: `node_id`, `ok`, `duration_ms`). Result: 6 node events put together
-> 1 delivered; a `run_progressed` also evicts a queued graph event. The editor
sets each node's status from these events (`frontend/js/editor/executions.js`
~line 166), so live node status is wrong for fast graphs.

**Do.** Replace the single `PROGRESS_EVENT_TYPES` rule with delivery classes:
- state: `run_progressed` -> coalesce **per run** (`run_id`): a new frame
  replaces the queued one with the same key. Never evict other kinds/keys.
- delta: `graph_execution_progressed` -> never coalesced; all delivered, in
  order.
- everything else is lifecycle: never dropped for the sake of progress.
- Overflow (buffer at `QUEUE_MAX`): drop the oldest *state* frame first; if none,
  drop the oldest *delta*; only if the buffer is all lifecycle drop the oldest
  frame. Every drop is counted and logged (keep `coalesced`/`dropped` counters,
  add `dropped_delta`).
Compute the key where the event object is available (`on_event`) and pass it to
`put(event_type, payload, key=None)`. Keep `serialize_event` unchanged.

**Tests (must fail before, pass after).** In the existing SSE/ClientBuffer test
file (`grep -ln ClientBuffer backend/tests`) add:
```python
async def test_graph_deltas_all_delivered():
    b = ClientBuffer()
    for n in "ABCDEF":
        b.put("graph_execution_progressed", json.dumps({"node_id": n}))   # adapt to your signature
    got = [json.loads(await b.get(0.1))["node_id"] for _ in range(len(b))]
    check(got == list("ABCDEF"), got)
```
plus: a `run_progressed` never evicts a queued graph frame; two
`run_progressed` for the same run keep only the newest; lifecycle frames
survive a flood of progress frames; overflow drops state first, then delta, and
increments the right counter.
**Done when** the tests pass and `grep -rn PROGRESS_EVENT_TYPES backend` shows
no remaining use of the old rule.

### WP-02 Dataset file endpoint serves any file (N-05) [Track A]
**Problem (reproduced).** `GET /api/v1/datasets/{name}/files/{path}`
(`presentation/api/datasets.py::read_dataset_file` ->
`application/use_cases/read_dataset_file.py` ->
`infrastructure/dataset_files.py::FsDatasetFiles.read`) is documented as
"preview image bytes" but returns *any* file inside the dataset directory:
`metadata.db`, multi-GB `.safetensors` shards (read fully into RAM with
`read_bytes()`), and `.svg` as `image/svg+xml` on the app's own origin, with no
`X-Content-Type-Options`, CSP or `Content-Disposition`. Traversal is already
blocked correctly; keep that.

**Do.**
- Allowlist extensions `.png .jpg .jpeg .webp` (case-insensitive). Anything else
  -> the same `DatasetFileNotFoundError` (404 `dataset_file_not_found`) as a
  missing file. Remove `.svg` and `.gif` from `_MEDIA_TYPES`.
- Size cap (constant, 32 MB): larger -> 404 (do not read it).
- Response headers on this route: `X-Content-Type-Options: nosniff`,
  `Content-Security-Policy: default-src 'none'; sandbox`,
  `Cache-Control: private, max-age=60`.
- Before changing anything, `grep -rn "/files/" frontend/js` and confirm the UI
  only requests preview images; adjust nothing else.
**Tests.** In `test_api_datasets.py` (or the file that already tests this
route): `metadata.db`, a `.safetensors` shard, `x.svg` and an oversized png all
return 404; a small png, jpg and webp return 200 with the right media type and
the three headers; `../` traversal and a symlink pointing outside still 404.
**Done when** those pass and the route's docstring states the allowlist.

### WP-03 Upload: stream to disk, do not buffer (N-02, N-14) [Track A]
**Problem (measured).** `presentation/api/assets.py::upload_asset` is
`async def`; it appends every chunk to a list, `b"".join()`s it, then
`FileAssetStore.save_upload` calls `partial.write_bytes(content)` on the event
loop. With a 600 MB upload a trivial request took up to 739 ms (17 ms idle) and
memory grew by about 2x the file. The cap is 8 GiB, so the peak at the cap is
about 16 GiB. The docstring claims it cannot "buffer the server into swap",
which is false. Also (N-14) an existing `.safetensors` of the same name is
silently replaced.

**Do.**
- Change the port (`application/ports/asset_store.py`) and adapter to a
  streaming write: begin -> write chunks to `<name>.part` -> finish (atomic
  `replace`) or abort (delete the `.part`). Enforce the size cap while writing;
  on exceeding it or on any error remove the `.part`. Keep all existing
  validation that runs before touching the filesystem (kind, safe path, suffix).
- In the async handler, run each blocking write (`write`, `finish`, `abort`) off
  the event loop (`anyio.to_thread.run_sync` / `run_in_threadpool`). Never hold
  more than one chunk.
- Overwrite protection: if the target exists return 409 (new error code
  `asset_exists`, add it next to the other asset errors and map it in
  `presentation/errors.py`) unless the query parameter `overwrite=true` is
  given. Check how the frontend uploads (`grep -rn "assets/" frontend/js`); if a
  UI upload exists, make it surface the 409 message instead of failing silently.
- Fix the docstring so it states what is true.
**Tests.** (a) Upload 48 MB in 1 MB chunks and assert `tracemalloc` peak growth
is under 8 MB. (b) Exceeding the cap leaves no `.part` and no final file.
(c) A failure while writing (monkeypatch the writer to raise) leaves no `.part`
and no final file. (d) Existing target -> 409; with `overwrite=true` -> 201 and
replaced. (e) A concurrent `GET /api/v1/health` completes while a (slow,
chunked) upload is in flight; build this deterministically with an async
generator body that awaits between chunks (use
`httpx.AsyncClient(transport=httpx.ASGITransport(app=app))`), not with timing
thresholds.
**Done when** all pass and no code path builds a `bytes` of the whole body.

### WP-04 Run-detail page repeats fixed mistakes; shared event module (N-06) [Track A]
**Problem (read from code, no browser).** `frontend/js/views/run.js` does
`sse("/events", { onMessage: handleEvent })` with no `onOpen` resync (`/events`
has no replay, so a missed `run_completed` leaves the page on "running"),
ignores the `nonfinite` marker (server sends a diverged loss as `null` plus
`nonfinite: {loss: "nan"}`; the run page shows an em dash), and has
`try { e = JSON.parse(raw.data); } catch { return; }` (silent). It also copies
`fmtDuration`/`fmtRel` from `dashboard.js`.
`dashboard.js` (see its header comment and `onOpen: () => { startSafetyPoll();
resync(); }`) and `lib/value.js` (`setValue`, `nonfiniteText`) show the correct
pattern.

**Do.**
1. Create `frontend/js/lib/events.js` exporting `subscribeEvents({ onEvent,
   onResync, onNotice })`: wraps the existing `sse()` from `api.js`; parses each
   frame; a parse failure is **counted and reported through `onNotice`**
   (rate-limited), never swallowed; `onResync` runs on **every** `open`
   including the first. Optionally export `startSafetyPoll(fn, ms)`.
2. Create `frontend/js/lib/format.js` with `fmtDuration`, `fmtRel`, `fmtTime`
   moved out of `dashboard.js`/`run.js` (keep behaviour identical; import them
   in both).
3. `run.js`: use `subscribeEvents`; `onResync` = `loadRun().then(loadLog)`;
   handle `run.nonfinite` exactly as `dashboard.js:429` does and render the loss
   with `setValue`/`nonfiniteText` so a diverged loss shows a loud state.
4. Migrate `dashboard.js` and `editor.js` to `subscribeEvents` **only if** it
   is a mechanical change; otherwise leave them and say so in the report.
5. `grep -rn "catch *{ *return" frontend/js` and fix every remaining silent
   catch (rule 7). `editor.js:56` (corrupted localStorage -> defaults) is
   acceptable; add a `console.warn`.
**Tests.** Add `frontend/tests/` with `node:test` unit tests (no browser):
`format.test.mjs`; `events.test.mjs` using a fake `EventSource` (assert: resync
called on open and on reconnect, a malformed frame calls `onNotice` and does not
throw, a valid frame reaches `onEvent`). Add `node --test frontend/tests` to
`scripts/full_gate.sh` (skip with a message if `node` is missing).
**Done when** the tests pass and `run.js` has no silent catch.

### WP-05 Monitor bus can raise into the training thread (N-08) [Track A]
**Problem (read).** Repo-root `monitor_bus.py::MonitorBus.report` runs
`json.dumps(data)` on the *calling* thread (the trainer), after appending to
history, but only once a subscriber exists. A non-serializable value would raise
into the training loop, and only while a dashboard is open. `subscribe()` replays
history with the same `json.dumps`.
**Do.** Wrap both serializations in `try/except (TypeError, ValueError)`: log a
warning once per exception type, skip that frame, keep the bus alive. Do not
change the frame format. Keep `backend/infrastructure/monitor_bus.py`'s
`sanitize` call.
**Tests.** In `nodes/smoke_tests/smoke_test_monitor_bus.py` (the file that pins
this class): with a subscriber attached, `report()` of a frame containing an
object that cannot be serialized does not raise, and the next valid frame is
delivered; replay with such a frame in history does not crash.

### WP-06 Orphan shard files after a failed discard (N-12) [Track A]
**Problem.** `infrastructure/dataset_library.py::discard` correctly commits row
deletion first and then removes files, but a failed unlink leaves an orphan
shard on disk forever (only a log line).
**Do.** Add `sweep_orphan_shards(name) -> int` to the library port/adapter:
delete files under `shards/` that no `shards` row references. Call it at the end
of `discard()` (best-effort, logged) and from `delete`. Do not add an HTTP
endpoint.
**Tests.** Extend `test_dataset_library.py`: make one unlink fail during
`discard` (monkeypatch `Path.unlink`), assert rows are gone, the orphan remains,
then a second call to `sweep_orphan_shards` removes it and returns 1; a file
referenced by a row is never removed.

---------------------------------------------------------------------------

## 2. Phase 1B: the `core.cli` training-run route (Track B; see 0.4)

Check `ls core/cli.py` first. If absent, stop and report.

### WP-07 A run that finished while the server was down is marked failed (N-03)
**Problem (reproduced).** `application/use_cases/reconcile_runs.py` adopts a
trainer only if its pid is alive. For a dead pid it calls `gateway.kill()`
(False) and marks the run `failed`, "process already gone", without reading
the progress file. A progress file with steps 1..100 and the trainer's
`{"phase":"finished"}` line produced `status=failed done=0/100`. The adoption
path (`application/supervisor.py::_finalize`) already treats that terminal line
as authoritative; reconcile must use the same evidence.
**Do.** Inject the progress source into `ReconcileRuns` (bootstrap and
`tests/support.py::build_services` both construct it; update both). For a
`running` row with a dead/unowned pid: read the whole progress file
(`ProgressSource.read_new` on a fresh reader starts at 0), fold the samples
(last step/total/loss/phase; last terminal verdict), apply them with
`run.record_progress(...)`, then finalise with the same rule as adoption:
`finished` -> `mark_completed`; `error` -> `mark_failed("trainer reported an
error")`; no terminal line -> `mark_failed` as today but **with the real
`done_steps`**. Keep the CAS (`update_if_status(..., expected=RUNNING)`) and the
event publishing. Reuse the supervisor's verdict logic through a shared helper
instead of copying it.
**Test seed** (put in `test_start_stop.py` or a new `test_reconcile_runs.py`):
```python
run = seed_run(repo, clock, total_steps=100, start=True, pid=4242)
prog = runs_dir / f"run_{run.id}" / "log.progress.jsonl"; prog.parent.mkdir(parents=True)
lines = [{"phase":"training_start","total_steps":100}] + \
        [{"phase":"step","step":s,"total":100,"loss":0.1,"avg":0.1,"lr":1e-4} for s in range(1,101)] + \
        [{"phase":"finished"}]
prog.write_text("".join(json.dumps(l)+"\n" for l in lines))
gw.alive.discard(4242)
svc.reconcile_runs.execute()
r = repo.get(run.id)
check(r.status.value == "completed" and r.done_steps == 100, (r.status, r.done_steps))
```
Add variants: `{"phase":"error"}` last -> `failed` with done_steps from the file;
no terminal line -> `failed` with done_steps from the file; empty/missing
progress file -> `failed` (as today). Also assert a `RunCompleted` event is
published and `ReconcileResult.cleaned` counts it.

### WP-08 Liveness of an adopted pid is checked by number only (N-07)
**Problem (read).** `infrastructure/subprocess_gateway.py::is_alive` is
`os.kill(pid, 0)` for any pid this process did not spawn. After adoption, if
the trainer dies and the number is reused, the supervisor sees "alive" forever,
the row stays `running`, and `stop()` is refused as "not our trainer", so only a
backend restart clears it. `PermissionError` is also treated as alive.
**Do.** For a pid not in `self._procs`: after `os.kill(pid, 0)` succeeds (or
raises `PermissionError`), return `owns(pid)`'s verdict via
`cmdline_mentions(pid, marker)`: `False` -> not alive (recycled number); `True`
or `None` (cannot tell) -> alive. Keep the fast path for spawned children.
**Test.** Start `sleep 30` *outside* the gateway; `SubprocessTrainingGateway(
..., cmdline_marker="definitely-not-in-its-cmdline").is_alive(pid)` is `False`;
with `cmdline_marker="sleep"` it is `True`. Kill the sleeper in a `finally`.

### WP-09 Garbled log note (N-09)
**Problem (reproduced).** `application/supervisor.py::_start` writes
`--- RUN  -- server watching pid N ---` for a normal start and
`--- RUN REAPTIED (adopted after a server restart) -- ...` after adoption.
Users read this log.
**Do.** Normal: `--- server watching pid {pid} ---`. Adopted:
`--- server re-attached to pid {pid} after a restart ---`.
**Test.** Assert both exact strings via the artifacts fake; check no existing
test pins the old text.

### WP-10 Hermetic tests (N-01)
**Problem (reproduced).** `backend/tests/test_training_adapter.py::
test_signal_safety` spawns the real gateway, which needs a ComfyUI directory; on
a fresh clone `run_all.py` reports 1/22 failed (passes with `COMFY_DIR` set).
Also, after running the suites an untracked `checkpoints/` directory appears in
the repo root (observed; origin not confirmed).
**Do.** Give the test its own world: a temp project dir and a
`WorkspaceLayout(..., settings_kv=...)` that supplies `comfy_dir`, with env vars
restored in `finally`. Find which test creates `checkpoints/` in the repo root
(run each test file and `git status --short`) and make it use a temp dir.
**Done when** `env -u COMFY_DIR python3 backend/tests/run_all.py` is green from a
fresh clone in any directory and `git status --short` is clean afterwards.

---------------------------------------------------------------------------

## 3. Phase 2: quality infrastructure (Track A)

### WP-11 Add ruff + mypy and a single check command (Q1, N-11)
**Facts (measured).** No linter/type checker is configured, yet the code has
`# noqa: BLE001` and `# type: ignore` markers. Baseline in `backend/` (non-test):
mypy 50 errors (arg-type 14, attr-defined 10, valid-type 10, union-attr 5,
name-defined 5, override 1); ruff (rules `E,F,B,BLE,UP,SIM,C901,RET,ARG`,
ignoring E501) 92 findings: B008 50 (FastAPI idiom), UP037 9, ARG001 9, C901 8,
SIM105 4, UP035 3, F401 3, BLE001 2, F821 2, ARG002 2. Real defects among them:
`application/dto.py` uses `GraphExecution`, `DatasetInfo`, `DatasetStats` in
annotations without importing them (works only because annotations are
deferred); `application/use_cases/start_training.py:44` uses `RunSupervisor`
unimported; unused imports.
**Do.**
1. Add `pyproject.toml`: ruff (`select = E,F,B,BLE,UP,SIM,C901,RET,ARG,ASYNC`;
   `ignore = E501,B008`; max-complexity 12; tests exempt from ARG/BLE) and mypy
   (`packages = ["backend"]`, `ignore_missing_imports = true`,
   `follow_imports = "silent"`, exclude `backend/tests`). Match the Python
   version in `docs/setup.md`.
2. Fix the cheap real ones: F401, F821, the `dto.py`/`start_training.py` missing
   names (import under `if TYPE_CHECKING:`), and apply ruff's safe auto-fixes
   (`UP037`, `UP035`).
3. Create `scripts/check_quality.py` that runs ruff and mypy with JSON output and
   compares per-file/per-rule counts with a committed
   `scripts/quality_baseline.json`: fail if any count **increases**, print where
   it decreased. Commit the baseline from the post-fix state.
4. Create `scripts/check.sh` running: `check_quality.py`, backend tests,
   `run_tests.py`, frontend `node --check` and `node --test`; call it from
   `scripts/full_gate.sh` and remove any hardcoded home path from that script.
**Done when** `scripts/check.sh` is green and a deliberately added unused import
makes it fail.

### WP-12 Remove the `list` shadowing; type the run id (Q2)
**12a.** Methods named `list` on `RunRepository`, `GraphExecutionRepository`,
`DatasetLibrary` and their adapters/fakes shadow the builtin inside their own
class body, producing 10 mypy `valid-type` errors on `list[...]` annotations.
Rename to `list_runs`, `list_executions`, `list_datasets` (ports, SQLite
adapters, `tests/support.py` fakes, use cases, all call sites, tests). Use
`mypy` and the test suite to find every site. Do not change HTTP routes.
**12b.** `Run.id` is `RunId | None`, causing five
`# type: ignore[arg-type]`. Add `Run.require_id() -> RunId` (raises
`RuntimeError` if unpersisted) and use it at those sites; remove the ignores.
**Done when** mypy `valid-type` and the removed `arg-type` ignores are gone and
the baseline file is updated (counts only go down).

### WP-13 Contract tests for fakes vs real adapters (Q9)
`backend/tests/support.py` contains in-memory fakes that can drift from the
SQLite adapters. Write each repository contract once
(`backend/tests/contracts/run_repository_contract.py`: add/get, ordering and
limit of the list method, `update_if_status` success and lost-CAS, `find_active`
counting `created`, `list_unfinished`, `delete_all`) and run it against both
`InMemoryRunRepository` and `SqliteRunRepository`. Same for
`GraphExecutionRepository` if time allows. Any behavioural difference you find
is a bug in one of them; fix the fake unless the SQLite adapter is wrong.

### WP-14 Resource budgets and default pagination (Q10, F-14 residual)
Create `backend/limits.py` holding the scattered constants (upload cap, SSE
queue/heartbeat, dataset page size, log tail default, stop grace, history
caps), imported where they are used. Dataset item listing
(`application/use_cases/list_dataset_items.py`, `infrastructure/dataset_library.py`)
currently returns **every row** unless `limit` is given and the UI asks for
all of them. Make a default page size (e.g. 500) with `total` and `next_offset`
in the response, and make `frontend/js/views/datasets.js` load more pages
(a "Load more" button is enough). Add a test that a dataset with 1,200 items
returns 500 per page and that paging covers all items exactly once.

### WP-15 Frontend shared core (Q8)
Twelve helper names are defined in more than one JS file (`boot`, `log`,
`logError`, `showError`, `errText`/`_errText`, `fmtTime`, `fmtRel`,
`fmtDuration`, `buildWidget`, `showTab`, `handleEvent`). After WP-04, move the
remaining duplicates into `frontend/js/lib/` (`format.js`, `log.js`,
`errors.js`) and import them. Add `// @ts-check` and JSDoc types to the new
`lib/` modules and run `npx tsc --noEmit --checkJs` over `frontend/js/lib` if
`npx` is available (do not add `node_modules` to the repo). Extend
`frontend/tests/` for the new modules.

### WP-16 Property-based tests at the boundaries (Q5)
Add `hypothesis` as a **dev** dependency (`requirements-dev.txt`). Tests:
(a) progress reader: for a valid multi-line file, truncating at **every byte
offset** and then appending the rest yields exactly the original samples, no
exception, none lost or duplicated; random garbage lines never raise;
(b) `json_safe.sanitize`/`strict_dumps`: output always parses with a strict
JSON parser; (c) `security.host_name` and the Host/Origin guard: random header
strings never raise; (d) the asset path sandbox: random relative paths never
resolve outside the base.

### WP-17 Decisions as ADRs and a doc check (Q11)
Create `docs/decisions/` with short ADRs (context, decision, consequences,
the test that pins it) for: no authentication + what the Host/Origin guard does
and does not protect; in-process graph training; adoption of surviving
trainers; B580-only device hardcoding. Add `scripts/check_docs.py` that fails if
a documented number is stale: the endpoint count claimed in
`docs/design/backend/03-migration-strategy.md` versus
`len(app.openapi()["paths"] operations)` (currently 48 documented operations),
and the test-file counts quoted in `docs/status/progress.md`. Wire it into
`scripts/check.sh`.

### WP-18 Measure test quality (Q12) [report only]
Run `mutmut` (dev tool, do not commit its cache) on
`backend/application/supervisor.py`,
`backend/infrastructure/jsonl_progress_source.py`,
`backend/application/use_cases/reconcile_runs.py`,
`backend/presentation/sse.py`. Do not fix survivors blindly: for each survivor
either add a test that kills it or explain in `docs/status/mutation-notes.md`
why it is equivalent. Add branch-coverage output for `backend/application` and
`backend/infrastructure` to `scripts/check.sh` (report, no threshold).

---------------------------------------------------------------------------

## 4. Phase 3: large refactors (do NOT start until the `core/` removal is merged)

Write a short design note first (`docs/design/backend/09-<topic>.md`: goal,
current state, steps, how each step stays green), then implement in small
commits behind the existing tests. Stop and report if a step needs a decision
you cannot take from this document.

- **WP-19 Split large units (Q7).** Targets: `presentation/schemas.py` (997
  lines) split per router; `infrastructure/dataset_library.py` (534) into
  queries / commands / files; `infrastructure/graph/runtime.py::validate`
  (171 lines, complexity about 32) into one function per validation pass;
  `frontend/js/views/datasets.js` (1339) into modules with a small state store;
  `tests/support.py` (1197) and `tests/visual_smoke.py` (1287) by domain. Rule:
  no file over about 400 lines, no function over about 60 lines or complexity 12.
  Behaviour must not change; the baseline in `quality_baseline.json` only goes
  down.
- **WP-20 One supervisor abstraction (Q3).** `RunSupervisor` and
  `GraphExecutionSupervisor` hand-roll the same lifecycle (thread, guard, crash
  repair, final drain, terminal-state CAS). Extract the shared parts into one
  helper/base so a fix lands once; extract a `ProcessGateway` base for
  signalling, identity and escalation shared by the training and dataset-task
  gateways.
- **WP-21 Event contract (Q4).** Give every domain event a monotonically
  increasing `seq`; keep a bounded replay ring for **lifecycle** events;
  honour `Last-Event-ID` on `/events`; generate the event JSON Schema from the
  dataclasses and have a test that fails when a frontend handler references a
  field the schema lacks; document each kind's delivery class (state / delta /
  lifecycle) in one table next to the dataclasses.
- **WP-22 Process isolation for graph training (Q6).** Graph execution runs in
  the API server process (`infrastructure/graph/runtime.py`): a device fault, OOM
  kill or restart takes the server and the run down, and training competes with
  the event loop. Use `infrastructure/dataset_task_gateway.py` +
  `dataset_task_worker.py` as the template: run the runtime in a child process
  behind the same port, stream node events over a file/pipe, and reuse the
  adoption/reconcile logic for restarts. Do this last and in small steps.

---------------------------------------------------------------------------

## 5. Final report (required)

At the end, write a message (and `docs/status/fixes-r2.md`) with one line per WP:
`done | partial | skipped`, the commit hash, what test proves it, and anything
you were unsure about. Also list: tests you changed (and why), traps from 0.2
you were tempted to touch, anything you found but did not fix. Put every fix
that was not run on real hardware into `docs/known-issues/pending-testing.md`.
Do not claim a WP is done unless its "Done when" / tests exist and pass.
