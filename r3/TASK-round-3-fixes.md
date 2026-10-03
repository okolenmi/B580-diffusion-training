# TASK: fix the round-3 findings in `backend/`

Repository `okolenmi/B580-diffusion-training`, start from `main` at `13566c2`
or later (`git pull` first). Single user, one Intel Arc B580 (12 GB). The
backend supervises LoRA training that now runs as **graph executions in a
child process**, reporting through an append-only event file.

This document is self-contained. If `docs/design/backend/10-review-round-3.md`
exists it has the long versions. Work through the work packages (WP) in order;
you may stop after any WP; each one must leave the repo green and committed.

---------------------------------------------------------------------------

## 0. How to work

**Git.** Branch `fixes/round-3`. One commit per WP, message
`R3-NN: <summary>` with the finding id and the tests that prove it in the body.
Run the verification commands (0.3) before every commit; never commit red.

**Scope.**
- Minimal diffs; no reformatting or renaming outside the WP.
- Do not change HTTP response shapes or event payloads unless a WP says so. If
  you must, update the frontend caller, the schemas and the tests in the same
  commit (`backend/presentation/event_schema.py` and
  `backend/tests/test_event_contract.py` pin the event shapes).
- No new runtime dependencies. Never weaken, skip or delete an existing test
  to get green; if a test pins behaviour a WP deliberately changes, change the
  test and say so in the commit body.
- If something is red *before* your change, check with `git stash`; if the
  baseline is red too, note it in the final report and move on.
- If unsure, take the smaller change and write the doubt in the report.

**0.1 Rules (each one comes from a real bug in this codebase).**
1. A recovery path must use every piece of evidence the normal path uses.
2. Treat everything from outside the process as hostile: files the child
   writes, request bodies, settings. Parse per record; one bad record must not
   end a loop or thread.
3. Never mark a row terminal while its child process may still be alive, unless
   you have delivered stop and kill and confirmed it is gone.
4. Validate first, then act; DB commits before file deletions.
5. No silent swallowing: every `except Exception` logs with context.
6. A comment that says "safe", "never", "atomic", "bounded" is a claim; add a
   test that fails if the claim is false.
7. **Fix the pattern, not the instance:** after a fix, `grep` for siblings of
   the same pattern and fix or test each.
8. Tests build their own world: no dependence on the developer's directories,
   environment variables or the contents of model folders; tests must never
   write outside temporary directories.
9. Event kinds have delivery classes (see `application/event_delivery.py`):
   lifecycle (never dropped), delta (every one must arrive), state (newest
   wins). Decide the class before touching queues.

**0.2 Intentional. Do not "fix".**
- The event-file design: append-only, one `write(2)` per record, whole-line
  reads, `outcome` record, truncating the event file at launch, argv-based
  child identity (`find_by_argv`), refusing to signal a process whose cmdline
  lacks the marker, the final drain after `is_alive` says no.
- `STATE_EVENT_TYPES` being empty (no state events exist right now).
- `InProcessGraphTaskGateway` and the `graph_execution_mode` rollback switch.
- `except Exception:  # noqa: BLE001 -- <reason>` at thread boundaries.
- `check(True, ...)` inside `except X:` in tests (the failing branch is a
  `check(False, ...)` inside the `try`).
- `backend/presentation/security.py` (Host/Origin guard): reviewed, correct.
- `device="xpu"` hardcoded in the dataset worker (the project targets a B580).

**0.3 Environment and commands.**
```
pip install -r requirements.txt -r requirements-dev.txt pillow httpx uvicorn pydantic-settings tomli_w hypothesis
python3 backend/tests/run_all.py            # backend suite (parallel)
python3 run_tests.py                        # nodes + manager
python3 scripts/check_quality.py            # ruff + mypy regression gate (counts may only go down)
```
Until R3-04 is done, set `COMFY_DIR` to an empty temp dir for the backend
suite (`mkdir -p /tmp/fakecomfy && export COMFY_DIR=/tmp/fakecomfy`) and delete
any `checkpoints/` or `loras/` that appear in the repo root before committing.
Tests use the project's `check(cond, msg)` style, not pytest. Helpers:
`backend/tests/support.py` (`build_services`, `asgi_request`, fakes) and the
doubles in `backend/tests/test_graph_adoption.py` (`RecordingGateway`,
`RecordingWriter`, `StubExecutions`, `_supervisor(...)`).
The project targets Python 3.14. If you run an older Python, one test
(`test_event_contract.py`) fails for an unrelated reason (R3-06); do not chase it.

---------------------------------------------------------------------------

## 1. Work packages

### R3-01 Watcher errors must not orphan a live child (N3-01)
**Where.** `backend/application/graph_supervisor.py`, `_watch` and `_finish`.
**Problem (reproduced).** `_watch` calls `self._executions.get(execution_id)`
every poll with no individual guard. One transient error (e.g. SQLite
"database is locked") reaches the outer `except Exception`, then
`finally: _finish(..., saw_outcome=False)` marks the row failed with "exited
without reporting an outcome (crashed, or a device fault killed it)" and
`_release` forgets the pid. The child is alive, nobody signals it, the
single-active check now allows a second run (two trainers, one 12 GB GPU), and
the first run's real outcome is never read.
**Do.**
1. Guard each loop iteration (the `poll` + `_apply` + `get` body): on an
   exception log with the execution id, count consecutive failures, back off
   (0.5 s doubling to a 5 s cap) and continue. Reset the counter on success.
2. After 20 consecutive failures while the child is alive: `request_stop(pid)`,
   start the existing escalation, and finish the row failed with the message
   `supervision failed (<last error>); the child was stopped`.
3. In the final failure path never call `_finish(saw_outcome=False)` with the
   "crashed" text while `gateway.is_alive(pid)` is true. If the loop is exiting
   because of an unexpected exception and the child is alive, stop it first
   (INT, then the existing KILL escalation) and use the message from step 2.
   The "crashed, or a device fault" message is only for a child that is gone.
**Test seed** (adapt to the doubles; this is the verified scenario):
```python
class FlakyExecutions(StubExecutions):          # fails exactly one read inside the watch loop
    def __init__(self, **kw): super().__init__(**kw); self.reads = 0
    def get(self, execution_id):
        self.reads += 1
        if self.reads == 4: raise RuntimeError("database is locked")
        return super().get(execution_id)
gw = RecordingGateway(events_path, alive=True); gw.found = 777
sup = _supervisor(gw, writer=RecordingWriter(), executions=FlakyExecutions(graph=g))
sup.adopt(1); time.sleep(0.8)
# BEFORE the fix: row status 'error', gw.alive True, gw.stopped == [] and gw.killed == []
# AFTER: the row is still running, the watcher is still alive, and a later
#        outcome record finishes it correctly.
```
Add: (b) 20+ consecutive failures with a live child -> `gw.stopped` non-empty
and the row failed with the "supervision failed" message; (c) child dead +
exception -> still the old "crashed" message.

### R3-02 A run that finished while the server was down (N3-02)
**Where.** `GraphExecutionSupervisor.adopt`, `ReconcileGraphExecutions`
(`application/use_cases/reconcile_graph_executions.py`), `_finish`.
**Problem (reproduced).** `adopt()` returns `None` as soon as `find_running`
finds no live child, and reconcile fails the row as "server restarted while the
execution was in flight". If the child exited cleanly while the server was down,
its event file holds the remaining node results and a clean `outcome`; none is
read. Reproduced: file with nodes a, b, c and `outcome(error=None)` gave
`status=error results=0/3`.
**Do.**
1. Add one routine, e.g. `GraphExecutionSupervisor.recover(execution_id) ->
   bool`, used by reconcile when `adopt()` returns `None` and the event file
   exists and is non-empty. It reads the whole file with a fresh
   `ExecutionEventTail`, persists node records the row does not have yet
   (dedupe by `node_id`; respect the domain rule "no more results than
   nodes"), then decides from the last `outcome` record using the **same
   rule `_finish` uses**: clean outcome -> finished; outcome with error ->
   failed with that error; no outcome -> failed with "process exited without
   reporting an outcome (the server was not running)" and **keep the partial
   results**. Reuse `_finish`'s decision code rather than copying it (extract a
   function both call).
2. Reconcile order: adopt (live child) -> recover (dead child, evidence on
   disk) -> current "restarted while in flight" failure (no evidence). A
   `queued` row whose child never wrote anything keeps today's message.
3. `ReconcileResult` gets a `recovered` count (add to the dto, the log line and
   the startup log); do not change the existing `cleaned`/`adopted` meanings.
**Tests** (use the real `SqliteGraphExecutionRepository`,
`ExecutionLifecycleWriter`, `ExecutionEventWriter`; see how
`backend/tests/test_graph_adoption.py` wires them): (a) nodes + clean outcome
-> `finished`, all results present; (b) nodes + outcome with an error ->
`failed` with that error, results present; (c) nodes, no outcome -> `failed`
"without reporting an outcome", partial results kept; (d) the row already holds
a, b and the file has a, b, c + outcome -> exactly 3 results, no duplicates;
(e) event file missing -> today's message; (f) torn last line ignored; (g) the
`finished` event is published once.

### R3-03 The harness must not pass with zero checks (N3-03)
**Where.** `backend/tests/support.py` (`check`, `finish`, `FAILURES`), the
equivalent helpers in `nodes/smoke_tests` and `manager/smoke_tests` if they
have their own, and a new `scripts/check_tests_called.py`.
**Problem (reproduced).** `python3 -c "from backend.tests.support import
finish; finish()"` prints "ALL CHECKS PASSED" and exits 0 with no check run.
That is how two files "ran none of their tests" unnoticed.
**Do.**
1. Count executed checks in `check()`. `finish()` exits non-zero and prints
   `no checks were executed` when the count is zero; always print the count.
2. `scripts/check_tests_called.py`: AST-scan `backend/tests`,
   `nodes/smoke_tests`, `manager/smoke_tests`; fail if a file defines
   `test_*`/`check_*` functions that are never referenced anywhere in the file.
   (A scan for this today finds none; it must keep finding none.) Call it from
   `scripts/full_gate.sh` and document it in the gate section of the docs.
3. In `backend/tests/run_all.py`, print the number of checks per file and fail
   a file whose output contains no check lines.
**Tests.** A tiny test that runs a throwaway script calling only `finish()` and
asserts a non-zero exit; a test of the AST scan on a temp file with an uncalled
`test_x`.

### R3-04 Hermetic tests; nothing written outside temp dirs (N3-05)
**Problem (reproduced).** On a bare checkout (no `COMFY_DIR`)
`test_graph_task_gateway.py` (4 checks), `test_graph_adoption.py` and
`test_process_identity.py` fail: the real gateway needs a ComfyUI directory.
With `COMFY_DIR` set they pass. Separately `backend/tests/test_api_datasets.py`
(~line 418) writes `m.safetensors` into the *resolved* checkpoints directory
(`<ComfyUI>/models/checkpoints/`, or `<repo>/checkpoints/` when there is no
ComfyUI): on the developer's machine that is their real model folder.
**Do.**
1. In `backend/tests/support.py` add `hermetic_world()` (or similar) that
   creates a temporary ComfyUI-shaped tree (`models/checkpoints`,
   `models/loras`), sets `COMFY_DIR`, `CHECKPOINTS_DIR`, `LORAS_DIR` for the
   duration, and restores the environment afterwards. Apply it automatically
   for every backend test file (import-time in `support.py`, undone at exit) so
   no test can forget it.
2. Fix `test_api_datasets.py` (and `grep -rn "safetensors" backend/tests` for
   any other write to a resolved path) to write only under that temp tree.
3. In `backend/tests/run_all.py`: run each file with a scrubbed environment
   (remove `COMFY_DIR`, `CHECKPOINTS_DIR`, `LORAS_DIR`, `VENV_PYTHON`) so the
   fixture is proven to be sufficient; and snapshot the repo root's untracked
   entries (`git status --porcelain` when inside a git checkout) before and
   after the whole run, failing with the names of anything new.
**Done when** `env -u COMFY_DIR python3 backend/tests/run_all.py` is green on a
fresh clone in any directory, and `git status --short` is clean afterwards.

### R3-05 Event ids carry a process epoch (N3-04)
**Where.** `backend/infrastructure/events/callback_event_bus.py`,
`backend/presentation/sse.py` (`_last_event_id`, `_frame`, `stream_opened`),
`frontend/js/lib/events.js`, `backend/presentation/event_schema.py`,
`backend/tests/test_event_contract.py`.
**Problem (reproduced).** Sequence numbers restart at 1 in each server process;
`replay_since` recognises a foreign id only when `last_seq > newest`. On a new
process that already published 60 events, `replay_since(40)` from a *previous*
process returns `complete=True` with events 41..60, so the client skips its
resync and shows stale state.
**Do.**
1. `CallbackEventBus` gets `epoch` (8 hex chars from `uuid.uuid4()`, fixed at
   construction) and `replay_since(last_seq, epoch=None)`: a non-matching
   epoch returns `Replay(events=(), complete=False)`.
2. SSE frame ids become `"{epoch}:{seq}"`; `_last_event_id` parses that form and
   still accepts a bare integer (treated as epoch-unknown -> `complete=False`
   unless the bus has published nothing yet). `stream_opened` includes `epoch`.
3. Keep `seq` in the JSON payload as an integer (the schema and frontend read
   it). The browser sends `Last-Event-ID` back automatically, so
   `frontend/js/lib/events.js` should need no change except to remember the
   epoch if it tracks a watermark; check it and add a unit test in
   `frontend/tests/`.
**Tests.** Same epoch, in range -> normal replay; different epoch with an
in-range seq -> `complete=False`; bare integer id -> `complete=False`; the
event-contract test still passes.

### R3-06 Declare the Python floor (N3-06)
**Problem (reproduced).** `pyproject.toml` targets py314 (ruff, mypy) but has no
`requires-python`; nothing checks at startup; `docs/setup.md` names no version.
`event_schema.py` decides "is this `X | None`" with `get_origin(t) is Union`
(true only on 3.14); on 3.12 `schema_for("run_progressed")` raises and
`test_event_contract.py` fails. `config_schema.py` uses the same identity check.
**Do.**
1. Add `requires-python = ">=3.14"` (use whatever floor `pyproject.toml`
   already targets) under `[project]` if the file has one, otherwise add the
   minimum `[project]` table needed; state the version in `docs/setup.md`.
2. At the top of `backend/cli.py` (before heavy imports) and `run_tests.py`:
   if `sys.version_info` is below the floor, print one clear sentence naming
   the required and the running version and exit with status 2.
3. Add `backend/typing_compat.py::is_union(tp)` accepting both `typing.Union`
   and `types.UnionType`, and use it in `event_schema.py`, `config_schema.py`
   and the test helper `_sample`, so those modules work on any supported
   Python. Test with both `Optional[int]` and `int | None`.

### R3-07 Scratch lifecycle (N3-07)
**Where.** `graph_supervisor.py` (`_paths_for`), `use_cases/delete_graph_executions.py`,
`infrastructure/graph_event_stream.py::ExecutionEventTail.poll`,
`application/limits.py`, `bootstrap.py`.
**Problem.** `graph_executions/execution_N.{graph.json,events.jsonl,log}` are
never removed; deleting executions deletes rows only; the event file holds
every monitor report of the run. `poll()` does one `read()` to EOF.
**Do.**
1. A `GraphScratch` port (small: `remove(execution_id)`, `sweep(keep_ids)`,
   `size_bytes()`), implemented next to the supervisor's path logic. Deleting
   executions removes their scratch (skip any execution whose child is alive).
2. Startup sweep (after reconcile): remove scratch for ids that have no row,
   and keep at most `GRAPH_SCRATCH_KEEP` (new constant in `limits.py`, default
   50) newest terminal executions' files.
3. `poll()` reads at most `EVENT_TAIL_CHUNK_BYTES` (4 MiB, `limits.py`) per
   call and the watcher loops until `caught_up`, so adoption replay never loads
   an arbitrarily large file at once. Keep whole-line semantics.
**Tests.** delete removes files; a live child's files survive a delete; sweep
keeps the newest N and removes orphans; a 20 MiB event file is consumed in
chunks with identical records and no record split across chunks.

### R3-08 Child signal handling (N3-08)
**Where.** `backend/infrastructure/graph_task_worker.py::main`.
**Problem.** The SIGINT handler is installed after argument parsing and the
`nodes.xpu_env` import; a Stop in that window hits Python's default handler
(traceback, no `outcome`) and the row reads "crashed". SIGTERM is unhandled.
**Do.** Install handlers for SIGINT and SIGTERM as the first statements of
`main`, before any non-stdlib import; both set the cancel event. If the cancel
event is already set when the runtime starts, the run must end with a clean
`outcome` (error None) so the supervisor labels it "stop requested". Wrap the
body so `KeyboardInterrupt`/`SystemExit` still write an `outcome`.
**Test.** Spawn the real child module with a trivial graph, send SIGINT 50 ms
after spawn and again at random delays; the event file must end with an
`outcome` record and the process must exit without being killed by the signal.

### R3-09 Orphan-child reaper (defence in depth)
**Where.** `GraphTaskGateway` port (`list_children() -> list[tuple[int, int]]`
of `(pid, execution_id)`), both gateways, `GraphExecutionSupervisor`,
`bootstrap.py`.
**Do.** A daemon thread started at bootstrap (interval `GRAPH_REAP_INTERVAL`,
default 30 s, in `limits.py`) lists live children via `find_by_argv`. For each
child whose execution has no row, or a terminal row, and that this supervisor is
not watching: log a WARNING naming pid and execution, `request_stop`, then
escalate to kill after the grace period. Never touch a child whose row is
`running`/`queued`. The in-process gateway returns an empty list.
**Tests** with `RecordingGateway`: terminal row + live child -> stop then kill;
running row -> untouched; watched child -> untouched; no row -> reaped.

### R3-10 Fault-injection invariant tests for the supervisor
After R3-01 and R3-02, add `backend/tests/test_graph_supervisor_faults.py`.
Build a scripted run on the doubles (spawn, three node records, one monitor
record, a clean outcome, child exit). For each collaborator call the supervisor
makes (`executions.get`, `writer.commit`, `tail.poll`, `gateway.is_alive`,
`events.publish`) inject a single exception at call number *n* for *n* = 1..K
(K = the number of calls in the clean run), re-run, and assert:
1. the row is never terminal while the fake child is alive;
2. after the child exits and the file is complete, the row is terminal and
   agrees with the last `outcome` record;
3. results are a duplicate-free subset of the graph's nodes;
4. no watcher thread is still running ten poll intervals after the row is
   terminal;
5. no exception escapes a thread (use `threading.excepthook`).
Run the same for the adopt path and the recover path (R3-02). Any failure the
harness finds is a real bug: fix it in the supervisor, not in the test.

---------------------------------------------------------------------------

## 2. Optional, only if time remains

- **R3-11 Soak script** `scripts/soak_graph.py` (not in the default gate; add to
  `full_gate.sh --slow`): real uvicorn + real child on cheap nodes (find one
  with `grep -rn "class .*Node" nodes | head`), a large graph; then (1) `kill
  -9` the server mid-run and restart: the run must be adopted and finish;
  (2) let a child finish while the server is down: row `finished` (R3-02);
  (3) Stop at startup, mid-run and after the last node; (4) delete executions
  and check scratch (R3-07). Print one line per scenario, exit non-zero on any
  failure.
- **R3-12 Heartbeat**: the child appends a `heartbeat` record every 30 s; the
  watcher exposes "last heard from" on the execution; the UI shows a warning
  after 5 minutes of silence. Requires adding the record kind to
  `EventKind`, the reader (unknown kinds are skipped today), the event schema
  and one frontend line.
- **R3-13 Split `graph_supervisor.py`** (530 lines) into watcher, finaliser and
  adopter modules behind the same public class, with no behaviour change.

---------------------------------------------------------------------------

## 3. Final report (required)

Write the report as the last chat message and as `docs/status/fixes-r3.md`: one
line per WP (`done | partial | skipped`), commit hash, which test proves it, and
what you were unsure about. List the existing tests you changed and why, the
traps from 0.2 you were tempted to touch, and anything you found but did not
fix. Put every change not exercised on real hardware into
`docs/known-issues/pending-testing.md`. Do not mark a WP done unless its tests
exist, fail without the fix, and pass with it.
