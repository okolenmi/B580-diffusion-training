*[← backend design docs](README.md)*

# Backend/frontend: third review (2026-10-03)

Reviewed at `13566c2` (73 commits after the round-2 review: WP-01..22,
the `core/` archive, the structure audit, event sequence numbers and replay,
and graph execution moved into a child process).

**Evidence.** `[R]` reproduced by running project code (CPU only; Python 3.12.3
in my sandbox, no XPU, no browser). `[C]` read from the source, not executed.
**Severity.** High = data loss or an app state that blocks work. Medium = wrong
behaviour in a plausible situation. Low = latent or hygiene.

Suites at this commit in my sandbox: backend suite 21/25 files pass on a bare
checkout and 24/25 with a ComfyUI directory set (the remaining failure is
Python-version related, N3-06). The quality gate holds its baseline (ruff 14,
mypy 22). I did not re-run `run_tests.py` (nodes/manager) or the browser smoke.

## 1. Round-2 findings: what I re-verified

| Round-2 | Now |
|---|---|
| N-02 upload blocks loop, 2x RAM | **Fixed [R]**: worst concurrent request 11 ms (was 739), RSS flat during a 600 MB upload |
| N-03 finished run recorded failed | Fixed for the TOML route, **but that route was removed**; the same defect exists on the new graph route (N3-02) |
| N-04 coalescing dropped node events | **Fixed [C]**: delivery classes in `application/event_delivery.py`, delta never coalesced |
| N-05 file endpoint served any file | **Fixed [R]**: db, shard, svg all 404 |
| N-06 run page repeated mistakes | **Moot [C]**: the run page was removed; `lib/events.js` now owns resync |
| N-07, N-09 | **Moot**: the `core.cli` run supervisor was removed |
| N-11 no linter/type checker | **Fixed [R]**: `scripts/check_quality.py`, baseline holds |
| N-12 orphan shards | **Fixed [R]**: failed discard then swept |
| Round-1 F-05, F-06, F-10, F-16 | **Still fixed [R]** (re-ran; write chain blocked, discard order, foreign dir 409, atomic migrations) |
| N-01 hermetic tests | **Regressed** (N3-05) |
| N-08, N-14 | Committed with tests per the log; not re-verified |

Round-2's first and most useful lesson ("fix the pattern, not the instance")
was applied well in most places. The new problems below are mostly the same
two patterns on the **new** route: *recovery paths that ignore evidence the
normal path uses*, and *tests that are green without proving what they claim*.

## 2. New findings

| ID | Sev | Ev | Problem |
|---|---|---|---|
| N3-01 | Med-High | R | A transient error inside the graph watcher marks the row failed while the child keeps running on the GPU, unwatched |
| N3-02 | Med-High | R | A graph run that finished while the server was down is recorded failed "server restarted", its results discarded |
| N3-03 | Medium | R | `finish()` reports "ALL CHECKS PASSED" with zero checks run; the cleanup removed instances, not the mechanism |
| N3-04 | Low-Med | R | `Last-Event-ID` from a previous server process can be accepted as valid (process-local sequence, no epoch) |
| N3-05 | Low-Med | R | Tests are not hermetic: three new files need a ComfyUI dir; one test writes into the resolved model directory |
| N3-06 | Low | R | An undeclared Python 3.14 floor: production schema code and a test fail on 3.12/3.13 |
| N3-07 | Low-Med | C | Per-execution scratch files (graph, event history, log) are never deleted; adoption reads a whole file in one call |
| N3-08 | Low | C | The child installs its SIGINT handler late and ignores SIGTERM |
| N3-09 | Info | C | Every run pays full node discovery and torch import in a fresh child |

### N3-01 Watcher error leaves a live child unwatched and the row failed [R]
`GraphExecutionSupervisor._watch` re-reads the row every poll
(`self._executions.get`). That call is not guarded individually. One transient
error (for example SQLite "database is locked") lands in the outer
`except Exception`, which logs and falls into `finally: _finish(...,
saw_outcome=False)`. `_finish` marks the row failed ("exited without reporting
an outcome (crashed, or a device fault killed it)") and `_release` forgets the
pid. Nothing checks whether the child is alive, and nothing signals it.
Reproduced with the project's own doubles (`/tmp/repro/r14`): row status
`error`, child alive, stop signals sent `[]`, kill signals sent `[]`. The
single-active check then permits a second run, i.e. two trainers on one 12 GB
GPU, and the first child's real outcome is never read. The node-recording path
already has its own guard ("recording result for node ... failed"); the poll
loop does not.
**Fix.** Guard each loop iteration and retry with backoff; never finalise a row
terminal while `is_alive(pid)` unless stop and kill have been delivered and the
process confirmed gone; make the error message true ("supervision failed;
child stopped" vs "process died").

### N3-02 Finished while the server was down is recorded failed [R]
`adopt()` returns `None` at once when `find_running` finds no live child, and
`ReconcileGraphExecutions` then fails the row as "server restarted while the
execution was in flight". If the child exited cleanly while the server was down
(overnight crash, reboot, update), its event file holds the remaining node
results and a clean `outcome` record, and none of it is read. Reproduced
(`r16`): event file with nodes a, b, c and `outcome(error=None)` gave
`status=error results=0/3`. This is round-2 N-03 on the new route, and it
breaks round-2 rule 5 ("a recovery path must use every piece of evidence the
normal path uses"): `_finish` already knows how to decide from an outcome
record.
**Fix.** One shared "finalise from the event file" routine used by the watcher,
adopt and reconcile: persist node records the row lacks (dedupe by node id),
then decide from the last `outcome`; only "no outcome" means "died".

### N3-03 The harness can pass with zero checks [R]
`backend/tests/support.py::finish()` prints "ALL CHECKS PASSED" and exits 0
even if no `check()` ever ran (demonstrated). The latest commit found two test
files that ran no tests ("test_pages.py ran none of its tests... green on zero
checks"). They were fixed, and an AST scan finds no defined-but-uncalled test
functions today (453 across 75 files), but nothing prevents it recurring.
**Fix.** Count executed checks and fail on zero; add the AST scan to the gate;
fail a file that defines `test_*` functions it never references.

### N3-04 Stale `Last-Event-ID` across a restart [R]
Sequence numbers are process-local. `replay_since` only recognises a foreign id
when `last_seq > newest`. After a restart, once the new process has published
more events than the client's old id, the old id looks valid: demonstrated
(`r15`), `replay_since(40)` on a new process with 60 events returns
`complete=True`, events 41..60. The client now resyncs only when the server
says `resync_required`, so it would skip the refetch and show stale state.
Window is narrow (late reconnect after a restart that has already published
many events), hence Low-Medium.
**Fix.** Put a per-process `epoch` in the event id (`"{epoch}:{seq}"`) and in
`stream_opened`; a different epoch means `complete=False`.

### N3-05 Hermeticity regressed, and a test pollutes the model directory [R]
On a bare checkout (no `COMFY_DIR`) `test_graph_task_gateway` (4 checks),
`test_graph_adoption` and `test_process_identity` fail because the real gateway
needs a ComfyUI directory; with `COMFY_DIR` set all three pass, three runs in a
row. Round-2 WP-10 fixed exactly this for another file. Separately,
`test_api_datasets.py` (line ~418) writes `m.safetensors` into the *resolved*
checkpoints directory: with `COMFY_DIR` set that was
`<ComfyUI>/models/checkpoints/m.safetensors`. On your machine that is your real
model folder, every run, and a real checkpoint with that name would be
overwritten by 2 bytes.
**Fix.** A shared test fixture that builds a temporary ComfyUI-shaped tree and
points every path at it; assert after the run that nothing outside temp
directories changed.

### N3-06 Undeclared Python 3.14 floor [R]
`pyproject.toml` targets py314 for ruff and mypy, but there is no
`requires-python`, no startup check and `docs/setup.md` names no version.
`backend/presentation/event_schema.py` decides "is this `X | None`" with
`get_origin(t) is Union`, whose comment says "PEP 604 at runtime is
typing.Union"; that is true only from 3.14. On 3.12 `schema_for("run_progressed")`
raises `TypeError`, and `test_event_contract.py` fails deterministically
(`KeyError: str | None`). Production code does not call it at runtime (only
tests and tooling), so impact is low, but `config_schema.py` uses the same
identity check.
**Fix.** Declare the floor and check it at startup, or add one `is_union()`
helper that accepts both `typing.Union` and `types.UnionType`.

### N3-07 Scratch files grow without bound [C]
Each execution leaves `execution_N.graph.json`, `.events.jsonl` (the full
monitor history, one record per report) and `.log` in `graph_executions/`.
`DeleteGraphExecutions` removes rows, not files, and there is no retention.
Long training runs make large event files (order of tens of MB per 12 h run;
an estimate, not measured). `ExecutionEventTail.poll` does one
`handle.read()` from the offset to EOF, so adoption replay loads the whole file
and parses it in one go.
**Fix.** Delete scratch with the execution, add retention and a startup sweep,
read in bounded chunks.

### N3-08 Child signal handling [C]
`graph_task_worker.main` installs the SIGINT handler after parsing arguments
and importing `nodes.xpu_env`. A Stop in that first moment hits Python's
default handler (traceback, no `outcome`), and the row reads "crashed".
SIGTERM is not handled at all. Window is small, effect is a wrong label.
**Fix.** Install SIGINT and SIGTERM handlers as the first statements, before any
non-stdlib import.

### N3-09 Per-run startup cost [C, design trade-off]
The child discovers the node registry and imports torch for every execution
("a few seconds", documented). Fine for training; noticeable for small
diagnostic graphs. A warm worker is an option if it becomes annoying.

## 3. What is good (keep)
- WP-22 is carefully designed: a file instead of a pipe so a server restart
  cannot kill a run; an `outcome` record that separates "failed" from "died";
  identity by argv instead of a stored pid; refusal to signal strangers; a
  final drain after liveness says no; truncating a stale event file at launch.
- The authors ran a 3,000-node graph through a real server and fixed two bugs
  the tests had passed. That habit is the most valuable one here.
- Every round-1 and round-2 fix I re-ran still holds.
- Mutation testing, branch coverage and a regression-only quality gate exist.
- Decisions are written down (`docs/decisions/`) and linked to tests.

## 4. Improvements, including large ones

**I1. Invariant-based fault injection for supervisors (M).** N3-01 and N3-02
came from failures nobody injected. Build a small harness that, for each
collaborator call a supervisor makes (`executions.get`, `writer.commit`,
`tail.poll`, `gateway.is_alive`, `events.publish`), raises once at the *n*-th
call of a scripted run, for every *n*, and then asserts invariants: (1) a row is
never terminal while its child is alive; (2) once the child has exited and the
file is complete, the row is terminal and agrees with the last `outcome`;
(3) results are a duplicate-free subset of the graph's nodes; (4) no watcher
thread outlives its row; (5) no exception escapes a thread. This finds a whole
class of defects mechanically.

**I2. One evidence-based finaliser, tested as a table (M).** Replace the three
places that decide how a run ended (watcher `_finish`, `adopt`, reconcile) with
one function whose inputs are (row, process state, event-file contents). Test
it table-driven over process state {alive, dead, unknown} x file {empty, nodes
only, nodes + clean outcome, nodes + error outcome, torn tail, missing}.

**I3. An orphan-child reaper and health signal (S-M).** Periodically list live
children by argv marker and compare with rows. A child with a terminal or
missing row is a leaked GPU holder: log loudly, signal it (INT, then KILL).
Expose "children / watchers / unfinished rows" counts in the health response.
This is a defence in depth that protects against N3-01 and any future cause.

**I4. Make the test harness incapable of a vacuous pass (S).** Count checks and
fail on zero (N3-03); run the suite once per gate in a scrubbed environment
(`env -i`, no `COMFY_DIR`) and snapshot the repo root and model directories
before and after (N3-05); add a minimum-checks budget per file so a test that
silently stops running is noticed.

**I5. An automated soak test (M).** Script what the authors did by hand: real
uvicorn plus a real child on cheap nodes, a large graph, then `kill -9` the
server mid-run and restart it; let a child finish while the server is down;
send Stop at startup, mid-node and after the last node; delete executions and
check scratch. Run in the slow gate (not on every commit). This covers N3-02,
N3-07, N3-08 and the adoption design together.

**I6. Event-stream epoch and an end-to-end reconnect test (S).** N3-04, plus a
test that drives a real client reconnect against a restarted app.

**I7. Python version policy (S).** Declare `requires-python`, check at startup
with a clear message, and put union detection in one helper.

**I8. Child lifecycle hardening (M).** Handlers first (N3-08); write an
`outcome` on `KeyboardInterrupt`/`SystemExit`; add a periodic *heartbeat*
record so the watcher can tell "alive and silent" from "alive and working" and
the UI can warn after N minutes of silence (a hung XPU kernel currently looks
identical to a slow node until the 15 s stop escalation).

**I9. Scratch lifecycle (S-M).** N3-07: delete with the execution, retention,
sweep, bounded reads, and consider downsampling stored monitor history (the
dashboard does not need every frame of a 12 h run).

**I10. Keep shrinking the large units (M).** Largest remaining files:
`datasets.js` 1,383 lines, `schemas.py` 898, `dataset_library.py` 628,
`graph_supervisor.py` 530 (it now mixes launching, adoption, tailing,
persisting node results, finalising and stop escalation). Splitting the
supervisor into watcher, finaliser and adopter makes I1/I2 simpler and each
piece testable alone.

**I11. Optional warm worker (L).** Only if N3-09 bothers you: a long-lived,
isolated worker that keeps the discovered registry and torch loaded, with the
same event-file protocol. Weigh against the isolation you just gained: a worker
that survives many runs also keeps device memory fragmentation.

## 5. Not reviewed
The editor canvas and datasets UI in a browser, `visual_smoke` (185 checks),
the training nodes, the first-run installer plan (`docs/design/11-first-run-*`),
the archived `core/` and `server/`, and all behaviour on XPU hardware.

## Appendix: reproductions
`r14_watcher_crash_live_child.py` (N3-01), `r16_graph_finished_while_down.py`
(N3-02), `r15_stale_event_id.py` (N3-04). N3-03 is the one-liner
`python3 -c "from backend.tests.support import finish; finish()"`. N3-05 is
`env -u COMFY_DIR python3 backend/tests/run_all.py` and, with `COMFY_DIR` set,
`find $COMFY_DIR -name m.safetensors` after `test_api_datasets.py`. N3-06 is
`python3 -c "from backend.presentation.event_schema import schema_for;
schema_for('run_progressed')"` on Python 3.12/3.13.
