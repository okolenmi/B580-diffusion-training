# TASK: round-4 fixes (three findings) in `backend/`

Repository `okolenmi/B580-diffusion-training`, start from `main` at `82738e2` or
later. Same working rules as the round-3 task: one commit per WP
(`R4-NN: <summary>`, finding id and proving tests in the body), minimal diffs,
never weaken a test, run `python3 backend/tests/run_all.py`,
`python3 run_tests.py` and `python3 scripts/check_quality.py` before each
commit, fix the pattern not just the instance, every `except Exception` logs.
Do not touch the intentional designs listed in the round-3 task section 0.2
(event-file protocol, argv-based child identity, `security.py`, etc.).
Python 3.14+ is required (the project enforces it); tests that spawn a real
child cannot run on older Pythons.

Each finding below was reproduced. The repro scripts are in
`repro-scripts-round4.tar.gz`; use them as test seeds.

---------------------------------------------------------------------------

### R4-01 `GET /api/v1/installer/readiness` fans out into torch processes
**Where.** `backend/application/use_cases/check_requirements.py`,
`backend/application/ports/environment.py` (`TorchDeviceProbe.report`),
`backend/presentation/api/installer.py`.
**Problem (reproduced).** Every GET spawns its own subprocess that imports torch
and calls `is_available()` and `get_device_properties(current_device())`. There
is no cache and no concurrency limit: 8 concurrent requests produced 8
simultaneous torch-importing processes (15.3 s wall on a 1-core box). A plain
GET is not covered by the Origin check, so any web page the user has open can
trigger it with an `<img src=".../readiness">`. On the B580 each probe also
initialises the accelerator runtime, which can take VRAM from a run that is
using the same 12 GB card (not measured here; treat as a risk).
**Do.**
1. Single-flight: one probe at a time; concurrent callers wait for it and share
   the result.
2. Cache the last result for `READINESS_CACHE_SECONDS` (new constant in
   `limits.py`, default 30). A query parameter `refresh=true` bypasses the cache
   but is still single-flight and still rate-limited to one real probe per 5 s.
3. While a graph execution is active (`executions.find_active()` is not None)
   never start a new device probe: return the cached result if there is one,
   otherwise the report with `device_checked=false` and a reason that says a run
   is active and the accelerator was deliberately not touched.
4. The package-presence part of the report (no device) may stay uncached; it must
   not import torch in the server process.
**Tests.** With a counting fake probe: 8 concurrent `check.execute()` calls cause
exactly 1 probe; a second call within the TTL causes 0; `refresh=true` causes 1
more but not more than one per 5 s (use the injected clock); with an active
execution no probe runs, with and without a cached result. Add a real-subprocess
variant of `/tmp/repro/r17_readiness_fanout.py` that asserts a peak of 1.

### R4-02 The startup sweep deletes the log of a crashed run
**Where.** `backend/application/use_cases/sweep_execution_scratch.py`,
`backend/application/graph_supervisor.py` (`_finish`).
**Problem (reproduced).** The sweep removes `execution_N.events.jsonl`,
`.graph.json` and `.log` for every terminal run. For a child that died without an
`outcome` record the row only says "exited without reporting an outcome
(crashed, or a device fault killed it)"; the real error (a traceback, a device
loss) exists only in `execution_N.log`, which the next server start deletes.
Repro: a failed row plus a log containing a traceback -> the log is gone after
the sweep.
**Do.**
1. When `_finish` records a failure with no outcome, append the last 4 KiB of
   the child's log to the row's `error` text (bounded, decoded with
   `errors="replace"`, prefixed by a separator line), so the evidence survives
   in the database.
2. The sweep keeps the `.log` of runs that ended `error` or `cancelled`, subject
   to `GRAPH_FAILED_LOG_KEEP` (`limits.py`, default 20 newest); it still removes
   the event and graph files, and removes logs of `finished` runs. Explicit
   `DeleteGraphExecutions` removes everything for the deleted ids.
3. The sweep's docstring must say what it keeps and why.
**Tests.** failed run: log survives a sweep, events and graph files do not;
finished run: all three removed; 25 failed runs: only the newest 20 logs remain;
a crash with no outcome puts the log tail into the row's error; a log with
invalid UTF-8 does not raise; explicit delete removes the kept logs too.

### R4-03 Make hermeticity automatic
**Where.** `backend/tests/support.py` (`use_temporary_comfy_dir`),
`backend/tests/run_all.py`.
**Problem (reproduced).** The temporary-ComfyUI fixture is opt-in. Only
`test_graph_adoption.py` and `test_graph_task_gateway.py` call it. On a machine
with no ComfyUI directory `test_config.py`, `test_installer.py` and
`test_settings.py` fail with "Cannot find ComfyUI directory" (all three pass with
`COMFY_DIR` set). Every new test that forgets the call repeats the bug.
**Do.**
1. In `support.py`, call `use_temporary_comfy_dir()` at import time unless
   `BACKEND_TESTS_REAL_COMFY=1` is set, so no test can forget it. Keep the
   function public for tests that need the returned path.
2. `run_all.py` already scrubs the environment per file; add a final self-check
   that runs two representative files with no `COMFY_DIR` anywhere and fails the
   run if either errors.
3. Add `scripts/check_bare_checkout.sh` that does `git archive HEAD | tar -x -C
   <tmp>` and runs `python3 backend/tests/run_all.py` there with a scrubbed
   environment; mention it in the gate docs. It is a slow check, not part of the
   default gate.
**Done when** `env -u COMFY_DIR python3 backend/tests/test_config.py`,
`test_installer.py` and `test_settings.py` pass directly from a fresh clone.

---------------------------------------------------------------------------

## Still open from round 3 (improvements, not defects)
These were in the round-3 task as R3-09..R3-13 and are not in the tree or the
trackers: the **orphan-child reaper**, the **fault-injection invariant tests**
for the supervisor, the **soak script**, the **child heartbeat**, and splitting
`graph_supervisor.py`. Do them after R4-01..03, in that order of value:
fault-injection tests first (they would have found the watcher bugs
mechanically), then the reaper, then the soak script. Specs are in the round-3
task file.

## Final report
As before: one line per WP (`done | partial | skipped`), commit, proving test,
doubts, tests you changed and why. Put everything not run on real hardware in
`docs/known-issues/pending-testing.md`.
