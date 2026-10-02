# 0003 — Trainers that outlive a server restart are adopted

## Context

A training run is a child process that writes its own progress file. The
server supervising it is not: it can be stopped, restarted, or crash.

If the server died and came back, the options are to leave the run
`running` because nobody knows what happened, or to look at the world and
decide.

The trainer is the only source of truth about itself. It writes
`log.progress.jsonl` and, when it finishes, writes a line saying so —
including on the exit path, which is why a run that completed while the
server was down can be recorded as *completed* rather than guessed at.

## Decision

On startup, `reconcile_runs` sweeps rows that are not terminal and binds a
supervisor to each trainer still alive, keyed on the pid recorded in the
row. `RunSupervisor` marks these `_adopted` so it knows the process is not
one it spawned.

An adopted trainer's exit code is unreadable — it is not a child of this
process. Its progress file's `finished`/`error` line is therefore the exit
evidence, and the supervisor reads it.

Signalling an adopted pid is gated on the same identity check as any other:
`owns()` requires the pid's cmdline to still mention this project's entry
point before `stop()` or `kill()` will touch it.

## Consequences

* **The supervisor watches a pid it did not spawn.** It cannot `waitpid`,
  so liveness comes from `os.kill(pid, 0)` *plus* an identity re-check: a
  number alone would report a recycled pid as a live trainer forever, and
  `stop()` would then be refused as "not our trainer" — a run stuck
  `running` that only a restart clears.
* **A trainer that dies without writing a terminal line** is finalised from
  what the file does contain. Absent evidence means failed, which is the
  safe direction: a completed-looking run that actually died is worse than
  the reverse.
* Adoption is best-effort by construction. A row whose pid is long gone
  and whose file says nothing is failed, and the reason is recorded on the
  row.

## What pins it

* `backend/tests/test_start_stop.py::test_reconcile_reads_the_progress_file_of_a_dead_run`
  — a run that finished while the server was down records *completed*,
  not failed.
* `backend/tests/test_start_stop.py` — reconcile is idempotent for an
  adopted run, and the re-attachment note is recorded in the run's log.
* `backend/tests/test_training_adapter.py::test_signal_safety` — owns() gates
  stop/kill for a stranger pid; and a live process that is *not* our
  trainer is reported as gone rather than watched.