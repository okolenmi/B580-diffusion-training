# Process isolation for graph execution

*[← design index](README.md)* · see also
[`11-core-removal.md`](11-core-removal.md),
[`12-training-modes.md`](12-training-modes.md)

Graph execution used to run in a thread inside the API server process.
It now runs in a child, behind a port, supervised by tailing a file.

The problem that prompted it: a device fault, an OOM kill or a driver
reset took the server **and** the run down together, and a long node
build competed with everything else in the process. With the run out
here, the worst outcome is that *that* process dies and the row says so.

## What moves and what does not

| | Before | After |
|---|---|---|
| `ReflectedGraphRuntime.execute` | a `GraphExecutionSupervisor` thread | `backend/infrastructure/graph_task_worker.py`, a child |
| `GraphRuntime.validate` | in-process | **in-process, unchanged** — it is pure, cheap, and the editor needs it on every keystroke |
| live monitor reports | `SharedMonitorBus`, in-process | appended to the event file, republished by the watcher onto the server's own bus |
| node results | `on_node_done` callback | `node` records in the event file |
| how it ended | the return value | an `outcome` record |

Both gateways call the *same* producer,
`graph_task_worker.run_execution`. The alternative was two
implementations of "run a graph and report it", and they would not have
stayed identical.

## The event file is the only channel

A pipe would couple the two lifecycles in exactly the way isolation is
meant to break: close the parent and the child gets `EPIPE` on its next
write, so a server restart would kill the run it was supposed to
survive. A file has no such coupling.

Four record kinds, one JSON object per line:

| kind | meaning |
|---|---|
| `node` | a node finished; the payload is a described `NodeResult` |
| `monitor` | a live monitor report: `monitor_id`, `data` |
| `memory` | one memory frame: `reserved_mb`, `allocated_mb`, `peak_mb`, `budget_mb` (null = no budget stated) |
| `outcome` | the run ended: `error`, `results_count` |

`outcome` is the one that earns its keep. Without it, "the graph
failed" and "the process died" are the same absence, and a run killed by
a device fault would be reported as a **success** because nobody wrote
down a problem.

`memory` is how the server learns what a run actually reached: the
watcher files `peak_mb` into the peak store under the fingerprint key
admission stored on the row (MEM-04 #2), one statement per rise, and
admission reads that store back as the next run's observed demand.

Records are written per record with a fresh append-mode handle rather
than through a buffer held for the whole run. That costs one
open/write/close per report and buys two things a held-open handle does
not: no user-space buffer, so a `SIGKILL` cannot lose an
already-written record; and each append is one `write(2)` under
`O_APPEND`, which the kernel does as a single step, so records cannot
interleave with another writer's.

The reader still consumes whole lines only, and still tolerates a torn
tail. "Far more often than not" is not "always" — the deleted
`JsonlProgressSource` had the same rule for the same reason (docs 07
F-07).

### Why the monitor reports go through the file

`SharedMonitorBus` keeps its history in a per-process `deque` and hands
each subscriber an `asyncio.Queue`. Neither is reachable from a child.
Routing reports through the file keeps the server's own bus as the
single source of what a dashboard sees — including "opened the page
mid-run and saw the whole curve", because the file *is* the history.

## Stop

Cooperative first. `request_stop` sends `SIGINT` to the child's process
group; the child's handler sets the cancel event and the runtime notices
at the next step boundary. A stop that the child ignores is escalated to
`SIGKILL` after a grace period (15 s), by the supervisor, so exactly one
place decides a run is not stopping.

Sending `SIGINT` rather than killing outright is what lets a stopping
run flush its event file and run `release_memory` — the same reason the
retired training route used the same escalation (docs 07 F-12).

Both signals are refused for a pid whose cmdline does not name our
worker module. `kill(pid, 0)` answers "does *a* process hold this
number", and after a reboot it will happily answer yes about something
else. The test for this is the server itself: with the guard removed,
`test_graph_task_gateway.py` SIGINTs its own test runner.

## Adoption across a restart

A child in its own session outlives the server that started it. So a
restart is no longer the end of a run that is still going, and
`ReconcileGraphExecutions` offers each unfinished row to the launcher
before failing it.

**The pid is discovered, not stored.** A pid kept in a row is a claim
about the past; read it after a reboot and it names whatever process the
number was recycled to. Instead `find_by_argv` reads live `/proc` and
matches the child's *own* argv: the worker module as an exact token,
plus `--execution <id>` as an exact `flag value` pair. Exactness is the
point — a substring test would let `--execution 1` match the child
running `--execution 15`, and an adopted run gets its history replayed
into a row that is not its own.

**The replay re-reads the file from the top, and skips node records
while it catches up.** The monitor history is the reason: it is the only
trace of the run that lives outside the row, and it is what makes a
dashboard opened after the restart show the curve instead of an empty
chart. Node records are skipped for the opposite reason — the row
already holds their results, so replaying them would append each twice,
and the domain refuses to load a row with more results than nodes. It
would also re-announce progress events the event store is about to
replay to reconnecting clients on its own (docs 07 F-08).

The replay window closes on `ExecutionEventTail.caught_up` — "every
byte written so far has been consumed as whole records" — and not on "the
last poll returned nothing", which is also true while the child is
mid-line. Closing it early would record node results the row already
has.

**Not adopted is not the same as failed.** A run that outlived the server
has two distinguishable endings, and only one of them is a crash:

* it **finished**, and said so — an outcome record on disk, plus every
  node result it produced. The row is settled from that record.
* it was **killed** — no outcome record. Then absence is all the evidence
  there is, and the row fails with the reason that describes it.

Asking what the run said, before drawing any conclusion from the absence of
a process, is not a refinement. Without it, a run that completed while
nobody was watching was reported as a crash and its results thrown away.
Measured, on a 4000-node run with the server `SIGKILL`ed mid-flight: the
run completed all 4000 nodes and wrote
`{"kind": "outcome", "error": null, "results_count": 4000}`, and the
startup sweep reported the row `error` with zero results. Recovering it is
now verified end to end — verdict `finished`, all 4000 results back, in the
order the run recorded them, outputs intact.

This is round-2 finding N-03 ("a run that finished while the server was
down is marked failed"), recorded as moot when the run route was removed.
It was moot *for that route*; isolation carried the same assumption into
newer code. The general form of it: **a row's terminal state is not
implied by the absence of a process**, and where the work left a record,
the record is the better witness.

One limit worth stating: recovery reads the event file, so it costs a
replay of that run's records — the same order of work adoption already
does, and bounded by the same file. A run whose file was removed while the
server was down is unrecoverable, and correctly reported as debris.

## Choosing the mode

`BACKEND_GRAPH_EXECUTION=child|inprocess`, default **`child`**.

`child` is the default because isolation is the entire reason for putting
the run in another process, and the cost is measured rather than assumed
(below): ~1.95 s of startup per run, against runs that take minutes.

The in-process gateway is not "the old way round a new interface": it
calls the same `run_execution` with a `threading.Event` where the child
has a `SIGINT` handler. It exists for two reasons.

*Rollout.* Until the child path is the default, the supervisor, the
watcher, the event format and the reconcile all have to be exercisable on
every run. If only the child path were exercised, the code that replaced
the old path would be untested until the moment it became the only path.
Being able to select either makes the flip one line rather than a leap.

*Rollback.* If the child path breaks something on hardware nobody tested,
the default goes back without a revert.

An unrecognised value falls back to the default rather than raising: this
is read during start-up, and a typo in a convenience variable should not
stop a server that is otherwise fine from starting.

## Limits, measured

*Child startup is ~1.95 s*, nearly all of it importing torch and walking
`nodes/` (measured on this machine: 1.98 s and 1.95 s on two runs of a
two-node graph). This is per run, not per node, so it is noise against a
training step and visible only for short graphs. It is why the child
tests run concurrently — five serial startups took the backend suite from
6.3 s to 15 s.

**Adoption cannot resurrect a run that never wrote anything.** The event
file is what the watcher reads, so a run whose child produced no output
is not observable. Its row fails, and its child is left alone rather than
killed: the supervisor does not own a process it did not start, and a
user can still find it in `ps`.

**Node results written between the last successful CAS and the restart
are lost.** They are in the event file, but the replay deliberately
skips them to avoid double-counting. Bounded by one record per node.

****`inprocess` is not adoptable, by construction.** A thread cannot
outlive its process, and `InProcessGraphTaskGateway.find_running`
returns `None` — the honest answer, not a limitation to apologise for. So
setting `BACKEND_GRAPH_EXECUTION=inprocess` gives up adoption as well as
isolation; that is the price of the rollback, and worth knowing before
reaching for it.

**One active execution, unchanged.** Nothing here altered that; see
`start_graph_execution.py` for why.

## Verified against a running server

Not only in tests. On this machine, against a live server on port 8791
with `BACKEND_GRAPH_EXECUTION=child`:

| what was done | what happened |
|---|---|
| a 3-node graph through `POST /api/v1/graphs/run` | `finished` in 2.01 s, all three node results present (start to finish: 2.01 s, which is the startup) |
| `POST .../stop` on a 20 000-node run | `stopped`, "stop requested"; the child ran a few more nodes before noticing, which is what cooperative means |
| `SIGKILL` the child mid-run | row went to `error`: *"execution process exited without reporting an outcome (crashed, or a device fault killed it) -- see the execution log"*. **The server kept serving**, and a new run started immediately |
| `SIGKILL` the **server** mid-run, restart it | `adopted graph execution 1, still running as pid 538971`, then `adopted 1 still-running graph execution(s)`; results kept climbing under the new server, and a later `stop` ended it cleanly |

That third row is the property the whole change exists for: before it, a
`SIGKILL` of the running process was the server's own death.

## Two failure modes worth knowing about

Both of these were found by running a 3000-node graph through a live
server rather than by the suite, and both passed every test that existed
at the time. They are recorded because the shapes recur — the first is a
property of any watcher that polls a file a writer is still appending to,
and the second of any path named after a database id.

**A successful run reported as a hardware fault.** The watcher polls, then
checks liveness, then loops. The child writes its outcome record and
*then* exits — so by the time liveness says no, everything it ever said
is on disk. But a watcher still busy persisting the batch it had just
read would break without a final poll, and never see that outcome. One
CAS per node result is slow enough on a long graph that a fast child
finishes underneath it: a 3000-node graph ended with a clean
`"outcome", "error": null, "results_count": 3000` in its event file, and
the row said *"execution process exited without reporting an outcome
(crashed, or a device fault killed it)"*. The watcher now drains once
more when the child is gone.

**A new run inherited the previous one's results.** Records are appended,
which is what makes a `SIGKILL` unable to tear the last one — and which
means a new run at a path that still holds an old run's records reads
them as its own. Row ids are not reused — except that they are: delete
`backend.db` and ids restart at 1, so the next execution 1 lands on the
previous execution 1's file. The watcher then read 6000 results for a
3000-node graph, which the domain refuses to load — and because every
subsequent read *and* write raised, the row stayed `running` forever,
answering the executions list with a 500. `launch` truncates the event
file before spawning. Adoption is the one path that must keep the old
records, and it does not go through `launch`.

The second one is the more interesting failure: the domain's "no more
results than nodes" rule is what caught it, by refusing to load a row the
persistence layer had happily written.

## Where the pieces are

| file | role |
|---|---|
| `application/ports/graph_task_stream.py` | the record protocol (`EventKind`, `ExecutionEvent`) |
| `application/ports/graph_task_gateway.py` | `spawn` / `request_stop` / `kill` / `is_alive` / `find_running` |
| `application/graph_supervisor.py` | claims the row, tails, finalises, adopts |
| `infrastructure/graph_event_stream.py` | the file: `ExecutionEventWriter`, `ExecutionEventTail` |
| `infrastructure/graph_task_worker.py` | the child, and the shared producer `run_execution` |
| `infrastructure/graph_task_gateway.py` | the two gateways |
| `infrastructure/process_identity.py` | the pid-reuse guard and pid discovery |

The protocol lives in `application/ports/` and the file does not, because
the supervisor has to know what a `node` record means without importing a
file reader — `application` may not import `infrastructure`.
