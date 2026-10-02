# Round-2 review: outcomes

An external review of the backend/frontend redesign produced 12 findings
(`N-01`..`N-14`) and 22 work packages (`WP-01`..`WP-22`). This is the record
of what was decided for each one and why.

The review's own material is kept at
[`archive/review-r2/`](../../archive/review-r2/), where the finding
IDs are traceable. Three of its five reproductions have since been
invalidated by the removal of the supervised-subprocess route — see that
directory's README — so the table below cites **tests that still exist**
rather than the reproduction scripts.

The review's rules were sound. Its picture of the current state often was
not, because it was reviewing work that had not been finished: several
items were already fixed when it was written, and each was re-verified
before being worked on.

## Findings and quality items

| WP | Finding | Outcome | What holds it now |
| --- | --- | --- | --- |
| WP-01 | N-04 SSE buffer drops per-node graph events | fixed | `backend/tests/test_json_safe.py` — the client buffer and the per-delivery-class assignment |
| WP-02 | N-05 dataset file endpoint serves any file | fixed | `backend/tests/test_api_datasets.py` — the allowlist and header cases |
| WP-03 | N-02 upload buffers the body and doubles RAM; N-14 silent overwrite | fixed | `backend/tests/test_assets.py` — streaming peak, abort and overwrite cases |
| WP-04 | N-06 run-detail page repeats fixed mistakes | fixed; partly moot | `frontend/tests/events.test.mjs` owns the shared client module. The run page itself is gone with the route. |
| WP-05 | N-08 the monitor bus can raise into the caller's thread | fixed | `nodes/smoke_tests/smoke_test_monitor_bus.py` |
| WP-06 | N-12 orphan shard files after a failed discard | fixed | `backend/tests/test_dataset_library.py` |
| WP-07 | N-03 a run that finished while the server was down is marked failed | moot | The run domain was removed (`11-core-removal.md`). The equivalent rule for graph executions is `ReconcileGraphExecutions`. |
| WP-08 | N-07 liveness of an adopted pid is checked by number only | fixed, and outlived the route | `backend/tests/test_process_identity.py`. The guard is `infrastructure/process_identity.py`, now used by three gateways. |
| WP-09 | N-09 garbled log note | moot | The string was in the removed run domain. |
| WP-10 | N-01 a test needs a real ComfyUI directory | fixed | The case passes with the repository `.env` hidden, which is what used to make it fail. |
| WP-11 | Q1/N-11 lint and type checks behind one command | fixed | `scripts/full_gate.sh` and `scripts/quality_baseline.json`. Verified failing on an injected error. |
| WP-12 | Q2 `list` shadows a builtin; run id untyped | fixed | `require_id()` instead of `id` plus an ignore. No `valid-type` remains in the mypy output. |
| WP-13 | Q9 contract tests for fakes vs real adapters | fixed | `backend/tests/test_dataset_tasks.py` — the fake and the SQLite adapter pass one contract. |
| WP-14 | Q10 resource budgets and default pagination | fixed | `backend/application/limits.py`, exercised by `backend/tests/test_api_datasets.py`. |
| WP-15 | Q8 one frontend console and message box | fixed | `frontend/tests/` — the suite runs with `node --test`, no browser. |
| WP-16 | Q5 property-based tests at the untrusted boundaries | fixed | `backend/tests/test_property_boundaries.py`. It found a NUL byte turning a 422 into a 500. |
| WP-17 | Q11 decisions as records, plus a doc check | fixed | `docs/decisions/`, `scripts/check_docs.py`, `scripts/check_doc_links.py`. |
| WP-18 | Q12 measure test quality (report only) | reported | `docs/status/mutation-notes.md`. |
| WP-19 | Q7 split large units | declined | Every target is a test file or a schema module. See below. |
| WP-20 | Q3 one supervisor abstraction | done, mostly by removal | See below. |
| WP-21 | Q4 event contract | done | `backend/tests/test_event_contract.py`, `test_event_replay.py`, `backend/presentation/event_schema.py`. See below. |
| WP-22 | Q6 process isolation for graph execution | done | `docs/design/13-process-isolation.md`. |

## The three that needed a decision

### WP-19 declined

Every target was a test file or a schema module, and the ask was to split
them to hit a line count. Splitting a file to satisfy a metric rather than
to fix a comprehension problem adds indirection and produces a large
reviewable diff for no measured gain. Line count is not the property the
finding was reaching for.

### WP-20 mostly dissolved by removal

The finding named two pairs as duplicated: `RunSupervisor` against
`GraphExecutionSupervisor`, and a `ProcessGateway` base shared by the
training and dataset-task gateways. Both pairs lost a member when the
supervised-subprocess route went, so most of it was resolved by deletion
rather than refactoring, and inventing an abstraction from what remained
would have unified things that are not the same:

* `GraphExecutionSupervisor` owns a run's life (a thread, cancellation,
  per-node progress, a terminal compare-and-swap). `DatasetTaskSweeper`
  judges rows it finds already dead (a repair pass, no thread, no events).
  A sweeper and a supervisor are not two copies of one thing.
* `SubprocessDatasetTaskGateway` is now the only process gateway that
  predates WP-22, and the identity logic it would have shared already
  lived once, in `infrastructure/process_identity.py`.

What was genuinely duplicated: the graph-execution terminal-repair rule
was written twice in one package — `GraphExecutionSupervisor`'s crash
repair and `ReconcileGraphExecutions`' startup sweep both read the status,
marked failed with a note, and compare-and-swapped from what they read.
That is `ExecutionLifecycleWriter.fail_if_unfinished`, which takes a
clock and keeps the note caller-supplied, because a crash and a restart
are different facts.

A second instance of the same rule was in the dataset-task adapter, where
`finish_if_active` / `fail_if_active` / `kill_if_active` were three port
methods over two near-identical bodies, plus two more hand-rolled guarded
`UPDATE`s in `update_progress` — five copies of one statement in a single
file. All of it is now one `finalize_if_active` over
`infrastructure/persistence/cas.py`, which the graph-execution repository
calls too.

One deliberate split remains: the statement (`cas.py`) and the
announcement (`application/lifecycle_writer.py`) are separate. Dataset
tasks have no events, so they use the first without the second. Giving
them events is a feature decision, not a cleanup.

### WP-21 done, with one deviation

The review asked for the delivery-class table "next to the dataclasses".
It is in `application/event_delivery.py` instead, because
`domain/events.py` opens by stating that events have "no knowledge of
JSON, SSE, or any transport" — and a delivery class is exactly that.
`application/` is where this project puts policy, the infrastructure bus
can import it without inverting the layering, and having one location was
the part that mattered.

### WP-22 done

Cost and benefit, both measured rather than argued, are in
`docs/design/13-process-isolation.md`. The short form: a `SIGKILL` of a
run's process now fails that run's row and leaves the server serving,
where before it killed the server too; and a run survives the server being
killed underneath it. The cost is about two seconds of child startup per
run. `BACKEND_GRAPH_EXECUTION=inprocess` is the rollback, and it gives up
adoption along with isolation.

## Working notes for this repository

* **The error table in `docs/design/backend/02-api-reference.md` is a build
  input.** `backend/tests/test_error_contract.py` parses it and fails if an
  error code exists without a documented row, so adding an error class means
  editing that table in the same commit.
* **`env -u COMFY_DIR` does not simulate a fresh clone.** `path_tiers` loads
  the repository's gitignored `.env`, which sets it. Hide `.env` as well
  when checking whether a test depends on the developer's machine.
* **A fixture that writes by relative path will write into the developer's
  tree.** One such draft overwrote a dataset's real `metadata.db` and broke
  every later case in that file through an unrelated path.
* **Palette counts in the graph tests are derived, not literal**
  (`support.concrete_node_classes()`), so adding or retiring a node does not
  require editing a test.