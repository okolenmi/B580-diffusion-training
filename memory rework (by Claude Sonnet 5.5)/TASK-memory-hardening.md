# TASK: MEM-03 hardening (do this before MEM-04's remaining wiring and MEM-05)

Repository `okolenmi/B580-diffusion-training`, `main` at `383a1e8` or later.
Short on purpose: read only `backend/application/memory_ledger.py`,
`backend/domain/memory_settings.py`, `backend/presentation/schemas.py`
(`MemorySettingsIn`, `MemoryOverridesIn`, `GraphRunIn`),
`backend/application/use_cases/start_graph_execution.py` (the admission block,
around line 100-140) and `backend/application/use_cases/start_dataset_task.py`
(around line 160-200). Same working rules as `TASK-memory-rework.md`: one
commit per package (`MEM-03H-NN`), never weaken a test, run
`python3 backend/tests/run_all.py` and `python3 scripts/check_quality.py` before
each commit.

What is already good and must not regress: device-MB accounting
(`demand + process_overhead_mb`), claim under a pending owner then rename,
release-exactly-once paths, rebuild from rows, the real-process uvicorn race
test, exploratory-exclusive for unknown demand, the SQLite peak store (verified:
150 trials x 6 writer processes, 0 lost updates).

## MEM-03H-01 Reject bad numbers at every layer (reproduced)
**Problem.** Through the real endpoint `POST /api/v1/graphs/run`:
`memory: {"vram_max_mb": -9000}` -> HTTP 201, and the ledger then reports
**19,592 MB free on a capacity of 11,192**. The same with
`memory_overrides: {"vram_max_mb": -9000}`. At the ledger itself:
`reserve("a", float("nan"))` is granted and makes `held_mb()` and `free_mb()` NaN;
afterwards `reserve("b", 99999)` is **granted** (`demand > NaN` is False).
`reserve(x, -9000)` is granted and then two claims totalling 16,000 MB are
admitted on 11,192. Zero is accepted as a claim. Nothing in the chain checks:
`MemorySettingsIn` uses a bare `float`, `MemorySettings.from_dict` calls
`float(...)`, and the ledger compares with `>`.
**Do (three independent layers; each must hold on its own).**
1. API: `vram_min_mb` and numeric `vram_max_mb` / `ram_max_mb` must be finite,
   `>= 0`; `vram_min_mb <= vram_max_mb` when max is numeric; reject `bool`
   (it is an `int` in Python). Use pydantic `Field(ge=0, allow_inf_nan=False)`
   and a model validator for the ordering. Same for `MemoryOverridesIn`.
   422 with the field name.
2. Domain: `MemorySettings.__post_init__` (and `from_dict`) raise
   `ValueError` for the same conditions, so a stored or hand-edited graph file
   cannot smuggle a bad number past the API.
3. Ledger: `reserve()` refuses (returns a `Refusal` whose `reason` names the
   value) any `demand_mb` that is not finite or is `<= 0`; never mutates state
   on a refusal. `rebuild_from_rows` skips rows with such values and logs them.
   Add `assert held <= capacity` style checks to the test helper that inspects
   the ledger after every operation.
**Tests.** Seeds are in `r23_ledger_bad_demand.py`. Add: each bad value
(`-1`, `0`, `nan`, `inf`, `-inf`, `True`, `"lots"`) is rejected at API, domain
and ledger independently; after a rejected submission `free == capacity` and no
row is written; a property test (hypothesis) over random sequences of
`reserve/release/rename` with arbitrary floats never lets
`held_mb() > capacity_mb` or produces a non-finite total.

## MEM-03H-02 `vram_min_mb` is carried but never used
`grep -rn vram_min_mb backend` shows it is validated, stored and shown, and
never read by admission. The design says "the least the graph can run in
(refuse to start below it)". Implement it: when the effective demand is
**unknown** (exploratory) or `auto`, refuse with the breakdown if
`ledger.free_mb() < vram_min_mb + process_overhead_mb`, saying so ("this graph
needs at least X; Y is free"). When demand is stated or observed, `min` must not
exceed it (already enforced by MEM-03H-01 for stated). If you decide not to
enforce it yet, remove it from the API schema and editor until it is real and
say so in the ADR; do not leave a setting that looks like protection and is not.
**Tests.** min above free -> refused with both numbers in the message; min
below free -> admitted; min ignored for a stated demand that already fits.

## MEM-03H-03 Write down what an unknown device total means
The commit says a container with no device total refuses **every** start
("never a zero claim"). That is right for a GPU graph on the B580 and wrong for
a graph that uses no accelerator (CPU-only machine, a fresh install before
torch, or a server restarted while an adopted child holds the probe busy).
Decide and document in ADR 0005: (a) which node classes are GPU-using
(`uses_accelerator = True` on the class, default False for pure-data nodes) and
whether a graph with none of them skips admission; (b) what `/health` and the
refusal say when the total is unknown because the probe is busy (name the
adopted holder from the rows instead of "unknown"). Implement (b) at least:
the refusal for an unknown total must list the holders recovered from rows.
**Tests.** Unknown total + an adopted running row -> the refusal names it.

## After this
Continue with the remainder of `TASK-memory-rework.md`: MEM-04 (child
`memory` record and the server watcher applying peaks with MAX), then MEM-05.
Keep each work package in its own session if context gets tight; paste only the
package you are on plus section 1 (rules) of that file.
