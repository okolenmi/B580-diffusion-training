# Mutation notes

How the round-2 review's WP-18 was carried out, and what survived.

**The point is not zero survivors.** It is that every survivor is either
killed by a test or has a written reason it is equivalent. A surviving
mutation says a test passes without checking the thing the test is about. An
*equivalent* mutation says the code states the same thing twice, which is a
milder finding and not worth a test.

Produced by `scripts/mutation_report.py` against the four modules the review
named:

| Module | Mutations | Killed | Survived |
|---|---|---|---|
| `backend/application/supervisor.py` | 231 | _see the run log_ | |
| `backend/infrastructure/jsonl_progress_source.py` | | | |
| `backend/application/use_cases/reconcile_runs.py` | | | |
| `backend/presentation/sse.py` | | | |

How it runs, and two things worth knowing before trusting a number from it:

* **It drives mutmut's mutation engine by hand**, because mutmut's runner
  is pytest and these tests are scripts (`python backend/tests/test_x.py`
  with a `check()`/`finish()` pair). Importing them into pytest puts their
  module-level work at collection time, where a raised assertion is a
  collection *error* — which kills mutants, and tells nobody which
  assertion fired.
* **Which tests can kill a mutation is measured, not guessed.** A targeted
  coverage pass records which test files execute lines in the target, and
  for each mutation only the tests that execute *the function that mutation
  sits in* are run. A hand-written map goes stale and then reports a mutant
  as surviving because nobody ran the test that would have killed it.

## Survivors

_Classified below once the run completes._

## What the runner got wrong on the way, and what it cost

Worth recording because both failures were silent in the way that matters.

* **The coverage mapping reported that nothing covers `supervisor.py`**,
  when three test files plainly do. It used `--parallel-mode` and read the
  result with the `CoverageData` API; switching to one `--data-file` per
  test plus `coverage json` fixed it. A mapping that says "no coverage"
  when there is coverage is the worst failure available for a component
  whose only job is to say which tests to run — it turns every mutant into
  a false survivor.
* **Matching on any executed line said 25 of 26 test files covered
  `supervisor.py`**, because *importing* it runs its module-level lines,
  and an import cannot kill a mutation inside `_guard`. Narrowing to lines
  inside a function body cut it to 11, and narrowing further to the
  mutated function itself cut the per-mutant cost by an order of magnitude.
* **`MetadataWrapper` deep-copies by default**, so matching a mutation's
  `original_node` by identity silently matched nothing: 57 mutations, 57
  no-ops, reported as 0% killed. `unsafe_skip_copy=True` keeps identity.
* **The first full run took four minutes on one mutant** and would have
  taken hours, because a mutant that makes a test *hang* waits out the
  timeout, and the default was 300s. A hang is a legitimate kill — a
  mutation that stops a test finishing is an observable change — but the
  price per hang has to match the tests' normal runtime, not dwarf it.
* **Being killed partway through left a mutant in the working tree.** A
  tool that rewrites source files in place has to restore them on every
  exit path, including a signal; it did not, and it did once.

The last two are the same lesson: the tool rewrites real files and runs
real tests, so its failure modes are silent ones unless they are handled
explicitly.