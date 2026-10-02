# Mutation notes

How the round-2 review's WP-18 was carried out, and what survived.

**The point is not zero survivors.** It is that every survivor is either
killed by a test or has a written reason it is equivalent. A surviving
mutation says a test passes without checking the thing the test is about. An
*equivalent* mutation says the code states the same thing twice, which is a
milder finding and not worth a test.

Produced by `scripts/mutation_report.py` against the four modules the review
named:

| Module | Mutations | Killed | Survived | Before the tests below |
|---|---|---|---|---|
| `application/supervisor.py` | 231 | 168 | 63 | 73% killed |
| `infrastructure/jsonl_progress_source.py` | 260 | 178 | 82 | 70% killed, up from 58% |
| `application/use_cases/reconcile_runs.py` | 156 | 78 | 78 | 51% killed |
| `presentation/sse.py` | 198 | 105 | 93 | 56% killed |
| **Total** | **845** | **529** | **316** | **63% killed** |

## What the survivors turned out to be

The number alone is not the finding; the *shape* of the survivors is.
Classified by hand from the 316:

| Kind | Share | Verdict |
|---|---|---|
| Mutations inside `logger.*(...)` — message text and arguments | ~65% | **Equivalent.** No test asserts log wording, and none should. `"skipping non-JSON progress line in %s"` -> `"SKIPPING..."` is not a behaviour change. |
| Unreachable — the mutation is in a function no test executes | 23 of 845 | A real coverage gap, reported as such rather than as a mutant. |
| Degenerate numeric comparisons | the rest | **Equivalent.** `last_newline == -1` -> `== 1` describes a two-byte chunk. |
| Genuine coverage gaps | 4 | **Fixed.** Below. |

So the headline is not "37% of mutations survive". It is: *the tests do
not check log messages, which is correct*, and four real holes existed.

## The four holes, and what closed them

All four were in `jsonl_progress_source.py`, all found by mutation, none
visible to line coverage — every one of those statements *is* executed.

1. **The cache phases were never read by the real reader.**
   `test_supervisor.py` covers `cache_start` / `cache` / `cache_done` —
   through a **fake** progress source. `JsonlProgressSource._sample` never
   saw a cache line, so mutating `data.get("total")` to `data.get("TOTAL")`
   in all three branches changed nothing observable. A fake and the thing
   it stands in for drift quietly, and only this found it.

2. **`total_steps: 0` — zero was indistinguishable from absent.**
   `_int_field` rejects negatives but must keep zero, and the `or 0` / `or 1`
   expressions at most call sites rewrite a falsy counter *before* the
   coercer sees it. So `number >= 0` -> `> 0` survived: no record in the
   suite carried a `0` that reached the coercer.

3. **A file that shrank was never tested.** The reader keeps a byte offset;
   if the file is replaced with a shorter one, `size < offset` has to reset
   it, or the reader reads nothing forever. `offset = 0` -> `offset = 1`
   survived because nothing ever made a file smaller.

4. **A blank line, and an unexpected exception inside `_sample`, both ended
   the tail.** `continue` -> `break` survived in *three* separate skip
   branches. All-garbage input hides it (`continue` and `break` both yield
   nothing); all-good input never reaches the branch. The fix is mixed input
   with the good records *after* the bad ones, and a blank line — which is
   what a flushed write leaves behind. The third branch is the defensive
   `except` whose whole purpose is "never end the tail", and which cannot
   be reached by feeding bad input at all because `_sample` is written not
   to raise; it is exercised by making it raise.

`backend/tests/test_progress_reader.py` now covers all of it, and the
reader's kill rate went from **58% to 70%**.

One correction the process caught: a first draft of that file asserted a
negative loss becomes `None`. It does not — only `_int_field` rejects
negatives, because the *domain* rejects a negative step. The test was wrong
about the code, not the reverse, and "fixing" the code would have changed a
behaviour nobody asked to change. The test now pins the asymmetry from both
sides.

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
* **It was serial, on a machine with six idle cores.** 58.8% idle while
  making progress: exactly one test subprocess alive at a time. The tests
  were never the cost — they run in 0.35-0.98s and import no torch — it was
  231 mutants x one interpreter each, in a row. Giving each worker its own
  copy of the repository (which also stops the tool touching the real tree
  at all) took supervisor.py from ~22 minutes to **3m08s**. The coverage
  pass that maps tests to targets then became the bottleneck, so it is
  parallel too: sse.py went 2m01 -> 1m20 for all 198.
  Verified equivalent rather than assumed: `--serial` runs mutants in ROOT
  the old way, and on a 40-mutation slice of sse.py the two paths produce
  **byte-identical verdicts**.

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