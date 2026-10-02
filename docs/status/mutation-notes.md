# Test-quality measurement

Line coverage answers "was this code run". It does not answer the question
that matters about a test: **if the code were wrong, would this test notice?**

Mutation testing answers that by breaking the code on purpose and seeing
what notices. A mutation that survives tells you a test passes without
checking the thing the test is about.

The distinction that matters is between a survivor that is a **gap** and one
that is **equivalent**. A gap is a test that is not testing what it appears
to test. An equivalent mutant is a change that does not change behaviour —
usually a message string, or a default that nothing can reach. The second is
not a bug and does not deserve a test; a report that does not separate them
is just a number.

## Running it

```bash
/home/okolenmi/comfy/venv/bin/python scripts/mutation_report.py              # every target
/home/okolenmi/comfy/venv/bin/python scripts/mutation_report.py graph.py    # one
```

Targets are `TARGETS` in the script. They are pure-logic modules with no
torch, no device and no server, and each is chosen because tests reach most
of its lines — a module nothing calls makes every mutant "survive" by
default, which measures nothing. `scripts/mutation_report.py`'s own docstring
has the mechanics and the two mutmut behaviours that matter.

Targets must also have real function bodies. See the limitation below.

## Current measurement

Measured 2026-10-03 by running the command above with no arguments:

| Target | Killed | Survived | Testable killed |
| --- | --- | --- | --- |
| `backend/json_safe.py` | 33 | 1 | 100% |
| `backend/domain/graph.py` | 89 | 29 | 75% |
| `backend/infrastructure/graph_event_stream.py` | 102 | 64 | 61% |
| `backend/domain/lifecycle.py` | 20 | 28 | 42% |
| **Total** | **244** | **122** | **67%** |

Reproduce with the command above; the numbers move as the tests move, which
is the point of measuring rather than asserting.

## Classifying the survivors

Of the 122:

* **32 are inside `logger.*` calls.** Equivalent. No test asserts log
  wording and none should.
* **A handful are reported as *not evaluated* rather than uncovered** — see
  the limitation below. That label is a fix; the old one was a lie.
* **The rest are real.** They are listed below rather than left as a count,
  because a count of "38 survivors" tells the next person nothing about
  which line to go and read.

## What it found

### `domain/lifecycle.py` — id validation is only half pinned

`StatusMachine.require_id` validates with:

```python
if not isinstance(entity_id, int) or isinstance(entity_id, bool) or entity_id < 1:
```

Two mutations of that line survive: `or` → `and` in either position. One of
them would let `True` through as an id (`True` is an `int` in Python, and the
explicit `bool` guard is the only thing rejecting it); the other would make a
string raise `TypeError` from `"a" < 1` instead of the intended
`DomainError`.

The production code is correct — both guards are there. **The tests do not
cover either branch**, so the correctness is a fact about the source rather
than about the suite. That is exactly the difference mutation testing exists
to surface, and line coverage cannot see it: the line is covered.

`StatusMachine.ends_lifecycle` is never executed by any test, and neither is
its `__repr__`.

### `domain/graph.py` — `from_dict`'s tolerant defaults are never exercised

All 29 survivors are the same shape: a default in
`str(raw.get(key, ""))` mutated to `None`, `""` or a junk string, for every
field of `GraphDefinition.from_dict`.

`from_dict` is documented as tolerant by design — missing `params` defaults
to `{}`, unknown keys are ignored — and the authoritative shape check is
`validate()`. So the defaults are load-bearing for a malformed payload and no
test supplies one. One case feeding a payload with the keys absent would
close all 29 at once.

## A limitation worth stating

The tool assigns candidate tests to a mutation by looking for tests that
executed the **function body** the mutation sits in. That narrowing exists
because it is otherwise useless: importing a module runs its module-level
lines, so "this line executed" is true of any test that merely imports the
file, and every mutant would look covered by the whole suite.

The cost is that module-level mutations are invisible to it. A module that is
mostly module-level data therefore cannot be measured by this tool at all.

This was not a hypothetical. An earlier target list included
`domain/value_objects.py`, which is 97% covered — and the tool reported it as
**0% killed, 62 survivors, "NO TEST EXECUTES IT"** for every one. The module is
enums and transition tables; almost all of its mutations are at module level;
the filter never asked the question it appeared to be answering. The message
now says *not evaluated* for those, which is the truth, and the module is no
longer a target.

Two consequences to keep in mind:

* A kill rate from this tool is a floor on what the tests check, not a
  measure of the whole file.
* `backend/domain/value_objects.py` and `backend/domain/graph.py`'s transition
  tables are **not** measured here. The lifecycle machine that enforces them
  is covered, and `test_value_objects.py` exercises the tables through it —
  but that is a separate argument, not a number this tool produced.

## A reporting bug this surfaced

Before the fix, a mutant that no test was assigned to was reported as
`NO TEST EXECUTES IT` — including mutants the tool had never tried to
evaluate. A mapping that reports "no coverage" where coverage exists is the
worst failure available for a component whose only job is to say which tests
to run, because it converts every mutant into a false survivor. It is now
split into *no test executes it* (a real gap) and *not evaluated* (outside the
tool's reach).