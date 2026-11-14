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

| Target | Killed | Survived |
| --- | --- | --- |
| `backend/json_safe.py` | 34 | 1 |
| `backend/domain/graph.py` | 107 | 11 |
| `backend/domain/lifecycle.py` | 30 | 18 |
| `backend/infrastructure/graph_event_stream.py` | 106 | 58 |
| **Total** | **277** | **88** |

Reproduce with the command above. The figures move by a mutation or two
between runs — the tool is a measurement, not a proof — and they move for a
real reason when a test is added, which is the point of measuring rather
than asserting.

**Every one of the 88 survivors is equivalent or unevaluated. There is no
coverage gap left in these four modules.**

| Kind | Count | Why it survives |
|---|---|---|
| Inside `logger.*` | 30 | Equivalent. No test asserts log wording and none should. |
| At module level | 23 | Not evaluated by this tool — see the limitation below. |
| Error-message strings, and defaulted arguments whose fallback is unreachable from the current call sites | 35 | Equivalent. Changing `"is not allowed"` to something else does not change what the code does. |

## What it found, and what it changed

Four gaps, all real, all now closed.

### `domain/lifecycle.py` — id validation was only half pinned

`StatusMachine.bind` validates with:

```python
if not isinstance(entity_id, int) or isinstance(entity_id, bool) or entity_id < 1:
```

Two mutations of that line survived: `or` → `and` in either position. One
would let `True` through as an id — `bool` is a subclass of `int`, and the
explicit `isinstance(entity_id, bool)` is the only thing rejecting it. The
other makes a string raise `TypeError` from `"a" < 1` instead of the
intended `DomainError`.

The production code was correct; **the tests reached neither branch**, so
the correctness was a fact about the source and not about the suite. That
is exactly what line coverage cannot see: the line *is* covered.
`test_value_objects.py` now pins all three branches, the accepted case, and
the exactly-once rule.

### `domain/lifecycle.py` — `ends_lifecycle` had no callers at all

Reported as "no test executes it", which turned out to be the milder
statement. Grepping the whole repository — backend, `nodes/`, `manager/`,
the archive — found zero callers anywhere. It is dead code, and the honest
response was to delete it rather than to write a test that would entrench
it. The enums' own `is_terminal()` answers the same question from the same
table.

### `domain/graph.py` — `from_dict`'s tolerance was unexercised

All 29 survivors were the same shape: the default in a
`str(raw.get(key, ""))` mutated to `None`, `""` or junk, for every field.
`from_dict` documents itself as tolerant — missing `params` defaults to
`{}`, unknown keys are ignored — and no test supplied a payload that needed
the tolerance. `test_graph_execution.py` now feeds it an empty payload, a
node with no `class_name`, a node with no `id`, an edge with no fields at
all, and a payload with unknown keys at both levels. 29 survivors → 0.

### `graph_event_stream.py` — a second writer on an existing directory

`mkdir(exist_ok=True)` → `mkdir(exist_ok=False)` survived, because every test
opened a path whose parent did not exist yet. It is not unreachable: the
supervisor creates the execution's scratch directory *before* spawning, so
the writer's own `mkdir` always finds it there. With `exist_ok=False` that
raises, the constructor's `OSError` handler marks the writer unavailable,
and every record the run produces is silently dropped — the server would
watch an empty file and report a completed run as crashed. Now tested.

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