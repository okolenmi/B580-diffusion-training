"""JsonlProgressSource, against real files.

Run directly: python backend/tests/test_progress_reader.py

Three of these cases exist because mutation testing found them missing,
which line coverage could not: every one of the statements below *is*
executed by some test, so a coverage report showed them green. What no
test did was feed the **real reader** the input that distinguishes the
right answer from the wrong one.

The clearest example: the cache phases (`cache_start`, `cache`,
`cache_done`) are covered by `test_supervisor.py`, but through the fake
progress source. `JsonlProgressSource._sample` never saw a cache line,
so every string in those branches -- `"cache"`, `"done"`, `"total"` --
could be mutated and nothing noticed. A fake and the thing it stands in
for drift quietly; this file is the real thing.

The other two: `training_start` with `total_steps: 0` (zero is a value,
not an absence -- the coercer rejects negatives, so it must not reject
zero), and a file that *shrinks* under a reader that has already read it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.jsonl_progress_source import JsonlProgressSource
from backend.tests.support import check, finish

SCRATCH = Path(tempfile.mkdtemp(prefix="progress-reader-"))


def write(name: str, records: list[dict]) -> Path:
    path = SCRATCH / name
    path.write_bytes(
        b"".join((json.dumps(r) + "\n").encode("utf-8") for r in records)
    )
    return path


def read(path: Path) -> list:
    return JsonlProgressSource().read_new(path)


# --------------------------------------------------------------------------
# The cache phases, through the real reader
# --------------------------------------------------------------------------

def test_cache_phases() -> None:
    """`_sample`'s cache branches had no coverage at all.

    Covered elsewhere by a *fake* progress source, so the strings, the
    `or 0` / `or 1` fallbacks and the `_int_field` calls here were never
    exercised. Mutation rewrote `data.get("total")` to `data.get("TOTAL")`
    in every one of them and the suite stayed green.
    """
    print("\n== cache phases, read from a file ==")

    path = write("cache.jsonl", [
        {"phase": "cache_start", "est_trajs": 1000},
        {"phase": "cache", "done": 500, "total": 1000},
        {"phase": "cache_done", "total": 1000},
    ])
    samples = read(path)
    check(len(samples) == 3, f"three cache records read (got {len(samples)})")

    start = samples[0]
    check(start.phase == "cache", f"cache_start is the cache phase (got {start.phase!r})")
    check(start.cache_done == 0, f"nothing done yet (got {start.cache_done})")
    check(start.cache_total == 1000,
          f"the estimate becomes the total (got {start.cache_total})")

    middle = samples[1]
    check(middle.cache_done == 500, f"progress is read (got {middle.cache_done})")
    check(middle.cache_total == 1000, f"the total is read (got {middle.cache_total})")

    end = samples[2]
    check(end.cache_done == 1000 and end.cache_total == 1000,
          f"cache_done reports the total on both sides "
          f"(got {end.cache_done}/{end.cache_total})")

    # The keys are the whole point of the mutation above.
    check(end.cache_total != 1,
          "the total came from the record, not from the `or 1` fallback")


def test_cache_records_with_missing_counters() -> None:
    """A cache line that omits its counters falls back rather than raising.

    The fallbacks are `or 0` / `or 1`, so a zero `total` also lands on the
    fallback -- deliberate, since a cache total of zero has nothing to
    divide by -- and this pins that as the behaviour rather than an
    accident of the expression.
    """
    print("\n== cache records with missing counters ==")
    path = write("cache-sparse.jsonl", [
        {"phase": "cache"},
        {"phase": "cache", "done": 3},
        {"phase": "cache_done"},
    ])
    samples = read(path)
    check(len(samples) == 3, f"all three read (got {len(samples)})")
    check(samples[0].cache_done == 0 and samples[0].cache_total == 1,
          f"a bare cache line falls back to 0/1 "
          f"(got {samples[0].cache_done}/{samples[0].cache_total})")
    check(samples[1].cache_done == 3 and samples[1].cache_total == 1,
          f"a missing total still falls back (got {samples[1].cache_total})")
    check(samples[2].cache_done == 1,
          f"cache_done with no total falls back to 1 (got {samples[2].cache_done})")


# --------------------------------------------------------------------------
# Zero is a value, not an absence
# --------------------------------------------------------------------------

def test_zero_is_preserved_not_treated_as_missing() -> None:
    """`_int_field` rejects negatives and must keep zero.

    The mutation `number >= 0` -> `number > 0` survived every test: no
    record in the suite carried a `0` that mattered, because a falsy
    counter is rewritten to the fallback by the `or` expressions at the
    call site *before* the coercer sees it. `training_start.total_steps`
    has no such `or`, so it is the one place zero reaches the coercer --
    and `total_steps: 0` is exactly the malformed record the docstring
    says must not rewind the run.
    """
    print("\n== zero is preserved, not treated as absent ==")
    path = write("zero.jsonl", [
        {"phase": "training_start", "total_steps": 0},
        {"phase": "step", "step": 0, "total": 0, "loss": 0.0, "avg": 0.0},
    ])
    samples = read(path)
    check(len(samples) == 2, f"both read (got {len(samples)})")

    start = samples[0]
    check(start.total == 0,
          f"training_start keeps a zero total rather than dropping it "
          f"(got {start.total!r})")
    check(start.step == 0, f"and the step stays zero (got {start.step!r})")

    step = samples[1]
    check(step.step == 0, f"a zero step is a step (got {step.step!r})")
    check(step.loss == 0.0,
          f"a zero loss is a measurement, not a missing one (got {step.loss!r})")


def test_integers_reject_negatives_and_floats_do_not() -> None:
    """The asymmetry is deliberate and worth pinning from both sides.

    `_int_field` rejects a negative integer because the *domain* rejects
    it: a negative step would raise `DomainError` and kill the supervisor
    thread. `_float_field` has no such constraint -- a negative loss is
    not something the domain forbids, it is a number the trainer reported
    -- so it passes through. Only the first half of that was being tested.

    An earlier draft of this file asserted that a negative loss became
    None. That was the test being wrong about the code, not the other way
    round: nothing in `_float_field` promises it, and inventing the
    expectation would have "fixed" a behaviour nobody wanted changed.
    """
    print("\n== integers reject negatives, floats do not ==")
    path = write("negative.jsonl", [
        {"phase": "step", "step": -5, "total": 100},
        {"phase": "step", "step": 7, "total": 100, "loss": -1.0},
        {"phase": "step", "step": 8, "total": 100, "loss": "not a number"},
    ])
    samples = read(path)
    check(len(samples) == 3, f"all three lines are still read (got {len(samples)})")
    check(samples[0].step is None,
          f"a negative step becomes None, not -5 (got {samples[0].step!r})")
    check(samples[1].loss == -1.0,
          f"a negative loss is a reported number and passes through "
          f"(got {samples[1].loss!r})")
    check(samples[2].loss is None,
          f"a malformed loss is None, which is the other half "
          f"(got {samples[2].loss!r})")


# --------------------------------------------------------------------------
# A file that shrinks under a reader that has already read it
# --------------------------------------------------------------------------

def test_a_file_that_shrinks_restarts_instead_of_resuming() -> None:
    """A rotated or truncated tail file must be read from the beginning.

    The reader remembers a byte offset. If the file is replaced with a
    shorter one, that offset is past the end, and resuming from it would
    read nothing forever. `size < offset` resets to zero -- and the
    mutation `offset = 0` -> `offset = 1` survived, because no test ever
    made a file smaller.
    """
    print("\n== a file that shrinks restarts ==")
    path = write("shrink.jsonl", [{"phase": "step", "step": i, "total": 100}
                                  for i in range(1, 21)])
    source = JsonlProgressSource()
    first = source.read_new(path)
    check(len(first) == 20, f"twenty records to begin with (got {len(first)})")

    # Replace it with a shorter file, as a rotation would.
    path.write_bytes(
        b"".join(
            (json.dumps({"phase": "step", "step": i, "total": 100}) + "\n").encode()
            for i in (90, 91)
        )
    )
    second = source.read_new(path)
    check(len(second) == 2,
          f"the shorter file is read whole, not skipped (got {len(second)})")
    check(second and second[0].step == 90,
          f"starting from its first record (got {second[0].step if second else None})")

    # And it keeps working from the new end afterwards.
    with path.open("ab") as handle:
        handle.write((json.dumps({"phase": "step", "step": 92, "total": 100})
                      + "\n").encode())
    third = source.read_new(path)
    check(len(third) == 1 and third[0].step == 92,
          f"and resumes correctly afterwards (got {[s.step for s in third]})")


def test_a_file_that_grows_is_only_read_past_its_end() -> None:
    """The ordinary case, pinned so the reset above cannot be satisfied by
    simply always starting over."""
    print("\n== a growing file is read incrementally ==")
    path = write("grow.jsonl", [{"phase": "step", "step": 1, "total": 100}])
    source = JsonlProgressSource()
    check(len(source.read_new(path)) == 1, "first record")

    with path.open("ab") as handle:
        for step in (2, 3):
            handle.write((json.dumps({"phase": "step", "step": step, "total": 100})
                          + "\n").encode())
    second = source.read_new(path)
    check(len(second) == 2, f"only the new records (got {len(second)})")
    check([s.step for s in second] == [2, 3],
          f"in order (got {[s.step for s in second]})")

    # The counters carried on a step record, checked against absolute
    # values rather than for self-consistency. The truncation property in
    # test_property_boundaries.py compares one whole-file read against a
    # prefix-plus-rest read, so it is *invariant*: a mutation that changes
    # what every read reports passes it. This assertion is what killed
    # `data.get("total")` -> `data.get("TOTAL")` on the step branch --
    # and it was added after that mutant survived a first pass of this
    # very file, which had read step records without checking them.
    check(all(s.total == 100 for s in second),
          f"the total comes from each record "
          f"(got {[s.total for s in second]})")
    check(all(s.phase == "training" for s in second),
          f"and the phase is the step phase (got {[s.phase for s in second]})")


def test_one_bad_record_does_not_end_the_tail() -> None:
    """A malformed line is skipped; the ones after it are still read.

    The reader's whole contract is that one bad record cannot stop the
    tail (docs 07 F-01), and nothing tested it: `continue` -> `break` in
    that loop survived, because every existing test fed either all-good
    or all-garbage input. All-garbage hides it -- both `continue` and
    `break` yield no samples -- and all-good never reaches the branch.

    So the input has to be *mixed*, and the good records have to come
    after the bad ones.
    """
    print("\n== one bad record does not end the tail ==")
    good = {"phase": "step", "step": 7, "total": 100}
    path = write("mixed.jsonl", [
        good,
        {"phase": "step", "step": 8, "total": 100},
        {"this": "is not a progress record"},          # no phase
        {"phase": "step", "step": "not a number", "total": 100},
        good,
        good,
    ])
    # A blank line is the case a flushed write leaves behind, and it is
    # skipped by the same `continue`. `continue` -> `break` there survived
    # every other test because none of them had one: an all-good file has
    # no blank line, and an all-garbage file looks the same either way.
    path.write_bytes(path.read_bytes() + b"\n" + (json.dumps(good) + "\n").encode())
    samples = read(path)
    # Six lines, one of which has an unrecognised phase and is dropped;
    # the malformed step still yields a sample (step=None), because the
    # coercer turns a bad value into "leave unchanged".
    check(len(samples) == 6,
          f"every line but the unrecognisable one is read, across a blank "
          f"line (got {len(samples)})")
    check([s.step for s in samples] == [7, 8, None, 7, 7, 7],
          f"including the ones *after* the bad line and the blank "
          f"(got {[s.step for s in samples]})")

    # And a non-JSON line in the middle, which is a different skip.
    path.write_bytes(
        b"".join([
            (json.dumps(good) + "\n").encode(),
            b"{not json at all\n",
            (json.dumps({"phase": "step", "step": 9, "total": 100}) + "\n").encode(),
        ])
    )
    mixed = read(path)
    check(len(mixed) == 2,
          f"a non-JSON line is skipped too (got {len(mixed)})")
    check([s.step for s in mixed] == [7, 9],
          f"and the tail after it survives (got {[s.step for s in mixed]})")


def test_an_unexpected_failure_inside_the_coercer_does_not_end_the_tail() -> None:
    """The defensive `except` around `_sample` is the belt to the braces.

    Its entire purpose is that nothing raised while interpreting one
    record may end the tail (docs 07 F-01) -- which makes it the one branch
    that cannot be reached by feeding bad *input*, because `_sample` is
    written not to raise. So it is exercised by making it raise.

    Worth doing precisely because it is otherwise untestable through the
    public interface: `continue` -> `break` in that handler survived every
    other test in the suite.
    """
    print("\n== an unexpected failure does not end the tail ==")
    good = {"phase": "step", "step": 3, "total": 100}
    path = write("coercer-boom.jsonl", [good, good, good])

    calls = {"n": 0}
    real = JsonlProgressSource._sample

    def exploding(data: dict):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected: the coercer fell over")
        return real(data)

    JsonlProgressSource._sample = staticmethod(exploding)
    try:
        samples = read(path)
    finally:
        JsonlProgressSource._sample = staticmethod(real)

    check(len(samples) == 2,
          f"only the record that raised is lost (got {len(samples)})")
    check([s.step for s in samples] == [3, 3],
          f"and the one after it is still read (got {[s.step for s in samples]})")


def main() -> int:
    test_cache_phases()
    test_cache_records_with_missing_counters()
    test_zero_is_preserved_not_treated_as_missing()
    test_integers_reject_negatives_and_floats_do_not()
    test_a_file_that_shrinks_restarts_instead_of_resuming()
    test_a_file_that_grows_is_only_read_past_its_end()
    test_one_bad_record_does_not_end_the_tail()
    test_an_unexpected_failure_inside_the_coercer_does_not_end_the_tail()
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())