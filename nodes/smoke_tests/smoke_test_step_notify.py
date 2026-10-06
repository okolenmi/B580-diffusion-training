"""Checks nodes/train/step_notify.py -- passing the step's latent shape to a
step callback without breaking callbacks that do not want it.

`on_step` is public and this repo already calls it in three shapes: None, a
two-argument lambda (`smoke_test_managed_trainer.py`'s
`lambda s, l: on_steps.append(s)`), and now a three-argument one
(`hw_validate.py`'s, which records `latent_shape` into steps.jsonl). The shape
is what made the multi-shape throughput cost measurable at all -- see
docs/known-issues/pending-testing.md -- so it is worth carrying; it is not
worth breaking a working caller for, which is exactly what happened when both
MonitoringPhase classes started passing a third argument unconditionally.

Everything here is about the boundary between "the caller cannot accept a
shape" and "the caller raised": the first must fall back, the second must
propagate. A blanket try/except TypeError would conflate them and re-run a
callback that legitimately raised, which is how a real bug gets hidden behind a
compatibility shim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.train.step_notify import notify_step


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_none_callback_is_a_noop():
    print("[on_step=None: notify_step does nothing and does not raise]")
    notify_step(None, 1, 0.5, "64x64")
    print("    PASS")


def check_two_argument_callback_still_works():
    print("[a two-argument callback is called with (step, loss) -- the case that "
          "regressed smoke_test_managed_trainer.py]")
    got = []
    notify_step(lambda s, l: got.append((s, l)), 7, 0.25, "64x64")
    check(got == [(7, 0.25)], got)
    print("    PASS")


def check_three_argument_callback_receives_the_shape():
    print("[a three-argument callback receives the shape unchanged]")
    got = []
    notify_step(lambda s, l, sh: got.append((s, l, sh)), 3, 0.5, "48x64")
    check(got == [(3, 0.5, "48x64")], got)
    print("    PASS")


def check_unknown_shape_is_passed_through_as_none():
    print("[a shape that could not be read arrives as None, not as a fabricated "
          "string -- the caller's 'no source' branch is its own to have]")
    got = []
    notify_step(lambda s, l, sh: got.append(sh), 1, 0.5, None)
    check(got == [None], got)
    print("    PASS")


def check_varargs_callback_gets_the_shape():
    print("[a *args callback is given all three -- it can accept the shape, so "
          "withholding it would lose the measurement for no reason]")
    got = []

    def cb(*args):
        got.append(args)

    notify_step(cb, 2, 0.5, "64x85")
    check(got == [(2, 0.5, "64x85")], got)
    print("    PASS")


def check_bound_method_arity_is_respected():
    print("[a bound method is inspected on its underlying function, so a "
          "two-argument method is not called with three]")
    seen = []

    class Reporter:
        def two(self, step, loss):
            seen.append(("two", step, loss))

        def three(self, step, loss, shape):
            seen.append(("three", step, loss, shape))

    r = Reporter()
    notify_step(r.two, 1, 0.5, "64x64")
    notify_step(r.three, 2, 0.5, "64x64")
    check(seen == [("two", 1, 0.5), ("three", 2, 0.5, "64x64")], seen)
    print("    PASS")


def check_keyword_only_shape_is_not_positional():
    print("[a callback whose third parameter is keyword-only is called with the "
          "two-argument form -- a third POSITIONAL would be a TypeError]")
    seen = []

    def cb(step, loss, *, shape=None):
        seen.append((step, loss, shape))

    notify_step(cb, 4, 0.5, "64x64")
    check(len(seen) == 1, seen)
    check(seen[0][:2] == (4, 0.5), seen)
    print("    PASS")


def check_a_typeerror_from_inside_the_callback_propagates():
    print("[a TypeError raised INSIDE a three-argument callback propagates -- it "
          "is the callback's bug, not an arity mismatch, and must not be "
          "swallowed into a silent two-argument retry]")
    def cb(step, loss, shape):
        raise TypeError("deliberate, from inside the callback")

    raised = False
    try:
        notify_step(cb, 1, 0.5, "64x64")
    except TypeError as exc:
        raised = True
        check("deliberate" in str(exc), str(exc))
    check(raised, "the callback's own TypeError must reach the caller")
    print("    PASS")


def check_a_typeerror_from_a_two_argument_callback_propagates():
    print("[the same holds for a two-argument callback: its own TypeError is not "
          "an arity signal either]")
    def cb(step, loss):
        raise TypeError("deliberate, from a two-argument callback")

    raised = False
    try:
        notify_step(cb, 1, 0.5, "64x64")
    except TypeError as exc:
        raised = True
        check("deliberate" in str(exc), str(exc))
    check(raised, "must propagate")
    print("    PASS")


def check_a_valueerror_from_inside_the_callback_propagates():
    print("[a non-TypeError from inside a callback propagates untouched]")
    def cb(step, loss, shape):
        raise ValueError("deliberate")

    raised = False
    try:
        notify_step(cb, 1, 0.5, "64x64")
    except ValueError:
        raised = True
    check(raised, "must propagate")
    print("    PASS")


def check_defaults_cover_exactly_three_positional_slots():
    print("[a callback with defaults for its 2nd and 3rd parameters still counts "
          "as shape-capable -- it binds three arguments fine]")
    got = []

    def cb(step, loss=0.0, shape=None):
        got.append((step, loss, shape))

    notify_step(cb, 9, 0.5, "64x64")
    check(got == [(9, 0.5, "64x64")], got)
    print("    PASS")


def check_the_same_callback_is_not_called_twice():
    print("[one notify_step call means exactly one callback invocation -- the "
          "fallback must not re-run the callback after a binding failure]")
    calls = []

    def cb(step, loss):
        calls.append(step)

    notify_step(cb, 1, 0.5, "64x64")
    notify_step(cb, 2, 0.5, "48x64")
    check(calls == [1, 2], calls)
    check(len(calls) == 2, f"expected exactly 2 calls, got {len(calls)}")
    print("    PASS")


def main():
    check_none_callback_is_a_noop()
    check_two_argument_callback_still_works()
    check_three_argument_callback_receives_the_shape()
    check_unknown_shape_is_passed_through_as_none()
    check_varargs_callback_gets_the_shape()
    check_bound_method_arity_is_respected()
    check_keyword_only_shape_is_not_positional()
    check_a_typeerror_from_inside_the_callback_propagates()
    check_a_typeerror_from_a_two_argument_callback_propagates()
    check_a_valueerror_from_inside_the_callback_propagates()
    check_defaults_cover_exactly_three_positional_slots()
    check_the_same_callback_is_not_called_twice()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
