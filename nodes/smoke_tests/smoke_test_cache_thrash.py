"""Checks PrimitiveCacheThrashDetector: the runtime warning for a oneDNN
primitive cache too small for the shapes a run uses.

The detector exists because the capacity constant in `nodes/xpu_env.py` came
from one measured configuration, and rank, DoRA, targets and optimizer
strategy all change how many primitives a step creates. A cache that is too
small has a measured signature on this card -- a revisit of an
already-compiled shape takes 4.34x a repeat, and *first* sightings do not
improve when the cache is raised (they are JIT, which no cache size fixes).

So the checks here are about the two ways this detector could be wrong, and
both would be silent:

  1. **Fires when it should not.** Warning about a healthy run teaches people
     to ignore the warning, and the healthy cases are the common ones: a
     single-shape dataset has no revisits at all, and a well-sized cache makes
     revisits and repeats indistinguishable.
  2. **Stays quiet when it should not.** The whole value is that it catches a
     mis-sized cache at runtime, in a configuration nobody measured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.train.cache_thrash import (CAPACITY_ENV_VAR,
                                      PrimitiveCacheThrashDetector,
                                      format_warning)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _feed(detector, observations):
    """Feed (shape, seconds) pairs, returning every warning produced."""
    return [w for w in (detector.observe(s, d) for s, d in observations) if w]


def check_thrash_is_detected_and_names_the_knob():
    print("[a cache that evicts: revisits slow, first sightings normal -- "
          "warns, and names the env var to raise]")
    # Measured shape of the problem: repeat 0.873 s, revisit 3.790 s, and
    # first sightings that the capacity fix does NOT help.
    d = PrimitiveCacheThrashDetector()
    obs = [("64x64", 0.87), ("96x64", 3.99), ("64x64", 3.79),
           ("96x64", 3.79), ("64x64", 3.79), ("64x64", 0.87)]
    warnings = _feed(d, obs)
    check(len(warnings) == 1, f"expected exactly one warning, got {warnings}")
    w = warnings[0]
    check(CAPACITY_ENV_VAR in w,
          f"the warning must name the variable to raise, got: {w}")
    check("4.35x" in w or "4.3" in w,
          f"the warning must carry the observed ratio, got: {w}")
    check(d.warned is True and d.warning == w, "warned/warning must agree")
    print(f"    {w[:110]}...")
    print("    PASS")


def _mixed(fast: float, revisit: float, pairs: int, first: float = None):
    """An observation list shaped like the real thing, with a real baseline.

    One established shape "A", with a *different* shape dropped between
    visits. That drop is what makes each return to A a **revisit** rather than
    a repeat: the loader clumps same-shape batches (nodes/train/step_pipeline
    .py's MonitoringPhase comment on why a shape belongs in the step record at
    all), and a thrashing cache is slow on the step after a transition, not on
    the step that continues one.

    The pattern per round, and what each step is classified as:

        A            repeat   fast          -- the baseline
        B<i>         first    slow          -- a new shape, JIT, unavoidable
        A            revisit  revisit_slow  -- the thrash under test

    Two earlier versions of this helper were wrong in ways that made the tests
    pass without exercising anything: strictly alternating shapes produce no
    repeats at all (so a detector with no baseline is silent for the wrong
    reason), and doubling the *slow* shape put the slow values in `repeat`,
    inverting the ratio. The classification is now asserted from `report()`
    in each check rather than assumed.
    """
    if first is None:
        first = revisit
    obs = [("A", fast)]
    for i in range(pairs):
        obs += [("A", fast), (f"B{i}", first), ("A", revisit)]
    return obs


def check_it_warns_only_once():
    print("[once per run, not once per step -- the ratio does not change after "
          "the first few revisits, so a rate would be noise]")
    d = PrimitiveCacheThrashDetector()
    warnings = _feed(d, _mixed(0.87, 3.8, pairs=15))
    check(len(warnings) == 1,
          f"15 thrash rounds produced {len(warnings)} warnings; must be 1")
    rep = d.report()
    check(rep["revisit"] >= 3 and rep["repeat"] >= 3 and rep["first"] >= 3,
          f"the pattern must produce a real baseline in every bucket: {rep}")
    check(rep["revisit_median_sec"] > rep["repeat_median_sec"],
          f"the helper must put the slow values in revisits: {rep}")
    # And the detector must not re-evaluate either: after warning, observe()
    # is a no-op, so even feeding it clean data changes nothing.
    before = d.report()["revisit"]
    for _ in range(5):
        d.observe("A", 0.87)
    check(d.report()["revisit"] == before,
          "observe() kept classifying after it warned")
    print(f"    {rep['revisit']} revisits at {rep['revisit_ratio']:.2f}x, "
          f"1 warning")
    print("    PASS")


def check_a_healthy_run_is_silent():
    print("[a healthy run says nothing: revisits and repeats are "
          "indistinguishable when the cache is big enough]")
    d = PrimitiveCacheThrashDetector()
    warnings = _feed(d, _mixed(1.02, 1.03, pairs=20))
    check(not warnings,
          f"a run whose revisits match its repeats warned: {warnings}")
    rep = d.report()
    check(rep["revisit"] >= 3 and rep["repeat"] >= 3,
          f"the pattern must produce a baseline to be a real test: {rep}")
    check(abs(rep["revisit_ratio"] - 1.0) < 0.05,
          f"expected a ratio near 1.0, got {rep['revisit_ratio']}")
    check(rep["warned"] is False, "warned must stay False")
    print(f"    ratio {rep['revisit_ratio']:.3f} over {rep['revisit']} "
          f"revisits and {rep['repeat']} repeats, silent")
    print("    PASS")


def check_a_single_shape_dataset_is_silent():
    print("[a single-shape dataset has no revisits at all and must not be "
          "made to warn about a cache it cannot thrash]")
    d = PrimitiveCacheThrashDetector()
    warnings = _feed(d, [("64x64", 1.0)] * 40)
    check(not warnings, f"one shape, no revisits, warned anyway: {warnings}")
    rep = d.report()
    check(rep["revisit"] == 0 and rep["repeat"] == 39,
          f"expected 39 repeats and 0 revisits, got {rep}")
    check(rep["distinct_shapes"] == 1, f"one shape, got {rep}")
    print(f"    {rep['repeat']} repeats, {rep['revisit']} revisits, silent")
    print("    PASS")


def check_only_slow_first_sightings_is_a_different_problem():
    print("[slow FIRST sightings alongside FAST revisits do not warn: the "
          "first sightings are JIT, no cache size fixes them, and pointing at "
          "the cache would send someone to the wrong knob]")
    d = PrimitiveCacheThrashDetector()
    # New shapes arrive slow while the established shape stays fast whichever
    # way it is reached, so first_median is high and the revisit/repeat ratio
    # is 1.0 -- the exact run that must stay silent. Reuses _mixed so the
    # baseline is real rather than absent.
    obs = _mixed(fast=0.88, revisit=0.88, pairs=5, first=3.9)
    warnings = _feed(d, obs)
    check(not warnings,
          f"a run whose cost is in first sightings warned about the cache: "
          f"{warnings}")
    rep = d.report()
    check(rep["first"] == 6, f"expected 6 first sightings, got {rep}")
    check(rep["revisit"] >= 3 and rep["repeat"] >= 3,
          f"the pattern must produce a real baseline: {rep}")
    check(rep["first_median_sec"] > 2.0,
          f"the first sightings must actually be the slow ones: {rep}")
    check(abs(rep["revisit_ratio"] - 1.0) < 0.05,
          f"revisits were fast, so the ratio must be ~1.0: {rep}")
    print(f"    {rep['first']} first sightings at "
          f"{rep['first_median_sec']:.2f} s, revisit ratio "
          f"{rep['revisit_ratio']:.2f}, silent")
    print("    PASS")


def check_too_few_revisits_does_not_warn():
    print("[three revisits is the floor: below it one slow step sets the "
          "median, and the 'median' of two numbers is not evidence]")
    d = PrimitiveCacheThrashDetector()
    # Two revisits at 9 s against a 0.87 s repeat -- an enormous ratio, and
    # still not enough to decide.
    obs = [("a64x64", 0.87), ("b96x64", 9.0), ("a64x64", 9.0),
           ("a64x64", 0.87), ("b96x64", 9.0)]
    warnings = _feed(d, obs)
    check(not warnings,
          f"warned on {len(warnings)} revisit(s); the floor is 3")
    check(d.report()["revisit"] == 2, f"expected 2 revisits: {d.report()}")
    check(d.report()["repeat"] == 1, f"expected 1 repeat: {d.report()}")
    # The third revisit crosses the floor and it fires.
    third = d.observe("a64x64", 9.0)
    check(third is not None,
          "the third revisit must be enough to decide; it is the floor, not "
          "a sample size to exceed comfortably")
    check(CAPACITY_ENV_VAR in third, f"warning must name the knob: {third}")
    print("    2 revisits silent, 3rd warns")
    print("    PASS")


def check_no_repeat_baseline_means_no_warning():
    print("[with no repeat to compare against there is no ratio, so no "
          "warning -- rather than inventing a denominator and firing on every "
          "shape-alternating run]")
    d = PrimitiveCacheThrashDetector()
    obs = [("a64x64", 3.8)]
    for i in range(8):
        obs += [(f"b{i}x64", 3.8), ("a64x64", 3.8)]
    warnings = _feed(d, obs)
    check(not warnings, f"a run that never repeats a shape warned: {warnings}")
    rep = d.report()
    check(rep["revisit"] >= 3,
          f"the pattern must actually produce revisits: {rep}")
    check(rep["repeat"] == 0,
          f"expected no repeats in this pattern, got {rep}")
    print(f"    {rep['revisit']} revisits, {rep['repeat']} repeats, silent")
    print("    PASS")


def check_unusable_observations_are_ignored():
    print("[an unknown shape or a broken clock reading is ignored, not "
          "classified -- either would put a wrong number into a median]")
    d = PrimitiveCacheThrashDetector()
    # None shape: the step's shape was not reported.
    d.observe(None, 9.0)
    # None / non-numeric / zero / negative / NaN / inf seconds.
    d.observe("64x64", None)
    d.observe("64x64", "not a number")
    d.observe("64x64", 0.0)
    d.observe("64x64", -1.0)
    d.observe("64x64", float("nan"))
    d.observe("64x64", float("inf"))
    rep = d.report()
    check(rep["repeat"] == 0 and rep["revisit"] == 0 and rep["first"] == 0,
          f"nothing should have been classified, got {rep}")
    check(rep["distinct_shapes"] == 0,
          f"an unusable observation must not mark a shape as seen: {rep}")
    # And the detector still works afterwards. _mixed, because a pattern with
    # no repeats would leave it silent for the wrong reason.
    warnings = _feed(d, _mixed(0.87, 3.8, pairs=5))
    check(len(warnings) == 1, f"detector broken after bad input: {warnings}")
    print("    PASS")


def check_the_first_shape_is_a_sighting():
    print("[the first shape of a run has no predecessor, so it is a first "
          "sighting -- counting it as a repeat would make every run look "
          "healthy by one sample]")
    d = PrimitiveCacheThrashDetector()
    d.observe("64x64", 0.87)
    d.observe("64x64", 0.87)
    rep = d.report()
    check(rep["first"] == 1, f"expected 1 first sighting, got {rep}")
    check(rep["repeat"] == 1, f"expected 1 repeat, got {rep}")
    print(f"    first {rep['first']}, repeat {rep['repeat']}")
    print("    PASS")


def check_thresholds_are_validated():
    print("[a ratio at or below 1.0 is refused: it would make every run that "
          "repeats a shape warn]")
    for bad in (1.0, 0.5, -1.0):
        try:
            PrimitiveCacheThrashDetector(revisit_ratio=bad)
        except ValueError as e:
            check("revisit_ratio" in str(e), f"unhelpful refusal: {e}")
        else:
            raise AssertionError(f"revisit_ratio={bad} was accepted")
    for bad in (0, -1):
        try:
            PrimitiveCacheThrashDetector(min_revisits=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"min_revisits={bad} was accepted")
    # A stricter threshold is honoured: the same data that warned at 1.5 does
    # not at 3.0, and does at 1.1.
    obs = _mixed(0.87, 1.3, pairs=4)
    check(not _feed(PrimitiveCacheThrashDetector(revisit_ratio=3.0), obs),
          "a 3.0 threshold fired on a 1.5x ratio")
    check(_feed(PrimitiveCacheThrashDetector(revisit_ratio=1.1), obs),
          "a 1.1 threshold did not fire on a 1.5x ratio")
    print("    PASS")


def check_both_monitoring_phases_feed_it():
    print("[both trainer routes feed the detector, and both measure the step "
          "duration themselves -- a route that does not would be silently "
          "unmonitored]")
    import inspect
    from nodes.train import managed, step_pipeline
    for mod in (step_pipeline, managed):
        src = inspect.getsource(mod)
        check("cache_thrash" in src or "PrimitiveCacheThrashDetector" in src,
              f"{mod.__name__} does not reference the detector")
        mon = inspect.getsource(getattr(mod, "MonitoringPhase"))
        check("observe(" in mon and "shape" in mon,
              f"{mod.__name__}.MonitoringPhase must call detector.observe() "
              f"with the step's shape and duration")
        # The print goes through the shared formatter, so the warning is
        # prefixed and cannot be traced to nothing in an interleaved log.
        check("format_warning" in mon or "cache_thrash" in mon,
              f"{mod.__name__}.MonitoringPhase must use the shared formatter")
    print("    PASS")


def check_report_is_available_whether_or_not_it_warned():
    print("[report() is always populated, so 'it did not warn' is "
          "distinguishable from 'it was not looking']")
    d = PrimitiveCacheThrashDetector()
    check(d.report()["distinct_shapes"] == 0, "empty detector must report")
    _feed(d, [("64x64", 0.9), ("96x64", 0.9)])
    rep = d.report()
    for key in ("distinct_shapes", "first", "repeat", "revisit",
                "repeat_median_sec", "revisit_median_sec", "warned"):
        check(key in rep, f"report() is missing {key}")
    check(rep["warned"] is False, "this run should not have warned")
    check(rep["revisit_ratio"] is None,
          f"no revisits means no ratio, not a ratio of 1: {rep}")
    # And the formatter prefixes, or returns None.
    check(format_warning(d) is None, "format_warning must return None if unwarned")
    warned = PrimitiveCacheThrashDetector()
    _feed(warned, _mixed(0.87, 3.8, pairs=5))
    text = format_warning(warned)
    check(text is not None and "PrimitiveCacheThrash" in text,
          f"a warning must carry its source prefix, got {text!r}")
    print("    PASS")


def main():
    check_thrash_is_detected_and_names_the_knob()
    check_it_warns_only_once()
    check_a_healthy_run_is_silent()
    check_a_single_shape_dataset_is_silent()
    check_only_slow_first_sightings_is_a_different_problem()
    check_too_few_revisits_does_not_warn()
    check_no_repeat_baseline_means_no_warning()
    check_unusable_observations_are_ignored()
    check_the_first_shape_is_a_sighting()
    check_thresholds_are_validated()
    check_both_monitoring_phases_feed_it()
    check_report_is_available_whether_or_not_it_warned()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
