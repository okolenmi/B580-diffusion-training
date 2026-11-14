"""Checks the oneDNN primitive-cache sizing in nodes/xpu_env.py.

Sizing a cache to a dataset is the kind of function that looks trivially
correct and is wrong in three ways that only show up on hardware: sized to the
optimistic end of a measurement, silently answering for a dataset whose shape
count is unknown, and sized for one model while being applied to another. Each
of those has a check here.

The measured basis is in the module: 44 distinct shapes need a capacity in
(1280, 1536], i.e. 29.1 to 34.9 primitives per shape, and peak host RSS was
byte-identical from capacity 1024 to 262144 -- so capacity is a ceiling, unused
entries cost nothing, and headroom is free.

What is deliberately NOT checked: that the ratio is right. That is a hardware
measurement (docs/known-issues/open.md), not something a unit test can assert
without re-measuring the card inside a test. What this pins is that the
*function* uses the wide end of the bracket and that it says "unknown" when it
does not know.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes import xpu_env
from nodes.xpu_env import (
    DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
    PRIMITIVES_PER_SHAPE_MEASURED,
    SHAPES_WHERE_GROUPING_WINS,
    primitive_cache_capacity_for_shapes,
)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_the_measured_bracket_is_still_what_the_comment_claims():
    print("[the 44-shape measurement is bracketed, and the bracket is what the "
          "function sizes from]")
    lo, hi = PRIMITIVES_PER_SHAPE_MEASURED
    check(lo < hi, f"the bracket collapsed: {PRIMITIVES_PER_SHAPE_MEASURED}")
    # 44 shapes measured to need (1280, 1536]; 44 * 35 = 1540, which must cover
    # the bracket's top or the sizing is on the wrong side of the measurement.
    check(44 * hi >= 1536,
          f"44 shapes at {hi}/shape = {44 * hi}, which does not cover the "
          f"measured upper bound of 1536")
    check(44 * lo < 1536,
          f"44 shapes at {lo}/shape = {44 * lo} already exceeds the measured "
          f"upper bound -- the bracket is not tight enough to be a measurement")
    print(f"    {lo}-{hi} primitives per shape; 44 shapes -> "
          f"{44 * lo}..{44 * hi}")
    print("    PASS")


def check_sizing_is_from_the_wide_end_not_the_narrow_one():
    print("[sizing uses the WIDE end of the bracket, so a dataset is never sized "
          "to the optimistic end of a 1.2x-wide measurement]")
    for n in (44, 128, 512, 4096):
        got = primitive_cache_capacity_for_shapes(n)
        want = max(DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
                   n * PRIMITIVES_PER_SHAPE_MEASURED[1])
        check(got == want, f"{n} shapes -> {got}, expected {want}")
    print("    PASS")


def check_a_measured_dataset_is_covered():
    print("[`non-square` (44 shapes) gets a capacity that covers its measured "
          "requirement, and the default already does]")
    got = primitive_cache_capacity_for_shapes(44)
    check(got >= 1536,
          f"44 shapes sized to {got}, below the measured upper bound 1536")
    check(got == DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
          f"44 shapes sized to {got} but the default is "
          f"{DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY} -- one of these two is "
          f"wrong and they should agree for the dataset that was measured")
    print("    PASS")


def check_an_unknown_shape_count_is_not_a_number():
    print("[an unknown shape count returns None, not a default -- 'we do not "
          "know how many shapes this dataset has' is a different fact from "
          "'this dataset has few shapes' and must not be answered with one]")
    for value in (None, 0, -1, -100):
        check(primitive_cache_capacity_for_shapes(value) is None,
              f"{value!r} should be unknown, got "
              f"{primitive_cache_capacity_for_shapes(value)!r}")
    print("    PASS")


def check_a_non_shape_count_type_is_unknown_not_an_error():
    print("[a non-integer shape count (bool, float, str) is unknown rather than "
          "an exception or a silent number -- True is an int in Python and must "
          "not size a cache to one shape]")
    for value in (True, False, 3.5, 44.0, "44", [44], {"n": 44}):
        got = primitive_cache_capacity_for_shapes(value)
        check(got is None, f"{value!r} should be unknown, got {got!r}")
    print("    PASS")


def check_the_default_is_never_smaller_than_the_formula_for_small_datasets():
    print("[the function never returns less than the default, so a one-shape "
          "dataset does not get a smaller cache than the measured baseline]")
    for n in (1, 2, 10, 43, 44):
        got = primitive_cache_capacity_for_shapes(n)
        check(got >= DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
              f"{n} shapes -> {got}, below the default "
              f"{DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY}")
    print("    PASS")


def check_the_growth_is_linear_in_shape_count():
    print("[doubling the shape count doubles the requirement once past the "
          "default's headroom -- the relationship the 4096-shape case relies on]")
    base = primitive_cache_capacity_for_shapes(512)
    dbl = primitive_cache_capacity_for_shapes(1024)
    # Past the default the function is exactly n * per-shape, so the scaling is
    # plain doubling -- no default offset to carry. Asserted where the default
    # is already out of the way, which is the regime that matters for large
    # shape counts.
    check(base > DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
          f"512 shapes should be past the default headroom, got {base}")
    check(dbl == 2 * base,
          f"512 -> {base}, 1024 -> {dbl}: not exactly double")
    print(f"    512 -> {base}, 1024 -> {dbl}")
    print("    PASS")


def check_the_grouping_threshold_is_a_shape_count_not_a_capacity():
    print("[the grouping recommendation is expressed in shapes, because that is "
          "what a user can act on -- and it names a threshold rather than "
          "leaving 'a lot' in prose]")
    check(SHAPES_WHERE_GROUPING_WINS > 0, "must be positive")
    check(isinstance(SHAPES_WHERE_GROUPING_WINS, int), "must be an int")
    # At the threshold, grouping 44->3 shapes is the cheaper lever. The
    # constant is a recommendation boundary, not a cliff, so this only pins
    # that it sits above the dataset that was measured -- otherwise the
    # measured case would be told to group.
    check(SHAPES_WHERE_GROUPING_WINS >= 44,
          f"the threshold {SHAPES_WHERE_GROUPING_WINS} is below the 44-shape "
          f"dataset whose cost was actually measured, so it would recommend "
          f"grouping for a case that is known to be fine")
    print(f"    threshold: {SHAPES_WHERE_GROUPING_WINS} distinct shapes")
    print("    PASS")


def check_the_dataset_node_triggers_the_sizing():
    print("[ManagedDatasetSourceNode.build() sizes the cache to its dataset -- "
          "the trigger, because it is the first node that knows the shape count]")
    import os
    from nodes import xpu_env
    # The graph child calls set_xpu_perf_env_vars() before torch, so by the
    # time a dataset node builds the variable already holds the default. This
    # check reproduces that order rather than starting from a clean env.
    os.environ.pop("ONEDNN_PRIMITIVE_CACHE_CAPACITY", None)
    xpu_env._capacity_was_explicit = False
    xpu_env.set_xpu_perf_env_vars()
    before = os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]

    from nodes.dataset.managed import ManagedDatasetSourceNode
    out = ManagedDatasetSourceNode(None).build(
        dataset_root="non-square", batch_size=2, shuffle=True)
    after = os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]

    # The count the node uses over-counts (63 rows vs 44 trained shapes), which
    # is the safe direction: capacity is free, an under-count is not.
    check(int(after) > int(before),
          f"the dataset node did not raise the capacity: {before} -> {after}")
    check(int(after) == 2205,
          f"expected 63 shapes x 35 = 2205, got {after}")
    # And the node still does its actual job afterwards: a mis-sized cache must
    # not cost the run its batches.
    batch = next(iter(out["batches"]))
    check(batch["x_t"] is not None, "batches stopped iterating")
    print(f"    capacity {before} -> {after}; batches still iterate")
    print("    PASS")


def check_an_explicitly_exported_value_is_not_overridden_by_the_dataset():
    print("[a capacity an operator exported survives the dataset node -- a knob "
          "that cannot be deliberately set smaller is not a knob -- but the "
          "shortfall is logged, because slow revisits look like a bug]")
    import io
    import logging
    import os
    from nodes import xpu_env
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "2048"
    try:
        xpu_env._capacity_was_explicit = False
        xpu_env.set_xpu_perf_env_vars()  # sees the export, marks it explicit
        check(xpu_env._capacity_was_explicit,
              "an exported value must be recorded as explicit, or the dataset "
              "node cannot tell an operator's choice from our default")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("nodes.xpu_env")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.WARNING)
        try:
            from nodes.dataset.managed import ManagedDatasetSourceNode
            ManagedDatasetSourceNode(None).build(
                dataset_root="non-square", batch_size=2, shuffle=True)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

        check(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] == "2048",
              f"the dataset node overrode an explicit value: "
              f"{os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY']}")
        check("distinct latent shapes need" in stream.getvalue(),
              f"the shortfall must be logged, got: {stream.getvalue()!r}")
    finally:
        os.environ.pop("ONEDNN_PRIMITIVE_CACHE_CAPACITY", None)
        xpu_env._capacity_was_explicit = False
    print("    PASS")


def check_an_unknown_shape_count_leaves_the_default_alone():
    print("[apply_primitive_cache_capacity_for_shapes(None) changes nothing and "
          "reports nothing applied]")
    import os
    from nodes import xpu_env
    os.environ.pop("ONEDNN_PRIMITIVE_CACHE_CAPACITY", None)
    xpu_env._capacity_was_explicit = False
    xpu_env.set_xpu_perf_env_vars()
    before = os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]
    got = xpu_env.apply_primitive_cache_capacity_for_shapes(None)
    check(got is None, f"None shape count should report nothing applied, got {got}")
    check(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] == before,
          "an unknown shape count must not change the capacity")
    print("    PASS")


def main():
    check_the_measured_bracket_is_still_what_the_comment_claims()
    check_sizing_is_from_the_wide_end_not_the_narrow_one()
    check_a_measured_dataset_is_covered()
    check_an_unknown_shape_count_is_not_a_number()
    check_a_non_shape_count_type_is_unknown_not_an_error()
    check_the_default_is_never_smaller_than_the_formula_for_small_datasets()
    check_the_growth_is_linear_in_shape_count()
    check_the_grouping_threshold_is_a_shape_count_not_a_capacity()
    check_the_dataset_node_triggers_the_sizing()
    check_an_explicitly_exported_value_is_not_overridden_by_the_dataset()
    check_an_unknown_shape_count_leaves_the_default_alone()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
