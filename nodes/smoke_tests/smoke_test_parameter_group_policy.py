"""Correctness check for ComposedOptimizerHandle's ParameterGroupPolicy
fix (docs/training_pipeline_design.md section 3.4).

The bug being fixed: update_lr() used to unconditionally do
`self.param_lr = [new_lr] * len(self.params)`, silently erasing any
per-parameter ratio set at construction on the very next step. The crux
check here is check_ratio_survives_update_lr() -- everything else is
supporting context.

Run this directly: `python nodes/smoke_tests/smoke_test_parameter_group_policy.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.composed import (ComposedOptimizerHandle, LoRAPlusGroups,
                                       ParameterGroupPolicy, UniformGroups)
from nodes.optimizer.composed_fused import ComposedFusedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy

DEVICE = "cpu"
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _params():
    torch.manual_seed(0)
    return [torch.randn(4, 4, requires_grad=True) for _ in range(3)]


def _algorithm():
    return AdamWAlgorithm(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)


def check_uniform_groups_reproduces_old_behavior():
    print("\n=== UniformGroups (default) reproduces today's exact [lr]*len(params) ===")
    params = _params()
    handle = ComposedOptimizerHandle(_algorithm(), SimpleLoopStrategy(), params,
                                      lr=1e-3, device=DEVICE)
    record(handle.param_lr == [1e-3] * len(params),
           "param_lr == [lr]*len(params) at construction, no group_policy given",
           detail=str(handle.param_lr))
    handle.update_lr(5e-4)
    record(handle.param_lr == [5e-4] * len(params),
           "param_lr == [new_lr]*len(params) after update_lr(), no group_policy given",
           detail=str(handle.param_lr))
    record(UniformGroups().group_ratios(params) == [1.0] * len(params),
           "UniformGroups().group_ratios() is all 1.0")


def check_ratio_survives_update_lr():
    """The actual bug. Before this fix: param_lr collapsed to a single
    uniform value on the very next update_lr() call, silently discarding
    whatever ratio group_policy set at construction."""
    print("\n=== THE FIX: a per-group ratio survives update_lr(), not just construction ===")
    params = _params()
    is_b_matrix = lambda p: p is params[1]  # noqa: E731 -- mark the middle param as "B"
    handle = ComposedOptimizerHandle(_algorithm(), SimpleLoopStrategy(), params,
                                      lr=1e-3, device=DEVICE,
                                      group_policy=LoRAPlusGroups(is_b_matrix, ratio=16.0))

    expected_at_construction = [1e-3, 1e-3 * 16.0, 1e-3]
    record(handle.param_lr == expected_at_construction,
           "param_lr reflects the 16x ratio at construction",
           detail=str(handle.param_lr))

    handle.update_lr(5e-4)
    expected_after_update = [5e-4, 5e-4 * 16.0, 5e-4]
    record(handle.param_lr == expected_after_update,
           "param_lr STILL reflects the 16x ratio after update_lr() -- this is the fix",
           detail=str(handle.param_lr))

    handle.update_lr(2e-4)
    expected_second_update = [2e-4, 2e-4 * 16.0, 2e-4]
    record(handle.param_lr == expected_second_update,
           "ratio survives a second update_lr() call too (not a one-time fluke)",
           detail=str(handle.param_lr))


def check_fused_variant_threads_group_policy():
    print("\n=== ComposedFusedOptimizerHandle accepts and applies group_policy too ===")
    params = _params()
    is_b_matrix = lambda p: p is params[0]  # noqa: E731
    handle = ComposedFusedOptimizerHandle(
        _algorithm(), params, lr=1e-3, device=DEVICE,
        group_policy=LoRAPlusGroups(is_b_matrix, ratio=8.0))
    expected = [1e-3 * 8.0, 1e-3, 1e-3]
    record(handle.param_lr == expected, "fused variant's param_lr reflects the ratio",
           detail=str(handle.param_lr))
    handle.update_lr(1e-4)
    record(handle.param_lr == [1e-4 * 8.0, 1e-4, 1e-4],
           "fused variant's ratio also survives update_lr()", detail=str(handle.param_lr))


def check_custom_policy_contract():
    print("\n=== ParameterGroupPolicy is a real ABC, group_ratios() is enforced ===")

    class BadPolicy(ParameterGroupPolicy):
        pass

    try:
        BadPolicy()
        ok = False
    except TypeError:
        ok = True
    record(ok, "can't instantiate a ParameterGroupPolicy that doesn't implement group_ratios()")


def main():
    print("Device: cpu")
    check_uniform_groups_reproduces_old_behavior()
    check_ratio_survives_update_lr()
    check_fused_variant_threads_group_policy()
    check_custom_policy_contract()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
