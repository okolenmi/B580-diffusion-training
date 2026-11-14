"""Numerical equivalence check: ShapeGroupedForeachStrategy vs.
SimpleLoopStrategy, across all three Algorithms.

ShapeGroupedForeachStrategy's own body is just wiring together two
pieces each already separately bit-exact-verified elsewhere
(shape_grouping.py's compute_grouped_updates(), exercised by
smoke_test_adafactor_shape_grouped_equivalence.py/
smoke_test_adamw_shape_grouped_equivalence.py/
smoke_test_shape_grouped_equivalence.py; base.py's
apply_updates_batched(), exercised by
smoke_test_foreach_strategy_equivalence.py) -- so the real thing this
test needs to prove is that the *wiring* is correct, not that either
piece's own math is (already covered elsewhere). Checked with
torch.equal(), a bit-exact match, matching both of the strategies this
combines rather than falling back to a tolerance.

Run this directly:
`python nodes/smoke_tests/smoke_test_shape_grouped_foreach_equivalence.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle, LoRAPlusGroups
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.shape_grouped_foreach import ShapeGroupedForeachStrategy

DEVICE = "cpu"
_ALGORITHM_FACTORIES = {
    "adamw": lambda: AdamWAlgorithm(weight_decay=1e-2),
    "adafactor_scale_false": lambda: AdafactorAlgorithm(weight_decay=0.5, scale_parameter=False),
    "came": lambda: CAMEAlgorithm(weight_decay=0.01),
}
# A real mix: two groups of same-shape parameters (2D and 1D) plus one
# unpaired shape each -- exercises both the batched-group path and the
# singleton fallback in the same run, across two (device, dtype) buckets
# once dtype varies too.
_SHAPES = [(12, 20), (12, 20), (12, 20), (64,), (64,), (7, 33), (31,)]

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def run_case(algorithm_name: str, dtype, n_steps: int = 25) -> bool:
    torch.manual_seed(7)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in _SHAPES]

    params_simple = [w.clone().requires_grad_(True) for w in inits]
    handle_simple = ComposedOptimizerHandle(
        algorithm=_ALGORITHM_FACTORIES[algorithm_name](), strategy=SimpleLoopStrategy(),
        params=params_simple, lr=0.01, device=DEVICE,
    )

    params_grouped = [w.clone().requires_grad_(True) for w in inits]
    handle_grouped = ComposedOptimizerHandle(
        algorithm=_ALGORITHM_FACTORIES[algorithm_name](), strategy=ShapeGroupedForeachStrategy(),
        params=params_grouped, lr=0.01, device=DEVICE,
    )

    for step in range(n_steps):
        torch.manual_seed(3000 + step)
        grads = [(torch.randn(s) * 0.05).to(dtype) for s in _SHAPES]

        for p, g in zip(params_simple, grads):
            p.grad = g.clone()
        handle_simple.step()
        handle_simple.zero_grad()

        for p, g in zip(params_grouped, grads):
            p.grad = g.clone()
        handle_grouped.step()
        handle_grouped.zero_grad()

    return all(torch.equal(a.detach(), b.detach())
               for a, b in zip(params_simple, params_grouped))


def check_lr_aware_grouping():
    print("\n=== LoRAPlusGroups (real policy) still separates groups correctly "
          "with both axes batched ===")
    torch.manual_seed(9)
    shape = (10, 16)
    w_a = torch.randn(shape) * 0.1
    w_b = w_a.clone()
    params = [w_a.clone().requires_grad_(True), w_b.clone().requires_grad_(True)]
    is_b_matrix = lambda p: p is params[1]
    handle = ComposedOptimizerHandle(
        algorithm=AdamWAlgorithm(), strategy=ShapeGroupedForeachStrategy(),
        params=params, lr=0.01, device=DEVICE,
        group_policy=LoRAPlusGroups(is_b_matrix=is_b_matrix, ratio=16.0),
    )
    for step in range(15):
        torch.manual_seed(600 + step)
        g = torch.randn(shape) * 0.05
        params[0].grad = g.clone()
        params[1].grad = g.clone()
        handle.step()
        handle.zero_grad()
    moved_a = (params[0].detach() - w_a).abs().sum().item()
    moved_b = (params[1].detach() - w_b).abs().sum().item()
    record(moved_b > moved_a, "the 16x-lr group (B matrix) moved further than the 1x group",
           detail=f"moved_a={moved_a:.4f} moved_b={moved_b:.4f}")


def check_scale_parameter_true_with_weight_decay_raises_not_corrupts():
    print("\n=== AdafactorAlgorithm(scale_parameter=True, weight_decay!=0) through a "
          "batched strategy: raises clearly instead of silently using the wrong decay "
          "for some group members (the real bug this session found and fixed in "
          "algorithms/base.py's default compute_update_batched()) ===")
    torch.manual_seed(13)
    shape = (12, 20)
    params = [torch.randn(shape).requires_grad_(True) for _ in range(3)]
    handle = ComposedOptimizerHandle(
        algorithm=AdafactorAlgorithm(weight_decay=0.5, scale_parameter=True),
        strategy=ShapeGroupedForeachStrategy(),
        params=params, lr=0.01, device=DEVICE,
    )
    for p in params:
        p.grad = torch.randn(shape) * 0.05
    try:
        handle.step()
        record(False, "should have raised RuntimeError (per-member decay silently "
                       "collapsed to one shared value), but completed without error")
    except RuntimeError as e:
        record("decay varies across this group" in str(e),
               "raised the expected, specific RuntimeError", detail=str(e)[:100])


def main():
    print(f"Device: {DEVICE} (bit-exact equivalence check -- real hardware not required)")

    for algorithm_name in _ALGORITHM_FACTORIES:
        print(f"\n=== algorithm = {algorithm_name!r} ===")
        for dtype in (torch.float32, torch.bfloat16):
            ok = run_case(algorithm_name, dtype)
            record(ok, f"dtype={dtype}, mixed groups + singleton fallback")

    check_lr_aware_grouping()
    check_scale_parameter_true_with_weight_decay_raises_not_corrupts()

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
