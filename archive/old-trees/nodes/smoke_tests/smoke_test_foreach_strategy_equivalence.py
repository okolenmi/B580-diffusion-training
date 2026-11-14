"""Numerical equivalence check: ForeachApplyStrategy vs. SimpleLoopStrategy.

Run this directly:
`python nodes/smoke_tests/smoke_test_foreach_strategy_equivalence.py`

ForeachApplyStrategy doesn't restructure any algorithm's math -- it only
batches the mechanical `param.data *= decay; param.data -= delta` step
across parameters that already share (device, dtype) (see
strategies/foreach.py). Every individual foreach_mul_/foreach_sub_ call
performs the identical elementary op, on the identical operands, as
SimpleLoopStrategy's own per-parameter `.mul_()`/`.sub_()` -- so unlike
the bf16-tolerance checks elsewhere in this package, this is checked with
torch.equal(), a bit-exact match, not a tolerance. Run against all three
Algorithms in this package (CAME, Adafactor, AdamW) -- the whole point of
an ExecutionStrategy is that it doesn't need algorithm-specific testing,
so this is what "generic" is actually being asked to prove -- with a
deliberately mixed parameter set (2D + 1D, two different shapes each) so
the (device, dtype) grouping logic is genuinely exercised, not vacuously
correct on a single-parameter list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.foreach import ForeachApplyStrategy

DEVICE = "cpu"
_ALGORITHM_FACTORIES = {
    "adamw": lambda: AdamWAlgorithm(weight_decay=1e-2),
    "adafactor": lambda: AdafactorAlgorithm(weight_decay=0.5),
    "came": lambda: CAMEAlgorithm(weight_decay=0.01),
}
_SHAPES = [(13, 17), (9, 5), (23,), (8,)]


def run_case(algorithm_name: str, dtype, n_steps: int = 25) -> bool:
    torch.manual_seed(7)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in _SHAPES]

    params_simple = [w.clone().requires_grad_(True) for w in inits]
    handle_simple = ComposedOptimizerHandle(
        algorithm=_ALGORITHM_FACTORIES[algorithm_name](), strategy=SimpleLoopStrategy(),
        params=params_simple, lr=0.02, device=DEVICE,
    )

    params_foreach = [w.clone().requires_grad_(True) for w in inits]
    handle_foreach = ComposedOptimizerHandle(
        algorithm=_ALGORITHM_FACTORIES[algorithm_name](), strategy=ForeachApplyStrategy(),
        params=params_foreach, lr=0.02, device=DEVICE,
    )

    for step in range(n_steps):
        torch.manual_seed(500 + step)
        grads = [(torch.randn(s) * 0.05).to(dtype) for s in _SHAPES]

        for p, g in zip(params_simple, grads):
            p.grad = g.clone()
        handle_simple.step()
        handle_simple.zero_grad()

        for p, g in zip(params_foreach, grads):
            p.grad = g.clone()
        handle_foreach.step()
        handle_foreach.zero_grad()

    return all(torch.equal(a.detach(), b.detach())
               for a, b in zip(params_simple, params_foreach))


def main():
    print(f"Device: {DEVICE} (bit-exact equivalence check -- pure numerical "
          f"comparison, real hardware not required)")
    failures = []
    for algorithm_name in _ALGORITHM_FACTORIES:
        for dtype in (torch.float32, torch.bfloat16):
            ok = run_case(algorithm_name, dtype)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: algorithm={algorithm_name}, dtype={dtype}")
            if not ok:
                failures.append(f"algorithm={algorithm_name}, dtype={dtype}: "
                                 f"foreach and simple strategies diverged")

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
