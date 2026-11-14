"""Bit-exact verification: CAMEAlgorithm's in-place (axis-2 scratch reuse)
code path vs. its original out-of-place path.

Run this directly: `python nodes/smoke_tests/smoke_test_came_inplace_equivalence.py`

See algorithms/came.py's module docstring for the full design writeup --
this file is the executable form of the claim made there: the two paths
(`_compute_update_safe`, used whenever `SimpleLoopStrategy` drives
training since it never passes `scratch`; `_compute_update_inplace`,
used by `ChunkedScratchBufferStrategy`, which always does) perform the
identical sequence of elementary floating-point operations, just with
in-place APIs instead of allocating fresh tensors -- so they should
produce bit-exact results, not merely close ones. Checks that directly
with `torch.equal()`, across:

  - float32 AND bf16 parameters. float32 matters specifically because
    that's the dtype where `SimpleLoopStrategy`'s `grad = p.grad.detach()
    .float()` can alias `p.grad` itself (confirmed directly -- see the
    module docstring) -- if the in-place path were ever accidentally
    taken under that strategy, this is where it would show up as
    corrupted training, not just a rounding difference.
  - weight_decay on and off.
  - factored (2D) and non-factored (1D) parameters, deliberately
    awkward/non-round shapes (37x53, 431) rather than convenient round
    numbers, so no accidental shape-alignment coincidence could hide a
    real bug.
  - A direct, surgical check (not just "training still converges") that
    `p.grad` itself is never mutated by a `SimpleLoopStrategy` step, even
    with float32 parameters -- the exact hazard this design was built to
    avoid, checked directly rather than inferred from matching final
    numbers alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.chunked import ChunkedScratchBufferStrategy

DEVICE = "cpu"


def run(strategy_cls, dtype, weight_decay: float, n_steps: int = 50):
    torch.manual_seed(7)
    # Deliberately awkward, non-round shapes -- factored (2D) and
    # non-factored (1D) in the same run.
    W1_init = (torch.randn(37, 53) * 0.15).to(dtype)
    W2_init = (torch.randn(431) * 0.15).to(dtype)

    W1 = W1_init.clone().requires_grad_(True)
    W2 = W2_init.clone().requires_grad_(True)
    algorithm = CAMEAlgorithm(weight_decay=weight_decay)
    strategy = strategy_cls()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=strategy,
                                      params=[W1, W2], lr=0.01, device=DEVICE)

    history = []
    for step in range(n_steps):
        torch.manual_seed(2000 + step)
        g1 = (torch.randn(37, 53) * 0.05).to(dtype)
        g2 = (torch.randn(431) * 0.05).to(dtype)
        W1.grad = g1.clone()
        W2.grad = g2.clone()
        handle.step()
        handle.zero_grad()
        history.append((W1.detach().clone(), W2.detach().clone()))
    return history


def check_bit_exact(failures: list) -> None:
    print("\n[1] Bit-exact equivalence: SimpleLoopStrategy (out-of-place) "
          "vs. ChunkedScratchBufferStrategy (in-place):")
    for dtype in (torch.float32, torch.bfloat16):
        for wd in (0.0, 0.02):
            h_simple = run(SimpleLoopStrategy, dtype, wd)
            h_chunked = run(ChunkedScratchBufferStrategy, dtype, wd)
            first_mismatch = None
            for step, ((s1, s2), (c1, c2)) in enumerate(zip(h_simple, h_chunked)):
                if not (torch.equal(s1, c1) and torch.equal(s2, c2)):
                    first_mismatch = step
                    break
            ok = first_mismatch is None
            status = "PASS" if ok else f"FAIL (first mismatch at step {first_mismatch})"
            print(f"    {status}: dtype={dtype}, weight_decay={wd}")
            if not ok:
                failures.append(f"dtype={dtype}, weight_decay={wd}: "
                                 f"diverged at step {first_mismatch}")


def check_grad_not_corrupted(failures: list) -> None:
    print("\n[2] p.grad is never mutated by SimpleLoopStrategy, even with "
          "float32 params (the exact hazard this design avoids):")
    torch.manual_seed(0)
    W = (torch.randn(20, 30, dtype=torch.float32) * 0.1).requires_grad_(True)
    algorithm = CAMEAlgorithm()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=SimpleLoopStrategy(),
                                      params=[W], lr=0.01, device=DEVICE)

    grad_in = torch.randn(20, 30, dtype=torch.float32) * 0.05
    W.grad = grad_in.clone()
    grad_snapshot = grad_in.clone()
    handle.step()

    ok = torch.equal(W.grad, grad_snapshot)
    print(f"    {'PASS' if ok else 'FAIL'}: p.grad unchanged after step()")
    if not ok:
        failures.append("p.grad was mutated by a SimpleLoopStrategy step -- "
                         "the in-place path is being taken somewhere it shouldn't be")


def main():
    print(f"Device: {DEVICE} (numerical equivalence check -- pure computation, "
          f"real hardware not required)")
    failures: list = []
    check_bit_exact(failures)
    check_grad_not_corrupted(failures)

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
