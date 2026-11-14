"""Bit-exact verification: AdafactorAlgorithm's in-place (axis-2 scratch
reuse) code path vs. its original out-of-place path.

Run this directly: `python nodes/smoke_tests/smoke_test_adafactor_inplace_equivalence.py`

See algorithms/adafactor.py's module docstring for the full design
writeup -- this file is the executable form of the claim made there.
Mirrors smoke_test_came_inplace_equivalence.py's structure and
reasoning closely (see that file for the fuller explanation of why
bit-exact, not just close, is the right target here), covering:

  - float32 AND bf16 parameters -- float32 matters specifically because
    that's the dtype where `SimpleLoopStrategy`'s `grad = p.grad.detach()
    .float()` can alias `p.grad` itself (see algorithms/adafactor.py and
    algorithms/came.py's docstrings for the confirmed mechanism).
  - scale_parameter on and off, weight_decay on and off, momentum
    (beta1) on and off -- the full configuration surface, not just the
    default case.
  - factored (2D) and non-factored (1D) parameters, deliberately
    awkward/non-round shapes.
  - A direct check that `p.grad` itself is never mutated by a
    SimpleLoopStrategy step, even with float32 parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.chunked import ChunkedScratchBufferStrategy

DEVICE = "cpu"


def run(strategy_cls, dtype, scale_parameter: bool, weight_decay: float,
        beta1, n_steps: int = 50):
    torch.manual_seed(7)
    W1_init = (torch.randn(37, 53) * 0.15).to(dtype)   # factored, awkward shape
    W2_init = (torch.randn(431) * 0.15).to(dtype)       # non-factored

    W1 = W1_init.clone().requires_grad_(True)
    W2 = W2_init.clone().requires_grad_(True)
    algorithm = AdafactorAlgorithm(scale_parameter=scale_parameter,
                                    weight_decay=weight_decay, beta1=beta1)
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
    configs = [
        (torch.float32, False, 0.0, None),
        (torch.bfloat16, False, 0.0, None),
        (torch.float32, True, 0.0, None),   # scale_parameter=True
        (torch.float32, False, 0.02, None),  # weight_decay != 0
        (torch.float32, False, 0.0, 0.9),   # momentum
        (torch.bfloat16, True, 1.0, 0.9),   # legacy's own default combo, plus momentum
    ]
    for dtype, scale_parameter, wd, beta1 in configs:
        h_simple = run(SimpleLoopStrategy, dtype, scale_parameter, wd, beta1)
        h_chunked = run(ChunkedScratchBufferStrategy, dtype, scale_parameter, wd, beta1)
        first_mismatch = None
        for step, ((s1, s2), (c1, c2)) in enumerate(zip(h_simple, h_chunked)):
            if not (torch.equal(s1, c1) and torch.equal(s2, c2)):
                first_mismatch = step
                break
        ok = first_mismatch is None
        status = "PASS" if ok else f"FAIL (first mismatch at step {first_mismatch})"
        print(f"    {status}: dtype={dtype}, scale_parameter={scale_parameter}, "
              f"weight_decay={wd}, beta1={beta1}")
        if not ok:
            failures.append(f"dtype={dtype}, scale_parameter={scale_parameter}, "
                             f"weight_decay={wd}, beta1={beta1}: diverged at step {first_mismatch}")


def check_grad_not_corrupted(failures: list) -> None:
    print("\n[2] p.grad is never mutated by SimpleLoopStrategy, even with "
          "float32 params (the exact hazard this design avoids):")
    torch.manual_seed(0)
    W = (torch.randn(20, 30, dtype=torch.float32) * 0.1).requires_grad_(True)
    algorithm = AdafactorAlgorithm(scale_parameter=True, beta1=0.9)
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
