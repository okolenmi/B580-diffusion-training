"""Numerical equivalence check: AdamWAlgorithm vs. the legacy
core.optimizers.CPUAdamW it's a fresh reimplementation of.

Run this directly: `python nodes/smoke_tests/smoke_test_adamw_equivalence.py`

Covers weight_decay on/off, float32/bf16, all three ExecutionStrategies
(simple/chunked/foreach), across a 2D and a 1D parameter together (AdamW's
math doesn't branch on shape the way CAME/Adafactor's does, so this
doesn't need the factored/non-factored split those tests use).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import CPUAdamW
from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.composed_adamw import _STRATEGIES

DEVICE = "cpu"
# Each key: (dtype, weight_decay).
_TOLERANCES = {
    (torch.float32, 0.0): 1e-5,
    (torch.float32, 1e-2): 1e-5,
    (torch.bfloat16, 0.0): 1e-2,
    (torch.bfloat16, 1e-2): 1e-2,
}


def run_case(strategy_name: str, dtype, weight_decay: float, n_steps: int = 40) -> float:
    torch.manual_seed(42)
    W1_init = (torch.randn(30, 20) * 0.1).to(dtype)
    W2_init = (torch.randn(50) * 0.1).to(dtype)

    W1_ref = W1_init.clone().requires_grad_(True)
    W2_ref = W2_init.clone().requires_grad_(True)
    legacy = CPUAdamW(params=[W1_ref, W2_ref], lr=0.01, weight_decay=weight_decay)

    W1_new = W1_init.clone().requires_grad_(True)
    W2_new = W2_init.clone().requires_grad_(True)
    algorithm = AdamWAlgorithm(weight_decay=weight_decay)
    strategy = _STRATEGIES[strategy_name]()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=strategy,
                                      params=[W1_new, W2_new], lr=0.01, device=DEVICE)

    max_diff = 0.0
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        g1 = (torch.randn(30, 20) * 0.05).to(dtype)
        g2 = (torch.randn(50) * 0.05).to(dtype)

        W1_ref.grad = g1.clone()
        W2_ref.grad = g2.clone()
        legacy.step()
        legacy.zero_grad()

        W1_new.grad = g1.clone()
        W2_new.grad = g2.clone()
        handle.step()
        handle.zero_grad()

        d1 = (W1_ref.detach().float() - W1_new.detach().float()).abs().max().item()
        d2 = (W2_ref.detach().float() - W2_new.detach().float()).abs().max().item()
        max_diff = max(max_diff, d1, d2)

    return max_diff


def main():
    print(f"Device: {DEVICE} (equivalence check -- pure numerical comparison, "
          f"real hardware not required)")
    failures = []
    for strategy_name in _STRATEGIES:
        print(f"\n=== strategy = {strategy_name!r} ===")
        for (dtype, weight_decay), tol in _TOLERANCES.items():
            diff = run_case(strategy_name, dtype, weight_decay)
            ok = diff <= tol
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: dtype={dtype}, weight_decay={weight_decay}: "
                  f"max abs diff over 40 steps = {diff:.3e} (tolerance {tol:.0e})")
            if not ok:
                failures.append(f"[{strategy_name}] dtype={dtype}, weight_decay={weight_decay}: "
                                 f"diff {diff:.3e} exceeds tolerance {tol:.0e}")

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
