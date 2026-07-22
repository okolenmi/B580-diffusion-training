"""Numerical equivalence check: AdafactorAlgorithm vs. the legacy
core.optimizers.ChunkedXPUAdafactor it's a fresh reimplementation of.

Run this directly: `python nodes/smoke_tests/smoke_test_adafactor_equivalence.py`

See algorithms/adafactor.py's module docstring for the full verification
writeup and the real findings this check is built around -- most
importantly:

  - Parameters here are deliberately >= 10,000 elements. Below that, the
    legacy reference routes through its tiny-parameter batching fast
    path -- a different code path entirely, not what AdafactorAlgorithm
    implements (a strategy/batching concern, not algorithm math -- see
    algorithms/base.py's module docstring). A smaller toy size would
    silently compare against the wrong reference code path.
  - scale_parameter=False, weight_decay=0 on the reference side --
    AdafactorAlgorithm's documented scope (see algorithms/adafactor.py).
  - The beta1 (momentum) case is checked with bf16 parameters, matching
    real training usage, not float32 -- float32 params trigger a real,
    separate, previously-undocumented aliasing quirk in the legacy
    reference's momentum handling (see algorithms/adafactor.py's
    docstring and docs/suspicious_findings.md) that has nothing to do
    with whether this port is correct, and would make an otherwise
    apples-to-apples comparison misleading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import ChunkedXPUAdafactor
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.chunked import ChunkedScratchBufferStrategy

DEVICE = "cpu"
_STRATEGIES = {"simple": SimpleLoopStrategy, "chunked": ChunkedScratchBufferStrategy}
# Loose but real: bf16 has ~3 decimal digits of precision, and this compares
# two independently-implemented, sequentially-compounding 40-step recurrences
# -- see this file's module docstring and algorithms/adafactor.py for why a
# non-trivial bf16 tolerance is the honest, expected bound here, not a
# loosened check to force a pass.
_TOLERANCES = {(None, torch.float32): 1e-4, (0.9, torch.bfloat16): 1e-2,
               (None, torch.bfloat16): 1e-2}


def run_case(strategy_name: str, beta1, dtype, n_steps: int = 40) -> float:
    torch.manual_seed(42)
    W1_init = (torch.randn(120, 100) * 0.1).to(dtype)   # factored, 12000 elem
    W2_init = (torch.randn(11000) * 0.1).to(dtype)      # non-factored, 11000 elem

    W1_ref = W1_init.clone().requires_grad_(True)
    W2_ref = W2_init.clone().requires_grad_(True)
    legacy = ChunkedXPUAdafactor(
        params=[W1_ref, W2_ref], lr=0.01, eps=(1e-8, 1e-3),
        clip_threshold=1.0, beta1=beta1, weight_decay=0.0,
        scale_parameter=False, device=DEVICE,
    )

    W1_new = W1_init.clone().requires_grad_(True)
    W2_new = W2_init.clone().requires_grad_(True)
    algorithm = AdafactorAlgorithm(eps=(1e-8, 1e-3), clip_threshold=1.0, beta1=beta1)
    strategy = _STRATEGIES[strategy_name]()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=strategy,
                                      params=[W1_new, W2_new], lr=0.01, device=DEVICE)

    max_diff = 0.0
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        g1 = (torch.randn(120, 100) * 0.05).to(dtype)
        g2 = (torch.randn(11000) * 0.05).to(dtype)

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
        for (beta1, dtype), tol in _TOLERANCES.items():
            diff = run_case(strategy_name, beta1, dtype)
            ok = diff <= tol
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: beta1={beta1}, dtype={dtype}: "
                  f"max abs diff over 40 steps = {diff:.3e} (tolerance {tol:.0e})")
            if not ok:
                failures.append(f"[{strategy_name}] beta1={beta1}, dtype={dtype}: "
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
