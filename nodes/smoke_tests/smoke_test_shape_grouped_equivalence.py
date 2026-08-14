"""Numerical equivalence check: ShapeGroupedBatchStrategy vs. SimpleLoopStrategy.

Run this directly:
`python nodes/smoke_tests/smoke_test_shape_grouped_equivalence.py`

Unlike smoke_test_foreach_strategy_equivalence.py, this is NOT a bit-exact
(torch.equal()) check -- ShapeGroupedBatchStrategy genuinely restructures
CAMEAlgorithm's math (adds a batch axis to every reduction, replaces the
host-sync clip with a device-tensor torch.clamp()), so a small amount of
floating-point reduction-order difference is expected (summing k members'
worth of numbers in a different order than k separate per-member sums
isn't guaranteed bit-identical). This test checks that difference stays
within a tolerance sized to that expectation using real torch tensors on
real hardware, not loosened further to make a real divergence pass
silently, and not tightened to demand bit-exactness this restructuring
was never expected to have.

Three things this specifically needs to prove, each with its own case
below:
1. Basic correctness -- a group of same-shape parameters, batched math
   matches the per-parameter reference, across many steps (state
   evolution staying synchronized, not just one step's output).
2. The size-1 fallback path (a shape with no siblings in the parameter
   set) still produces correct results -- this is the untested-by-the-
   numpy-check code path, since that check only ever exercised the
   batched branch directly.
3. The lr-aware grouping key actually does its job: two same-shaped
   parameters given genuinely different effective lr (via
   ComposedOptimizerHandle's real LoRAPlusGroups policy, not a mock)
   must NOT get silently merged into one group and given the wrong lr --
   verified by confirming their two final values differ from each other
   in the direction their respective lr predicts, not just that the
   strategy runs without crashing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle, LoRAPlusGroups
from nodes.optimizer.strategies.simple import SimpleLoopStrategy
from nodes.optimizer.strategies.shape_grouped import ShapeGroupedBatchStrategy

DEVICE = "cpu"
# float32 reduction-order noise, sized off the numpy pre-check's own
# measured ~7e-7 max relative difference -- generous enough to not fail
# on harmless reordering, tight enough that a real algebra bug (wrong
# axis, wrong broadcast) would still fail it by orders of magnitude.
_RTOL, _ATOL = 1e-4, 1e-6


def _came():
    return CAMEAlgorithm(weight_decay=0.01)


def run_grouped_case(shapes, dtype, n_steps: int = 30) -> tuple[bool, float]:
    torch.manual_seed(11)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in shapes]

    params_simple = [w.clone().requires_grad_(True) for w in inits]
    handle_simple = ComposedOptimizerHandle(
        algorithm=_came(), strategy=SimpleLoopStrategy(),
        params=params_simple, lr=0.02, device=DEVICE,
    )

    params_grouped = [w.clone().requires_grad_(True) for w in inits]
    handle_grouped = ComposedOptimizerHandle(
        algorithm=_came(), strategy=ShapeGroupedBatchStrategy(),
        params=params_grouped, lr=0.02, device=DEVICE,
    )

    max_rel = 0.0
    for step in range(n_steps):
        torch.manual_seed(900 + step)
        grads = [(torch.randn(s) * 0.05).to(dtype) for s in shapes]

        for p, g in zip(params_simple, grads):
            p.grad = g.clone()
        handle_simple.step()
        handle_simple.zero_grad()

        for p, g in zip(params_grouped, grads):
            p.grad = g.clone()
        handle_grouped.step()
        handle_grouped.zero_grad()

        for a, b in zip(params_simple, params_grouped):
            diff = (a.detach() - b.detach()).abs()
            denom = a.detach().abs() + 1e-8
            max_rel = max(max_rel, (diff / denom).max().item())

    ok = all(torch.allclose(a.detach(), b.detach(), rtol=_RTOL, atol=_ATOL)
             for a, b in zip(params_simple, params_grouped))
    return ok, max_rel


def run_lr_aware_grouping_case(dtype, n_steps: int = 20) -> bool:
    """Two same-shaped params, one flagged as a "B matrix" -> 16x lr via
    LoRAPlusGroups (real policy from composed.py, not mocked). If the
    grouping key ignored lr, both would get batched under one shared
    scalar lr and end up with identical-magnitude updates despite the
    16x lr difference -- this checks they don't."""
    torch.manual_seed(3)
    shape = (12, 20)
    w_a = (torch.randn(shape) * 0.1)
    w_b = w_a.clone()  # identical init -- any divergence is purely from lr

    params = [w_a.clone().requires_grad_(True), w_b.clone().requires_grad_(True)]
    is_b_matrix = lambda p: p is params[1]
    handle = ComposedOptimizerHandle(
        algorithm=_came(), strategy=ShapeGroupedBatchStrategy(),
        params=params, lr=0.02, device=DEVICE,
        group_policy=LoRAPlusGroups(is_b_matrix=is_b_matrix, ratio=16.0),
    )

    for step in range(n_steps):
        torch.manual_seed(700 + step)
        g = torch.randn(shape) * 0.05
        params[0].grad = g.clone()
        params[1].grad = g.clone()  # identical grad each step too
        handle.step()
        handle.zero_grad()

    # Identical init + identical grads every step + only lr differs ->
    # the two params must have moved by different amounts. If the
    # grouping key had ignored lr, they'd be identical here instead.
    moved_a = (params[0].detach() - w_a).abs().sum().item()
    moved_b = (params[1].detach() - w_b).abs().sum().item()
    diverged = abs(moved_a - moved_b) > 1e-6
    return diverged and moved_b > moved_a  # 16x lr -> moved further


def main():
    print(f"Device: {DEVICE} (float32/bf16 tolerance check -- real hardware "
          f"not required for this part; real-hardware validation is separate, "
          f"still needed, see composed_came.py's strategy port doc)")
    failures = []

    print("\n[1] Basic correctness -- grouped batching vs. per-parameter reference")
    for shapes, label in [
        ([(12, 20), (12, 20), (12, 20)], "group of 3, identical shape"),
        ([(12, 20)] * 20, "group of 20, identical shape"),
        ([(20, 12), (20, 12)], "group of 2, transposed shape"),
        ([(12, 20), (7, 33), (12, 20)], "mixed: group of 2 + one singleton shape"),
    ]:
        for dtype in (torch.float32, torch.bfloat16):
            ok, max_rel = run_grouped_case(shapes, dtype)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: {label}, dtype={dtype}, max_rel_diff={max_rel:.2e}")
            if not ok:
                failures.append(f"{label}, dtype={dtype}: max_rel_diff={max_rel:.2e} exceeds tolerance")

    print("\n[2] Size-1 fallback path (a shape with no siblings)")
    ok, max_rel = run_grouped_case([(9, 5)], torch.float32, n_steps=15)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: single unpaired shape, max_rel_diff={max_rel:.2e}")
    if not ok:
        failures.append(f"size-1 fallback: max_rel_diff={max_rel:.2e} exceeds tolerance")

    print("\n[3] lr-aware grouping key (LoRAPlusGroups, real policy, not mocked)")
    for dtype in (torch.float32,):
        ok = run_lr_aware_grouping_case(dtype)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: dtype={dtype} -- two same-shape params at 1x/16x lr diverged as expected")
        if not ok:
            failures.append(f"lr-aware grouping, dtype={dtype}: params did NOT diverge -- "
                             f"grouping key may be ignoring lr and silently sharing one scalar")

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")
        print("\nNOTE: this confirms correctness on CPU only. Real-hardware "
              "(XPU) validation -- and the actual speed measurement this "
              "strategy exists for -- is separate and still needed before "
              "promoting shape_grouped to any node's default.")


if __name__ == "__main__":
    main()
