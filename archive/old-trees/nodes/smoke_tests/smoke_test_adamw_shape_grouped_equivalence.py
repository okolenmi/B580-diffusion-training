"""Numerical equivalence check: AdamWAlgorithm.compute_update_batched()
(via ShapeGroupedBatchStrategy) vs. the per-parameter reference
(SimpleLoopStrategy), with real same-shape groups.

smoke_test_adamw_equivalence.py already runs shape_grouped as one of
_STRATEGIES, but its own two parameters (30x20 and a 50-vector) are
different shapes -- ShapeGroupedBatchStrategy correctly falls back to
the per-member path for a singleton group, so that test never actually
exercises compute_update_batched()'s real batched code. This test uses
real groups of 2+ identical-shape parameters specifically to close that
gap, mirroring smoke_test_adafactor_shape_grouped_equivalence.py's
structure.

Run this directly:
`python nodes/smoke_tests/smoke_test_adamw_shape_grouped_equivalence.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle, LoRAPlusGroups
from nodes.optimizer.strategies.shape_grouped import ShapeGroupedBatchStrategy
from nodes.optimizer.strategies.simple import SimpleLoopStrategy

DEVICE = "cpu"

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def run_grouped_case(shapes, dtype, weight_decay=0.0, n_steps=30) -> tuple[bool, float]:
    torch.manual_seed(11)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in shapes]

    params_simple = [w.clone().requires_grad_(True) for w in inits]
    handle_simple = ComposedOptimizerHandle(
        algorithm=AdamWAlgorithm(weight_decay=weight_decay), strategy=SimpleLoopStrategy(),
        params=params_simple, lr=0.01, device=DEVICE,
    )

    params_grouped = [w.clone().requires_grad_(True) for w in inits]
    handle_grouped = ComposedOptimizerHandle(
        algorithm=AdamWAlgorithm(weight_decay=weight_decay), strategy=ShapeGroupedBatchStrategy(),
        params=params_grouped, lr=0.01, device=DEVICE,
    )

    for step in range(n_steps):
        torch.manual_seed(2000 + step)
        grads = [(torch.randn(s) * 0.05).to(dtype) for s in shapes]

        for p, g in zip(params_simple, grads):
            p.grad = g.clone()
        handle_simple.step()
        handle_simple.zero_grad()

        for p, g in zip(params_grouped, grads):
            p.grad = g.clone()
        handle_grouped.step()
        handle_grouped.zero_grad()

    max_diff = max(
        (a.detach().float() - b.detach().float()).abs().max().item()
        for a, b in zip(params_simple, params_grouped)
    )
    ok = all(torch.allclose(a.detach(), b.detach(), rtol=1e-4, atol=1e-6)
             for a, b in zip(params_simple, params_grouped))
    return ok, max_diff


def check_lr_aware_grouping():
    print("\n=== LoRAPlusGroups (real policy, not mocked) still separates groups "
          "correctly with the batched strategy ===")
    torch.manual_seed(3)
    shape = (10, 16)
    w_a = torch.randn(shape) * 0.1
    w_b = w_a.clone()
    params = [w_a.clone().requires_grad_(True), w_b.clone().requires_grad_(True)]
    is_b_matrix = lambda p: p is params[1]
    handle = ComposedOptimizerHandle(
        algorithm=AdamWAlgorithm(), strategy=ShapeGroupedBatchStrategy(),
        params=params, lr=0.01, device=DEVICE,
        group_policy=LoRAPlusGroups(is_b_matrix=is_b_matrix, ratio=16.0),
    )
    for step in range(15):
        torch.manual_seed(500 + step)
        g = torch.randn(shape) * 0.05
        params[0].grad = g.clone()
        params[1].grad = g.clone()
        handle.step()
        handle.zero_grad()
    moved_a = (params[0].detach() - w_a).abs().sum().item()
    moved_b = (params[1].detach() - w_b).abs().sum().item()
    record(moved_b > moved_a, "the 16x-lr group (B matrix) moved further than the 1x group",
           detail=f"moved_a={moved_a:.4f} moved_b={moved_b:.4f}")


def main():
    print(f"Device: {DEVICE} (equivalence check -- pure numerical comparison, "
          f"real hardware not required)")

    for shapes, label in [
        ([(12, 20), (12, 20), (12, 20)], "group of 3, identical shape"),
        ([(12, 20)] * 15, "group of 15, identical shape"),
        ([(20, 12), (20, 12)], "group of 2, transposed shape"),
        ([(12, 20), (7, 33), (12, 20)], "mixed: group of 2 + one singleton shape"),
        ([(64,), (64,), (64,)], "group of 3, 1D"),
        ([(9, 5)], "single unpaired shape (singleton fallback)"),
    ]:
        for dtype in (torch.float32, torch.bfloat16):
            ok, diff = run_grouped_case(shapes, dtype)
            record(ok, f"{label}, dtype={dtype}", detail=f"max_diff={diff:.3e}")

    ok, diff = run_grouped_case([(12, 20), (12, 20)], torch.float32, weight_decay=0.01)
    record(ok, "group of 2 with weight_decay=0.01", detail=f"max_diff={diff:.3e}")

    check_lr_aware_grouping()

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
