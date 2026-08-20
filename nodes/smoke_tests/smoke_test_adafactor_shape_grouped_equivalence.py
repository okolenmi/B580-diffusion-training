"""Numerical equivalence check: AdafactorAlgorithm.compute_update_batched()
(via ShapeGroupedBatchStrategy) vs. the per-parameter reference
(SimpleLoopStrategy).

Mirrors smoke_test_shape_grouped_equivalence.py's CAME test closely --
same tolerance reasoning (a small, bounded floating-point difference from
reduction-order restructuring is expected, not a bug), same three real
things to prove. Two Adafactor-specific additions this test needs that
the CAME one doesn't:

1. **beta1 (momentum) on and off** -- Adafactor's momentum is optional
   (CAME's isn't), and the batched override's exp_avg handling only runs
   when beta1 is set. Both must be checked.
2. **scale_parameter=True falls back to the per-member default, and
   still produces correct results.** compute_update_batched() explicitly
   does NOT batch this case (see its own docstring for why -- alpha_t,
   and therefore decay, would vary per group member). The fallback is
   Algorithm.compute_update_batched()'s own default (loop + stack) --
   this needs its own check that the fallback path is actually taken
   and is itself correct, not just that scale_parameter=False works.

Run this directly:
`python nodes/smoke_tests/smoke_test_adafactor_shape_grouped_equivalence.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle, LoRAPlusGroups
from nodes.optimizer.strategies.shape_grouped import ShapeGroupedBatchStrategy
from nodes.optimizer.strategies.simple import SimpleLoopStrategy

DEVICE = "cpu"
# Same reasoning and same magnitude as smoke_test_shape_grouped_equivalence.py's
# CAME tolerance -- float32 reduction-order noise, not a correctness bug.
_RTOL, _ATOL = 1e-4, 1e-6

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _adafactor(scale_parameter=False, beta1=None, weight_decay=0.0):
    return AdafactorAlgorithm(scale_parameter=scale_parameter, beta1=beta1,
                               weight_decay=weight_decay)


def run_grouped_case(shapes, dtype, algorithm_factory, n_steps: int = 30) -> tuple[bool, float]:
    torch.manual_seed(21)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in shapes]

    params_simple = [w.clone().requires_grad_(True) for w in inits]
    handle_simple = ComposedOptimizerHandle(
        algorithm=algorithm_factory(), strategy=SimpleLoopStrategy(),
        params=params_simple, lr=0.02, device=DEVICE,
    )

    params_grouped = [w.clone().requires_grad_(True) for w in inits]
    handle_grouped = ComposedOptimizerHandle(
        algorithm=algorithm_factory(), strategy=ShapeGroupedBatchStrategy(),
        params=params_grouped, lr=0.02, device=DEVICE,
    )

    max_rel = 0.0
    for step in range(n_steps):
        torch.manual_seed(1900 + step)
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


def run_lr_aware_grouping_case(n_steps: int = 20) -> bool:
    """Same real-policy check as the CAME test -- LoRAPlusGroups, not
    mocked."""
    torch.manual_seed(4)
    shape = (12, 20)
    w_a = torch.randn(shape) * 0.1
    w_b = w_a.clone()

    params = [w_a.clone().requires_grad_(True), w_b.clone().requires_grad_(True)]
    is_b_matrix = lambda p: p is params[1]
    handle = ComposedOptimizerHandle(
        algorithm=_adafactor(), strategy=ShapeGroupedBatchStrategy(),
        params=params, lr=0.02, device=DEVICE,
        group_policy=LoRAPlusGroups(is_b_matrix=is_b_matrix, ratio=16.0),
    )

    for step in range(n_steps):
        torch.manual_seed(800 + step)
        g = torch.randn(shape) * 0.05
        params[0].grad = g.clone()
        params[1].grad = g.clone()
        handle.step()
        handle.zero_grad()

    moved_a = (params[0].detach() - w_a).abs().sum().item()
    moved_b = (params[1].detach() - w_b).abs().sum().item()
    diverged = abs(moved_a - moved_b) > 1e-6
    return diverged and moved_b > moved_a


def check_scale_parameter_true_falls_back_and_is_still_correct():
    print("\n=== scale_parameter=True: NOT batched (falls back to the per-member "
          "default), but the fallback itself is still correct ===")
    shapes = [(12, 20), (12, 20), (12, 20)]
    ok, max_rel = run_grouped_case(shapes, torch.float32,
                                    lambda: _adafactor(scale_parameter=True))
    record(ok, "grouped (fallback) matches per-parameter reference",
           detail=f"max_rel_diff={max_rel:.2e}")

    # Confirm the fallback path is actually what ran, not that
    # compute_update_batched() silently batched something it shouldn't
    # have: patch Algorithm.compute_update_batched (the base default) to
    # record that it was called.
    from nodes.optimizer.algorithms.base import Algorithm
    calls = []
    original = Algorithm.compute_update_batched

    def _recording(self, grad_stack, params, states, lr):
        calls.append(1)
        return original(self, grad_stack, params, states, lr)

    Algorithm.compute_update_batched = _recording
    try:
        params = [torch.randn(12, 20).requires_grad_(True) for _ in range(3)]
        handle = ComposedOptimizerHandle(
            algorithm=_adafactor(scale_parameter=True), strategy=ShapeGroupedBatchStrategy(),
            params=params, lr=0.02, device=DEVICE,
        )
        for p in params:
            p.grad = torch.randn(12, 20) * 0.05
        handle.step()
    finally:
        Algorithm.compute_update_batched = original
    record(len(calls) > 0,
           "the base class's default (fallback) implementation was really invoked",
           detail=f"calls={len(calls)}")


def check_scale_parameter_true_with_weight_decay_raises_not_corrupts():
    print("\n=== scale_parameter=True with weight_decay != 0: the case the check "
          "above never actually covered (it used weight_decay=0.0, where decay is "
          "always None regardless of alpha_t) -- raises clearly instead of silently "
          "using the wrong decay for some group members ===")
    torch.manual_seed(17)
    shapes = [(12, 20), (12, 20), (12, 20)]
    params = [torch.randn(s).requires_grad_(True) for s in shapes]
    handle = ComposedOptimizerHandle(
        algorithm=_adafactor(scale_parameter=True, weight_decay=0.5),
        strategy=ShapeGroupedBatchStrategy(),
        params=params, lr=0.02, device=DEVICE,
    )
    for p in params:
        p.grad = torch.randn(12, 20) * 0.05
    try:
        handle.step()
        record(False, "should have raised RuntimeError (per-member decay silently "
                       "collapsed to one shared value), but completed without error")
    except RuntimeError as e:
        record("decay varies across this group" in str(e),
               "raised the expected, specific RuntimeError", detail=str(e)[:100])


def main():
    print(f"Device: {DEVICE} (float32/bf16 tolerance check -- real hardware "
          f"not required for this part)")

    print("\n[1] Basic correctness, scale_parameter=False (the batched case), "
          "beta1=None -- grouped batching vs. per-parameter reference")
    for shapes, label in [
        ([(12, 20), (12, 20), (12, 20)], "group of 3, identical shape"),
        ([(12, 20)] * 20, "group of 20, identical shape"),
        ([(20, 12), (20, 12)], "group of 2, transposed shape"),
        ([(12, 20), (7, 33), (12, 20)], "mixed: group of 2 + one singleton shape"),
        ([(64,), (64,), (64,)], "group of 3, 1D (non-factored branch)"),
    ]:
        for dtype in (torch.float32, torch.bfloat16):
            ok, max_rel = run_grouped_case(shapes, dtype, lambda: _adafactor())
            record(ok, f"{label}, dtype={dtype}", detail=f"max_rel_diff={max_rel:.2e}")

    print("\n[2] beta1 (momentum) on -- exp_avg state must batch/scatter correctly too")
    for shapes, label in [
        ([(12, 20), (12, 20), (12, 20)], "group of 3, 2D, with momentum"),
        ([(64,), (64,), (64,)], "group of 3, 1D, with momentum"),
    ]:
        ok, max_rel = run_grouped_case(shapes, torch.float32,
                                        lambda: _adafactor(beta1=0.9))
        record(ok, f"{label}", detail=f"max_rel_diff={max_rel:.2e}")

    print("\n[3] weight_decay != 0 -- decay stays group-uniform as expected")
    ok, max_rel = run_grouped_case([(12, 20), (12, 20)], torch.float32,
                                    lambda: _adafactor(weight_decay=0.01))
    record(ok, "group of 2 with weight_decay=0.01", detail=f"max_rel_diff={max_rel:.2e}")

    print("\n[4] Size-1 fallback path (a shape with no siblings)")
    ok, max_rel = run_grouped_case([(9, 5)], torch.float32, lambda: _adafactor(), n_steps=15)
    record(ok, "single unpaired shape", detail=f"max_rel_diff={max_rel:.2e}")

    print("\n[5] lr-aware grouping key (LoRAPlusGroups, real policy, not mocked)")
    ok = run_lr_aware_grouping_case()
    record(ok, "two same-shape params at 1x/16x lr diverged as expected")

    check_scale_parameter_true_falls_back_and_is_still_correct()
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
