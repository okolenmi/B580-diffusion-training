"""Numerical equivalence check: ComposedFusedOptimizerHandle(AdafactorAlgorithm)
vs. the legacy core.optimizers.FusedXPUAdafactor it's a non-legacy
alternative to.

Run this directly: `python nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`

Unlike this package's other equivalence tests, this one calls real
`.backward()` and lets real backward hooks fire -- simulating gradients
with a hand-set `.grad` (as the other tests do) wouldn't actually exercise
`register_post_accumulate_grad_hook`, which is the entire mechanism under
test here.

**Deliberately uses only parameters with >= 10,000 elements.** Smaller
parameters hit FusedXPUAdafactor's TINY_NUMEL special case (a real
formula difference, not replicated here -- see composed_fused.py and
composed_fused_adafactor.py's module docstrings for why). This test
checks the regime where AdafactorAlgorithm and FusedXPUAdafactor compute
the same thing; it isn't testing the tiny-parameter path, on purpose.

**A second, more interesting divergence was found while writing this
test, and is deliberately NOT papered over with a looser tolerance:**
FusedXPUAdafactor's momentum (`beta1` set) path has a real, pre-existing
bug for float32 parameters. `g = self.exp_avg[i]` aliases the momentum
buffer (no copy); the very next line, `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`,
calls `.to(dtype=p.dtype)` -- which for a float32 parameter (state is
already float32) returns the *same object*, not a copy, since there's
nothing to convert -- and then `.mul_(alpha_t)` mutates it in place. Net
effect: every step, right after using the momentum buffer to compute that
step's update, the buffer is permanently shrunk by `alpha_t` (~lr) as an
unintended side effect -- confirmed directly (see
check_legacy_float32_momentum_bug() below), and confirmed to be
float32-specific: for a bf16 parameter, `.to(dtype=p.dtype)` performs a
real cast, producing a genuine copy, so the aliasing -- and the bug --
doesn't happen (also confirmed directly, same function). AdafactorAlgorithm
doesn't have this bug (its own momentum blend explicitly clones before
any further scaling -- see algorithms/adafactor.py). Replicating it here
would mean deliberately copying a bug into new code, which is exactly
what this session was told not to do -- so the equivalence checks below
compare momentum behavior only where the legacy reference itself isn't
corrupted (bf16), and check_legacy_float32_momentum_bug() /
check_new_momentum_not_corrupted() make the float32 divergence explicit
and understood rather than silently excluded.

What's checked, in order:
1. check_legacy_float32_momentum_bug(): isolates and confirms the bug
   above, directly, in the untouched legacy class.
2. check_new_momentum_not_corrupted(): confirms AdafactorAlgorithm's
   momentum buffer does NOT get this same corruption for float32
   parameters, across several real steps.
3. Single-pass equivalence (sub_steps=1) across weight_decay,
   scale_parameter, and beta1 (momentum) on/off, float32 and bf16 --
   same config surface smoke_test_adafactor_equivalence.py already covers
   for the non-fused path, now proven for the hook-driven path too.
   float32+beta1 is deliberately excluded from this grid, per the above.
4. Multi-pass accumulation (sub_steps=2, this codebase's actual
   conditional+unconditional distillation shape, beta1=None so this is
   purely about the accumulation mechanism, not entangled with the
   momentum finding above): two real backward() calls per logical step,
   accumulation left entirely to autograd's own default behavior on both
   sides, `prepare_next_pass()` between them -- checks that no update is
   applied after the first pass and that the final, single applied
   update after the second pass matches.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import FusedXPUAdafactor
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.composed_fused import ComposedFusedOptimizerHandle

DEVICE = "cpu"
_SHAPES = [(120, 120), (12000,)]  # both >= TINY_NUMEL (10_000)

# Each key: (dtype, scale_parameter, weight_decay, beta1). float32+beta1
# deliberately excluded -- see module docstring.
_TOLERANCES = {
    (torch.float32, False, 0.0, None): 1e-4,
    (torch.float32, True, 0.0, None): 1e-4,
    (torch.float32, False, 0.05, None): 1e-4,
    (torch.bfloat16, False, 0.0, None): 1e-2,
    (torch.bfloat16, True, 0.05, 0.9): 1e-2,
}


def check_legacy_float32_momentum_bug() -> bool:
    """Confirms the aliasing bug described in the module docstring,
    directly: after one update, legacy's stored exp_avg for a float32
    parameter equals *that step's applied delta*, not the true
    pre-alpha_t momentum value -- and does NOT for a bf16 parameter,
    isolating dtype as the actual cause."""
    torch.manual_seed(1)
    p_init = torch.randn(120, 120) * 0.1
    g = torch.randn(120, 120) * 0.05

    p = p_init.clone().requires_grad_(True)
    legacy = FusedXPUAdafactor(params=[p], lr=0.02, weight_decay=0.0,
                                scale_parameter=False, beta1=0.9, device=DEVICE)
    p.grad = g.clone()
    legacy.begin_step(1)
    legacy._update_param(p)
    applied_delta = p_init - p.detach()
    float32_corrupted = torch.allclose(legacy.exp_avg[0], applied_delta, atol=1e-6)

    p_bf16 = p_init.to(torch.bfloat16).clone().requires_grad_(True)
    legacy_bf16 = FusedXPUAdafactor(params=[p_bf16], lr=0.02, weight_decay=0.0,
                                     scale_parameter=False, beta1=0.9, device=DEVICE)
    p_bf16.grad = g.to(torch.bfloat16).clone()
    legacy_bf16.begin_step(1)
    legacy_bf16._update_param(p_bf16)
    applied_delta_bf16 = (p_init.to(torch.bfloat16) - p_bf16.detach()).float()
    bf16_not_corrupted = not torch.allclose(legacy_bf16.exp_avg[0].float(), applied_delta_bf16, atol=1e-6)

    return float32_corrupted and bf16_not_corrupted


def check_new_momentum_not_corrupted(n_steps: int = 5) -> bool:
    """Confirms AdafactorAlgorithm's momentum buffer, for a float32
    parameter, is NOT the same object as (and doesn't collapse to equal)
    that step's applied delta -- the exact corruption legacy has."""
    torch.manual_seed(1)
    p = torch.randn(120, 120) * 0.1
    algorithm = AdafactorAlgorithm(weight_decay=0.0, scale_parameter=False, beta1=0.9)
    state = algorithm.init_state((120, 120), torch.float32, DEVICE)

    for step in range(n_steps):
        algorithm.begin_step(1)
        torch.manual_seed(100 + step)
        g = torch.randn(120, 120) * 0.05
        exp_avg_before = state["exp_avg"].clone()
        delta, decay = algorithm.compute_update(g, p, state, 0.02)
        p = p - delta
        if torch.allclose(state["exp_avg"], delta, atol=1e-6):
            return False  # would mean the same corruption is present
        if step > 0 and torch.equal(state["exp_avg"], exp_avg_before):
            return False  # momentum buffer should actually be updating
    return True


def run_single_pass_case(dtype, scale_parameter, weight_decay, beta1, n_steps: int = 15) -> float:
    torch.manual_seed(11)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in _SHAPES]

    W_ref = [w.clone().requires_grad_(True) for w in inits]
    legacy = FusedXPUAdafactor(params=W_ref, lr=0.02, weight_decay=weight_decay,
                                scale_parameter=scale_parameter, beta1=beta1, device=DEVICE)
    legacy.register_hooks()

    W_new = [w.clone().requires_grad_(True) for w in inits]
    algorithm = AdafactorAlgorithm(weight_decay=weight_decay,
                                    scale_parameter=scale_parameter, beta1=beta1)
    handle = ComposedFusedOptimizerHandle(algorithm, W_new, lr=0.02, device=DEVICE)

    max_diff = 0.0
    for step in range(n_steps):
        torch.manual_seed(2000 + step)
        targets = [(torch.randn(s) * 0.05).to(dtype) for s in _SHAPES]

        legacy.begin_step(1)
        loss_ref = sum(((w - t) ** 2).sum() for w, t in zip(W_ref, targets))
        loss_ref.backward()

        handle.begin_step(1)
        loss_new = sum(((w - t) ** 2).sum() for w, t in zip(W_new, targets))
        loss_new.backward()

        for w_ref, w_new in zip(W_ref, W_new):
            diff = (w_ref.detach().float() - w_new.detach().float()).abs().max().item()
            max_diff = max(max_diff, diff)

    legacy.free_states()
    handle.free_states()
    return max_diff


def run_multi_pass_case(dtype, sub_steps: int = 2, n_steps: int = 8) -> dict:
    torch.manual_seed(13)
    inits = [(torch.randn(s) * 0.1).to(dtype) for s in _SHAPES]

    W_ref = [w.clone().requires_grad_(True) for w in inits]
    legacy = FusedXPUAdafactor(params=W_ref, lr=0.02, weight_decay=0.05,
                                scale_parameter=False, beta1=None, device=DEVICE)
    legacy.register_hooks()

    W_new = [w.clone().requires_grad_(True) for w in inits]
    algorithm = AdafactorAlgorithm(weight_decay=0.05, scale_parameter=False, beta1=None)
    handle = ComposedFusedOptimizerHandle(algorithm, W_new, lr=0.02, device=DEVICE)

    results = {"max_diff": 0.0, "no_premature_update": True}
    for step in range(n_steps):
        legacy.begin_step(sub_steps)
        handle.begin_step(sub_steps)

        for sub in range(sub_steps):
            torch.manual_seed(3000 + step * 10 + sub)
            targets = [(torch.randn(s) * 0.05).to(dtype) for s in _SHAPES]

            loss_ref = sum(((w - t) ** 2).sum() for w, t in zip(W_ref, targets))
            loss_ref.backward()
            loss_new = sum(((w - t) ** 2).sum() for w, t in zip(W_new, targets))
            loss_new.backward()

            if sub < sub_steps - 1:
                for w in (*W_ref, *W_new):
                    if w.grad is None:
                        results["no_premature_update"] = False
                legacy.prepare_next_pass()
                handle.prepare_next_pass()

        for w_ref, w_new in zip(W_ref, W_new):
            diff = (w_ref.detach().float() - w_new.detach().float()).abs().max().item()
            results["max_diff"] = max(results["max_diff"], diff)
            if w_ref.grad is not None or w_new.grad is not None:
                results["no_premature_update"] = False

    legacy.free_states()
    handle.free_states()
    return results


def main():
    print(f"Device: {DEVICE} (equivalence check -- real backward() and real "
          f"backward hooks, pure numerical comparison, real hardware not required)")
    failures = []

    print("\n=== legacy float32-momentum aliasing bug (found while writing this test) ===")
    ok = check_legacy_float32_momentum_bug()
    print(f"  {'PASS' if ok else 'FAIL'}: legacy exp_avg corrupted for float32, not for bf16 "
          f"(confirms the mechanism described in this file's module docstring)")
    if not ok:
        failures.append("check_legacy_float32_momentum_bug: expected corruption pattern not found "
                         "-- module docstring's explanation may now be wrong, needs re-checking")

    ok = check_new_momentum_not_corrupted()
    print(f"  {'PASS' if ok else 'FAIL'}: AdafactorAlgorithm's momentum buffer does NOT have "
          f"this corruption for float32 parameters")
    if not ok:
        failures.append("check_new_momentum_not_corrupted: AdafactorAlgorithm's momentum buffer "
                         "shows the same corruption pattern as the legacy bug -- would need fixing")

    print("\n=== single-pass (sub_steps=1) ===")
    for (dtype, scale_parameter, weight_decay, beta1), tol in _TOLERANCES.items():
        diff = run_single_pass_case(dtype, scale_parameter, weight_decay, beta1)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: dtype={dtype}, scale_parameter={scale_parameter}, "
              f"weight_decay={weight_decay}, beta1={beta1}: "
              f"max abs diff over 15 steps = {diff:.3e} (tolerance {tol:.0e})")
        if not ok:
            failures.append(f"dtype={dtype}, scale_parameter={scale_parameter}, "
                             f"weight_decay={weight_decay}, beta1={beta1}: "
                             f"diff {diff:.3e} exceeds tolerance {tol:.0e}")

    print("\n=== multi-pass (sub_steps=2, distillation shape, beta1=None) ===")
    for dtype in (torch.float32, torch.bfloat16):
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        results = run_multi_pass_case(dtype)
        diff_ok = results["max_diff"] <= tol
        status = "PASS" if (diff_ok and results["no_premature_update"]) else "FAIL"
        print(f"  {status}: dtype={dtype}: max abs diff over 8 steps = "
              f"{results['max_diff']:.3e} (tolerance {tol:.0e}), "
              f"no_premature_update={results['no_premature_update']}")
        if not diff_ok:
            failures.append(f"multi-pass dtype={dtype}: diff {results['max_diff']:.3e} "
                             f"exceeds tolerance {tol:.0e}")
        if not results["no_premature_update"]:
            failures.append(f"multi-pass dtype={dtype}: premature update or grad-clearing "
                             f"mismatch detected across the accumulation passes")

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
