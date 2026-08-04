"""Numerical equivalence check: nodes/components/diffusion.py's
NoiseSchedule/Parameterization/ModelInputTransform vs. the legacy
core.noise_schedule/core.model_io free functions they're fresh
reimplementations of, plus the invariants specific to the new pieces that
have no legacy equivalent (RescaledZeroTerminalSNRSchedule,
DiffusionProcess's parameterization guard).

Run this directly: `python nodes/smoke_tests/smoke_test_diffusion_equivalence.py`
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.model_io import comfy_input_transform
from core.noise_schedule import (eps_to_vpred, eps_to_x0, get_alpha_sigma,
                                  vpred_to_eps, vpred_to_x0)
from nodes.components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                         EpsParameterization, KarrasInputScaler,
                                         RescaledZeroTerminalSNRSchedule,
                                         VPredParameterization)

TOL = 1e-6
failures = []


def check(name: str, new, ref, tol: float = TOL):
    diff = (new.detach().float() - ref.detach().float()).abs().max().item()
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: {name}: max abs diff = {diff:.3e} (tolerance {tol:.0e})")
    if not ok:
        failures.append(f"{name}: diff {diff:.3e} exceeds tolerance {tol:.0e}")


def check_schedule():
    print("\n=== DiscreteLinearNoiseSchedule.alpha_sigma vs get_alpha_sigma ===")
    schedule = DiscreteLinearNoiseSchedule()

    t_int = 137
    a_new, s_new = schedule.alpha_sigma(t_int)
    a_ref, s_ref = get_alpha_sigma(t_int)
    check("int index, alpha", a_new, a_ref)
    check("int index, sigma", s_new, s_ref)

    t_batch = torch.tensor([0, 1, 500, 998, 999], dtype=torch.long)
    a_new, s_new = schedule.alpha_sigma(t_batch)
    a_ref, s_ref = get_alpha_sigma(t_batch)
    check("cpu tensor batch, alpha", a_new, a_ref)
    check("cpu tensor batch, sigma", s_new, s_ref)


def check_parameterizations():
    print("\n=== Parameterization.to_x0 / convert_to vs core.noise_schedule ===")
    torch.manual_seed(0)
    x_t = torch.randn(4, 4, 8, 8)
    raw = torch.randn(4, 4, 8, 8)
    alpha = torch.rand(4, 1, 1, 1) * 0.9 + 0.05
    sigma = torch.rand(4, 1, 1, 1) * 10.0 + 0.01

    eps = EpsParameterization()
    vpred = VPredParameterization()

    check("EpsParameterization.to_x0 vs eps_to_x0",
          eps.to_x0(raw, x_t, alpha, sigma), eps_to_x0(raw, x_t, alpha, sigma))
    check("VPredParameterization.to_x0 vs vpred_to_x0",
          vpred.to_x0(raw, x_t, alpha, sigma), vpred_to_x0(raw, x_t, alpha, sigma))
    check("Eps.convert_to(target=VPred) vs eps_to_vpred",
          eps.convert_to(raw, x_t, alpha, sigma, vpred), eps_to_vpred(raw, x_t, alpha, sigma))
    check("VPred.convert_to(target=Eps) vs vpred_to_eps",
          vpred.convert_to(raw, x_t, alpha, sigma, eps), vpred_to_eps(raw, x_t, alpha, sigma))

    # No legacy equivalent for same-type conversion (raw_to_target's else branch
    # covers this too, but only incidentally) -- checked as its own invariant.
    check("Eps.convert_to(target=Eps) is identity", eps.convert_to(raw, x_t, alpha, sigma, eps), raw)
    check("VPred.convert_to(target=VPred) is identity",
          vpred.convert_to(raw, x_t, alpha, sigma, vpred), raw)


def check_input_transform():
    print("\n=== KarrasInputScaler.scale_input vs comfy_input_transform ===")
    torch.manual_seed(1)
    x_t = torch.randn(4, 4, 8, 8)
    scaler = KarrasInputScaler()

    sigma_scalar = 3.7
    check("scalar sigma (float)",
          scaler.scale_input(x_t, sigma_scalar), comfy_input_transform(x_t, sigma_scalar))

    sigma_0d = torch.tensor(3.7)
    check("0-dim tensor sigma",
          scaler.scale_input(x_t, sigma_0d), comfy_input_transform(x_t, sigma_0d))

    sigma_batch = torch.rand(4) * 10.0 + 0.01
    check("per-sample sigma tensor",
          scaler.scale_input(x_t, sigma_batch), comfy_input_transform(x_t, sigma_batch))


def check_zero_terminal_snr_invariants():
    print("\n=== RescaledZeroTerminalSNRSchedule invariants (no legacy equivalent) ===")
    schedule = RescaledZeroTerminalSNRSchedule()
    alpha_T, sigma_T = schedule.alpha_sigma(999)
    ok = alpha_T.item() == 0.0
    print(f"  {'PASS' if ok else 'FAIL'}: alpha_t[-1] == 0.0 exactly (got {alpha_T.item()!r})")
    if not ok:
        failures.append(f"alpha_t[-1] expected exactly 0.0, got {alpha_T.item()!r}")

    ok = math.isinf(sigma_T.item())
    print(f"  {'PASS' if ok else 'FAIL'}: sigma_t[-1] == inf exactly (got {sigma_T.item()!r})")
    if not ok:
        failures.append(f"sigma_t[-1] expected inf, got {sigma_T.item()!r}")

    alpha_0, _ = schedule.alpha_sigma(0)
    lin_alpha_0, _ = DiscreteLinearNoiseSchedule().alpha_sigma(0)
    diff = abs(alpha_0.item() - lin_alpha_0.item())
    ok = diff <= TOL
    print(f"  {'PASS' if ok else 'FAIL'}: alpha_t[0] left ~unchanged by rescale "
          f"(diff = {diff:.3e})")
    if not ok:
        failures.append(f"alpha_t[0] rescale drift {diff:.3e} exceeds tolerance {TOL:.0e}")


def check_diffusion_process_guard():
    print("\n=== DiffusionProcess.__post_init__ parameterization guard ===")
    try:
        DiffusionProcess(RescaledZeroTerminalSNRSchedule(), EpsParameterization(),
                          KarrasInputScaler())
        ok = False
    except ValueError:
        ok = True
    print(f"  {'PASS' if ok else 'FAIL'}: rejects zero-terminal-SNR + eps")
    if not ok:
        failures.append("DiffusionProcess accepted zero-terminal-SNR schedule with "
                         "EpsParameterization; should have raised ValueError")

    try:
        DiffusionProcess(RescaledZeroTerminalSNRSchedule(), VPredParameterization(),
                          KarrasInputScaler())
        DiffusionProcess(DiscreteLinearNoiseSchedule(), EpsParameterization(),
                          KarrasInputScaler())
        ok = True
    except ValueError as e:
        ok = False
        failures.append(f"DiffusionProcess rejected a valid combination: {e}")
    print(f"  {'PASS' if ok else 'FAIL'}: accepts zero-terminal-SNR + vpred, and linear + eps")


def main():
    print("Device: cpu (equivalence check -- pure numerical comparison, "
          "real hardware not required)")
    check_schedule()
    check_parameterizations()
    check_input_transform()
    check_zero_terminal_snr_invariants()
    check_diffusion_process_guard()

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
