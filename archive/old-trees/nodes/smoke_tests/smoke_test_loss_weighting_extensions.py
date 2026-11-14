"""Correctness check for nodes/train/loss.py's Min-SNR v-prediction
completion and the new P2LossWeighting (docs/design/04-lora-adapter-mechanics-and-loss-weighting.md
section 4).

No legacy implementation exists for either (this is new math, not a
core.* rewrite), so what's checked is hand-computed reference values
against the papers' own formulas, not equivalence to prior code -- except
for MinSNRLossWeighting's epsilon branch, which must stay bit-identical
to what shipped before this change (a real regression risk, since
__init__'s signature changed).

Run this directly: `python nodes/smoke_tests/smoke_test_loss_weighting_extensions.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nodes.components.diffusion import EpsParameterization, VPredParameterization
from nodes.train.loss import (MinSNRLossWeighting, MinSNRLossWeightingNode,
                               P2LossWeighting, P2LossWeightingNode,
                               UniformLossWeighting)

TOL = 1e-9
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check(name: str, got: float, expected: float, tol: float = TOL):
    ok = abs(got - expected) <= tol
    record(ok, name, detail=f"got {got}, expected {expected}")


def check_eps_branch_unchanged():
    """Regression check: __init__'s signature changed (parameterization
    param added) -- the actual eps formula must not have moved at all."""
    print("\n=== MinSNRLossWeighting: eps branch, unchanged from before this change ===")
    gamma = 5.0
    default = MinSNRLossWeighting(gamma=gamma)  # no parameterization given
    explicit_eps = MinSNRLossWeighting(gamma=gamma, parameterization=EpsParameterization())

    for sigma in (0.1, 1.0, 5.0, 14.6):
        snr = 1.0 / (sigma ** 2 + 1e-8)
        expected = min(snr, gamma) / snr
        check(f"default (no parameterization) at sigma={sigma}", default.weight(sigma), expected)
        check(f"explicit EpsParameterization at sigma={sigma}",
              explicit_eps.weight(sigma), expected)


def check_vpred_branch():
    """min(SNR, gamma) / (SNR + 1) -- Hang et al. 2023's own derivation,
    cross-checked against huggingface/diffusers#5654's corrected formula."""
    print("\n=== MinSNRLossWeighting: v-prediction branch, hand-computed ===")
    gamma = 5.0
    vpred = MinSNRLossWeighting(gamma=gamma, parameterization=VPredParameterization())

    for sigma in (0.1, 1.0, 5.0, 14.6):
        snr = 1.0 / (sigma ** 2 + 1e-8)
        expected = min(snr, gamma) / (snr + 1.0)
        check(f"v-pred at sigma={sigma}", vpred.weight(sigma), expected)


def check_eps_vpred_relationship():
    """v_weight(sigma) == eps_weight(sigma) * snr/(snr+1) -- an algebraic
    identity between the two formulas, checked numerically as a
    cross-consistency invariant, not just each formula in isolation."""
    print("\n=== eps/v-pred formulas are internally consistent with each other ===")
    gamma = 5.0
    eps = MinSNRLossWeighting(gamma=gamma, parameterization=EpsParameterization())
    vpred = MinSNRLossWeighting(gamma=gamma, parameterization=VPredParameterization())

    for sigma in (0.1, 1.0, 5.0, 14.6):
        snr = 1.0 / (sigma ** 2 + 1e-8)
        check(f"v(sigma={sigma}) == eps(sigma) * snr/(snr+1)",
              vpred.weight(sigma), eps.weight(sigma) * (snr / (snr + 1.0)))


def check_p2_weighting():
    print("\n=== P2LossWeighting: 1/(k+SNR)^gamma, hand-computed ===")
    for k, gamma in ((1.0, 1.0), (0.5, 2.0), (1.0, 0.0)):
        p2 = P2LossWeighting(k=k, gamma=gamma)
        for sigma in (0.1, 1.0, 5.0):
            snr = 1.0 / (sigma ** 2 + 1e-8)
            expected = 1.0 / ((k + snr) ** gamma)
            check(f"k={k}, gamma={gamma}, sigma={sigma}", p2.weight(sigma), expected)

    record(P2LossWeighting(k=1.0, gamma=0.0).weight(3.7) == 1.0,
           "gamma=0 degenerates to constant weight 1.0 (x**0 == 1)")


def check_node_wrappers():
    print("\n=== Node wrappers construct the right types ===")
    result = MinSNRLossWeightingNode().build(gamma=3.0)
    w = result["weighting"]
    record(isinstance(w, MinSNRLossWeighting), "MinSNRLossWeightingNode builds a MinSNRLossWeighting")
    check("MinSNRLossWeightingNode default parameterization is eps, at sigma=1.0",
          w.weight(1.0), min(1.0, 3.0) / 1.0)
    check("MinSNRLossWeightingNode(gamma=3.0).weight(2.0) matches eps formula",
          w.weight(2.0), min(1.0 / (2.0 ** 2 + 1e-8), 3.0) / (1.0 / (2.0 ** 2 + 1e-8)))

    result = MinSNRLossWeightingNode().build(gamma=3.0, parameterization=VPredParameterization())
    w_v = result["weighting"]
    snr = 1.0 / (2.0 ** 2 + 1e-8)
    check("MinSNRLossWeightingNode(parameterization=VPredParameterization()).weight(2.0)",
          w_v.weight(2.0), min(snr, 3.0) / (snr + 1.0))

    result = P2LossWeightingNode().build(k=2.0, gamma=1.5)
    w_p2 = result["weighting"]
    record(isinstance(w_p2, P2LossWeighting), "P2LossWeightingNode builds a P2LossWeighting")
    snr = 1.0 / (2.0 ** 2 + 1e-8)
    check("P2LossWeightingNode(k=2.0, gamma=1.5).weight(2.0)",
          w_p2.weight(2.0), 1.0 / ((2.0 + snr) ** 1.5))


def check_loss_phase_weights_per_sample():
    """Both trainers' LossPhase must apply the weighting *per sample*
    (mean(w(sigma_i) * l_i)), not the older scalar form from the batch's
    mean sigma (w(mean sigma) * mean(l)). For a nonlinear weighting the
    two disagree on every mixed-t batch -- and batches carry per-sample t
    almost always -- so this is the actual behavioral change, with the
    uniform case staying bit-identical to the old math."""
    print("\n=== LossPhase applies weighting per sample (both routes) ===")
    from types import SimpleNamespace

    import torch

    from nodes.train.managed import LossPhase as ManagedLossPhase
    from nodes.train.step_pipeline import LossPhase as PipelineLossPhase

    # sample0 l = 9.0 (error 6 on one of four elements), sample1 l = 0.0.
    def make_state():
        return SimpleNamespace(extras={
            "pred": torch.tensor([[0.0, 0.0, 0.0, 6.0], [0.0, 0.0, 0.0, 0.0]]),
            "target": torch.zeros(2, 4),
            "sigma": torch.tensor([0.05, 4.0]),  # mixed-t: low noise + high noise
        })

    gamma = 5.0
    weighting = MinSNRLossWeighting(gamma=gamma)
    w0, w1 = weighting.weight(0.05), weighting.weight(4.0)
    expected_per_sample = (w0 * 9.0 + w1 * 0.0) / 2.0
    mean_sigma = (0.05 + 4.0) / 2.0
    old_scalar = weighting.weight(mean_sigma) * (9.0 + 0.0) / 2.0
    record(abs(old_scalar - expected_per_sample) > 0.1,
           "test numbers chosen so the two formulas genuinely differ "
           f"(old {old_scalar:.4f} vs per-sample {expected_per_sample:.4f})")

    for cls, label in ((PipelineLossPhase, "step_pipeline"), (ManagedLossPhase, "managed")):
        st = make_state()
        cls(weighting).run(st)
        got = float(st.extras["loss"])
        record(abs(got - expected_per_sample) <= TOL,
               f"{label} LossPhase: mean(w_i * l_i)",
               detail=f"got {got}, expected {expected_per_sample}")
        ps = st.extras.get("per_sample_loss")
        record(ps is not None and not ps.requires_grad
               and [round(float(v), 6) for v in ps] == [9.0, 0.0],
               f"{label} LossPhase: stashes detached per-sample raw MSE for the t buckets")

    # Uniform weighting must be bit-identical to the old scalar math.
    st = make_state()
    PipelineLossPhase(UniformLossWeighting()).run(st)
    check("uniform stays mean(l)", float(st.extras["loss"]), 4.5)

    # Scalar-sigma schedules take the old single-weight path.
    st = make_state()
    st.extras["sigma"] = torch.tensor(1.0)  # shared sigma, not per-sample
    PipelineLossPhase(weighting).run(st)
    check("scalar sigma collapses to one shared weight",
          float(st.extras["loss"]), weighting.weight(1.0) * 4.5)


def main():
    check_eps_branch_unchanged()
    check_vpred_branch()
    check_eps_vpred_relationship()
    check_p2_weighting()
    check_node_wrappers()
    check_loss_phase_weights_per_sample()

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
