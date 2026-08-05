"""Correctness check for LoRAScalingPolicy's effective-alpha seam
(docs/training_pipeline_design.md section 3.2,
nodes/model/lora_injector.py's module docstring for the derivation).

The crux: _effective_alpha() is checked against the REAL
core.lora.LoRALinear formula directly, not just its own algebra -- the
same discipline as every other equivalence test in this project.

Run this directly: `python nodes/smoke_tests/smoke_test_lora_scaling_policy.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch.nn as nn

from core.lora import LoRALinear
from nodes.model.lora_injector import (ClassicLoRAScaling, LoRAScalingPolicy,
                                        RankStabilizedScaling, _effective_alpha)

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_formulas():
    print("\n=== Policy formulas ===")
    alpha, rank = 2.0, 64
    record(ClassicLoRAScaling().scaling(alpha, rank) == alpha / rank,
           "ClassicLoRAScaling == alpha/rank")
    record(RankStabilizedScaling().scaling(alpha, rank) == alpha / (rank ** 0.5),
           "RankStabilizedScaling == alpha/sqrt(rank)")


def check_seam_against_real_core_lora():
    """The actual point of this test: _effective_alpha(), fed into a real
    core.lora.LoRALinear (not reimplemented, not mocked), must produce
    exactly the scaling the policy asked for -- checked against the
    legacy class's own formula executing for real, not against my own
    derivation of it."""
    print("\n=== _effective_alpha(), fed through the real core.lora.LoRALinear ===")
    original = nn.Linear(8, 8)
    alpha, rank, weight = 2.0, 64, 1.0

    for policy in (ClassicLoRAScaling(), RankStabilizedScaling()):
        eff_alpha = _effective_alpha(alpha, rank, policy)
        layer = LoRALinear(original, rank=rank, alpha=eff_alpha, weight=weight)
        expected = policy.scaling(alpha, rank) * weight
        record(abs(layer.scaling - expected) < 1e-9,
               f"{type(policy).__name__}: LoRALinear.scaling matches policy.scaling()*weight",
               detail=f"got {layer.scaling}, expected {expected}")


def check_classic_is_true_zero_behavior_change():
    """Not just numerically close -- the seam must produce IDENTICAL
    LoRALinear.scaling to never having gone through the seam at all, for
    the default policy. This is the "nothing wired to this Node today
    changes" claim, checked, not assumed."""
    print("\n=== ClassicLoRAScaling: bit-identical to not using the seam at all ===")
    original = nn.Linear(8, 8)
    alpha, rank, weight = 1.7, 64, 1.0

    eff_alpha = _effective_alpha(alpha, rank, ClassicLoRAScaling())
    record(eff_alpha == alpha, "effective_alpha == alpha exactly (identity)",
           detail=f"got {eff_alpha}")

    via_seam = LoRALinear(original, rank=rank, alpha=eff_alpha, weight=weight)
    direct = LoRALinear(original, rank=rank, alpha=alpha, weight=weight)
    record(via_seam.scaling == direct.scaling,
           "LoRALinear.scaling identical with and without the seam",
           detail=f"via_seam={via_seam.scaling}, direct={direct.scaling}")


def check_policy_contract():
    print("\n=== LoRAScalingPolicy is a real ABC ===")

    class BadPolicy(LoRAScalingPolicy):
        pass

    try:
        BadPolicy()
        ok = False
    except TypeError:
        ok = True
    record(ok, "can't instantiate a LoRAScalingPolicy that doesn't implement scaling()")


def main():
    check_formulas()
    check_seam_against_real_core_lora()
    check_classic_is_true_zero_behavior_change()
    check_policy_contract()

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
