"""Correctness check for nodes/model/adapter_strategy.py
(AdaptedLayer/AdapterStrategy/PlainLoRAAdapter).

The crux, same discipline as smoke_test_lora_scaling_policy.py: checked
against constructing a real core.lora.LoRALinear/LoRAConv2d directly, not
against my own derivation of what PlainLoRAAdapter.wrap() should do.

Run this directly: `python nodes/smoke_tests/smoke_test_adapter_strategy.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch.nn as nn

from core.lora import LoRAConv2d, LoRALinear
from nodes.model.adapter_strategy import AdaptedLayer, AdapterStrategy, PlainLoRAAdapter
from nodes.model.frozen_weight_store import BF16WeightStore, FrozenWeightStore
from nodes.model.lora_injector import ClassicLoRAScaling, RankStabilizedScaling

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_wrap_linear_matches_direct_construction():
    print("\n=== PlainLoRAAdapter.wrap() on nn.Linear matches core.lora.LoRALinear directly ===")
    original = nn.Linear(8, 6)
    rank, alpha = 4, 2.0
    adapted = PlainLoRAAdapter().wrap(
        original, BF16WeightStore(original.weight), rank=rank, alpha=alpha,
        scaling_policy=ClassicLoRAScaling())
    direct = LoRALinear(original, rank=rank, alpha=alpha)

    record(isinstance(adapted, LoRALinear), "returns a real LoRALinear instance")
    record(adapted.scaling == direct.scaling, "scaling matches (ClassicLoRAScaling is identity)",
           detail=f"got {adapted.scaling}, expected {direct.scaling}")
    record(adapted.rank == rank, "rank set correctly")
    record(adapted.base_weight.shape == original.weight.shape, "base_weight shape matches original")
    record(adapted.base_bias is not None and direct.base_bias is not None
           and adapted.base_bias.shape == direct.base_bias.shape,
           "bias carried through correctly")


def check_wrap_conv2d_matches_direct_construction():
    print("\n=== PlainLoRAAdapter.wrap() on nn.Conv2d matches core.lora.LoRAConv2d directly ===")
    original = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
    rank, alpha = 4, 1.5
    adapted = PlainLoRAAdapter().wrap(
        original, BF16WeightStore(original.weight), rank=rank, alpha=alpha,
        scaling_policy=ClassicLoRAScaling())
    direct = LoRAConv2d(original, rank=rank, alpha=alpha)

    record(isinstance(adapted, LoRAConv2d), "returns a real LoRAConv2d instance")
    record(adapted.scaling == direct.scaling, "scaling matches (ClassicLoRAScaling is identity)")
    record(adapted.stride == original.stride and adapted.padding == original.padding,
           "conv-specific shape params (stride/padding) carried through correctly")


def check_wrap_respects_rank_stabilized_scaling():
    print("\n=== wrap() actually applies a non-default LoRAScalingPolicy through the seam ===")
    original = nn.Linear(8, 6)
    rank, alpha = 64, 2.0
    adapted = PlainLoRAAdapter().wrap(
        original, BF16WeightStore(original.weight), rank=rank, alpha=alpha,
        scaling_policy=RankStabilizedScaling())
    expected = (alpha / (rank ** 0.5))  # RankStabilizedScaling's formula, weight=1.0 default
    record(abs(adapted.scaling - expected) < 1e-9,
           "resulting .scaling matches alpha/sqrt(rank), not alpha/rank",
           detail=f"got {adapted.scaling}, expected {expected}")


def check_wrap_rejects_non_bf16_weightstore():
    print("\n=== wrap() fails loudly on a FrozenWeightStore kind it can't honor ===")

    class _FakeNF4Store(FrozenWeightStore):
        def footprint_bytes(self):
            return 0

        def materialize(self):
            return None

    original = nn.Linear(4, 4)
    try:
        PlainLoRAAdapter().wrap(original, _FakeNF4Store(), rank=4, alpha=1.0,
                                 scaling_policy=ClassicLoRAScaling())
        ok = False
    except NotImplementedError:
        ok = True
    record(ok, "raises NotImplementedError for a non-BF16WeightStore, doesn't silently misbehave")


def check_wrap_rejects_unknown_module_type():
    print("\n=== wrap() rejects an original it doesn't know how to adapt ===")
    original = nn.LayerNorm(8)
    try:
        PlainLoRAAdapter().wrap(original, BF16WeightStore(original.weight), rank=4, alpha=1.0,
                                 scaling_policy=ClassicLoRAScaling())
        ok = False
    except TypeError:
        ok = True
    record(ok, "raises TypeError for an nn.LayerNorm (not Linear or Conv2d)")


def check_adapted_layer_registration():
    print("\n=== AdaptedLayer virtual-subclass registration ===")
    original = nn.Linear(4, 4)
    adapted = PlainLoRAAdapter().wrap(original, BF16WeightStore(original.weight), rank=2, alpha=1.0,
                                       scaling_policy=ClassicLoRAScaling())
    record(isinstance(adapted, AdaptedLayer),
           "a real LoRALinear returned by wrap() isinstance-checks as AdaptedLayer")
    record(hasattr(adapted, "get_lora_weights"), "and actually has get_lora_weights()")


def check_contracts():
    print("\n=== AdapterStrategy/AdaptedLayer are real, enforced ABCs ===")

    class BadStrategy(AdapterStrategy):
        pass

    class BadLayer(AdaptedLayer):
        pass

    try:
        BadStrategy()
        ok1 = False
    except TypeError:
        ok1 = True
    record(ok1, "can't instantiate an AdapterStrategy that doesn't implement wrap()")

    try:
        BadLayer()
        ok2 = False
    except TypeError:
        ok2 = True
    record(ok2, "can't instantiate an AdaptedLayer that doesn't implement get_lora_weights()")


def main():
    check_wrap_linear_matches_direct_construction()
    check_wrap_conv2d_matches_direct_construction()
    check_wrap_respects_rank_stabilized_scaling()
    check_wrap_rejects_non_bf16_weightstore()
    check_wrap_rejects_unknown_module_type()
    check_adapted_layer_registration()
    check_contracts()

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
