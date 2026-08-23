"""Correctness check for nodes/model/nf4_lora_layer.py's
NF4LoRALinear/NF4LoRAConv2d, and the frozen_weight_store wiring through
nodes/model/adapter_injection.py's adapter_strategy_scope.

Compares against an independent reference built directly from
frozen.materialize()'s own (deterministic -- see
smoke_test_nf4_weight_store.py) output, not against the true original
weight -- NF4's own quantization error is NF4WeightStore's own, already
tested concern (smoke_test_nf4_weight_store.py); this file is only
responsible for the LoRA forward/backward math built on top of whatever
materialize() returns.

Run this directly: `python nodes/smoke_tests/smoke_test_nf4_lora_layer.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

import core.lora as core_lora
from nodes.model.adapter_strategy import AdaptedLayer, PlainLoRAAdapter
from nodes.model.frozen_weight_store import BF16WeightStore
from nodes.model.lora_scaling import ClassicLoRAScaling
from nodes.model.nf4_lora_layer import NF4LoRAConv2d, NF4LoRALinear
from nodes.model.nf4_weight_store import NF4WeightStore

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _reference_lora_linear(x, dequantized_weight, base_bias, lora_A, lora_B, scaling):
    """Independent of nf4_lora_layer.py's own code -- the textbook LoRA
    forward, built directly from frozen.materialize()'s own fixed
    output, not NF4LoRALinear's internals."""
    base = F.linear(x, dequantized_weight, base_bias)
    lora = (x.to(lora_A.dtype) @ lora_A.T) @ (lora_B.T * scaling)
    return base + lora.to(base.dtype)


def check_identity_at_init_relative_to_quantized_weight():
    print("\n=== NF4LoRALinear at init: output exactly matches "
          "F.linear(x, frozen.materialize(), bias) -- lora_B starts at zero ===")
    torch.manual_seed(0)
    base = nn.Linear(64, 48)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRALinear(base, frozen, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    out = layer(x)
    ref = F.linear(x, frozen.materialize(), layer.base_bias)
    record(torch.allclose(out, ref, atol=1e-3),
           "NF4LoRALinear(x) == F.linear(x, frozen.materialize(), bias) at init",
           detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_matches_independent_reference_after_training_steps():
    print("\n=== NF4LoRALinear matches an independent reference built from "
          "frozen.materialize()'s own fixed output, after real training steps ===")
    torch.manual_seed(1)
    base = nn.Linear(64, 48)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRALinear(base, frozen, rank=4, alpha=8.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(3, 64, dtype=torch.bfloat16)
    out = layer(x)
    dequantized = frozen.materialize()
    ref = _reference_lora_linear(x, dequantized, layer.base_bias, layer.lora_A,
                                  layer.lora_B, layer.scaling)
    record(torch.allclose(out, ref, atol=1e-4),
           "matches the independent reference after training has moved lora_A/lora_B "
           "away from init", detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_conv2d_matches_independent_reference():
    print("\n=== NF4LoRAConv2d matches an independent reference after training steps ===")
    torch.manual_seed(2)
    base = nn.Conv2d(4, 6, kernel_size=3, padding=1)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRAConv2d(base, frozen, rank=2, alpha=4.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
    out = layer(x)
    dequantized = frozen.materialize()
    ref_base = F.conv2d(x, dequantized, layer.base_bias, layer.stride,
                         layer.padding, layer.dilation, layer.groups)
    adapter = F.conv2d(x.to(layer.lora_A.dtype), layer.lora_A, None, layer.stride,
                        layer.padding, layer.dilation, layer.groups)
    adapter = F.conv2d(adapter, layer.lora_B * layer.scaling)
    ref = ref_base + adapter.to(ref_base.dtype)
    record(torch.allclose(out, ref, atol=1e-3),
           "matches the independent reference", detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_gate_zero_produces_exactly_the_dequantized_base_output():
    print("\n=== gate=0 produces exactly F.linear(x, frozen.materialize(), bias), "
          "even after training ===")
    torch.manual_seed(3)
    base = nn.Linear(64, 48)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRALinear(base, frozen, rank=4, alpha=8.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(3, 64, dtype=torch.bfloat16)
    core_lora.set_lora_gate(torch.zeros(3))
    try:
        out = layer(x)
    finally:
        core_lora.set_lora_gate(None)
    ref = F.linear(x, frozen.materialize(), layer.base_bias)
    record(torch.allclose(out, ref, atol=1e-5),
           "gate=0 -> exactly the dequantized base's own forward",
           detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_gradients_and_parameter_membership():
    print("\n=== only lora_A/lora_B are trainable Parameters -- the frozen weight "
          "store holds no gradient-tracked state at all ===")
    base = nn.Linear(64, 48)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRALinear(base, frozen, rank=4, alpha=8.0)
    param_names = {n for n, _ in layer.named_parameters()}
    record(param_names == {"lora_A", "lora_B"},
           "exactly lora_A and lora_B are Parameters", detail=str(param_names))

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    layer(x).sum().backward()
    record(layer.lora_A.grad is not None, "lora_A received a gradient")
    record(layer.lora_B.grad is not None, "lora_B received a gradient (object exists)")
    record(not hasattr(frozen, "grad"), "NF4WeightStore itself has no .grad at all")


def check_footprint_is_real_not_hiding_a_bf16_copy():
    print("\n=== the frozen base is really compressed -- no hidden bf16 copy anywhere ===")
    base = nn.Linear(1280, 1280)
    frozen = NF4WeightStore(base.weight)
    layer = NF4LoRALinear(base, frozen, rank=32, alpha=32.0)
    param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in layer.buffers() if b is not None)
    bf16_equivalent_bytes = base.weight.numel() * 2
    record(frozen.footprint_bytes() < bf16_equivalent_bytes / 3,
           "frozen.footprint_bytes() is well under a third of the bf16-equivalent size",
           detail=f"nf4={frozen.footprint_bytes()} bf16_equiv={bf16_equivalent_bytes}")
    record(param_bytes + buffer_bytes < bf16_equivalent_bytes / 3,
           "the layer's own parameters+buffers (lora_A/B, bias) don't hide a bf16 "
           "copy of the base weight anywhere", detail=f"params+buffers={param_bytes + buffer_bytes}")


def check_adapted_layer_conformance_and_wrap_dispatch():
    print("\n=== AdaptedLayer conformance, and PlainLoRAAdapter.wrap() dispatches "
          "to NF4LoRALinear/NF4LoRAConv2d for NF4WeightStore ===")
    adapter = PlainLoRAAdapter()
    linear = nn.Linear(8, 8)
    conv = nn.Conv2d(4, 4, 3, padding=1)
    frozen_linear = NF4WeightStore(linear.weight)
    frozen_conv = NF4WeightStore(conv.weight)

    result = adapter.wrap(linear, frozen_linear, rank=2, alpha=4.0,
                           scaling_policy=ClassicLoRAScaling())
    record(isinstance(result, NF4LoRALinear), "wrap() on nn.Linear + NF4WeightStore "
           "returns an NF4LoRALinear")
    record(isinstance(result, AdaptedLayer), "NF4LoRALinear isinstance AdaptedLayer")

    result_conv = adapter.wrap(conv, frozen_conv, rank=2, alpha=4.0,
                                scaling_policy=ClassicLoRAScaling())
    record(isinstance(result_conv, NF4LoRAConv2d), "wrap() on nn.Conv2d + NF4WeightStore "
           "returns an NF4LoRAConv2d")

    # Confirm BF16WeightStore still takes the real core.lora.LoRALinear path
    # -- this class's new NF4 branch must not have disturbed the existing one.
    frozen_bf16 = BF16WeightStore(linear.weight)
    result_bf16 = adapter.wrap(linear, frozen_bf16, rank=2, alpha=4.0,
                                scaling_policy=ClassicLoRAScaling())
    LoRALinear, _ = core_lora.LoRALinear, core_lora.LoRAConv2d
    record(type(result_bf16).__name__ == "LoRALinear",
           "BF16WeightStore still dispatches to the real core.lora.LoRALinear, unchanged",
           detail=type(result_bf16).__name__)


def check_end_to_end_through_adapter_strategy_scope():
    print("\n=== frozen_weight_store_factory=NF4WeightStore through the real "
          "adapter_strategy_scope mechanism, with PlainLoRAAdapter (the default "
          "adapter_strategy) -- the exact case that would have silently done nothing "
          "under the old 'skip patching for PlainLoRAAdapter' rule ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smoke_test_adapter_injection import _MiniUNetLike, _same_fixture_pair
    from core.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.adapter_injection import adapter_strategy_scope

    model_bf16, model_nf4 = _same_fixture_pair()
    model_bf16 = model_bf16.to(torch.bfloat16)
    model_nf4 = model_nf4.to(torch.bfloat16)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)

    registry_bf16 = inject_lora_into_unet(model_bf16, config)

    before_linear = core_lora.LoRALinear
    with adapter_strategy_scope(PlainLoRAAdapter(), NF4WeightStore):
        record(core_lora.LoRALinear is not before_linear,
               "the patch IS installed for PlainLoRAAdapter + NF4WeightStore "
               "(the old 'always skip for PlainLoRAAdapter' rule would have been wrong "
               "here -- confirmed fixed)")
        registry_nf4 = inject_lora_into_unet(model_nf4, config)
    record(core_lora.LoRALinear is before_linear, "restored after the scope exits")

    names_bf16 = sorted(n for n, _, _, _ in registry_bf16)
    names_nf4 = sorted(n for n, _, _, _ in registry_nf4)
    record(names_bf16 == names_nf4,
           "NF4WeightStore targets exactly the same layers BF16WeightStore does")
    record(all(isinstance(layer, NF4LoRALinear) for _, _, _, layer in registry_nf4),
           "every targeted layer really is an NF4LoRALinear")

    x = torch.randn(2, 8, dtype=torch.bfloat16)
    out_bf16 = model_bf16(x.clone())
    out_nf4 = model_nf4(x.clone())
    # Not expected to match closely -- NF4 quantization error is real and
    # this fixture's tiny random weights amplify relative error further.
    # Just confirm both run and produce finite, distinct output.
    record(torch.isfinite(out_nf4).all(), "NF4 path produces finite output")
    record(not torch.allclose(out_bf16, out_nf4, atol=1e-6),
           "NF4 output measurably differs from BF16 (real quantization error, "
           "not a no-op)")


def main():
    check_identity_at_init_relative_to_quantized_weight()
    check_matches_independent_reference_after_training_steps()
    check_conv2d_matches_independent_reference()
    check_gate_zero_produces_exactly_the_dequantized_base_output()
    check_gradients_and_parameter_membership()
    check_footprint_is_real_not_hiding_a_bf16_copy()
    check_adapted_layer_conformance_and_wrap_dispatch()
    check_end_to_end_through_adapter_strategy_scope()

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
