"""Correctness/equivalence check for nodes/model/dora_layer.py's
DoRALinear/DoRAConv2d and nodes/model/adapter_strategy.py's DoRAAdapter.

Cross-checks against independent reference computations written directly
in this file (a naive "materialize the full merged+normalized weight,
then call F.linear/F.conv2d once" path) rather than re-testing the
module's own _weight_norm_linear/_weight_norm_conv2d helpers against
themselves -- two structurally different code paths for the same math,
not one path checked against its own arithmetic.

Run this directly: `python nodes/smoke_tests/smoke_test_dora_layer.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

import core.lora as core_lora
from nodes.model.adapter_strategy import AdaptedLayer, DoRAAdapter, PlainLoRAAdapter
from nodes.model.dora_layer import DoRAConv2d, DoRALinear
from nodes.model.frozen_weight_store import BF16WeightStore
from nodes.model.lora_scaling import ClassicLoRAScaling

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _reference_dora_linear(x, base_weight, base_bias, lora_A, lora_B, magnitude, scaling):
    """Independent of dora_layer.py's own code: materialize the full
    merged+normalized weight directly, one F.linear call -- the naive
    approach DoRALinear.forward()'s "efficient" decomposition is claimed
    to be algebraically equivalent to."""
    merged = base_weight.to(lora_A.dtype) + scaling * (lora_B @ lora_A)
    weight_norm = merged.norm(dim=1)
    w_dora = (magnitude / weight_norm).view(-1, 1) * merged
    return F.linear(x, w_dora.to(x.dtype), base_bias)


def _reference_dora_conv2d(x, base_weight, base_bias, lora_A, lora_B, magnitude, scaling,
                            stride, padding, dilation, groups, out_channels, rank):
    re_A = lora_A.reshape(rank, -1)
    re_B = lora_B.reshape(out_channels, rank)
    delta = (re_B @ re_A).view(base_weight.shape)
    merged = base_weight.to(lora_A.dtype) + scaling * delta
    dims = tuple(range(1, merged.dim()))
    weight_norm = merged.norm(p=2, dim=dims)
    w_dora = (magnitude / weight_norm).view(-1, 1, 1, 1) * merged
    return F.conv2d(x, w_dora.to(x.dtype), base_bias, stride, padding, dilation, groups)


def check_identity_at_init_linear():
    print("\n=== DoRALinear at init: output exactly matches the frozen base "
          "(lora_B starts at zero, magnitude starts at ||base_weight||) ===")
    torch.manual_seed(0)
    base = nn.Linear(6, 5)
    layer = DoRALinear(base, rank=3, alpha=6.0, dropout=0.0)
    x = torch.randn(2, 6)
    out = layer(x)
    ref = F.linear(x, base.weight, base.bias)
    record(torch.allclose(out, ref, atol=1e-5),
           "DoRALinear(x) == F.linear(x, base.weight, base.bias) at init",
           detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_identity_at_init_conv2d():
    print("\n=== DoRAConv2d at init: same identity property ===")
    torch.manual_seed(0)
    base = nn.Conv2d(4, 6, kernel_size=3, padding=1)
    layer = DoRAConv2d(base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(2, 4, 8, 8)
    out = layer(x)
    ref = F.conv2d(x, base.weight, base.bias, padding=1)
    record(torch.allclose(out, ref, atol=1e-4),
           "DoRAConv2d(x) == F.conv2d(x, base.weight, base.bias, ...) at init",
           detail=f"max diff={float((out - ref).abs().max().detach())}")


def check_linear_matches_independent_reference_after_training_steps():
    print("\n=== DoRALinear matches an independent reference implementation, "
          "after real training steps move lora_A/lora_B/magnitude away from init ===")
    torch.manual_seed(1)
    base = nn.Linear(6, 5)
    layer = DoRALinear(base, rank=3, alpha=6.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(4, 6)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(3, 6)
    out = layer(x)
    ref = _reference_dora_linear(x, layer._lora.base_weight, layer._lora.base_bias,
                                  layer._lora.lora_A, layer._lora.lora_B,
                                  layer.magnitude, layer.scaling)
    record(torch.allclose(out, ref, atol=1e-4),
           "DoRALinear's efficient forward matches the naive materialize-then-linear "
           "reference, after training has moved every parameter away from init",
           detail=f"max diff={float((out - ref).abs().max())}")


def check_conv2d_matches_independent_reference_after_training_steps():
    print("\n=== DoRAConv2d matches an independent reference implementation, "
          "after real training steps (also cross-checks the two-conv-composition "
          "trick against a single merged-kernel conv) ===")
    torch.manual_seed(2)
    base = nn.Conv2d(4, 6, kernel_size=3, padding=1)
    layer = DoRAConv2d(base, rank=2, alpha=4.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(2, 4, 8, 8)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(2, 4, 8, 8)
    out = layer(x)
    ref = _reference_dora_conv2d(x, layer._lora.base_weight, layer._lora.base_bias,
                                  layer._lora.lora_A, layer._lora.lora_B, layer.magnitude,
                                  layer.scaling, layer.stride, layer.padding, layer.dilation,
                                  layer.groups, layer.out_channels, layer.rank)
    record(torch.allclose(out, ref, atol=1e-3),
           "DoRAConv2d's efficient forward (two-conv composition for the LoRA term) "
           "matches the naive single-merged-kernel-conv reference",
           detail=f"max diff={float((out - ref).abs().max())}")


def check_gate_zero_produces_exactly_the_frozen_base_output():
    print("\n=== gate=0 produces exactly the frozen base output, matching "
          "LoRALinear's own gate semantics, even after training ===")
    torch.manual_seed(3)
    base = nn.Linear(6, 5)
    layer = DoRALinear(base, rank=3, alpha=6.0, dropout=0.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(4, 6)
        loss = layer(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    x = torch.randn(3, 6)
    core_lora.set_lora_gate(torch.zeros(3))
    try:
        out = layer(x)
    finally:
        core_lora.set_lora_gate(None)
    ref = F.linear(x, layer._lora.base_weight, layer._lora.base_bias)
    record(torch.allclose(out, ref, atol=1e-5),
           "gate=0 -> exactly base_weight's own forward, even though magnitude/lora_A/"
           "lora_B have all moved away from their init values",
           detail=f"max diff={float((out - ref).abs().max())}")


def check_magnitude_is_a_real_trainable_parameter():
    print("\n=== magnitude is a real Parameter: in .parameters(), receives a real "
          "gradient, base_weight/base_bias do not ===")
    base = nn.Linear(6, 5)
    layer = DoRALinear(base, rank=3, alpha=6.0, dropout=0.0)
    param_names = {n for n, _ in layer.named_parameters()}
    record("magnitude" in param_names, "magnitude is in named_parameters()",
           detail=str(param_names))
    record(not any("base_weight" in n or "base_bias" in n for n in param_names),
           "base_weight/base_bias are NOT in named_parameters() (they're buffers)",
           detail=str(param_names))

    x = torch.randn(2, 6)
    layer(x).sum().backward()
    record(layer.magnitude.grad is not None and bool((layer.magnitude.grad != 0).any()),
           "magnitude received a real, nonzero gradient")
    record(layer._lora.lora_A.grad is not None, "lora_A received a gradient")
    # lora_B starts at zero; at init its gradient can be legitimately zero
    # for some loss shapes, so just check it exists, not that it's nonzero.
    record(layer._lora.lora_B.grad is not None, "lora_B received a gradient (object exists)")


def check_adapted_layer_conformance():
    print("\n=== DoRALinear/DoRAConv2d conform to AdaptedLayer ===")
    # Registration happens lazily inside DoRAAdapter.wrap() (same pattern
    # as _register_legacy_adapted_layers() for core.lora's classes) --
    # constructing DoRALinear/DoRAConv2d directly, as this test does,
    # bypasses that, so trigger it explicitly rather than relying on
    # another test having already called wrap() first.
    from nodes.model.adapter_strategy import _register_dora_adapted_layers
    _register_dora_adapted_layers()

    linear_layer = DoRALinear(nn.Linear(4, 4), rank=2, alpha=4.0)
    conv_layer = DoRAConv2d(nn.Conv2d(4, 4, 3, padding=1), rank=2, alpha=4.0)
    record(isinstance(linear_layer, AdaptedLayer), "DoRALinear isinstance AdaptedLayer")
    record(isinstance(conv_layer, AdaptedLayer), "DoRAConv2d isinstance AdaptedLayer")
    a, b = linear_layer.get_lora_weights()
    record(a is linear_layer._lora.lora_A and b is linear_layer._lora.lora_B,
           "get_lora_weights() returns the real lora_A/lora_B")


def check_load_lora_weights_vs_load_dora_weights():
    print("\n=== load_lora_weights() recomputes magnitude; load_dora_weights() "
          "restores the real, independently-trained one ===")
    base = nn.Linear(6, 5)
    layer = DoRALinear(base, rank=3, alpha=6.0, dropout=0.0)
    new_A = torch.randn(3, 6)
    new_B = torch.randn(5, 3)

    layer.load_lora_weights(new_A, new_B)
    recomputed = layer.magnitude.clone()
    expected_recompute = (layer._lora.base_weight.to(new_A.dtype)
                           + layer.scaling * (new_B @ new_A)).norm(dim=1)
    record(torch.allclose(recomputed, expected_recompute, atol=1e-5),
           "load_lora_weights() recomputes magnitude from the new A/B, matching a "
           "direct reference computation")

    real_trained_magnitude = torch.rand(5) * 10 + 1
    layer.load_dora_weights(new_A, new_B, real_trained_magnitude)
    record(torch.allclose(layer.magnitude, real_trained_magnitude),
           "load_dora_weights() restores the given magnitude exactly, "
           "not the recomputed one", detail=f"{layer.magnitude} vs {real_trained_magnitude}")


def check_dora_adapter_wrap_contract():
    print("\n=== DoRAAdapter.wrap(): type checks and correct layer construction ===")
    adapter = DoRAAdapter()
    linear = nn.Linear(4, 4)
    conv = nn.Conv2d(4, 4, 3, padding=1)
    frozen_linear = BF16WeightStore(linear.weight)
    frozen_conv = BF16WeightStore(conv.weight)

    result = adapter.wrap(linear, frozen_linear, rank=2, alpha=4.0,
                           scaling_policy=ClassicLoRAScaling())
    record(isinstance(result, DoRALinear), "wrap() on nn.Linear returns a DoRALinear")

    result_conv = adapter.wrap(conv, frozen_conv, rank=2, alpha=4.0,
                                scaling_policy=ClassicLoRAScaling())
    record(isinstance(result_conv, DoRAConv2d), "wrap() on nn.Conv2d returns a DoRAConv2d")

    class _FakeFrozen:
        pass
    try:
        adapter.wrap(linear, _FakeFrozen(), rank=2, alpha=4.0, scaling_policy=ClassicLoRAScaling())
        record(False, "wrap() with a non-BF16WeightStore frozen should raise NotImplementedError")
    except NotImplementedError:
        record(True, "wrap() rejects a non-BF16WeightStore frozen, same as PlainLoRAAdapter")

    try:
        adapter.wrap(nn.Embedding(4, 4), frozen_linear, rank=2, alpha=4.0,
                      scaling_policy=ClassicLoRAScaling())
        record(False, "wrap() with an unsupported original type should raise TypeError")
    except TypeError:
        record(True, "wrap() rejects an unsupported original module type")


def check_dora_through_adapter_strategy_scope_end_to_end():
    print("\n=== DoRAAdapter through the real adapter_strategy_scope mechanism "
          "(not just standalone construction) -- same target set as PlainLoRAAdapter, "
          "identity at init holds end-to-end ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smoke_test_adapter_injection import _MiniUNetLike, _same_fixture_pair
    from core.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.adapter_injection import adapter_strategy_scope

    model_plain, model_dora = _same_fixture_pair()
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)

    registry_plain = inject_lora_into_unet(model_plain, config)
    with adapter_strategy_scope(DoRAAdapter()):
        registry_dora = inject_lora_into_unet(model_dora, config)

    names_plain = sorted(n for n, _, _, _ in registry_plain)
    names_dora = sorted(n for n, _, _, _ in registry_dora)
    record(names_plain == names_dora,
           "DoRAAdapter targets exactly the same layers PlainLoRAAdapter does "
           "(same core.lora._inject_lora targeting logic, unmodified either way)")
    record(all(isinstance(layer, DoRALinear) for _, _, _, layer in registry_dora),
           "every targeted layer really is a DoRALinear (this fixture has no Conv2d "
           "targets under default rules)")
    record(all(hasattr(layer, "magnitude") for _, _, _, layer in registry_dora),
           "every one has a real magnitude parameter")

    x = torch.randn(2, 8)
    out_plain = model_plain(x.clone())
    out_dora = model_dora(x.clone())
    record(torch.allclose(out_plain, out_dora, atol=1e-5),
           "at init, DoRA's output matches PlainLoRAAdapter's exactly -- both are "
           "identity transforms of the same frozen base at this point",
           detail=f"max diff={float((out_plain - out_dora).abs().max().detach())}")


def main():
    check_identity_at_init_linear()
    check_identity_at_init_conv2d()
    check_linear_matches_independent_reference_after_training_steps()
    check_conv2d_matches_independent_reference_after_training_steps()
    check_gate_zero_produces_exactly_the_frozen_base_output()
    check_magnitude_is_a_real_trainable_parameter()
    check_adapted_layer_conformance()
    check_load_lora_weights_vs_load_dora_weights()
    check_dora_adapter_wrap_contract()
    check_dora_through_adapter_strategy_scope_end_to_end()

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
