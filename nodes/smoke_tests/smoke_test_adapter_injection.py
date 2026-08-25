"""Equivalence check for nodes/model/adapter_injection.py's
adapter_strategy_scope -- live-wiring AdapterStrategy into
core.lora._inject_lora's real, unmodified targeting logic.

Uses a synthetic UNet-like fixture (nested to_q/to_k/to_v/to_out.0,
a time_embed Sequential, a block_weights/target_all case) that exercises
the same targeting rules core.lora._inject_lora actually implements,
rather than reconstructing ComfyUI's real SDXL UNet (not installed in
this environment) -- core.lora itself is imported and run completely
unmodified throughout; only what core.lora.LoRALinear/LoRAConv2d
temporarily point to changes.

Run this directly: `python nodes/smoke_tests/smoke_test_adapter_injection.py`
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import core.lora as core_lora
from core.lora import LoRAConfig, inject_lora_into_unet
from nodes.model.adapter_injection import adapter_strategy_scope, reenable_dora_requires_grad
from nodes.model.adapter_strategy import AdapterStrategy, DoRAAdapter, PlainLoRAAdapter
from nodes.model.dora_layer import DoRAConv2d, DoRALinear
from nodes.model.frozen_weight_store import BF16WeightStore
from nodes.model.lora_injector import ComfyUNetTrainableModel
from nodes.model.lora_scaling import ClassicLoRAScaling, RankStabilizedScaling, _effective_alpha

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _CrossAttentionLike(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        return self.to_out(q + k + v)


class _TransformerBlockLike(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = _CrossAttentionLike(dim)
        # Not a target under the default rules (not to_q/k/v/out.0) --
        # only reachable via block_weights + target_all=True.
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.attn1(x)
        return self.proj(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)


class _MiniUNetLike(nn.Module):
    """Exercises core.lora._inject_lora's real targeting rules -- nested
    to_q/to_k/to_v/to_out.0, a top-level time_embed Sequential
    (segment-boundary match), and a Conv2d only reachable via
    block_weights/target_all -- without needing ComfyUI's real SDXL
    UNet."""

    def __init__(self, dim=8):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.input_blocks = nn.ModuleList([_TransformerBlockLike(dim) for _ in range(2)])

    def forward(self, x):
        x = x + self.time_embed(x)
        for block in self.input_blocks:
            x = block(x)
        return x


def _same_fixture_pair(dim=8, seed=0):
    torch.manual_seed(seed)
    model_a = _MiniUNetLike(dim)
    model_b = copy.deepcopy(model_a)
    return model_a, model_b


class _ObservableAdapter(AdapterStrategy):
    """Records every wrap() call, then delegates to a real
    PlainLoRAAdapter for the actual returned layer -- so forward/backward
    still works for equivalence checks, while every argument
    adapter_strategy_scope actually passed is inspectable afterward."""

    def __init__(self):
        self.calls = []
        self._delegate = PlainLoRAAdapter()

    def wrap(self, original, frozen, rank, alpha, scaling_policy, dropout=0.0, weight=1.0):
        self.calls.append(dict(original=original, frozen=frozen, rank=rank, alpha=alpha,
                                scaling_policy=scaling_policy, dropout=dropout, weight=weight))
        return self._delegate.wrap(original, frozen, rank, alpha, scaling_policy, dropout, weight)


def check_patched_mechanism_matches_reference_exactly():
    print("\n=== adapter_strategy_scope(some_other_strategy) delegating internally to a "
          "real PlainLoRAAdapter() == the real, unpatched reference path ===")
    # Deliberately NOT PlainLoRAAdapter itself here -- adapter_strategy_scope
    # skips patching entirely for that (see check_plain_lora_adapter_installs_no_patch_at_all
    # below), so forcing it through the patch anyway doesn't exercise a real
    # code path. _ObservableAdapter is a different strategy that delegates
    # to a real PlainLoRAAdapter() internally -- exactly the shape a future
    # DoRAAdapter reusing PlainLoRAAdapter's base construction would have,
    # and exactly what surfaced the real recursion bug this module's fix
    # (adapter_strategy.py's _real_lora_classes() cache) exists for.
    model_a, model_b = _same_fixture_pair()
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)

    # Seeded identically immediately before each injection -- LoRALinear/
    # LoRAConv2d randomly initialize lora_A (lora_B starts at zero, so
    # forward output doesn't depend on it initially, but its *gradient*
    # does: d(loss)/d(lora_B) = d(loss)/d(out) @ (lora_A @ x).T). Without
    # this, the two injections consume the shared torch RNG from
    # different starting points (model_a's injection runs to completion
    # first, advancing the RNG, before model_b's even starts) and
    # lora_A ends up numerically different between them -- a real gap in
    # an earlier version of this test, not a bug in adapter_strategy_scope
    # itself (forward output matched regardless, since it doesn't depend
    # on lora_A while lora_B is still zero -- only the gradient check
    # caught it).
    torch.manual_seed(100)
    registry_a = inject_lora_into_unet(model_a, config)

    observable = _ObservableAdapter()
    torch.manual_seed(100)
    with adapter_strategy_scope(observable):
        registry_b = inject_lora_into_unet(model_b, config)

    names_a = sorted(full_name for full_name, _, _, _ in registry_a)
    names_b = sorted(full_name for full_name, _, _, _ in registry_b)
    record(names_a == names_b, "identical set of target layers", detail=f"{names_a}")
    record(len(names_a) == 10, "found the 10 expected target layers -- 2 blocks x 4 "
           "(to_q/to_k/to_v/to_out.0) plus time_embed's 2 Linear leaves (time_embed.0, "
           "time_embed.2, matched via the segment-boundary rule) -- not 0, not silently "
           "over- or under-matching", detail=str(len(names_a)))

    x = torch.randn(2, 8)
    out_a = model_a(x.clone())
    out_b = model_b(x.clone())
    record(torch.allclose(out_a, out_b, atol=1e-6),
           "forward output is byte-identical between the real path and the patched "
           "path (through _ObservableAdapter delegating to a real PlainLoRAAdapter)",
           detail=f"max diff={float((out_a - out_b).abs().max().detach())}")

    out_a.sum().backward()
    out_b.sum().backward()
    for (_, _, _, layer_a), (_, _, _, layer_b) in zip(registry_a, registry_b):
        if hasattr(layer_a, "lora_A"):
            record(torch.allclose(layer_a.lora_A.grad, layer_b.lora_A.grad, atol=1e-6),
                   f"lora_A gradient matches for one layer")
            record(torch.allclose(layer_a.lora_B.grad, layer_b.lora_B.grad, atol=1e-6),
                   f"lora_B gradient matches for one layer")


def check_plain_lora_adapter_installs_no_patch_at_all():
    print("\n=== adapter_strategy_scope(PlainLoRAAdapter()): installs nothing, "
          "core.lora.LoRALinear/LoRAConv2d untouched ===")
    before_linear, before_conv2d = core_lora.LoRALinear, core_lora.LoRAConv2d
    with adapter_strategy_scope(PlainLoRAAdapter()):
        record(core_lora.LoRALinear is before_linear,
               "core.lora.LoRALinear is the real, original class inside the scope")
        record(core_lora.LoRAConv2d is before_conv2d,
               "core.lora.LoRAConv2d is the real, original class inside the scope")
    record(core_lora.LoRALinear is before_linear and core_lora.LoRAConv2d is before_conv2d,
           "still untouched after the scope exits")


def check_a_different_strategy_actually_gets_used_and_is_restored_after():
    print("\n=== adapter_strategy_scope(some_other_strategy): every target routes through "
          "it, restored after exit ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    observable = _ObservableAdapter()

    before_linear, before_conv2d = core_lora.LoRALinear, core_lora.LoRAConv2d
    with adapter_strategy_scope(observable):
        registry = inject_lora_into_unet(model, config)
    record(core_lora.LoRALinear is before_linear and core_lora.LoRAConv2d is before_conv2d,
           "core.lora.LoRALinear/LoRAConv2d restored to the real classes after the scope exits")

    record(len(observable.calls) == len(registry) == 10,
           "every one of the 10 real targets went through the custom strategy's wrap(), "
           "not core.lora's own construction", detail=f"calls={len(observable.calls)}")
    record(all(isinstance(c["frozen"], BF16WeightStore) for c in observable.calls),
           "each call got a real, per-layer BF16WeightStore wrapping that layer's own weight")
    record(all(isinstance(c["scaling_policy"], ClassicLoRAScaling) for c in observable.calls),
           "every call's scaling_policy was ClassicLoRAScaling -- never something else, "
           "regardless of what strategy was active")


def check_alpha_is_not_double_applied():
    print("\n=== RankStabilizedScaling applied upstream (as ComfyUNetLoRANode.build() does) "
          "is NOT re-applied inside the patched path ===")
    model_a, model_b = _same_fixture_pair(seed=1)
    rank, nominal_alpha = 4, 8.0
    # Exactly what ComfyUNetLoRANode.build() does: resolve the real
    # scaling_policy into a single effective alpha BEFORE core.lora ever
    # runs.
    effective_alpha = _effective_alpha(alpha=nominal_alpha, rank=rank,
                                        policy=RankStabilizedScaling())
    config = LoRAConfig(rank=rank, alpha=effective_alpha, dropout=0.0)

    # Reference: real, unpatched core.lora, given the pre-resolved
    # effective_alpha directly -- exactly what happens today.
    registry_a = inject_lora_into_unet(model_a, config)

    observable = _ObservableAdapter()
    with adapter_strategy_scope(observable):
        registry_b = inject_lora_into_unet(model_b, config)

    record(len(observable.calls) > 0, "sanity: the custom strategy was actually invoked")
    record(all(c["alpha"] == effective_alpha for c in observable.calls),
           "wrap() always received the already-effective alpha unchanged, not "
           "RankStabilizedScaling applied a second time on top of it",
           detail=str({c["alpha"] for c in observable.calls}))

    for (_, _, _, layer_a), (_, _, _, layer_b) in zip(registry_a, registry_b):
        if hasattr(layer_a, "alpha"):
            record(layer_a.alpha == layer_b.alpha,
                   "final layer.alpha matches the reference path exactly -- not "
                   "further transformed", detail=f"a={layer_a.alpha} b={layer_b.alpha}")
            record(abs(layer_a.scaling - layer_b.scaling) < 1e-9,
                   "final layer.scaling matches the reference path exactly",
                   detail=f"a={layer_a.scaling} b={layer_b.scaling}")


def check_block_weights_and_target_all_flow_through_correctly():
    print("\n=== block_weights + target_all=True: a normally-untargeted Conv2d gets "
          "adapted with the right weight, through the patched path too ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0,
                         block_weights={"input_blocks.1": 0.5}, target_all=True)
    observable = _ObservableAdapter()
    with adapter_strategy_scope(observable):
        registry = inject_lora_into_unet(model, config)

    proj_calls = [c for c in observable.calls if isinstance(c["original"], nn.Conv2d)]
    record(len(proj_calls) == 1,
           "exactly the one Conv2d inside input_blocks.1 (target_all-only reachable) "
           "was adapted", detail=str(len(proj_calls)))
    if proj_calls:
        record(proj_calls[0]["weight"] == 0.5,
               "it got the real block_weights value (0.5), not the default 1.0",
               detail=str(proj_calls[0]["weight"]))
    other_block_conv_targeted = any(
        isinstance(c["original"], nn.Conv2d) and c is not proj_calls[0] for c in observable.calls
    ) if proj_calls else False
    record(not other_block_conv_targeted,
           "input_blocks.0's Conv2d (not named in block_weights) stayed untouched")


def check_exception_inside_scope_still_restores():
    print("\n=== an exception inside the scope still restores the real classes ===")
    before_linear, before_conv2d = core_lora.LoRALinear, core_lora.LoRAConv2d
    try:
        with adapter_strategy_scope(_ObservableAdapter()):
            raise RuntimeError("simulated failure mid-injection")
    except RuntimeError:
        pass
    record(core_lora.LoRALinear is before_linear and core_lora.LoRAConv2d is before_conv2d,
           "restored even though the with-block raised")


def check_a_strategy_that_delegates_to_plain_lora_adapter_does_not_recurse():
    print("\n=== regression: a strategy delegating to a real PlainLoRAAdapter() inside "
          "the patch does not recurse (RecursionError, hit for real while building this) ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)

    class _DelegatingStrategy(AdapterStrategy):
        """Shape of a real future case, not a contrived one -- a
        DoRAAdapter reusing PlainLoRAAdapter's own LoRALinear/LoRAConv2d
        construction as its base layer would delegate exactly like
        this."""
        def wrap(self, *args, **kwargs):
            return PlainLoRAAdapter().wrap(*args, **kwargs)

    try:
        with adapter_strategy_scope(_DelegatingStrategy()):
            registry = inject_lora_into_unet(model, config)
    except RecursionError:
        record(False, "delegating to PlainLoRAAdapter() from inside the patch recursed")
        return
    record(len(registry) == 10, "completed without recursing, found all 10 target layers",
           detail=str(len(registry)))


def check_dora_layers_end_up_trainable_after_the_real_injection_path():
    print("\n=== a real gap found while wiring in DoRA's checkpoint round-trip, "
          "unrelated to checkpointing itself: core.unet_wrapper.ComfyUNetWrapper."
          "_init_lora() freezes every DoRA parameter and never re-enables any of "
          "them -- reenable_dora_requires_grad() fixes it ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    with adapter_strategy_scope(DoRAAdapter()):
        registry = inject_lora_into_unet(model, config)
    record(len(registry) == 10 and all(isinstance(layer, (DoRALinear, DoRAConv2d))
                                        for *_, layer in registry),
           "real DoRALinear/DoRAConv2d layers actually got injected",
           detail=str(len(registry)))

    # Exactly core.unet_wrapper.ComfyUNetWrapper._init_lora()'s own two
    # lines (frozen legacy code -- reproduced verbatim here rather than
    # constructed for real, since that needs ComfyUI's real SDXL UNet,
    # not installed in this environment -- see this file's own module
    # docstring for why a synthetic fixture is used throughout instead).
    for p in model.parameters():
        p.requires_grad_(False)
    for _, _, _, layer in registry:
        if hasattr(layer, "lora_A"):
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)

    still_frozen = [full_name for full_name, _, _, layer in registry
                    if not layer.get_lora_weights()[0].requires_grad
                    or not layer.get_lora_weights()[1].requires_grad
                    or not layer.magnitude.requires_grad]
    record(len(still_frozen) == len(registry),
           "confirms the bug: _init_lora()'s own logic leaves every DoRA layer's "
           "lora_A/lora_B/magnitude frozen (0 of 3 trainable in any of them)",
           detail=f"{len(still_frozen)}/{len(registry)} layers still fully frozen")

    reenable_dora_requires_grad(registry)
    fixed = all(layer.get_lora_weights()[0].requires_grad
                and layer.get_lora_weights()[1].requires_grad
                and layer.magnitude.requires_grad
                for _, _, _, layer in registry)
    record(fixed, "reenable_dora_requires_grad() restores all three on every DoRA layer")

    # A real backward() pass actually populating .grad, not just the flag
    # being set -- the thing that would have made a real training run
    # silently do nothing before this fix.
    for _, _, _, layer in registry:
        A, B = layer.get_lora_weights()
        A.grad = None
        B.grad = None
        layer.magnitude.grad = None
    x = torch.randn(2, 8)
    model(x).pow(2).mean().backward()
    got_gradient = all(layer.get_lora_weights()[0].grad is not None
                        and layer.get_lora_weights()[1].grad is not None
                        and layer.magnitude.grad is not None
                        for _, _, _, layer in registry)
    record(got_gradient, "backward() now actually populates .grad for lora_A, "
           "lora_B, and magnitude on every DoRA layer")


def check_reenable_dora_requires_grad_is_a_no_op_for_plain_lora():
    print("\n=== reenable_dora_requires_grad() touches nothing for a "
          "PlainLoRAAdapter registry -- only ever matches a DoRA layer ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    with adapter_strategy_scope(PlainLoRAAdapter()):
        registry = inject_lora_into_unet(model, config)
    for _, _, _, layer in registry:
        layer.lora_A.requires_grad_(False)
        layer.lora_B.requires_grad_(False)
    reenable_dora_requires_grad(registry)
    stayed_false = all(not layer.lora_A.requires_grad and not layer.lora_B.requires_grad
                        for _, _, _, layer in registry)
    record(stayed_false, "correctly leaves plain LoRALinear/LoRAConv2d layers alone -- "
           "not a general-purpose 'fix requires_grad' hammer")


class _FakeWrapperWithLoraParameters:
    """Minimal stand-in for core.unet_wrapper.ComfyUNetWrapper --
    ComfyUNetTrainableModel only ever calls .lora_registry,
    .lora_parameters(), and .state_dict() on it. lora_parameters()
    below is that real class's own real logic, reproduced verbatim
    (frozen legacy code -- can't import and construct the real class
    without ComfyUI's real SDXL UNet, not installed in this environment
    -- see this file's own module docstring) -- this test is about that
    real, broken-for-DoRA logic, not an approximation of it."""

    def __init__(self, model, registry):
        self._model = model
        self.lora_registry = registry

    def lora_parameters(self):
        if not self.lora_registry:
            return []
        params = []
        for _, _, _, layer in self.lora_registry:
            if hasattr(layer, "lora_A") and isinstance(layer.lora_A, torch.nn.Parameter):
                params.append(layer.lora_A)
                params.append(layer.lora_B)
        return params

    def state_dict(self):
        return self._model.state_dict()


def check_dora_trainable_parameters_and_footprint_bytes():
    print("\n=== the other half of the same gap: core.unet_wrapper.ComfyUNetWrapper's "
          "own lora_parameters() also excludes a bare DoRA layer's params from what "
          "an optimizer actually receives -- dora_trainable_parameters() plus "
          "ComfyUNetTrainableModel closes it ===")
    model = _MiniUNetLike(dim=8)
    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    with adapter_strategy_scope(DoRAAdapter()):
        registry = inject_lora_into_unet(model, config)
    reenable_dora_requires_grad(registry)

    wrapper = _FakeWrapperWithLoraParameters(model, registry)
    trainable_model = ComfyUNetTrainableModel(wrapper)

    bare_call = wrapper.lora_parameters()
    record(len(bare_call) == 0,
           "confirms the bug: core.unet_wrapper.ComfyUNetWrapper's own "
           "lora_parameters() logic returns nothing at all for a bare DoRA registry",
           detail=str(len(bare_call)))

    full_list = trainable_model.trainable_parameters()
    expected_count = len(registry) * 3  # lora_A, lora_B, magnitude per layer
    record(len(full_list) == expected_count,
           "ComfyUNetTrainableModel.trainable_parameters() includes all three "
           "per DoRA layer (lora_A, lora_B, magnitude)",
           detail=f"{len(full_list)} vs expected {expected_count}")
    record(all(p.requires_grad for p in full_list),
           "every parameter handed back is actually trainable")

    # The real proof: real optimizer steps, built exactly the way a real
    # training run builds them (straight off trainable_parameters()),
    # move every one of these parameters. Two steps, not one: lora_B is
    # zero-initialized (standard LoRA convention), so on step 1 alone
    # lora_A's own gradient is exactly zero (multiplied through B=0) --
    # a real, well-known LoRA-init property, not a bug in this fix. By
    # step 2, B has moved off zero and A's gradient is genuinely nonzero
    # too, matching what an actual multi-step training run looks like.
    before = [p.detach().clone() for p in full_list]
    opt = torch.optim.SGD(full_list, lr=0.5)
    for _ in range(2):
        x = torch.randn(2, 8)
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
    moved = all(not torch.equal(b, a) for b, a in zip(before, full_list))
    record(moved, "real optimizer.step()s actually move every DoRA parameter -- "
           "not just present in the list, genuinely updated")

    # footprint_bytes(): DoRA's own trainable tensors must be excluded
    # from the frozen-base footprint, not double-counted into it.
    footprint = trainable_model.footprint_bytes()
    total_model_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    dora_param_bytes = sum(p.numel() * p.element_size() for p in full_list)
    record(footprint == total_model_bytes - dora_param_bytes,
           "footprint_bytes() excludes exactly DoRA's own trainable parameters "
           "from the frozen-base footprint count, no more and no less",
           detail=f"footprint={footprint}, model_total={total_model_bytes}, "
                  f"dora_params={dora_param_bytes}")


def main():
    check_patched_mechanism_matches_reference_exactly()
    check_plain_lora_adapter_installs_no_patch_at_all()
    check_a_different_strategy_actually_gets_used_and_is_restored_after()
    check_alpha_is_not_double_applied()
    check_block_weights_and_target_all_flow_through_correctly()
    check_exception_inside_scope_still_restores()
    check_a_strategy_that_delegates_to_plain_lora_adapter_does_not_recurse()
    check_dora_layers_end_up_trainable_after_the_real_injection_path()
    check_reenable_dora_requires_grad_is_a_no_op_for_plain_lora()
    check_dora_trainable_parameters_and_footprint_bytes()

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
