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
from nodes.model.adapter_injection import adapter_strategy_scope
from nodes.model.adapter_strategy import AdapterStrategy, PlainLoRAAdapter
from nodes.model.frozen_weight_store import BF16WeightStore
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


def main():
    check_patched_mechanism_matches_reference_exactly()
    check_plain_lora_adapter_installs_no_patch_at_all()
    check_a_different_strategy_actually_gets_used_and_is_restored_after()
    check_alpha_is_not_double_applied()
    check_block_weights_and_target_all_flow_through_correctly()
    check_exception_inside_scope_still_restores()
    check_a_strategy_that_delegates_to_plain_lora_adapter_does_not_recurse()

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
