"""ComfyUNetLoRANode.build() was refactored into a thin wrapper around
build_lora_injected_unet() (nodes/model/lora_injector.py) -- extracted
so the Resources Controller redesign's Phase 5 has a real function to
call rather than duplicating this construction logic later (see
docs/resources_controller_redesign_plan.md's "Consolidation" section
for why that matters).

Can't exercise this fully end to end -- core.unet_wrapper.ComfyUNetWrapper
needs ComfyUI's real SDXL UNet class, not installed in this environment
(every other smoke test in this project that touches UNet construction
has the same real constraint). So instead: patch ComfyUNetWrapper and
adapter_strategy_scope to record exactly what they were called with,
and check those recorded calls against what the *old*, pre-extraction
inline logic would have computed for the same inputs -- proving the
refactor is faithful, not just that it doesn't crash. reenable_dora_requires_grad
is left real, unpatched (the fake wrapper's lora_registry is empty, so
it's a real no-op call, not a mocked one).
"""

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import core.unet_wrapper as unet_wrapper_module
from core.lora import LoRAConfig
from nodes.model import adapter_injection
from nodes.model.gradient_checkpointing import FrozenParamSafeCheckpointing, NoCheckpointing
from nodes.model.handle import ModelWeights
from nodes.model.lora_injector import ComfyUNetLoRANode, build_lora_injected_unet
from nodes.model.lora_scaling import ClassicLoRAScaling, RankStabilizedScaling
from nodes.resource_policy import ResourcePolicy
from nodes.smoke_tests.smoke_test_gradient_checkpointing import _install_stub_comfy_checkpoint_module

# checkpointing_strategy.apply() (called inside build_lora_injected_unet,
# pre-existing behavior this test doesn't change) reaches into real
# ComfyUI internals to patch gradient checkpointing -- not installed in
# this sandbox. Reuses the same faithful stub
# smoke_test_gradient_checkpointing.py already built and verified,
# rather than a second, separate mock of the same real module.
_install_stub_comfy_checkpoint_module()


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _RecordingWrapper:
    """Stands in for core.unet_wrapper.ComfyUNetWrapper -- records its
    own construction args, exposes just enough (.lora_registry,
    .lora_parameters(), .model/.state_dict()) for
    reenable_dora_requires_grad(), ComfyUNetTrainableModel's
    trainable_parameters()/footprint_bytes(), and anything else that
    only needs a registry/state-dict-shaped object (not a real UNet
    forward pass) to work against it for real, not further mocked."""

    def __init__(self, unet_sd, device, dtype, use_checkpoint, lora_config):
        self.unet_sd = unet_sd
        self.device = device
        self.dtype = dtype
        self.use_checkpoint = use_checkpoint
        self.lora_config = lora_config
        self.lora_registry = []
        self.model = torch.nn.Linear(4, 4)  # real nn.Module -- real .state_dict()

    def lora_parameters(self):
        # Mirrors core.unet_wrapper.ComfyUNetWrapper.lora_parameters()'s
        # own real early-return for an empty registry -- this fixture's
        # lora_registry is always [] (no real LoRA layers get injected
        # against a fake unet_sd), so this is that same real code path,
        # not a separate guess at its behavior.
        return []

    def state_dict(self):
        return self.model.state_dict()


class _Recorder:
    def __init__(self):
        self.wrapper_calls = []
        self.scope_calls = []

    def install(self):
        recorder = self

        def fake_wrapper(unet_sd, device, dtype, use_checkpoint, lora_config):
            recorder.wrapper_calls.append(dict(
                unet_sd=unet_sd, device=device, dtype=dtype,
                use_checkpoint=use_checkpoint, lora_config=lora_config))
            return _RecordingWrapper(unet_sd, device, dtype, use_checkpoint, lora_config)

        @contextlib.contextmanager
        def fake_scope(adapter_strategy, frozen_weight_store_factory=None):
            recorder.scope_calls.append(dict(
                adapter_strategy=adapter_strategy,
                frozen_weight_store_factory=frozen_weight_store_factory))
            yield

        self._real_wrapper = unet_wrapper_module.ComfyUNetWrapper
        self._real_scope = adapter_injection.adapter_strategy_scope
        unet_wrapper_module.ComfyUNetWrapper = fake_wrapper
        adapter_injection.adapter_strategy_scope = fake_scope

    def uninstall(self):
        unet_wrapper_module.ComfyUNetWrapper = self._real_wrapper
        adapter_injection.adapter_strategy_scope = self._real_scope


def check_defaults_match_the_old_inline_logic():
    print("[build_lora_injected_unet(): defaults match exactly what the pre-extraction "
          "inline code in ComfyUNetLoRANode.build() used to compute]")
    rec = _Recorder()
    rec.install()
    try:
        weights = ModelWeights.from_state_dicts({"model.diffusion_model.x": "fake_tensor"}, {})
        model = build_lora_injected_unet(weights)
    finally:
        rec.uninstall()

    check(len(rec.wrapper_calls) == 1, f"expected 1 ComfyUNetWrapper call, got {len(rec.wrapper_calls)}")
    call = rec.wrapper_calls[0]
    check(call["device"] == "xpu", call["device"])
    import torch
    check(call["dtype"] == torch.bfloat16, call["dtype"])
    check(call["use_checkpoint"] is True, call["use_checkpoint"])
    # ClassicLoRAScaling is an identity -- effective_alpha == nominal alpha (1.0),
    # exactly matching the old inline code's default behavior.
    check(call["lora_config"] == LoRAConfig(rank=64, alpha=1.0, dropout=0.0), call["lora_config"])

    check(len(rec.scope_calls) == 1, f"expected 1 adapter_strategy_scope call, got {len(rec.scope_calls)}")
    from nodes.model.adapter_strategy import PlainLoRAAdapter
    check(isinstance(rec.scope_calls[0]["adapter_strategy"], PlainLoRAAdapter),
          rec.scope_calls[0]["adapter_strategy"])
    check(rec.scope_calls[0]["frozen_weight_store_factory"] is None,
          rec.scope_calls[0]["frozen_weight_store_factory"])
    check(model._wrapper is not None, "should return a real ComfyUNetTrainableModel")
    print("    PASS")


def check_resource_policy_overrides_use_checkpoint_and_scaling():
    print("[resource_policy given -> overrides use_checkpoint/scaling_policy, "
          "exactly like the old inline if/else did]")

    class _FakeResourcePolicy(ResourcePolicy):
        def checkpointing_strategy(self):
            return NoCheckpointing()

        def lora_scaling_policy(self):
            return RankStabilizedScaling()

        def parameter_group_policy(self):
            # Not read by build_lora_injected_unet() at all (it only calls
            # checkpointing_strategy()/lora_scaling_policy()) -- implemented
            # only because ResourcePolicy is an ABC requiring all three.
            raise NotImplementedError("not exercised by this test")

    rec = _Recorder()
    rec.install()
    try:
        weights = ModelWeights.from_state_dicts({}, {})
        build_lora_injected_unet(weights, rank=100, alpha=8.0, resource_policy=_FakeResourcePolicy(),
                                  use_checkpoint=True)  # deliberately contradicted by the policy
    finally:
        rec.uninstall()

    call = rec.wrapper_calls[0]
    check(call["use_checkpoint"] is False,
          "resource_policy's NoCheckpointing should win over the explicit use_checkpoint=True argument")
    expected_alpha = RankStabilizedScaling().scaling(8.0, 100) * 100
    check(abs(call["lora_config"].alpha - expected_alpha) < 1e-6,
          f"expected effective alpha {expected_alpha}, got {call['lora_config'].alpha}")
    print("    PASS")


def check_node_thin_wrapper_resolves_port_defaults_correctly():
    print("[ComfyUNetLoRANode.build() itself -- the thin wrapper -- resolves its own "
          "Port defaults into build_lora_injected_unet() correctly]")
    rec = _Recorder()
    rec.install()
    try:
        weights = ModelWeights.from_state_dicts({}, {})
        node = ComfyUNetLoRANode()
        result = node.build(weights=weights)
    finally:
        rec.uninstall()

    check("model" in result, result)
    call = rec.wrapper_calls[0]
    check(call["device"] == "xpu" and call["use_checkpoint"] is True, call)
    check(call["lora_config"] == LoRAConfig(rank=64, alpha=1.0, dropout=0.0), call["lora_config"])

    # And a non-default value actually flows through end to end, not just defaults.
    rec2 = _Recorder()
    rec2.install()
    try:
        node.build(weights=weights, rank=16, alpha=4.0, device="cpu", use_checkpoint=False)
    finally:
        rec2.uninstall()
    call2 = rec2.wrapper_calls[0]
    check(call2["device"] == "cpu", call2["device"])
    check(call2["use_checkpoint"] is False, call2["use_checkpoint"])
    check(call2["lora_config"] == LoRAConfig(rank=16, alpha=4.0, dropout=0.0), call2["lora_config"])
    print("    PASS")


def main():
    check_defaults_match_the_old_inline_logic()
    check_resource_policy_overrides_use_checkpoint_and_scaling()
    check_node_thin_wrapper_resolves_port_defaults_correctly()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
