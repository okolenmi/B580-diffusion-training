"""ComfyUNetLoRANode.build() was refactored into a thin wrapper around
build_lora_injected_unet() (nodes/model/lora_injector.py) -- extracted
so the Resources Controller redesign's Phase 5 has a real function to
call rather than duplicating this construction logic later (see
docs/design/resources-controller/08-consolidation.md
for why that matters).

Can't exercise this fully end to end -- unet_wrapper.ComfyUNetWrapper
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import nodes.model.unet_wrapper as unet_wrapper_module
from nodes.model.lora import LoRAConfig
from nodes.model.handle import ModelWeights
from nodes.model.lora_injector import ComfyUNetLoRANode, build_lora_injected_unet
from nodes.model.lora_scaling import ClassicLoRAScaling
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
    """Stands in for ComfyUNetWrapper -- records its
    own construction args, exposes just enough (.lora_registry,
    .lora_parameters(), .model/.state_dict()) for
    reenable_dora_requires_grad(), ComfyUNetTrainableModel's
    trainable_parameters()/footprint_bytes(), and anything else that
    only needs a registry/state-dict-shaped object (not a real UNet
    forward pass) to work against it for real, not further mocked."""

    def __init__(self, unet_sd, device, dtype, use_checkpoint, lora_config,
                 layer_classes=None):
        self.unet_sd = unet_sd
        self.device = device
        self.dtype = dtype
        self.use_checkpoint = use_checkpoint
        self.lora_config = lora_config
        self.layer_classes = layer_classes
        self.lora_registry = []
        self.model = torch.nn.Linear(4, 4)  # real nn.Module -- real .state_dict()

    def lora_parameters(self):
        # Mirrors ComfyUNetWrapper.lora_parameters()'s
        # own real early-return for an empty registry -- this fixture's
        # lora_registry is always [] (no real LoRA layers get injected
        # against a fake unet_sd), so this is that same real code path,
        # not a separate guess at its behavior.
        return []

    def state_dict(self):
        return self.model.state_dict()

    def to(self, device=None, **kwargs):
        # Mirrors ComfyUNetWrapper.to()'s own real
        # behavior exactly (self.model.to(...), then update self.device
        # only if a device was actually given) -- needed for
        # ComfyUNetTrainableModel.offload()/reload()/release(), all of
        # which call self._wrapper.to(device=...) directly.
        self.model.to(device=device, **kwargs)
        if device is not None:
            self.device = str(device)
        return self


class _Recorder:
    """Captures what build_lora_injected_unet() hands the UNet wrapper.

    Only the wrapper call needs intercepting now. It used to also patch
    `adapter_injection.adapter_strategy_scope`, because that context
    manager was how the adapter strategy was passed in; it is now an
    argument (`layer_classes=`) on the very same constructor call, so the
    strategy is observable right here without a second patch point.
    """

    def __init__(self):
        self.wrapper_calls = []

    def install(self):
        recorder = self

        def fake_wrapper(unet_sd, device, dtype, use_checkpoint, lora_config,
                         layer_classes=None):
            recorder.wrapper_calls.append(dict(
                unet_sd=unet_sd, device=device, dtype=dtype,
                use_checkpoint=use_checkpoint, lora_config=lora_config,
                layer_classes=layer_classes))
            return _RecordingWrapper(unet_sd, device, dtype, use_checkpoint,
                                     lora_config, layer_classes)

        self._real_wrapper = unet_wrapper_module.ComfyUNetWrapper
        unet_wrapper_module.ComfyUNetWrapper = fake_wrapper

    def uninstall(self):
        unet_wrapper_module.ComfyUNetWrapper = self._real_wrapper


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

    # The adapter strategy now reaches the wrapper as layer_classes=,
    # built by adapter_layer_classes(PlainLoRAAdapter(), None) -- the
    # default strategy, and no frozen_weight_store_factory override.
    #
    # Identity is not the assertion here: adapter_layer_classes() returns a
    # freshly-built class per call (that is what makes it safe to build
    # during a build at all), so two calls never compare equal. What is
    # asserted is behavior -- the pair is not the injection walk's own
    # defaults, and calling it really does route through PlainLoRAAdapter.
    import torch.nn as _nn
    from nodes.model.lora import LoRAConv2d, LoRALinear
    linear_cls, conv_cls = call["layer_classes"]
    check((linear_cls, conv_cls) != (LoRALinear, LoRAConv2d),
          "layer_classes is not the injection walk's own default pair, so the "
          "strategy really is in play")
    built = linear_cls(_nn.Linear(4, 4), rank=4, alpha=8.0)
    check(type(built) is LoRALinear,
          f"the passed linear_cls builds exactly a real LoRALinear via "
          f"PlainLoRAAdapter (got {type(built).__name__})")
    built_conv = conv_cls(_nn.Conv2d(2, 2, 1), rank=4, alpha=8.0)
    check(type(built_conv) is LoRAConv2d,
          f"the passed conv_cls builds exactly a real LoRAConv2d "
          f"(got {type(built_conv).__name__})")
    check(model._wrapper is not None, "should return a real ComfyUNetTrainableModel")
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
    check_node_thin_wrapper_resolves_port_defaults_correctly()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
