"""Checks nodes/model/sdxl_architecture.py's SDXLArchitecture and
nodes/model/lora_training_resources.py's LoRATrainingSkeleton/
SDXL_LoraTrainer -- Phase 4 of
docs/resources_controller_redesign_plan.md.

The central thing under test isn't any one method -- it's the
composition decision itself: SDXL_LoraTrainer(SDXLArchitecture,
LoRATrainingSkeleton) has to actually be instantiable, which depends on
base ordering in a way that's easy to get backwards with no error until
instantiation. check_wrong_base_order_fails_to_instantiate proves the
*negative* case (the naive, wrong order genuinely fails), not just that
the shipped order happens to work -- the only way to be sure the
ordering is load-bearing and not incidental.

Reuses smoke_test_lora_injector_extraction.py's exact _Recorder/
_RecordingWrapper (patches ComfyUNetWrapper/adapter_strategy_scope to
record real call args) rather than a second copy of the same mocking,
plus a new, equally-real patch of core.clip_encode.SDXLClipEncoder for
the same reason: neither is installed in this environment (same
constraint every UNet/CLIP-touching smoke test in this project already
has).
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import core.clip_encode as clip_encode_module
from nodes.model.lora_training_resources import LoRATrainingSkeleton, SDXL_LoraTrainer
from nodes.model.sdxl_architecture import SDXLArchitecture
from nodes.smoke_tests.smoke_test_gradient_checkpointing import _install_stub_comfy_checkpoint_module
from nodes.smoke_tests.smoke_test_lora_injector_extraction import _Recorder

_install_stub_comfy_checkpoint_module()


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _make_sdxl_checkpoint_sd():
    return {
        "model.diffusion_model.input_blocks.0.weight": torch.randn(2, 2, dtype=torch.bfloat16),
        "model.diffusion_model.out.weight": torch.randn(2, 2, dtype=torch.bfloat16),
        "conditioner.embedders.0.transformer.weight": torch.randn(2, 2, dtype=torch.float16),
        "conditioner.embedders.1.transformer.weight": torch.randn(2, 2, dtype=torch.float16),
        "first_stage_model.encoder.weight": torch.randn(3, 3, dtype=torch.float16),
        "first_stage_model.decoder.weight": torch.randn(3, 3, dtype=torch.float16),
    }


class _FakeClipEncoder:
    """Stands in for core.clip_encode.SDXLClipEncoder -- records its own
    construction args, exposes just enough (.clip_model, ._embedder,
    .device, .dtype, .unload()) for SDXLTextEncoder's real, unpatched
    offload()/reload()/release()/footprint_bytes() to work against it
    for real, not further mocked -- mirrors the real class's own
    unload() exactly (core/clip_encode.py's SDXLClipEncoder.unload():
    move clip_model to CPU, set self.device = "cpu")."""

    def __init__(self, clip_sd, device):
        self.clip_sd = clip_sd
        self.device = device
        self.dtype = torch.float16  # matches the real class's own hardcoded dtype
        self.clip_model = nn.Linear(4, 4)  # a real nn.Module, real .parameters()/.buffers()/.to()
        self._embedder = None

    def unload(self):
        self.clip_model = self.clip_model.cpu()
        if self._embedder:
            self._embedder = self._embedder.cpu()
        self.device = "cpu"


def check_split_checkpoint():
    print("[SDXLArchitecture.split_checkpoint(): every key ends up in exactly "
          "one of unet/clip/vae, matching resource_inspection's real prefixes]")
    sd = _make_sdxl_checkpoint_sd()
    arch = SDXLArchitecture()
    components = arch.split_checkpoint(sd)
    check(set(components["unet"]) == {"model.diffusion_model.input_blocks.0.weight",
                                        "model.diffusion_model.out.weight"}, components["unet"])
    check(set(components["clip"]) == {"conditioner.embedders.0.transformer.weight",
                                        "conditioner.embedders.1.transformer.weight"}, components["clip"])
    check(set(components["vae"]) == {"first_stage_model.encoder.weight",
                                       "first_stage_model.decoder.weight"}, components["vae"])
    total_keys = sum(len(v) for v in components.values())
    check(total_keys == len(sd), f"every key should end up in exactly one bucket, "
          f"got {total_keys} total vs {len(sd)} in the source")
    print("    PASS")


def check_build_text_encoder_delegates_correctly():
    print("[SDXLArchitecture.build_text_encoder(): delegates to SDXLClipEncoder, "
          "wraps it in the real, existing SDXLTextEncoder mask]")
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    try:
        arch = SDXLArchitecture()
        clip_sd = {"conditioner.embedders.0.transformer.weight": torch.randn(2, 2)}
        encoder = arch.build_text_encoder(clip_sd, device="cpu")
    finally:
        clip_encode_module.SDXLClipEncoder = real_clip_encoder

    from nodes.model.text_encoder import SDXLTextEncoder
    check(isinstance(encoder, SDXLTextEncoder), type(encoder))
    check(encoder._legacy.clip_sd is clip_sd, "should pass the exact clip_sd through")
    check(encoder._legacy.device == "cpu", encoder._legacy.device)
    # footprint_bytes() is real, unpatched logic -- proves the object this
    # returns is genuinely usable, not just structurally the right type.
    check(encoder.footprint_bytes() > 0, encoder.footprint_bytes())
    print("    PASS")


def check_inject_lora_delegates_to_build_lora_injected_unet():
    print("[SDXLArchitecture.inject_lora(): delegates to build_lora_injected_unet() "
          "with kwargs passed straight through]")
    rec = _Recorder()
    rec.install()
    try:
        arch = SDXLArchitecture()
        unet_sd = {"model.diffusion_model.x": torch.zeros(2)}
        model = arch.inject_lora(unet_sd, device="cpu", rank=16, alpha=4.0)
    finally:
        rec.uninstall()

    check(len(rec.wrapper_calls) == 1, len(rec.wrapper_calls))
    call = rec.wrapper_calls[0]
    check(call["device"] == "cpu", call["device"])
    check(call["lora_config"].rank == 16 and call["lora_config"].alpha == 4.0, call["lora_config"])
    check(model._wrapper is not None, "should return a real ComfyUNetTrainableModel")
    print("    PASS")


def check_sdxl_lora_trainer_full_construction():
    print("[SDXL_LoraTrainer end to end: real checkpoint -> real .unet/.clip/"
          ".vae_sd/.lora, footprint_bytes() sums correctly]")
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=32, alpha=2.0)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder

    check(rec.wrapper_calls[0]["lora_config"].rank == 32, rec.wrapper_calls[0]["lora_config"])
    check(trainer.unet is not None, trainer.unet)
    check(trainer.clip is not None and trainer.clip._legacy.device == "cpu", trainer.clip)
    check(set(trainer.vae_sd) == {"first_stage_model.encoder.weight",
                                    "first_stage_model.decoder.weight"}, trainer.vae_sd)
    check(trainer.lora is None,
          "continue-training isn't implemented yet -- .lora must stay None, not silently guess")

    vae_bytes = sum(t.numel() * t.element_size() for t in trainer.vae_sd.values())
    expected = trainer.unet.footprint_bytes() + trainer.clip.footprint_bytes() + vae_bytes
    check(trainer.footprint_bytes() == expected,
          f"footprint_bytes()={trainer.footprint_bytes()}, expected {expected}")
    print("    PASS")
    return trainer


def check_device_resident_offload_reload_release():
    print("[SDXL_LoraTrainer is a real DeviceResident -- offload()/reload()/"
          "release() genuinely move/drop unet, clip, AND vae_sd, not just "
          "the ones with an obvious object to delegate to]")
    from nodes.memory.handle import DeviceResident

    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8, alpha=1.0)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder

    check(isinstance(trainer, DeviceResident), type(trainer).__mro__)

    # No non-CPU device is available in this sandbox (no real XPU/CUDA),
    # so this can't prove an actual cross-device tensor move -- it proves
    # the real mechanics run correctly (the right calls happen, the right
    # attributes end up set), not that data crosses a real device
    # boundary. offload() -> reload(device="cpu") explicitly (not the
    # no-arg fallback) so both code paths through reload() get exercised.
    trainer.offload()
    check(trainer.unet._wrapper.device == "cpu", trainer.unet._wrapper.device)
    check(trainer.clip._legacy.device == "cpu", trainer.clip._legacy.device)
    check(all(t.device.type == "cpu" for t in trainer.vae_sd.values()),
          "vae_sd's own tensors must move too -- they aren't registered with the "
          "coordinator (not a DeviceResident), so offload() has to move them by hand")

    trainer.reload(device="cpu")
    check(trainer.unet._wrapper.device == "cpu", trainer.unet._wrapper.device)
    check(trainer.clip._legacy.device == "cpu", trainer.clip._legacy.device)
    check(all(t.device.type == "cpu" for t in trainer.vae_sd.values()), "still cpu")

    # reload() with no device given falls back to self._device (the
    # construction-time device) -- exercise that path explicitly too,
    # not just the explicit-device one above.
    trainer.offload()
    trainer.reload()
    check(trainer.unet._wrapper.device == "cpu", "no-arg reload() should fall back to self._device")

    # release(): unet/clip genuinely drop (not just move), vae_sd emptied,
    # not left holding stale tensors.
    trainer.release()
    check(trainer.unet._wrapper is None, "unet should be genuinely released, not just offloaded")
    check(trainer.clip._legacy is None, "clip should be genuinely released, not just offloaded")
    check(trainer.vae_sd == {}, trainer.vae_sd)
    check(trainer.footprint_bytes() == 0,
          f"a released trainer should report 0 footprint, got {trainer.footprint_bytes()}")
    print("    PASS")


def check_wrong_base_order_fails_to_instantiate():
    print("[the negative case: listing the abstract skeleton BEFORE the concrete "
          "architecture mixin genuinely fails to instantiate -- proves the shipped "
          "order is load-bearing, not incidental]")

    class _WrongOrder(LoRATrainingSkeleton, SDXLArchitecture):
        pass

    try:
        _WrongOrder({})
        raise AssertionError(
            "expected TypeError -- LoRATrainingSkeleton's abstract stubs should "
            "shadow SDXLArchitecture's real methods in this order, per Python's "
            "left-to-right MRO resolution"
        )
    except TypeError as e:
        check("abstract" in str(e).lower(), str(e))
        print(f"    PASS (correctly fails to instantiate): {e}")

    # And the shipped order genuinely doesn't have this problem -- not just
    # "doesn't raise on construction" (construction needs real args/mocks,
    # covered above), but abstractness itself resolved correctly.
    check(not SDXL_LoraTrainer.__abstractmethods__,
          f"SDXL_LoraTrainer should have zero unsatisfied abstract methods, "
          f"got {SDXL_LoraTrainer.__abstractmethods__}")
    print("    PASS: SDXL_LoraTrainer itself has no unsatisfied abstract methods")


def check_frozen_lora_merges_before_injection():
    print("[frozen_lora_sd merges into the unet's raw weights BEFORE injection -- "
          "the wrapper receives already-merged weights, not the original checkpoint's]")
    from nodes.model.lora_merge import merge_lora_into_state_dict

    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        original_out_weight = sd["model.diffusion_model.out.weight"].clone()
        frozen_lora_sd = {
            "lora_unet_out.lora_down.weight": torch.randn(2, 2, dtype=torch.bfloat16),
            "lora_unet_out.lora_up.weight": torch.randn(2, 2, dtype=torch.bfloat16),
            "lora_unet_out.alpha": torch.tensor([2.0]),
        }
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8,
                                    frozen_lora_sd=frozen_lora_sd, frozen_lora_strength=1.0)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder

    check(trainer.frozen_lora_merged_count == 1, trainer.frozen_lora_merged_count)
    received_weight = rec.wrapper_calls[0]["unet_sd"]["model.diffusion_model.out.weight"]
    check(not torch.equal(received_weight, original_out_weight),
          "the wrapper should have received the merged weight, not the original")

    expected_sd, _ = merge_lora_into_state_dict(
        {"model.diffusion_model.out.weight": original_out_weight.clone()},
        frozen_lora_sd, strength=1.0)
    torch.testing.assert_close(received_weight, expected_sd["model.diffusion_model.out.weight"])

    # No object holds the frozen LoRA afterward -- its effect is baked
    # into the weights, nothing else about it survives construction.
    check(not hasattr(trainer, "frozen_lora_sd"), "should not be stored as an attribute")
    print("    PASS")


def check_frozen_lora_none_is_a_true_no_op():
    print("[frozen_lora_sd=None (the default) behaves exactly as before -- "
          "the wrapper receives the checkpoint's own weights, unmerged]")
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        original_out_weight = sd["model.diffusion_model.out.weight"].clone()
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder

    check(trainer.frozen_lora_merged_count == 0, trainer.frozen_lora_merged_count)
    received_weight = rec.wrapper_calls[0]["unet_sd"]["model.diffusion_model.out.weight"]
    torch.testing.assert_close(received_weight, original_out_weight)
    print("    PASS")


def check_continue_lora_sd_loads_into_the_injected_registry():
    print("[continue_lora_sd calls load_lora_into_registry() against self.unet's own "
          "registry, after injection, and self.lora holds the exact dict given]")
    calls = []

    def fake_load_lora_into_registry(registry, state_dict, source_description="source"):
        calls.append(dict(registry=registry, state_dict=state_dict,
                           source_description=source_description))

    import nodes.model.lora_checkpoint_loader as loader_module
    real_fn = loader_module.load_lora_into_registry
    loader_module.load_lora_into_registry = fake_load_lora_into_registry

    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        continue_sd = {"lora_unet_out.lora_down.weight": torch.randn(2, 2)}
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8, continue_lora_sd=continue_sd)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder
        loader_module.load_lora_into_registry = real_fn

    check(len(calls) == 1, f"expected exactly 1 call, got {len(calls)}")
    check(calls[0]["registry"] is trainer.unet.raw.lora_registry,
          "should load into self.unet's own registry, after it was constructed")
    check(calls[0]["state_dict"] is continue_sd, "should pass continue_lora_sd through exactly")
    check(calls[0]["source_description"] == "continue_lora_sd", calls[0]["source_description"])
    check(trainer.lora is continue_sd, "self.lora should be the exact dict given, not a copy")
    print("    PASS")


def check_continue_lora_sd_none_is_a_true_no_op():
    print("[continue_lora_sd=None (the default) -- self.lora stays None, "
          "load_lora_into_registry never called]")
    import nodes.model.lora_checkpoint_loader as loader_module

    calls = []
    real_fn = loader_module.load_lora_into_registry
    loader_module.load_lora_into_registry = lambda *a, **kw: calls.append(1)

    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder
        loader_module.load_lora_into_registry = real_fn

    check(len(calls) == 0, "load_lora_into_registry should never be called")
    check(trainer.lora is None, trainer.lora)
    print("    PASS")


def check_continue_lora_sd_validation_errors_propagate():
    print("[a real validation error from load_lora_into_registry propagates out of "
          "construction -- not swallowed]")
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        # _RecordingWrapper's lora_registry is always [] -- load_lora_into_registry's
        # own real validation loop simply finds nothing to check against an empty
        # registry, so it wouldn't raise here. Proving propagation instead by
        # patching the function to raise directly, matching how construction should
        # behave if the real function ever does.
        import nodes.model.lora_checkpoint_loader as loader_module
        real_fn = loader_module.load_lora_into_registry

        def raising_fn(*a, **kw):
            raise ValueError("simulated: rank mismatch")
        loader_module.load_lora_into_registry = raising_fn
        try:
            SDXL_LoraTrainer(sd, device="cpu", rank=8, continue_lora_sd={"x": torch.zeros(1)})
            raise AssertionError("expected the ValueError to propagate")
        except ValueError as e:
            check("simulated" in str(e), str(e))
        finally:
            loader_module.load_lora_into_registry = real_fn
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder
    print("    PASS")


def check_frozen_and_continue_lora_both_apply_independently():
    print("[frozen_lora_sd and continue_lora_sd given together -- both apply, "
          "neither interferes with the other]")
    import nodes.model.lora_checkpoint_loader as loader_module

    calls = []
    real_fn = loader_module.load_lora_into_registry
    loader_module.load_lora_into_registry = lambda *a, **kw: calls.append(1)

    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        frozen_sd = {
            "lora_unet_out.lora_down.weight": torch.randn(2, 2, dtype=torch.bfloat16),
            "lora_unet_out.lora_up.weight": torch.randn(2, 2, dtype=torch.bfloat16),
            "lora_unet_out.alpha": torch.tensor([2.0]),
        }
        continue_sd = {"lora_unet_x.lora_down.weight": torch.randn(2, 2)}
        trainer = SDXL_LoraTrainer(sd, device="cpu", rank=8,
                                    frozen_lora_sd=frozen_sd, continue_lora_sd=continue_sd)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder
        loader_module.load_lora_into_registry = real_fn

    check(trainer.frozen_lora_merged_count == 1, trainer.frozen_lora_merged_count)
    check(len(calls) == 1, "continue_lora_sd's loader should still have been called")
    check(trainer.lora is continue_sd, trainer.lora)
    print("    PASS")


def main():
    check_split_checkpoint()
    check_build_text_encoder_delegates_correctly()
    check_inject_lora_delegates_to_build_lora_injected_unet()
    check_sdxl_lora_trainer_full_construction()
    check_frozen_lora_merges_before_injection()
    check_frozen_lora_none_is_a_true_no_op()
    check_continue_lora_sd_loads_into_the_injected_registry()
    check_continue_lora_sd_none_is_a_true_no_op()
    check_continue_lora_sd_validation_errors_propagate()
    check_frozen_and_continue_lora_both_apply_independently()
    check_device_resident_offload_reload_release()
    check_wrong_base_order_fails_to_instantiate()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
