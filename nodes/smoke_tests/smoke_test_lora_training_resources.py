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
    construction args, exposes just enough (.clip_model, ._embedder)
    for SDXLTextEncoder.footprint_bytes()'s real, unpatched logic to
    work against it for real, not mocked."""

    def __init__(self, clip_sd, device):
        self.clip_sd = clip_sd
        self.device = device
        self.clip_model = nn.Linear(4, 4)  # a real nn.Module, real .parameters()/.buffers()
        self._embedder = None


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


def main():
    check_split_checkpoint()
    check_build_text_encoder_delegates_correctly()
    check_inject_lora_delegates_to_build_lora_injected_unet()
    check_sdxl_lora_trainer_full_construction()
    check_wrong_base_order_fails_to_instantiate()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
