"""Checks nodes/model/trainer_unpack.py's TrainerResourcesUnpackNode --
the node that actually unblocks the Resources Controller route into
TrainerNode (see that module's own docstring for the full reasoning).

Builds a real SDXL_LoraTrainer (same construction this project already
verifies in smoke_test_lora_training_resources.py's own
check_sdxl_lora_trainer_full_construction, reusing its exact fakes
rather than a second copy of the same mocking -- same reasoning that
file already gives for its own cross-imports), then checks that
unpacking it hands back the identical .unet/.clip objects, not copies
or rebuilt equivalents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.clip_encode as clip_encode_module
from nodes.model.lora_training_resources import SDXL_LoraTrainer
from nodes.model.trainer_unpack import TrainerResourcesUnpackNode
from nodes.smoke_tests.smoke_test_gradient_checkpointing import _install_stub_comfy_checkpoint_module
from nodes.smoke_tests.smoke_test_lora_injector_extraction import _Recorder
from nodes.smoke_tests.smoke_test_lora_training_resources import _FakeClipEncoder, _make_sdxl_checkpoint_sd

_install_stub_comfy_checkpoint_module()


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _build_real_trainer() -> SDXL_LoraTrainer:
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    rec = _Recorder()
    rec.install()
    try:
        sd = _make_sdxl_checkpoint_sd()
        return SDXL_LoraTrainer(sd, device="cpu", rank=8, alpha=1.0)
    finally:
        rec.uninstall()
        clip_encode_module.SDXLClipEncoder = real_clip_encoder


def check_contracts():
    print("[contracts]")
    check(not getattr(TrainerResourcesUnpackNode, "__abstractmethods__", None),
          "must be concretely instantiable")
    check(set(TrainerResourcesUnpackNode.INPUTS) == {"trainer"}, TrainerResourcesUnpackNode.INPUTS)
    check(set(TrainerResourcesUnpackNode.OUTPUTS) == {"model", "text_encoder"},
          TrainerResourcesUnpackNode.OUTPUTS)
    print("    PASS")


def check_unpacks_the_real_unet_and_clip_by_identity():
    print("[build(): result['model'] is trainer.unet, result['text_encoder'] is "
          "trainer.clip -- the same objects, not copies]")
    trainer = _build_real_trainer()
    node = TrainerResourcesUnpackNode()

    result = node.build(trainer=trainer)

    check(result["model"] is trainer.unet, "must be the identical unet object")
    check(result["text_encoder"] is trainer.clip, "must be the identical clip object")
    print("    PASS")


def check_unpacked_model_is_a_real_trainable_model():
    print("[the unpacked model actually exposes trainable_parameters()/footprint_bytes() "
          "-- a real TrainableModel, usable by ModelParametersNode/a trainer node "
          "exactly as-is, not a rebuilt equivalent]")
    trainer = _build_real_trainer()
    node = TrainerResourcesUnpackNode()

    result = node.build(trainer=trainer)

    # _RecordingWrapper's own lora_registry is always [] (no real LoRA layers get
    # injected against a fake unet_sd -- see that fixture's own docstring), so
    # trainable_parameters() is legitimately empty here; the actual thing under
    # test is that it's the *same* call answering the *same* way on both sides,
    # not a rebuilt object that happens to also return [].
    check(list(result["model"].trainable_parameters()) == list(trainer.unet.trainable_parameters()),
          "must be the real object's own trainable_parameters(), not a rebuilt equivalent")
    check(result["model"].footprint_bytes() == trainer.unet.footprint_bytes(),
          "must be the real object, not a rebuilt equivalent")
    print("    PASS")


def check_missing_trainer_input_raises():
    print("[build() without trainer raises via validate_inputs]")
    node = TrainerResourcesUnpackNode()
    raised = False
    try:
        node.build()
    except ValueError:
        raised = True
    check(raised, "expected build() to raise ValueError (missing required input)")
    print("    PASS")


def main():
    check_contracts()
    check_unpacks_the_real_unet_and_clip_by_identity()
    check_unpacked_model_is_a_real_trainable_model()
    check_missing_trainer_input_raises()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
