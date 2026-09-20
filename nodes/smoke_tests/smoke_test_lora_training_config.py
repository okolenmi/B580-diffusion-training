"""Checks nodes/model/lora_training_config.py's LoRATrainingConfigNode --
Phase 6 of docs/design/resources-controller/06-phase-6-lora-training-config.md.

**Real gap this file closes.** Phase 6's own design doc describes real
manual/mock-level verification, but no smoke_test_*.py file for this
node existed anywhere in this project's history before this session
(checked directly: grepped nodes/smoke_tests/ and `git log --all` for
one) -- unlike every other phase in docs/design/resources-controller/,
none of that verification was ever captured as a repeatable test.
Found while planning the TrainerNode-integration work this session
actually landed (nodes/model/trainer_unpack.py,
nodes/train/budgeted.py) -- fixed alongside it rather than filed away
for later, since this node's own real behavior (rank-locking in
particular) is exactly what the new route now depends on being right.

Reuses smoke_test_lora_training_resources.py's exact construction
pattern (_FakeClipEncoder/_Recorder/_make_sdxl_checkpoint_sd) to build
a real SDXL_LoRATrainingResources -- LoRATrainingConfigNode's own
`resources` input -- then runs the real node against it. Same reasoning
that file already gives for its own cross-imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import core.clip_encode as clip_encode_module
from nodes.model.lora_training_config import LoRATrainingConfigNode
from nodes.model.lora_training_resources import SDXL_LoraTrainer, SDXL_LoRATrainingResources
from nodes.smoke_tests.smoke_test_gradient_checkpointing import _install_stub_comfy_checkpoint_module
from nodes.smoke_tests.smoke_test_lora_injector_extraction import _Recorder
from nodes.smoke_tests.smoke_test_lora_training_resources import _FakeClipEncoder, _make_sdxl_checkpoint_sd

_install_stub_comfy_checkpoint_module()


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _build_real_resources(continue_lora_sd=None) -> SDXL_LoRATrainingResources:
    real_clip_encoder = clip_encode_module.SDXLClipEncoder
    clip_encode_module.SDXLClipEncoder = _FakeClipEncoder
    try:
        sd = _make_sdxl_checkpoint_sd()
        return SDXL_LoRATrainingResources(sd, device="cpu", continue_lora_sd=continue_lora_sd)
    finally:
        clip_encode_module.SDXLClipEncoder = real_clip_encoder


def check_contracts():
    print("[contracts]")
    check(not getattr(LoRATrainingConfigNode, "__abstractmethods__", None),
          "must be concretely instantiable")
    check(set(LoRATrainingConfigNode.INPUTS) == {"resources", "rank", "alpha", "unet_weight_store"},
          LoRATrainingConfigNode.INPUTS)
    check(set(LoRATrainingConfigNode.OUTPUTS) == {"trainer"}, LoRATrainingConfigNode.OUTPUTS)
    print("    PASS")


def check_dispatches_to_the_matching_trainer_class_and_injects():
    print("[SDXL_LoRATrainingResources -> SDXL_LoraTrainer, real LoRA injection with "
          "the given rank/alpha]")
    resources = _build_real_resources()
    node = LoRATrainingConfigNode()
    rec = _Recorder()
    rec.install()
    try:
        result = node.build(resources=resources, rank=32, alpha=16.0)
    finally:
        rec.uninstall()

    check(isinstance(result["trainer"], SDXL_LoraTrainer), type(result["trainer"]))
    check(result["trainer"].unet is not None, "must have a real, injected unet")
    check(result["trainer"].clip is resources.clip,
          "clip is already built by resources -- from_resources() must reuse it, not rebuild")
    check(rec.wrapper_calls[0]["lora_config"].rank == 32, rec.wrapper_calls[0]["lora_config"])
    check(rec.wrapper_calls[0]["lora_config"].alpha == 16.0, rec.wrapper_calls[0]["lora_config"])
    print("    PASS")


def check_unregistered_resources_type_raises_a_clear_error():
    print("[a resources type with no _TRAINER_FOR_RESOURCES entry raises ValueError, "
          "not a cryptic dispatch failure]")
    node = LoRATrainingConfigNode()

    class _UnknownResources:
        pass

    raised = False
    try:
        node.build(resources=_UnknownResources())
    except ValueError as e:
        raised = True
        check("_TRAINER_FOR_RESOURCES" in str(e), str(e))
    check(raised, "expected a ValueError naming the missing dispatch entry")
    print("    PASS")


def check_rank_input_is_free_when_there_is_no_continuing_lora():
    print("[no continue_lora_sd: the given rank input is used exactly as given]")
    resources = _build_real_resources(continue_lora_sd=None)
    node = LoRATrainingConfigNode()
    rec = _Recorder()
    rec.install()
    try:
        node.build(resources=resources, rank=99)
    finally:
        rec.uninstall()
    check(rec.wrapper_calls[0]["lora_config"].rank == 99, rec.wrapper_calls[0]["lora_config"])
    print("    PASS")


def check_rank_input_is_ignored_and_overridden_when_continuing_a_lora():
    print("[continue_lora_sd given: the continuing LoRA's own detected rank wins, "
          "the rank input (even a very different value) is ignored entirely -- "
          "Phase 6's own central, 'impossible to override' rank-locking guarantee]")
    continue_sd = {"lora_unet_out.lora_down.weight": torch.randn(5, 3)}  # detected rank: 5
    resources = _build_real_resources(continue_lora_sd=continue_sd)
    node = LoRATrainingConfigNode()

    import nodes.model.lora_checkpoint_loader as loader_module
    real_fn = loader_module.load_lora_into_registry
    loader_module.load_lora_into_registry = lambda *a, **kw: None  # not under test here

    rec = _Recorder()
    rec.install()
    try:
        result = node.build(resources=resources, rank=64)  # 64 = the Port's own default
    finally:
        rec.uninstall()
        loader_module.load_lora_into_registry = real_fn

    check(rec.wrapper_calls[0]["lora_config"].rank == 5,
          f"expected the continuing LoRA's own detected rank (5), not the rank input "
          f"(64): got {rec.wrapper_calls[0]['lora_config'].rank}")
    check(result["trainer"].lora is continue_sd, "self.lora should be the exact dict given")
    print("    PASS")


def check_malformed_continue_lora_sd_raises_before_injecting():
    print("[continue_lora_sd whose modules disagree on rank raises a clear error, "
          "before any injection is attempted]")
    # Two lora_down.weight tensors with different first dims -- _lora_rank() can't
    # agree on a single rank, must return None, which this node must reject.
    continue_sd = {
        "lora_unet_a.lora_down.weight": torch.randn(4, 3),
        "lora_unet_b.lora_down.weight": torch.randn(8, 3),
    }
    resources = _build_real_resources(continue_lora_sd=continue_sd)
    node = LoRATrainingConfigNode()

    raised = False
    try:
        node.build(resources=resources, rank=64)
    except ValueError as e:
        raised = True
        check("rank" in str(e).lower(), str(e))
    check(raised, "expected a ValueError for a continue_lora_sd with no single agreed rank")
    print("    PASS")


def main():
    check_contracts()
    check_dispatches_to_the_matching_trainer_class_and_injects()
    check_unregistered_resources_type_raises_a_clear_error()
    check_rank_input_is_free_when_there_is_no_continuing_lora()
    check_rank_input_is_ignored_and_overridden_when_continuing_a_lora()
    check_malformed_continue_lora_sd_raises_before_injecting()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
