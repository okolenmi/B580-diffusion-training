"""Correctness check for nodes/model/frozen_weight_store.py
(FrozenWeightStore/BF16WeightStore) and the DeviceResident conformance it
enables on TrainableModel/ComfyUNetTrainableModel
(docs/training_pipeline_design.md sections 1.2, 3.3).

Built around real torch, real core.lora.LoRALinear, and a minimal fake
wrapper exposing only the surface ComfyUNetTrainableModel's new methods
actually touch (device/to/state_dict/lora_parameters) -- same lightweight-
real-objects approach as smoke_test_lora_checkpoint_loader.py and
smoke_test_lora_phase_split.py, not mocked math.

The crux is check_footprint_after_phase_split_counts_frozen_generation()
-- the real reason footprint_bytes() matches against lora_parameters() by
data_ptr() instead of by name, documented in
ComfyUNetTrainableModel.footprint_bytes()'s own docstring; this test
exercises the exact scenario that distinguishes the two.

Run this directly: `python nodes/smoke_tests/smoke_test_frozen_weight_store.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from core.lora import LoRALinear
from nodes.model.frozen_weight_store import BF16WeightStore
from nodes.model.lora_injector import ComfyUNetTrainableModel
from nodes.model.lora_phases import split_into_new_generation

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _FakeWrapper:
    """Minimal stand-in for core.unet_wrapper.ComfyUNetWrapper -- only the
    attributes/methods ComfyUNetTrainableModel's DeviceResident methods
    and lora_phases.split_into_new_generation actually touch."""

    def __init__(self, model: nn.Module, registry):
        self.model = model
        self.lora_registry = registry
        self.device = "cpu"

    def has_lora(self) -> bool:
        return len(self.lora_registry) > 0

    def to(self, device=None, **kwargs):
        self.model.to(device=device, **kwargs)
        if device is not None:
            self.device = str(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def lora_parameters(self):
        params = []
        for _, _, _, layer in self.lora_registry:
            if hasattr(layer, "lora_A") and isinstance(layer.lora_A, torch.nn.Parameter):
                params.append(layer.lora_A)
                params.append(layer.lora_B)
        return params


def _make_model(rank=4, alpha=1.0, in_f=8, out_f=6):
    torch.manual_seed(0)
    base = nn.Linear(in_f, out_f)
    layer = LoRALinear(base, rank=rank, alpha=alpha)
    root = nn.Module()
    root.proj = layer
    wrapper = _FakeWrapper(root, [("proj", root, "proj", layer)])
    return ComfyUNetTrainableModel(wrapper), root, layer


def check_bf16_weight_store_basics():
    print("\n=== BF16WeightStore basics ===")
    w = torch.randn(4, 8, dtype=torch.bfloat16)
    store = BF16WeightStore(w)
    record(store.footprint_bytes() == w.numel() * w.element_size(),
           "footprint_bytes() == numel * element_size")
    record(store.materialize() is w, "materialize() returns the same tensor object")


def check_footprint_excludes_only_trainable_lora():
    print("\n=== footprint_bytes() excludes exactly lora_A/lora_B (single generation) ===")
    model, root, layer = _make_model()
    expected = sum(
        t.numel() * t.element_size()
        for name, t in root.state_dict().items()
        if name not in ("proj.lora_A", "proj.lora_B")
    )
    record(model.footprint_bytes() == expected,
           "matches an independent name-based computation",
           detail=f"got {model.footprint_bytes()}, expected {expected}")
    record(model.footprint_bytes() > 0, "footprint is actually nonzero (base weight+bias)")


def check_footprint_after_phase_split_counts_frozen_generation():
    """THE CRUX: after a phase split, the OLD generation's lora_A/lora_B
    keep that exact name at a nested path (LoRAGeneration.inner is a real
    submodule) but are now frozen -- footprint_bytes() must count them,
    which only works because the exclusion is by data_ptr() against
    lora_parameters() (always exactly the current top-of-stack pair), not
    by name."""
    print("\n=== THE FIX: footprint_bytes() counts a frozen (split-off) generation's weights ===")
    model, root, layer = _make_model(rank=4)
    before_split = model.footprint_bytes()

    old_lora_A_bytes = layer.lora_A.numel() * layer.lora_A.element_size()
    old_lora_B_bytes = layer.lora_B.numel() * layer.lora_B.element_size()

    frozen_snapshot = split_into_new_generation(model.raw, rank=4, alpha=1.0)
    record(len(frozen_snapshot) == 1, "split froze exactly the one layer")

    after_split = model.footprint_bytes()
    expected_after = before_split + old_lora_A_bytes + old_lora_B_bytes
    record(after_split == expected_after,
           "footprint grew by exactly the old generation's lora_A+lora_B size",
           detail=f"before={before_split}, after={after_split}, expected={expected_after}")

    # And the NEW generation's own lora_A/lora_B must still be excluded --
    # if data_ptr() matching were broken, this would double-count or
    # under-count instead of landing exactly on expected_after above.
    new_layer = model.raw.lora_registry[0][3]
    record(hasattr(new_layer, "lora_A") and new_layer.lora_A.requires_grad,
           "the new top-of-stack generation is the one actually trainable now")
    record(not layer.lora_A.requires_grad,
           "the old (split-off) generation is frozen (requires_grad False)")


def check_offload_reload_round_trip():
    print("\n=== offload()/reload() remember and restore the device ===")
    model, root, layer = _make_model()
    model.raw.device = "cpu"  # this sandbox has no second real device to move to/from
    model.offload()
    record(model.raw.device == "cpu", "offload() moves to cpu")
    model.reload()  # no explicit device -- must recall what offload() remembered
    record(model.raw.device == "cpu",
           "reload() with no argument restores the pre-offload device")
    model.reload(device="cpu")  # explicit device still works
    record(model.raw.device == "cpu", "reload() with an explicit device argument works too")


def check_reload_without_prior_offload_raises():
    print("\n=== reload() with nothing to fall back to raises, not silently no-ops ===")
    model, root, layer = _make_model()
    try:
        model.reload()
        ok = False
    except RuntimeError:
        ok = True
    record(ok, "reload() with no device and no prior offload() raises RuntimeError")


def check_release_then_footprint_is_zero():
    print("\n=== release() drops the model; footprint_bytes() reports 0 afterward, doesn't raise ===")
    model, root, layer = _make_model()
    record(model.footprint_bytes() > 0, "footprint is nonzero before release()")
    model.release()
    try:
        fp = model.footprint_bytes()
        ok = fp == 0
    except Exception as e:
        ok = False
        record(ok, "footprint_bytes() returns 0 after release(), doesn't raise", detail=repr(e))
        return
    record(ok, "footprint_bytes() returns 0 after release(), doesn't raise")


def check_is_device_resident():
    print("\n=== TrainableModel is a real DeviceResident ===")
    from nodes.memory.handle import DeviceResident
    model, _, _ = _make_model()
    record(isinstance(model, DeviceResident), "ComfyUNetTrainableModel isinstance DeviceResident")


def main():
    check_bf16_weight_store_basics()
    check_footprint_excludes_only_trainable_lora()
    check_footprint_after_phase_split_counts_frozen_generation()
    check_offload_reload_round_trip()
    check_reload_without_prior_offload_raises()
    check_release_then_footprint_is_zero()
    check_is_device_resident()

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
