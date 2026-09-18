"""Verifies the footprint_bytes()-after-offload() fix
(docs/known-issues/pending-testing.md)
across all four real DeviceResident implementations in nodes/: none of
them checked whether their tensors were actually on a device before
summing numel()*element_size() -- so after offload() moved everything
to CPU, footprint_bytes() kept reporting the pre-offload byte total,
same as if nothing had moved. All four now track an explicit
offloaded flag (self._device_before_offload for the two that already
had a reason to remember the device; self._offloaded, added, for the
two that didn't) and return 0 from footprint_bytes() while it's set.

Two of the four (SDXLTextEncoder, ComfyUNetTrainableModel) need a real
torch environment and are exercised here against minimal fakes rather
than full core.clip_encode.SDXLClipEncoder/core.unet_wrapper.ComfyUNetWrapper
objects (heavier, legacy, ComfyUI-adjacent classes) -- the fakes satisfy
exactly the attributes/methods each footprint_bytes()/offload()/reload()
actually touches, confirmed by reading each directly, not guessed. The
other two (ComposedOptimizerHandle, AdafactorOptimizerHandle) reuse the
same real, already-proven construction patterns from
smoke_test_device_resident_retrofit.py.

Run this directly: `python nodes/smoke_tests/smoke_test_footprint_bytes_after_offload.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

DEVICE = "cpu"  # offload()/reload() round-trip cpu->cpu here, which is
                # enough to check the flag logic -- the actual data
                # movement isn't what's being verified.
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_composed_optimizer_handle():
    print("\n=== ComposedOptimizerHandle ===")
    from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
    from nodes.optimizer.composed import ComposedOptimizerHandle
    from nodes.optimizer.strategies.simple import SimpleLoopStrategy

    params = [torch.randn(8, 8, requires_grad=True)]
    algorithm = AdamWAlgorithm(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=SimpleLoopStrategy(),
                                      params=params, lr=1e-3, device=DEVICE)
    before = handle.footprint_bytes()
    record(before > 0, "footprint_bytes() > 0 before offload", detail=f"got {before}")

    handle.offload_states_to_cpu()
    after_offload = handle.footprint_bytes()
    record(after_offload == 0, "footprint_bytes() == 0 while offloaded",
           detail=f"got {after_offload}")

    handle.reload_states_to_device()
    after_reload = handle.footprint_bytes()
    record(after_reload == before, "footprint_bytes() back to the original value after reload",
           detail=f"got {after_reload}, expected {before}")


def check_adafactor_optimizer_handle():
    print("\n=== AdafactorOptimizerHandle ===")
    from core.optimizers import ChunkedXPUAdafactor
    from nodes.optimizer.adafactor import AdafactorOptimizerHandle

    params = [torch.randn(8, 8, requires_grad=True)]
    legacy = ChunkedXPUAdafactor(params, lr=1e-3, device=DEVICE)
    handle = AdafactorOptimizerHandle(legacy)
    for p in params:
        p.grad = torch.randn_like(p)
    handle.step()  # lazily allocates state -- footprint_bytes() is 0 before this

    before = handle.footprint_bytes()
    record(before > 0, "footprint_bytes() > 0 before offload", detail=f"got {before}")

    handle.offload_states_to_cpu()
    after_offload = handle.footprint_bytes()
    record(after_offload == 0, "footprint_bytes() == 0 while offloaded",
           detail=f"got {after_offload}")

    handle.reload_states_to_device()
    after_reload = handle.footprint_bytes()
    record(after_reload == before, "footprint_bytes() back to the original value after reload",
           detail=f"got {after_reload}, expected {before}")


class _FakeClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)


class _FakeLegacyEncoder:
    """Satisfies exactly what SDXLTextEncoder.footprint_bytes()/offload()/
    reload() touch: .clip_model, ._embedder, .device, .dtype, .unload()."""

    def __init__(self):
        self.clip_model = _FakeClipModel()
        self._embedder = None
        self.device = DEVICE
        self.dtype = torch.float32

    def unload(self):
        self.clip_model = self.clip_model.to("cpu")
        self.device = "cpu"


def check_sdxl_text_encoder():
    print("\n=== SDXLTextEncoder ===")
    from nodes.model.text_encoder import SDXLTextEncoder

    encoder = SDXLTextEncoder(_FakeLegacyEncoder())
    before = encoder.footprint_bytes()
    record(before > 0, "footprint_bytes() > 0 before offload", detail=f"got {before}")

    encoder.offload()
    after_offload = encoder.footprint_bytes()
    record(after_offload == 0, "footprint_bytes() == 0 while offloaded",
           detail=f"got {after_offload}")

    encoder.reload(DEVICE)
    after_reload = encoder.footprint_bytes()
    record(after_reload == before, "footprint_bytes() back to the original value after reload",
           detail=f"got {after_reload}, expected {before}")

    # Confirms the second half of the fix: reload() must reset
    # _device_before_offload, or a second offload()/reload() cycle
    # would still see a stale non-None flag from the first one.
    encoder.offload()
    encoder.reload(DEVICE)
    after_second_cycle = encoder.footprint_bytes()
    record(after_second_cycle == before,
           "footprint_bytes() correct after a second offload()/reload() cycle",
           detail=f"got {after_second_cycle}, expected {before}")


class _FakeUNetWrapper:
    """Satisfies exactly what ComfyUNetTrainableModel.footprint_bytes()/
    offload()/reload() touch: .lora_parameters(), .lora_registry,
    .state_dict(), .device, .to(device=...). No LoRA parameters at all
    (empty lists) -- the whole state_dict() counts as "frozen base",
    which is all this fix needs to check."""

    def __init__(self):
        self._module = torch.nn.Linear(16, 16)
        self.lora_registry = []
        self.device = DEVICE

    def lora_parameters(self):
        return []

    def state_dict(self):
        return self._module.state_dict()

    def to(self, device):
        self._module = self._module.to(device)
        self.device = device


def check_comfy_unet_trainable_model():
    print("\n=== ComfyUNetTrainableModel ===")
    from nodes.model.lora_injector import ComfyUNetTrainableModel

    model = ComfyUNetTrainableModel(_FakeUNetWrapper())
    before = model.footprint_bytes()
    record(before > 0, "footprint_bytes() > 0 before offload", detail=f"got {before}")

    model.offload()
    after_offload = model.footprint_bytes()
    record(after_offload == 0, "footprint_bytes() == 0 while offloaded",
           detail=f"got {after_offload}")

    model.reload(DEVICE)
    after_reload = model.footprint_bytes()
    record(after_reload == before, "footprint_bytes() back to the original value after reload",
           detail=f"got {after_reload}, expected {before}")

    model.offload()
    model.reload(DEVICE)
    after_second_cycle = model.footprint_bytes()
    record(after_second_cycle == before,
           "footprint_bytes() correct after a second offload()/reload() cycle",
           detail=f"got {after_second_cycle}, expected {before}")


def main():
    check_composed_optimizer_handle()
    check_adafactor_optimizer_handle()
    check_sdxl_text_encoder()
    check_comfy_unet_trainable_model()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
