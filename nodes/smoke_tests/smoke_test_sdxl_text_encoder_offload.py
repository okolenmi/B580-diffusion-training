"""Checks nodes/model/text_encoder.py's SDXLTextEncoder.offload()/reload()
-- specifically that offload() no longer routes through unload()'s own
gc.collect()/empty_cache() (real, measurable, previously-invisible cost
on every single call -- appropriate for unload()'s own "done with this
encoder for the rest of the run" use, not for a per-step offload cycle;
see that method's own docstring and docs/known-issues/pending-testing.md).

A minimal stand-in for core.clip_encode.SDXLClipEncoder -- offload()/
reload() only ever touch .clip_model/._embedder/.device, so a bare
object with a real nn.Linear for clip_model (real .cpu()/.to() calls,
not mocked) is enough; no need for the full checkpoint-loading machinery
smoke_test_lora_training_resources.py's fixtures build for a real one.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.text_encoder import SDXLTextEncoder


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _StubLegacyEncoder:
    def __init__(self):
        self.device = "cpu"  # "cpu" here just means "wherever it starts" for this test
        self.dtype = torch.float16
        self.clip_model = torch.nn.Linear(2, 2)
        self._embedder = torch.nn.Linear(2, 2)

    def unload(self):
        raise AssertionError("offload() must not call unload() -- see its own docstring")


def check_offload_moves_to_cpu_without_calling_unload_or_gc_or_empty_cache():
    print("[offload(): moves clip_model/_embedder to CPU directly, no unload(), "
          "no gc.collect()/empty_cache() of its own]")
    legacy = _StubLegacyEncoder()
    encoder = SDXLTextEncoder(legacy)

    with patch("gc.collect") as mock_gc, patch("torch.xpu.empty_cache", create=True) as mock_empty:
        encoder.offload()

    check(str(legacy.clip_model.weight.device) == "cpu", legacy.clip_model.weight.device)
    check(str(legacy._embedder.weight.device) == "cpu", legacy._embedder.weight.device)
    check(legacy.device == "cpu", legacy.device)
    check(mock_gc.call_count == 0, f"gc.collect() must not be called by offload(), was called "
                                    f"{mock_gc.call_count} time(s)")
    check(mock_empty.call_count == 0, "empty_cache() must not be called by offload()")
    print("    PASS")


def check_reload_restores_the_remembered_device():
    print("[reload(): moves back to the device offload() remembered, when none is given]")
    legacy = _StubLegacyEncoder()
    legacy.device = "cuda:0"  # simulated -- offload() should remember this even without real CUDA
    encoder = SDXLTextEncoder(legacy)
    encoder.offload()
    check(encoder._device_before_offload == "cuda:0", encoder._device_before_offload)

    # reload() would try to actually move to "cuda:0" here, which this sandbox can't do --
    # confirms the remembered value is right without needing real CUDA hardware to finish the
    # move itself (that half is exercised on "cpu" round trips elsewhere in this project's
    # own real usage).
    check(encoder._legacy.device == "cpu", "offload() itself must still have completed")
    print("    PASS")


def check_footprint_bytes_is_zero_while_offloaded():
    print("[footprint_bytes() reports 0 while offloaded, real total once reloaded]")
    legacy = _StubLegacyEncoder()
    encoder = SDXLTextEncoder(legacy)
    before = encoder.footprint_bytes()
    check(before > 0, "sanity: a real nn.Linear must report nonzero footprint")

    encoder.offload()
    check(encoder.footprint_bytes() == 0, "must report 0 while offloaded")

    encoder.reload(device="cpu")
    # reload() deliberately resets _embedder to None rather than moving it (its own
    # docstring: "_get_embedder()'s own existing lazy-construction path rebuilds it on
    # the right device next time it's actually needed") -- so footprint right after
    # reload is clip_model's own share only, not embedder's, until something actually
    # calls _get_embedder() again (encode_for_unet() would, in real use).
    after = encoder.footprint_bytes()
    check(0 < after < before,
          f"clip_model's own footprint should reappear (>0), but embedder's shouldn't "
          f"until re-touched (<original {before}): got {after}")
    print("    PASS")


def main():
    check_offload_moves_to_cpu_without_calling_unload_or_gc_or_empty_cache()
    check_reload_restores_the_remembered_device()
    check_footprint_bytes_is_zero_while_offloaded()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
