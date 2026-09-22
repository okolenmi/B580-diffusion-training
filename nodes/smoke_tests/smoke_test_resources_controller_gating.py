"""Checks nodes/model/resources_controller.py's LoRASDXLPreset.process()
gating for continue_lora_path/frozen_lora_path against their own
checkbox (continue_training/frozen_lora) -- specifically the bug a real
user hit: a hidden Port's leftover value (typed in before its checkbox
was unchecked -- the editor doesn't clear a widget's own stored value
just because visible_when hid its row) used to raise
"...isn't checked -- check it, or clear the path", blocking the whole
graph, with no way to actually clear a field the person couldn't see.
See LoRASDXLPreset's own docstring for the full reasoning; this file
is the regression test for it, not covered anywhere else (no smoke
test existed for LoRASDXLPreset/ResourcesControllerNode at all before
this -- a known, disclosed gap, still not fully closed here: this
covers exactly the gating bug, not the rest of process()'s real
loading/merging behavior).

Mocks the checkpoint/LoRA loading entirely (_inspect_checkpoint,
ProjectLayout, safetensors.torch.load_file, SDXL_LoRATrainingResources
itself) -- what's under test is process()'s own control flow around
continue_training/frozen_lora, not real file I/O, which a full
LoRASDXLPreset test would need actual checkpoint fixtures for (real,
separate, larger undertaking than this one bug fix warrants).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.model.resources_controller import LoRASDXLPreset


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _process(inputs: dict):
    fake_dtypes = {"unet": MagicMock(dtype="bf16"), "clip": MagicMock(dtype="fp16"),
                   "vae": MagicMock(dtype="fp32")}
    with patch("nodes.model.resources_controller._inspect_checkpoint", return_value=fake_dtypes), \
         patch("nodes.model.resources_controller.ProjectLayout") as MockLayout, \
         patch("nodes.model.resources_controller.SDXL_LoRATrainingResources") as MockResources, \
         patch("safetensors.torch.load_file", return_value={}):
        MockLayout.from_paths_module.return_value.resolve_safe_model_path.return_value = (
            "/fake/ckpt.safetensors")
        preset = LoRASDXLPreset()
        preset.process({"checkpoint_path": "x.safetensors", **inputs})
        return MockResources.call_args.kwargs


def check_unchecked_checkbox_with_a_stale_path_value_does_not_raise():
    print("[continue_training=False + a stale continue_lora_path value: must not raise, "
          "must be ignored entirely]")
    kwargs = _process({
        "continue_training": False, "continue_lora_path": "/stale/leftover.safetensors",
    })
    check(kwargs.get("continue_lora_sd") is None,
          "a stale, hidden continue_lora_path must never be loaded")
    print("    PASS")


def check_unchecked_frozen_lora_with_a_stale_path_value_does_not_raise():
    print("[frozen_lora=False + a stale frozen_lora_path value: must not raise, must be "
          "ignored entirely -- same bug, same fix, the other checkbox]")
    kwargs = _process({
        "frozen_lora": False, "frozen_lora_path": "/stale/leftover.safetensors",
    })
    check(kwargs.get("frozen_lora_sd") is None,
          "a stale, hidden frozen_lora_path must never be loaded")
    print("    PASS")


def check_checked_without_a_path_still_raises():
    print("[continue_training=True with no continue_lora_path: still a real, "
          "fixable-by-the-person mistake -- must still raise]")
    raised = False
    try:
        _process({"continue_training": True})
    except ValueError as e:
        raised = True
        check("continue_lora_path" in str(e), str(e))
    check(raised, "expected a ValueError")
    print("    PASS")


def check_checked_frozen_lora_without_a_path_still_raises():
    print("[frozen_lora=True with no frozen_lora_path: still must raise]")
    raised = False
    try:
        _process({"frozen_lora": True})
    except ValueError as e:
        raised = True
        check("frozen_lora_path" in str(e), str(e))
    check(raised, "expected a ValueError")
    print("    PASS")


def check_checked_with_a_real_path_actually_loads_it():
    print("[continue_training=True with a real path: still actually used, not "
          "accidentally swallowed by the same fix that ignores it when unchecked]")
    fake_lora_info = MagicMock(key_count=1, dtype="bf16", rank=8)
    with patch("nodes.model.resources_controller.inspect_lora", return_value=fake_lora_info):
        kwargs = _process({
            "continue_training": True, "continue_lora_path": "/real/lora.safetensors",
        })
    check(kwargs.get("continue_lora_sd") is not None,
          "a genuinely-checked continue_lora_path must still be loaded")
    print("    PASS")


def main():
    check_unchecked_checkbox_with_a_stale_path_value_does_not_raise()
    check_unchecked_frozen_lora_with_a_stale_path_value_does_not_raise()
    check_checked_without_a_path_still_raises()
    check_checked_frozen_lora_without_a_path_still_raises()
    check_checked_with_a_real_path_actually_loads_it()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
