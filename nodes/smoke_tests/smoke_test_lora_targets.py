"""LoRA target presets: config reachability + real module counts (C1).

Evidence class: config parsing (no weights) + meta-device SDXL UNet on CPU
(counts are architecture, not values -- injection wraps layers without
running them). The counts themselves (564/560/726/722) were measured on
real weights by lora_forensics (see MEASURED-targets-and-schedule.md);
this pins that the preset *strings* resolve to the module sets that
produce those counts, so a config typo or preset-map drift fails here
instead of silently training the wrong set.
"""

import sys
import os
sys.path.insert(0, os.environ.get("REPO", "."))

import torch


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_config_defaults_and_mapping():
    print("[config: new keys parse, default to current behaviour, mapped]")
    from nodes.config_model import CommonSettings, LoRATuning
    t = LoRATuning()
    check(t.target_modules == "attention",
          f"default preset must stay attention, got {t.target_modules!r}")
    check(t.target_conditioning_path is True,
          "cond path must default on (historical behaviour)")
    c = CommonSettings()
    check(c.lr_schedule == "cosine",
          f"schedule default must stay cosine, got {c.lr_schedule!r}")
    c2 = CommonSettings.model_validate({"lr_schedule": "constant"})
    check(c2.lr_schedule == "constant", "constant must parse")
    t2 = LoRATuning.model_validate({"target_modules": "kohya_plus",
                                    "target_conditioning_path": False})
    check(t2.target_modules == "kohya_plus"
          and t2.target_conditioning_path is False, "preset off-combo parse")
    try:
        LoRATuning.model_validate({"target_modules": "everything"})
    except Exception:
        pass
    else:
        check(False, "unknown preset string validated silently")
    from nodes.config_io import FLAT_TO_SECTION_MAP
    check(FLAT_TO_SECTION_MAP.get("lora_target_modules")
          == "tuning.target_modules", "flat mapping for target_modules")
    check(FLAT_TO_SECTION_MAP.get("lora_target_conditioning_path")
          == "tuning.target_conditioning_path", "flat mapping for cond path")
    print("    PASS")


def check_preset_module_counts():
    print("[presets on meta SDXL UNet: 564 / 560 / 726 / 722]")
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W
    cfg = dict(W.SDXL_CONFIG)
    cfg["use_checkpoint"] = False
    # (preset, cond path, expected count) -- the forensics-measured numbers.
    cases = [("attention", True, 564),
             ("attention", False, 560),
             ("kohya_default", True, 726),
             ("kohya_default", False, 722),
             ("kohya_plus", True, 726)]
    for preset, cond, want in cases:
        # Fresh model per case: injection wraps Linear layers in place, so a
        # second inject over the same object would find nothing to wrap.
        with torch.device("meta"):
            m = UNetModel(**cfg)
        m.to_empty(device="cpu")
        reg = inject_lora_into_unet(
            m, LoRAConfig(rank=2, alpha=1.0, target_modules=preset,
                          target_conditioning_path=cond))
        check(len(reg) == want,
              f"{preset}/cond={cond}: {len(reg)} modules, want {want}")
        print(f"    {preset}/cond={cond}: {len(reg)}")
    print("    PASS")


def main():
    check_config_defaults_and_mapping()
    check_preset_module_counts()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
