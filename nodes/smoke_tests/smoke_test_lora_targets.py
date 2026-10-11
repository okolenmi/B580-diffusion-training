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
    print("[config: defaults (constant/kohya_label/cond-off), old values parse, mapped]")
    from nodes.config_model import CommonSettings, LoRATuning
    t = LoRATuning()
    check(t.target_modules == "kohya_label",
          f"default preset must be kohya_label (724), got {t.target_modules!r}")
    check(t.target_conditioning_path is False,
          "cond path must default off (time_embed measured worse than label_emb)")
    c = CommonSettings()
    check(c.lr_schedule == "constant",
          f"C3 schedule default must be constant, got {c.lr_schedule!r}")
    t2 = LoRATuning.model_validate({"target_modules": "kohya_plus",
                                    "target_conditioning_path": False})
    check(t2.target_modules == "kohya_plus"
          and t2.target_conditioning_path is False, "preset off-combo parse")
    # The arm-5 config (722) must remain expressible as a stored value:
    # graphs and configs written before this default change keep theirs.
    t5 = LoRATuning.model_validate({"target_modules": "kohya_default"})
    check(t5.target_modules == "kohya_default",
          "the previous default must still parse as a stored value")
    t_old = LoRATuning.model_validate({"target_modules": "attention",
                                       "target_conditioning_path": True})
    check(t_old.target_modules == "attention"
          and t_old.target_conditioning_path is True,
          "old stored values must still parse (saved runs keep theirs)")
    c_old = CommonSettings.model_validate({"lr_schedule": "cosine"})
    check(c_old.lr_schedule == "cosine",
          "old stored cosine must still parse")
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


def check_node_port_defaults():
    print("[node Ports default to the 724 configuration]")
    from nodes.model.lora_injector import ComfyUNetLoRANode
    tm = ComfyUNetLoRANode.INPUTS["target_modules"].default
    cp = ComfyUNetLoRANode.INPUTS["target_conditioning_path"].default
    check(tm == "kohya_label",
          f"target_modules Port default must be kohya_label, got {tm!r}")
    check(cp is False,
          "target_conditioning_path Port default must stay False")
    print("    PASS")


def check_resources_route_passes_the_targets_through():
    print("[Resources Controller route: the 724 default reaches the injector]")
    # The managed route builds its adapter through LoRATrainingConfigNode, which
    # had no target ports and so injected with the injector function's bare
    # defaults (attention + time_embed + label_emb = 564), not the documented
    # kohya_label (724). Captured at the call that matters, with the real
    # node and the real from_resources() path; only the (heavy) injector is stubbed.
    import nodes.model.lora_injector as injector
    from nodes.core import ExecutionContext
    from nodes.model.lora_training_config import LoRATrainingConfigNode
    from nodes.model.lora_training_resources import SDXL_LoRATrainingResources

    class _Captured(Exception):
        pass

    def run(**extra):
        seen = {}

        def fake(weights, **kw):
            seen.update(kw)
            raise _Captured

        real = injector.build_lora_injected_unet
        injector.build_lora_injected_unet = fake
        try:
            res = SDXL_LoRATrainingResources.__new__(SDXL_LoRATrainingResources)
            res.unet_sd, res.vae_sd, res.clip = {}, {}, object()
            res.continue_lora_sd, res._device = None, "cpu"
            try:
                LoRATrainingConfigNode(ExecutionContext()).build(resources=res, **extra)
            except _Captured:
                pass
        finally:
            injector.build_lora_injected_unet = real
        return seen

    seen = run()
    check(seen.get("target_modules") == "kohya_label",
          f"default target_modules must reach the injector as kohya_label, got "
          f"{seen.get('target_modules', '<not passed>')!r}")
    check(seen.get("target_conditioning_path") is False,
          f"default cond path must reach the injector as False, got "
          f"{seen.get('target_conditioning_path', '<not passed>')!r}")
    seen = run(target_modules="attention", target_conditioning_path=True)
    check(seen.get("target_modules") == "attention"
          and seen.get("target_conditioning_path") is True,
          "an explicit choice must be passed through unchanged")
    for field in ("target_modules", "target_conditioning_path"):
        check(field in LoRATrainingConfigNode.memory_fields,
              f"{field} changes the peak VRAM and must be a memory field")
    print("    PASS")


def check_preset_module_counts():
    print("[presets on meta SDXL UNet: 564 / 560 / 726 / 724 / 722]")
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
             ("kohya_label", True, 726),
             ("kohya_label", False, 724),
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
    check_node_port_defaults()
    check_resources_route_passes_the_targets_through()
    check_preset_module_counts()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
