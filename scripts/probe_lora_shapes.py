#!/usr/bin/env python3
"""Diagnostic script: prints the exact input shape LoRALinear.forward()
receives at several different points in the SDXL UNet, during a real
forward pass with a batch of *different* timesteps per sample (matching
how a real training step works).

This exists to answer one question before implementing timestep-gated
LoRA: what shape does `x` actually have when it reaches to_q / to_k / to_v
/ to_out.0 in self-attention vs. cross-attention, at different block
depths -- and does its batch dimension (dim 0) line up 1:1 with the
timestep batch the way a gate value would need to?

Run from the ComfyUI root directory, same as convert.py:
    cd /path/to/ComfyUI
    python /path/to/this-project/scripts/probe_lora_shapes.py --config convert-cfg.toml

Paste the full output back.
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from paths import get_comfy_dir, set_comfy_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="convert-cfg.toml",
                    help="Path to a training config, used only to find base_model.")
    p.add_argument("--batch", type=int, default=4,
                    help="Batch size to simulate (use something >1 so per-sample "
                         "shape/ordering is actually visible, not just a batch of 1).")
    p.add_argument("--device", type=str, default=None,
                    help="cuda / xpu / cpu. Auto-detects if not given.")
    cli = p.parse_args()

    config_path = Path(cli.config).resolve()
    print(f"Resolved config path: {config_path}")
    if not config_path.exists():
        print(f"ERROR: no file exists at that path. Note: relative paths are "
              f"resolved against your current working directory ({Path.cwd()}), "
              f"the same convention core/cli.py uses -- not relative to wherever "
              f"this script itself lives. Pass an absolute path if unsure.")
        sys.exit(1)

    comfy_dir = get_comfy_dir()
    if str(comfy_dir) not in sys.path:
        sys.path.append(str(comfy_dir))

    import torch
    from core.config_io import read_config
    from core.lora import LoRAConfig, LoRALinear
    from core.unet_wrapper import ComfyUNetWrapper
    from safetensors.torch import load_file

    device = cli.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    config = read_config(config_path)
    base_model_path = config.paths.base_model
    if not base_model_path:
        print("ERROR: config.paths.base_model is empty -- point --config at a "
              "config that has a real base_model set.")
        sys.exit(1)

    bm_path = Path(base_model_path)
    if not bm_path.is_absolute():
        bm_path = comfy_dir / base_model_path
    print(f"Loading base model: {bm_path}")
    sd = load_file(str(bm_path))

    print("Building UNet + injecting LoRA (rank=4, tiny -- shapes are what matter, not quality)...")
    lora_cfg = LoRAConfig(rank=4, alpha=1.0)
    wrapper = ComfyUNetWrapper(
        unet_sd=sd, device=device, dtype=torch.bfloat16,
        use_checkpoint=False, adm_in_channels=2816,
        lora_config=lora_cfg,
    )
    wrapper.eval()

    # ------------------------------------------------------------------
    # Hook a handful of LoRALinear instances at different points: early
    # input_blocks (self-attn + cross-attn), middle_block, late
    # output_blocks, and a to_out.0 projection -- enough to see whether
    # shape/broadcast behavior is consistent everywhere or varies by
    # attention type / depth.
    # ------------------------------------------------------------------
    targets_seen = 0
    max_targets = 12
    hooks = []
    printed_paths = set()

    def make_hook(path):
        def hook(module, inputs):
            x = inputs[0]
            if path not in printed_paths:
                printed_paths.add(path)
                print(f"  {path:70s} x.shape={tuple(x.shape)} dtype={x.dtype}")
        return hook

    for name, module in wrapper.model.named_modules():
        if isinstance(module, LoRALinear):
            is_interesting = (
                "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name
            )
            if not is_interesting:
                continue
            if targets_seen >= max_targets:
                break
            h = module.register_forward_pre_hook(make_hook(name))
            hooks.append(h)
            targets_seen += 1

    print(f"\nHooked {len(hooks)} LoRALinear instances. Running one forward pass "
          f"with batch={cli.batch}, each sample at a DIFFERENT timestep "
          f"(mirrors real training, where a batch mixes noise levels)...\n")

    B = cli.batch
    x_t = torch.randn(B, 4, 128, 128, device=device, dtype=torch.bfloat16)
    # Deliberately spread across the full range, strictly increasing, so if
    # anything about batch-index<->timestep-index correspondence is wrong
    # it'll be obvious rather than accidentally masked by repeated values.
    timestep = torch.linspace(50, 950, B, device=device, dtype=torch.float32)
    context = torch.randn(B, 77, 2048, device=device, dtype=torch.bfloat16)
    y = torch.randn(B, 2816, device=device, dtype=torch.bfloat16)

    print(f"Input timestep tensor: shape={tuple(timestep.shape)} values={timestep.tolist()}")
    print()

    with torch.no_grad():
        out = wrapper.forward(x_t=x_t, timestep=timestep, context=context, y=y)

    print(f"\nUNet output shape: {tuple(out.shape)}")

    for h in hooks:
        h.remove()

    print("\nDone. Please paste everything above back.")


if __name__ == "__main__":
    main()
