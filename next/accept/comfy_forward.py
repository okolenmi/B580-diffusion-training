"""The save probe's inputs, run through ComfyUI itself: its checkpoint loader, its LoRA loader, its UNet and its text encoders.

Standalone on purpose (nothing here imports this repository), so it can run in the Python environment of a ComfyUI install:

    <comfy python> comfy_forward.py --comfy <ComfyUI folder> --checkpoint <file> --lora <file> --probe <x.probe.safetensors> --out <x.comfy.safetensors>

For every case of the probe it writes the prediction with the adapter and with our conditioning, with the adapter and ComfyUI's own
conditioning, and without the adapter; and ComfyUI's own conditioning, so the text side can be compared too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", required=True, help="the ComfyUI folder (the one holding the comfy package)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"))
    parser.add_argument("--unet-dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"), help="ComfyUI's own choice unless set")
    return parser.parse_args()


def main() -> int:
    args = parse()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]  # this folder's modules must not shadow ComfyUI's imports
    sys.path.insert(0, os.path.abspath(args.comfy))
    sys.argv = ["comfy"] + (["--cpu"] if args.device == "cpu" else []) + ([] if args.unet_dtype == "auto" else [f"--{args.unet_dtype}-unet"])
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    import comfy.lora
    import comfy.lora_convert
    import comfy.model_management as management
    import comfy.sd
    import comfy.utils

    with safe_open(args.probe, framework="pt") as handle:
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        meta = handle.metadata() or {}
    cases, size = json.loads(meta["cases"]), json.loads(meta["size"])
    base_patcher, clip = comfy.sd.load_checkpoint_guess_config(args.checkpoint, output_vae=False, output_clip=True)[:2]
    lora = comfy.utils.load_torch_file(args.lora, safe_load=True)
    patched, _ = comfy.sd.load_lora_for_models(base_patcher, None, lora, 1.0, 0.0)
    key_map = comfy.lora.model_lora_keys_unet(base_patcher.model, {})
    matched = len(comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map, log_missing=False))
    layers = sum(1 for k in lora if k.endswith(".lora_down.weight"))

    def unet(patcher, x, t, ctx, y):
        management.load_models_gpu([patcher])
        model = patcher.model
        dtype = getattr(model, "get_dtype_inference", model.get_dtype)()
        device = patcher.load_device
        with torch.no_grad():
            out = model.diffusion_model(x.to(device=device, dtype=dtype), t.to(device=device, dtype=torch.float32), context=ctx.to(device=device, dtype=dtype),
                                        y=y.to(device=device, dtype=dtype), control=None, transformer_options={})
        return out.float().cpu()

    def own_conditioning(prompt: str):
        cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
        y = base_patcher.model.encode_adm(pooled_output=pooled, width=size[1], height=size[0], crop_w=0, crop_h=0, target_width=size[1], target_height=size[0])
        return cond.float().cpu(), y.float().cpu()

    results: dict[str, torch.Tensor] = {}
    own = {c["prompt"]: own_conditioning(c["prompt"]) for c in cases}
    x = tensors["x"]
    for n in range(len(cases)):  # the unpatched model first for every case, then the patched one: ComfyUI swaps the weights per call
        results[f"comfy_base{n}"] = unet(base_patcher, x, tensors[f"t{n}"], tensors[f"ctx{n}"], tensors[f"y{n}"])
    for n, case in enumerate(cases):
        results[f"comfy_ours{n}"] = unet(patched, x, tensors[f"t{n}"], tensors[f"ctx{n}"], tensors[f"y{n}"])
        ctx, y = own[case["prompt"]]
        results[f"comfy_own{n}"] = unet(patched, x, tensors[f"t{n}"], ctx, y)
        results[f"comfy_ctx{n}"], results[f"comfy_y{n}"] = ctx.clone(), y.clone()
    try:
        from comfyui_version import __version__ as version
    except ImportError:
        version = "unknown"
    dtype = getattr(base_patcher.model, "get_dtype_inference", base_patcher.model.get_dtype)()
    save_file({k: v.contiguous() for k, v in results.items()}, args.out,
              metadata={"comfy_version": str(version), "unet_dtype": str(dtype), "device": str(base_patcher.load_device),
                        "lora_layers": str(layers), "matched_layers": str(matched), "cases": str(len(cases))})
    print(f"ComfyUI: {matched} of {layers} adapter layers matched, UNet dtype {dtype}, {len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
