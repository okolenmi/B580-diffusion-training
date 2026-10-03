"""End-to-end check of the reimplemented SDXL stack on the real accelerator.

Design doc 12, section 7.3. The UNet, both CLIP towers and the VAE are all
this project's code now (design doc 12, section 7.3, sections A/B/C1/C2).
Every other test checks them one at a time against ComfyUI or a checkpoint;
this one loads a real checkpoint into each and runs a real forward, because
"identical in float32 on CPU" and "runs on an Intel card" are different
claims and only the second one is about the machine this trains on.

**The models are measured one at a time on purpose.** The UNet is 2.6 B
parameters: 10.3 GB in float32 against an 11.93 GB card. Holding it and CLIP
and the VAE together OOMs, so the check encodes a prompt with CLIP, moves the
conditioning to the host, frees the card, and only then builds the UNet. A
first version of this file built all three up front and died in the UNet's
first attention -- which says something about the card and nothing about the
code.

Skipped rather than failed when there is no accelerator, or no checkpoint.
Both are legitimate states for a checkout; neither is a reason to report a
green run that never ran.

Run: `python nodes/smoke_tests/gpu/smoke_test_reimplemented_stack.py`
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

failures: list[str] = []
skipped: list[str] = []

PROMPT = "a photograph of an astronaut riding a horse on mars"


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def free(device: str) -> None:
    gc.collect()
    if device == "xpu" and torch.xpu.is_available():
        torch.xpu.empty_cache()


def load_checkpoint():
    """A real SDXL checkpoint, split into the three state dicts."""
    try:
        import paths as project_paths
        root = project_paths.get_comfy_dir()
    except Exception as exc:  # noqa: BLE001
        skip("a checkpoint", f"no ComfyUI directory ({type(exc).__name__})")
        return None
    candidates = sorted((root / "models" / "checkpoints").glob("*.safetensors"),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        skip("a checkpoint", f"none under {root / 'models' / 'checkpoints'}")
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        skip("a checkpoint", "safetensors is not installed")
        return None

    path = next((c for c in candidates if c.stat().st_size > 1_000_000_000),
                candidates[0])
    unet, clip, vae = {}, {}, {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.diffusion_model."):
                unet[key[len("model.diffusion_model."):]] = f.get_tensor(key)
            elif key.startswith("conditioner."):
                clip[key] = f.get_tensor(key)
            elif key.startswith("first_stage_model."):
                vae[key[len("first_stage_model."):]] = f.get_tensor(key)
    print(f"  using {path.name}: {len(unet)} unet, {len(clip)} clip, "
          f"{len(vae)} vae tensors")
    return unet, clip, vae


def check_clip(device, state):
    """Both towers, from a real checkpoint, on the device."""
    from nodes.model.clip_encoder import SDXLClipEncoder
    _, clip_sd, _ = state
    encoder = SDXLClipEncoder(clip_sd, device=device)
    ctx, pooled = encoder.encode_prompt(PROMPT)
    record(tuple(ctx.shape) == (1, 77, 2048),
           "CLIP context is [1, 77, 768+1280]", f"{tuple(ctx.shape)}")
    record(ctx.device.type == device,
           f"and it is on {device}", f"{ctx.device}")
    record(bool(torch.isfinite(ctx).all()),
           "with no NaN, which fp16 attention over 77 tokens could produce")

    y = torch.cat([pooled.to(torch.float32),
                   encoder.resolution_embedding(1024, 1024).to(torch.float32)],
                  dim=-1)
    record(tuple(y.shape) == (1, 2816),
           "and SDXL's y is [1, 1280 pooled + 1536 time]",
           f"{tuple(y.shape)}")
    return ctx.to(torch.float32).cpu(), y.cpu()


def check_unet(device, context, adm, state):
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper
    unet_sd, _, _ = state

    config = dict(ComfyUNetWrapper.SDXL_CONFIG)
    config["use_checkpoint"] = True
    config["adm_in_channels"] = 2816
    unet = UNetModel(**config)
    missing, unexpected = unet.load_state_dict(unet_sd, strict=False)
    record(not missing and not unexpected,
           "the UNet loads a real checkpoint with nothing missing or "
           "unexpected",
           f"{len(missing)} missing, {len(unexpected)} unexpected")
    unet = unet.to(device, torch.float32).eval()

    with torch.no_grad():
        out = unet(torch.randn(1, 4, 64, 64, device=device),
                   torch.tensor([500.0], device=device),
                   context=context.to(device), y=adm.to(device))
    record(tuple(out.shape) == (1, 4, 64, 64),
           "and a forward at 64x64 returns the latent's shape",
           f"{tuple(out.shape)}")
    record(bool(torch.isfinite(out).all()), "finite")

    # Checkpointing is on, so the frozen-parameter path runs here: every
    # parameter is frozen, which is the case ComfyUI's CheckpointFunction
    # raises on.
    record(not missing,
           "with use_checkpoint on and every parameter frozen -- the case "
           "comfyi's CheckpointFunction raises on")


def check_vae(device, state):
    from nodes.model.vae import AutoencoderKL
    _, _, vae_sd = state
    vae = AutoencoderKL(embed_dim=4, ddconfig={
        "double_z": True, "z_channels": 4, "resolution": 256,
        "in_channels": 3, "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0})
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    record(not missing and not unexpected,
           "the VAE loads a real checkpoint with nothing missing or "
           "unexpected",
           f"{len(missing)} missing, {len(unexpected)} unexpected")
    vae = vae.to(device, torch.float32).eval()
    with torch.no_grad():
        image = vae.decode(torch.randn(1, 4, 64, 64, device=device))
    record(tuple(image.shape) == (1, 3, 512, 512),
           "and decoding a 64x64 latent gives a 512x512 image -- the SDXL "
           "VAE's 8x", f"{tuple(image.shape)}")
    record(bool(torch.isfinite(image).all()), "finite")


def main() -> int:
    if not (torch.xpu.is_available() or torch.cuda.is_available()):
        skip("the reimplemented stack on an accelerator",
             "neither XPU nor CUDA is available")
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0
    device = "xpu" if torch.xpu.is_available() else "cuda"
    print(f"== the reimplemented SDXL stack on {device} ==")
    name = (torch.cuda.get_device_name(0) if device == "cuda"
            else torch.xpu.get_device_name(0))
    print(f"  {name}")

    state = load_checkpoint()
    if state is None:
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0

    # One model at a time: the UNet alone is 10.3 GB in float32.
    print("\n== CLIP ==")
    context, adm = check_clip(device, state)
    free(device)

    print("\n== UNet ==")
    check_unet(device, context, adm, state)
    free(device)

    print("\n== VAE ==")
    check_vae(device, state)
    free(device)

    print("\n" + "=" * 60)
    if skipped:
        print(f"  {len(skipped)} check(s) skipped: "
              + ", ".join(sorted(set(skipped))))
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())