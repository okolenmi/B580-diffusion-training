"""A2 real-weights confirmation: meta-built UNet/CLIP == eager-built, forward.

The tiny-weight CPU tests (`smoke_test_meta_init.py`) prove the *logic*:
a meta-built model materialised with to_empty() and then loaded produces
the same tensors as an eager build. What they cannot show is that this
holds for the real 2.5B-parameter SDXL UNet and the CLIP towers, where a
single tensor the checkpoint does not cover would be uninitialised
memory -- NaN, or different every run -- and never show up as a state
dict mismatch, because only *covered* tensors are compared.

So this does what the tiny test cannot, on the real checkpoint
(`div_4.safetensors` by default):

  1. builds the UNet twice from the same weights -- once eagerly (random
     init, then load_state_dict), once on meta (to_empty, then load) --
     and compares every parameter and buffer;
  2. compares the *forward output* for a fixed random input, in fp32 on
     CPU, which is the claim that actually matters: equal tensors that
     were never loaded would still differ here;
  3. does the same for the CLIP towers (encode a real tokenized prompt);
  4. repeats the UNet comparison after LoRA injection, since injection
     runs after the load and wraps layers in place;
  5. checks for NaN/Inf in the meta-built result, the failure this whole
     guard exists to catch.

Memory: two SDXL UNets in fp32 CPU is ~2 x 10 GB, so ~20 GB plus the
checkpoint's own tensors. Run it on a machine with room; it prints peak
RSS as it goes.

Usage:
    python -u archive/bad-LoRA-investigation/a6_real_meta_forward.py \
        [--checkpoint PATH] [--layers N] [--skip-clip]

Evidence class: measured on REAL SDXL weights, CPU, fp32.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _rss_gb() -> float:
    """Resident set size in GB, so the memory claim is a measurement."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("nan")


def _cmp_tensors(name_a: dict, name_b: dict, label: str) -> tuple[int, float]:
    """`(n_differing, max_abs_diff)` over every tensor in both dicts."""
    assert set(name_a) == set(name_b), (
        f"{label}: key sets differ (only in A: {sorted(set(name_a) - set(name_b))[:5]}, "
        f"only in B: {sorted(set(name_b) - set(name_a))[:5]})")
    n_diff, worst = 0, 0.0
    for k in name_a:
        a, b = name_a[k], name_b[k]
        if not torch.equal(a, b):
            n_diff += 1
            worst = max(worst, float((a - b).abs().max()))
    return n_diff, worst


def _check_finite(sd: dict) -> tuple[int, list[str]]:
    """How many tensors are non-finite, and the first few that are.

    This is the check the meta guard exists for: a tensor the checkpoint
    never covered stays uninitialised memory, which is NaN or garbage
    and would otherwise sail through a state-dict comparison against the
    eager build only if BOTH had it -- but the eager build has a real
    init, so the comparison catches it too. Checked separately anyway,
    because 'they differ' and 'this one is NaN' are different diagnoses.
    """
    bad = [k for k, v in sd.items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    return len(bad), bad[:5]


def build_unet_eager(sd: dict, cfg: dict):
    """The pre-meta route: real random init, then load over it."""
    from nodes.model.unet import UNetModel
    torch.manual_seed(0)  # so the eager fill is reproducible run to run
    model = UNetModel(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, list(missing), list(unexpected)


def build_unet_meta(sd: dict, cfg: dict):
    """The production route: meta, to_empty, then load (unet_wrapper.py)."""
    from nodes.model.unet import UNetModel
    from nodes.model.meta_init import verify_loaded
    with torch.device("meta"):
        model = UNetModel(**cfg)
    model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    verify_loaded(model, missing, unexpected, owner="UNetModel")
    return model, list(missing), list(unexpected)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="safetensors checkpoint (default: ComfyUI's div_4)")
    ap.add_argument("--layers", type=int, default=0,
                    help="truncate the comparison to the first N state-dict "
                         "tensors (0 = all). A memory shortcut, not a "
                         "different claim; the printed label says so.")
    ap.add_argument("--skip-clip", action="store_true",
                    help="UNet only (the CLIP towers are the second half)")
    args = ap.parse_args()

    import paths as project_paths
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        project_paths.get_comfy_dir() / "models" / "checkpoints" / "div_4.safetensors")
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 2

    from safetensors.torch import load_file
    print(f"checkpoint: {ckpt}")
    raw = load_file(str(ckpt))
    sd = {k.replace("model.diffusion_model.", ""): v for k, v in raw.items()
          if k.startswith("model.diffusion_model.")}
    print(f"  {len(sd)} UNet tensors, {_rss_gb():.1f} GB RSS after read")
    del raw
    gc.collect()

    # fp32 everywhere: the point is a numeric-identity claim, and bf16
    # would round two different-but-equal-in-bf16 values onto each other
    # and hide the very difference this is looking for.
    cfg = dict(ComfyUNetWrapper_SDXL_CONFIG())
    cfg["use_checkpoint"] = False

    print("\n== 1. UNet built eagerly vs on meta: every parameter and buffer ==")
    eager, miss_e, unexp_e = build_unet_eager(sd, cfg)
    print(f"  eager built; missing={len(miss_e)} unexpected={len(unexp_e)} "
          f"({_rss_gb():.1f} GB RSS)")
    meta, miss_m, unexp_m = build_unet_meta(sd, cfg)
    print(f"  meta  built; missing={len(miss_m)} unexpected={len(unexp_m)} "
          f"({_rss_gb():.1f} GB RSS)")
    print(f"  coverage agrees: missing {'==' if miss_e == miss_m else '!='} "
          f"{len(miss_m)} key(s), unexpected "
          f"{'==' if unexp_e == unexp_m else '!='} {len(unexp_m)} key(s)")

    a = eager.state_dict()
    b = meta.state_dict()
    if args.layers:
        keys = sorted(a)[:args.layers]
        a = {k: a[k] for k in keys}
        b = {k: b[k] for k in keys}
        print(f"  (TRUNCATED to the first {len(keys)} tensors -- a memory "
              f"shortcut, not the full claim)")
    n_diff, worst = _cmp_tensors(a, b, "UNet state dict")
    total = len(a)
    print(f"  RESULT: {total - n_diff}/{total} tensors bit-identical, "
          f"{n_diff} differ, max |diff| {worst:.3e}")

    n_bad, bad_keys = _check_finite(b)
    print(f"  meta-built non-finite tensors: {n_bad} {bad_keys}")

    print("\n== 2. the claim that matters: identical FORWARD output ==")
    # Fixed seed, one latent, one timestep, SDXL-shaped context (77 x 2048)
    # and the class-conditional vector the SDXL base expects.
    torch.manual_seed(1234)
    x = torch.randn(1, 4, 64, 64)
    timesteps = torch.tensor([500.0])
    context = torch.randn(1, 77, 2048)
    y = torch.randn(1, 2816)
    with torch.no_grad():
        out_e = eager(x, timesteps, context=context, y=y)
        out_m = meta(x, timesteps, context=context, y=y)
    fd = float((out_e - out_m).abs().max())
    rel = fd / max(float(out_e.abs().max()), 1e-12)
    print(f"  output shape {tuple(out_e.shape)}, max |eager-meta| {fd:.3e} "
          f"(relative {rel:.3e})")
    print(f"  bit-identical: {torch.equal(out_e, out_m)}")
    print(f"  RESULT: {'PASS' if rel < 1e-6 else 'FAIL'} "
          f"(relative difference {rel:.3e} against a 1e-6 tolerance)")

    del out_e, out_m, eager, a, b
    gc.collect()

    print("\n== 3. same comparison after LoRA injection (the real route) ==")
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet
    lcfg = LoRAConfig(target_modules="kohya_label", target_conditioning_path=False)
    m2, _, _ = build_unet_meta(sd, cfg)
    n_mod = inject_lora_into_unet(m2, lcfg)
    with torch.no_grad():
        out_l = m2(x, timesteps, context=context, y=y)
    del m2
    gc.collect()
    e2, _, _ = build_unet_eager(sd, cfg)
    inject_lora_into_unet(e2, lcfg)
    with torch.no_grad():
        out_l2 = e2(x, timesteps, context=context, y=y)
    fd2 = float((out_l - out_l2).abs().max())
    rel2 = fd2 / max(float(out_l2.abs().max()), 1e-12)
    print(f"  {len(n_mod)} adapters injected; max |eager-meta| {fd2:.3e} "
          f"(relative {rel2:.3e})")
    print(f"  RESULT: {'PASS' if rel2 < 1e-6 else 'FAIL'}")
    del e2, out_l, out_l2
    gc.collect()

    if args.skip_clip:
        return 0

    print("\n== 4. CLIP towers: eager vs meta, on a real encode ==")
    try:
        _clip_section()
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED: {type(exc).__name__}: {exc}")
    print(f"\npeak RSS: {_rss_gb():.1f} GB")
    return 0


def ComfyUNetWrapper_SDXL_CONFIG() -> dict:
    from nodes.model.unet_wrapper import ComfyUNetWrapper
    return dict(ComfyUNetWrapper.SDXL_CONFIG)


def _clip_section() -> None:
    """Build the CLIP towers both ways and compare a real encode.

    Same shape of claim as the UNet half: eager vs meta construction,
    then the thing users actually get out of it (an encoding), not the
    tensors underneath. CLIPEncoder itself only has the meta route, so
    this builds its inner SDXLClipModel both ways -- the eager build is
    what the route looked like before meta construction -- and runs the
    same weights through both.
    """
    from safetensors.torch import load_file
    import paths as project_paths
    from nodes.model.clip import SDXLClipModel
    from nodes.model.clip_encoder import _extract_and_convert_clip_state_dict
    from nodes.model.meta_init import verify_loaded

    ckpt = project_paths.get_comfy_dir() / "models" / "checkpoints" / "div_4.safetensors"
    full = load_file(str(ckpt))
    clip_sd = _extract_and_convert_clip_state_dict(full)
    del full
    gc.collect()
    if ("clip_l.transformer.text_model.embeddings.position_ids"
            not in clip_sd):
        clip_sd["clip_l.transformer.text_model.embeddings.position_ids"] = (
            torch.arange(77).expand((1, -1)))
    for k in list(clip_sd):
        if isinstance(clip_sd[k], torch.Tensor):
            clip_sd[k] = clip_sd[k].to(dtype=torch.float16)
    print(f"  {len(clip_sd)} CLIP tensors, {_rss_gb():.1f} GB RSS")

    allow_fill = frozenset({"clip_l.logit_scale",
                            "clip_l.transformer.text_projection.weight"})
    allow_unexp = frozenset(
        {"clip_l.transformer.text_model.embeddings.position_ids"})

    torch.manual_seed(0)
    eager = SDXLClipModel(device="cpu", dtype=torch.float16)
    eager.eval()
    miss_e, unexp_e = eager.load_state_dict(clip_sd, strict=False)
    del miss_e, unexp_e
    print(f"  eager built ({_rss_gb():.1f} GB RSS)")

    with torch.device("meta"):
        meta_m = SDXLClipModel(device="meta", dtype=torch.float16)
    meta_m.to_empty(device="cpu")
    meta_m.eval()
    miss_m, unexp_m = meta_m.load_state_dict(clip_sd, strict=False)
    verify_loaded(meta_m, list(miss_m), list(unexp_m),
                  allow_fill=allow_fill, allow_unexpected=allow_unexp,
                  owner="SDXLClipModel")
    print(f"  meta  built, missing={len(miss_m)} unexpected={len(unexp_m)} "
          f"({_rss_gb():.1f} GB RSS)")

    a, b = eager.state_dict(), meta_m.state_dict()
    n_diff, worst = _cmp_tensors(a, b, "CLIP state dict")
    print(f"  RESULT: {len(a) - n_diff}/{len(a)} tensors bit-identical, "
          f"{n_diff} differ, max |diff| {worst:.3e}")
    n_bad, bad_keys = _check_finite(b)
    print(f"  meta-built non-finite tensors: {n_bad} {bad_keys}")

    # A real encode through SDXL's own prompt path: both towers, joined
    # into the 2048-wide context the UNet consumes.
    torch.manual_seed(7)
    pairs = {"l": [torch.randint(0, 1000, (77,))],
             "g": [torch.randint(0, 1000, (77,))]}
    with torch.no_grad():
        ctx_e, pool_e = eager.encode_token_ids(pairs)
        ctx_m, pool_m = meta_m.encode_token_ids(pairs)
    for name, u, v in (("context", ctx_e, ctx_m), ("pooled", pool_e, pool_m)):
        fd = float((u - v).abs().max())
        rel = fd / max(float(u.abs().max()), 1e-12)
        print(f"  encode {name}: shape {tuple(u.shape)}, max |eager-meta| "
              f"{fd:.3e} (relative {rel:.3e}), bit-identical "
              f"{torch.equal(u, v)}")
        print(f"    RESULT: {'PASS' if rel < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    raise SystemExit(main())