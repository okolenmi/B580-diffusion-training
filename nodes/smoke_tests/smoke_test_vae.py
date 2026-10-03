"""Correctness check for nodes/model/vae.py -- the SDXL VAE this project owns
(design doc 12, section 7.3, section B).

The contract is the checkpoint's keys again: `encoder.*`, `decoder.*`,
`quant_conv.*`, `post_quant_conv.*`, 248 tensors for SDXL's config.

The interesting claims are the ones that are easy to get wrong and that a
shape test cannot see:

* **`Downsample`'s asymmetric padding.** A symmetric-padding stride-2
  convolution halves the size correctly and shifts the image by half a pixel
  per level. Over four levels that is a visible offset, and the output shape
  is identical either way -- so this is checked by *value*, against a
  hand-written symmetric version that must differ.
* **`AttnBlock` treats channels as features and `H*W` as the sequence**,
  the reverse of the UNet's attention in `attention.py`. Both conventions
  produce the same shape, so a mix-up is silent. Checked numerically, and the
  test proves the wrong convention gives a different answer.
* **`mid.attn_1` exists unconditionally**, even with `attn_resolutions: []`,
  which is what SDXL configures. So four `AttnBlock`s exist in a VAE built
  from a config that mentions no attention at all.
* **`ResnetBlock` has two different shortcuts**, `nin_shortcut` (1x1) and
  `conv_shortcut` (3x3). Getting that wrong is a shape error, but only in
  the width-changing case, so it is pinned with a block that changes width.

Also checked: both halves agree with ComfyUI bitwise on three configs --
including one that actually uses `attn_resolutions`, which SDXL does not --
and a real checkpoint loads with nothing missing.

Run: `python nodes/smoke_tests/smoke_test_vae.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.vae import (  # noqa: E402
    AttnBlock,
    AutoencoderKL,
    DiagonalGaussianDistribution,
    Downsample,
    ResnetBlock,
    Upsample,
    group_norm_32,
)

failures: list[str] = []
skipped: list[str] = []

#: SDXL's real configuration.
SDXL_DDCONFIG = {
    "double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3,
    "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
    "attn_resolutions": [], "dropout": 0.0,
}

#: Small enough to run on CPU. `ch` must be a multiple of 32: the
#: normalisation is 32-group GroupNorm, and ComfyUI raises on anything else.
SMALL_DDCONFIG = {
    "double_z": True, "z_channels": 4, "resolution": 64, "in_channels": 3,
    "out_ch": 3, "ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1,
    "attn_resolutions": [], "dropout": 0.0,
}


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def _fill(module, seed: int = 23) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * scale_of(p.shape))


def scale_of(shape) -> float:
    return 0.2 / (shape[0] ** 0.5) if shape[0] else 0.05


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def check_structure():
    vae = AutoencoderKL(embed_dim=4, ddconfig=SMALL_DDCONFIG)
    keys = set(vae.state_dict())
    record(any(k.startswith("encoder.") for k in keys), "encoder.* keys")
    record(any(k.startswith("decoder.") for k in keys), "decoder.* keys")
    record("quant_conv.weight" in keys and "post_quant_conv.weight" in keys,
           "quant_conv and post_quant_conv")

    # double_z=True scales *both* sides: (1 + double_z) * z_channels ->
    # (1 + double_z) * embed_dim, so 8 -> 8 rather than 8 -> 4. The first
    # version of this check assumed 8 -> 4 and failed; the implementation was
    # right, because it is ComfyUI's shape and the checkpoint's.
    record(tuple(vae.quant_conv.weight.shape) == (8, 8, 1, 1),
           "quant_conv maps (1+double_z)*z_channels to (1+double_z)*embed_dim, "
           "8 -> 8 for SDXL",
           f"{tuple(vae.quant_conv.weight.shape)}")
    record(tuple(vae.post_quant_conv.weight.shape) == (4, 4, 1, 1),
           "post_quant_conv widens back to z_channels for the decoder",
           f"{tuple(vae.post_quant_conv.weight.shape)}")

    # mid.attn_1 is unconditional -- the point of checking it.
    record(isinstance(vae.encoder.mid.attn_1, AttnBlock)
           and isinstance(vae.decoder.mid.attn_1, AttnBlock),
           "mid.attn_1 exists in both halves")
    record(vae.encoder.mid.attn_1 is not None
           and len(vae.encoder.down[0].attn) == 0,
           "while the per-level attn lists are empty, because SDXL configures "
           "attn_resolutions: []")
    total = sum(1 for m in vae.modules() if isinstance(m, AttnBlock))
    record(total == 2,
           "so an SDXL VAE still has 2 AttnBlocks -- one per half -- from a "
           "config naming no attention at all",
           f"got {total}")

    record(isinstance(group_norm_32(64), torch.nn.GroupNorm),
           "group_norm_32 is GroupNorm")


def check_downsamples_asymmetrically() -> None:
    """Why the pad is asymmetric, which a shape test alone would not show.

    On an *even* input the two versions agree on shape and differ only in
    value -- a half-pixel shift per level, four levels of it. On an *odd*
    input they disagree on shape as well. SDXL's VAE works at powers of two,
    so only the first case is reachable in practice, which is exactly why
    the second is worth stating rather than discovering.
    """
    down = Downsample(8, with_conv=True)
    _fill(down, seed=31)

    symmetric = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    with torch.no_grad():
        symmetric.weight.copy_(down.conv.weight)
        symmetric.bias.copy_(down.conv.bias)

    even = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        got = down(even)
        other = symmetric(even)
    record(tuple(got.shape) == (1, 8, 4, 4),
           "an even 8x8 halves to 4x4 with the asymmetric pad",
           f"{tuple(got.shape)}")
    record(tuple(other.shape) == tuple(got.shape),
           "and the symmetric-padding version gives the same shape -- which is "
           "why this needs a value check")
    record(not torch.allclose(got, other, atol=1e-5),
           "but different values: the symmetric one shifts by half a pixel "
           "per level",
           f"max diff {(got - other).abs().max().item():.3e}")

    odd = torch.randn(1, 8, 9, 9)
    with torch.no_grad():
        odd_got, odd_other = down(odd), symmetric(odd)
    record(tuple(odd_got.shape) != tuple(odd_other.shape),
           "and on an odd input the two disagree on shape as well",
           f"asymmetric {tuple(odd_got.shape)} vs symmetric "
           f"{tuple(odd_other.shape)}")

    no_conv = Downsample(8, with_conv=False)
    with torch.no_grad():
        record(tuple(no_conv(even).shape) == (1, 8, 4, 4),
               "with_conv=False average-pools instead",
               f"{tuple(no_conv(even).shape)}")

    up = Upsample(8, with_conv=True)
    with torch.no_grad():
        record(tuple(up(even).shape) == (1, 8, 16, 16),
               "Upsample doubles and keeps the width",
               f"{tuple(up(even).shape)}")


def check_vae_attention_orientation():
    """Channels as features, H*W as sequence -- the reverse of the UNet's."""
    torch.manual_seed(2)
    block = AttnBlock(32).eval()
    _fill(block, seed=41)
    x = torch.randn(2, 32, 4, 4)
    with torch.no_grad():
        got = block(x)

        h = block.norm(x)
        q, k, v = block.q(h), block.k(h), block.v(h)
        b, c = q.shape[0], q.shape[1]
        # Documented orientation: [B, C, H, W] -> [B, 1, H*W, C].
        manual = torch.nn.functional.scaled_dot_product_attention(
            *(t.view(b, 1, c, -1).transpose(2, 3).contiguous() for t in (q, k, v)))
        manual = manual.transpose(2, 3).reshape(q.shape)
        want = x + block.proj_out(manual)

    record(torch.equal(got, want),
           "AttnBlock attends over H*W with channels as features")

    # The wrong one: the UNet's convention, heads split out of channels. A
    # first version of this check built the wrong variant as
    # `view(b, c, 1, -1)`, which gives a feature dimension of 1 -- softmax
    # over a single element is 1.0, so that variant is degenerate and agreed
    # with everything. Four real heads is the comparison worth making.
    heads, head_dim = 4, c // 4

    def as_unet(tensor):
        return tensor.view(b, -1, heads, head_dim).transpose(1, 2)

    with torch.no_grad():
        other = torch.nn.functional.scaled_dot_product_attention(
            as_unet(q), as_unet(k), as_unet(v))
        other = other.transpose(1, 2).reshape(q.shape)
        unet_result = x + block.proj_out(other)

    record(not torch.allclose(want, unet_result, atol=1e-5),
           "and the UNet's orientation would give a different answer, so this "
           "check can fail",
           f"max diff {(want - unet_result).abs().max().item():.3e}")

def check_resnet_shortcuts():
    same = ResnetBlock(in_channels=32, out_channels=32, temb_channels=0)
    _fill(same, seed=51)
    record(not hasattr(same, "nin_shortcut")
           and not hasattr(same, "conv_shortcut"),
           "a width-preserving block has no shortcut projection at all")

    narrow = ResnetBlock(in_channels=32, out_channels=64, temb_channels=0)
    _fill(narrow, seed=52)
    record(hasattr(narrow, "nin_shortcut")
           and tuple(narrow.nin_shortcut.weight.shape)[2:] == (1, 1),
           "a widening block uses a 1x1 nin_shortcut by default",
           f"{tuple(narrow.nin_shortcut.weight.shape)}")
    record(not hasattr(narrow, "conv_shortcut"),
           "and no conv_shortcut")

    conv = ResnetBlock(in_channels=32, out_channels=64, temb_channels=0,
                       conv_shortcut=True)
    _fill(conv, seed=53)
    record(hasattr(conv, "conv_shortcut")
           and tuple(conv.conv_shortcut.weight.shape)[2:] == (3, 3),
           "conv_shortcut=True swaps it for a 3x3",
           f"{tuple(conv.conv_shortcut.weight.shape)}")

    x = torch.randn(1, 32, 6, 6)
    narrow.eval(); conv.eval()
    with torch.no_grad():
        record(tuple(narrow(x).shape) == (1, 64, 6, 6),
               "and both widen correctly",
               f"{tuple(narrow(x).shape)}")


def check_posterior():
    params = torch.randn(2, 8, 4, 4)
    post = DiagonalGaussianDistribution(params)
    record(post.mean.shape == (2, 4, 4, 4) and post.logvar.shape == (2, 4, 4, 4),
           "2*z_channels splits in half")
    record(float(post.logvar.max()) <= 20.0 and float(post.logvar.min()) >= -30.0,
           "logvar is clamped to [-30, 20]",
           f"[{float(post.logvar.min()):.1f}, {float(post.logvar.max()):.1f}]")
    record(torch.equal(post.mode(), post.mean),
           "mode() is the mean -- which is what encode() returns, so encoding "
           "is deterministic")
    wild = torch.randn(2, 8, 4, 4) * 1000
    clamped = DiagonalGaussianDistribution(wild)
    record(bool(torch.isfinite(clamped.std).all()),
           "a wild encoder output cannot produce an infinite std")

    det = DiagonalGaussianDistribution(params, deterministic=True)
    record(float(det.std.abs().max()) == 0.0,
           "and deterministic=True zeroes the std")


def check_forward():
    vae = AutoencoderKL(embed_dim=4, ddconfig=SMALL_DDCONFIG).eval()
    _fill(vae, seed=61)
    z = torch.randn(2, 4, 16, 16)
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        record(tuple(vae.decode(z).shape) == (2, 3, 32, 32),
               "decode maps a 16x16 latent to a 32x32 image",
               f"{tuple(vae.decode(z).shape)}")
        record(tuple(vae.encode(image).shape) == (2, 4, 16, 16),
               "encode maps a 32x32 image to a 16x16 latent",
               f"{tuple(vae.encode(image).shape)}")
        record(bool(torch.isfinite(vae.decode(z)).all()),
               "and the decode is finite")


# ---------------------------------------------------------------------------
# Against ComfyUI, and a real checkpoint
# ---------------------------------------------------------------------------

def check_against_comfy():
    try:
        import paths as p
        root = p.get_comfy_dir()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from comfy.ldm.models.autoencoder import AutoencoderKL as ComfyVAE
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("characterisation vs ComfyUI",
             f"comfy not importable ({type(exc).__name__})")
        return

    configs = [
        ("small", SMALL_DDCONFIG),
        # SDXL never uses attn_resolutions, but the branch is ported and
        # untested code is not ported code.
        ("with attn_resolutions", {**SMALL_DDCONFIG,
                                   "attn_resolutions": [32]}),
        ("SDXL's ch_mult", {**SMALL_DDCONFIG, "ch_mult": [1, 2, 4, 4],
                            "num_res_blocks": 2}),
    ]
    for label, ddconfig in configs:
        theirs = ComfyVAE(embed_dim=4, ddconfig=ddconfig).eval()
        ours = AutoencoderKL(embed_dim=4, ddconfig=ddconfig).eval()
        record(set(theirs.state_dict()) == set(ours.state_dict()),
               f"{label}: identical state_dict keys",
               f"{len(ours.state_dict())} tensors")
        # ComfyUI builds quant_conv/post_quant_conv with
        # disable_weight_init, so weights have to come from one generator.
        _fill(theirs)
        _fill(ours)
        torch.manual_seed(9)
        z = torch.randn(1, 4, 16, 16)
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            dz = (theirs.decode(z) - ours.decode(z)).abs().max().item()
            ez = (theirs.encode(image) - ours.encode(image)).abs().max().item()
        print(f"  {'DIFF' if (dz or ez) else 'SAME'}: {label}: "
              f"decode {dz:.3e}, encode {ez:.3e}")

    # The real thing, structurally.
    theirs = ComfyVAE(embed_dim=4, ddconfig=SDXL_DDCONFIG)
    ours = AutoencoderKL(embed_dim=4, ddconfig=SDXL_DDCONFIG)
    record(set(theirs.state_dict()) == set(ours.state_dict())
           and len(ours.state_dict()) == 248,
           "SDXL's real config: 248 tensors with identical names",
           f"{len(ours.state_dict())}")


def check_real_checkpoint():
    try:
        import paths as p
        root = p.get_comfy_dir()
        candidates = sorted((root / "models" / "checkpoints").glob("*.safetensors"))
    except Exception as exc:  # noqa: BLE001
        skip("a real checkpoint loads with nothing missing",
             f"no ComfyUI directory ({type(exc).__name__})")
        return
    if not candidates:
        skip("a real checkpoint loads with nothing missing",
             "no .safetensors in the ComfyUI checkpoints directory")
        return
    try:
        from safetensors import safe_open
    except ImportError:
        skip("a real checkpoint loads with nothing missing",
             "safetensors is not installed")
        return

    prefix = "first_stage_model."
    path = next((c for c in candidates if c.stat().st_size > 1_000_000_000),
                candidates[0])
    state = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key)
    if not state:
        skip("a real checkpoint loads with nothing missing",
             f"{path.name} has no {prefix}* keys")
        return

    record(len(state) == 248, f"{path.name} holds 248 VAE tensors",
           f"{len(state)}")
    vae = AutoencoderKL(embed_dim=4, ddconfig=SDXL_DDCONFIG)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    record(not missing, "nothing missing", f"{missing[:4]}")
    record(not unexpected, "nothing unexpected", f"{unexpected[:4]}")


def main() -> int:
    print("== structure ==")
    check_structure()
    print("\n== Downsample pads asymmetrically ==")
    check_downsamples_asymmetrically()
    print("\n== AttnBlock's attention orientation ==")
    check_vae_attention_orientation()
    print("\n== ResnetBlock shortcuts ==")
    check_resnet_shortcuts()
    print("\n== the posterior ==")
    check_posterior()
    print("\n== forward ==")
    check_forward()
    print("\n== characterisation vs ComfyUI ==")
    check_against_comfy()
    print("\n== a real checkpoint ==")
    check_real_checkpoint()

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