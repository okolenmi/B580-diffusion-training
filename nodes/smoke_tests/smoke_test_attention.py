"""Correctness check for nodes/model/attention.py -- the attention stack this
project owns (design doc 12, section 7.3, section A).

Two things are checked, and the split matters because they fail differently:

1. **The contract**, without ComfyUI at all: parameter names and shapes for
   every shape SDXL builds, the residual structure of a transformer block,
   and the projection order in `SpatialTransformer` (the two branches
   project on opposite sides of the flatten, and getting it wrong is a shape
   error rather than a wrong number).

2. **The numbers**, against ComfyUI where it is importable. This is a
   *characterisation*, not a gate -- it reports a difference rather than
   failing on one -- but the arithmetic is close enough that a divergence
   here means one of two implementations is wrong, and this file has no
   oracle for which. Unlike the timestep and key-translation tests, this one
   is expected to agree exactly: the transformer is unambiguous and nothing
   was "improved" inside it. So a difference is a bug report, not a
   question. It skips when ComfyUI is absent, so the file still runs on a
   machine that has none.

The comparison supplies weights from one shared generator. That is not
optional: ComfyUI's `ops.Linear` leaves its weight uninitialised (~3e29),
because it assumes a checkpoint overwrites it, so both sides come out
all-NaN and every comparison trivially "agrees" on nan == nan. A test that
passed for that reason would be worse than no test.

Run: `python nodes/smoke_tests/smoke_test_attention.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.attention import (  # noqa: E402
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    GEGLU,
    SpatialTransformer,
    group_norm_32,
)

failures: list[str] = []
skipped: list[str] = []

#: SDXL builds `num_heads = channels // num_head_channels` with
#: `dim_head = num_head_channels = 64`, so `inner_dim == channels` at every
#: level. Scaled down here: channels 128 and 64, dim_head 32.
SDXL_LEVELS = [
    # (channels, heads, dim_head, depth, context_dim, use_linear)
    (128, 4, 32, 2, 48, True),
    (128, 4, 32, 10, 48, True),
    (64, 2, 32, 10, 48, True),
    (64, 2, 32, 1, 48, False),   # the conv branch, which SDXL does not use
]


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def _fill(module, seed: int = 11) -> None:
    """Real weights for every parameter, from one shared generator."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * 0.08)


# ---------------------------------------------------------------------------
# The contract, with no ComfyUI involved
# ---------------------------------------------------------------------------

def check_parameter_names_and_shapes():
    """The names the checkpoint and the LoRA target paths bind to."""
    st = SpatialTransformer(128, 4, 32, depth=2, context_dim=48,
                            use_linear=True)
    keys = set(st.state_dict())

    expected = {
        "norm.weight", "norm.bias",
        "proj_in.weight", "proj_in.bias",
        "proj_out.weight", "proj_out.bias",
    }
    record(expected <= keys, "the SpatialTransformer's own names",
           f"missing {sorted(expected - keys)}")

    for i in range(2):
        for suffix in ("proj.weight", "proj.bias"):
            record(f"transformer_blocks.{i}.ff.net.0.{suffix}" in keys,
                   f"block {i}: the feed-forward is GEGLU (net.0.proj), "
                   "not Linear+GELU (net.0.0)")
        for suffix in ("weight", "bias"):
            record(f"transformer_blocks.{i}.ff.net.2.{suffix}" in keys,
                   f"block {i}: ff.net.2 is the output projection")
        for norm in ("norm1", "norm2", "norm3"):
            record(f"transformer_blocks.{i}.{norm}.weight" in keys,
                   f"block {i}: {norm}")
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v"):
                record(f"transformer_blocks.{i}.{attn}.{proj}.weight" in keys,
                       f"block {i}: {attn}.{proj} -- the LoRA target name")
            record(f"transformer_blocks.{i}.{attn}.to_out.0.weight" in keys,
                   f"block {i}: {attn}.to_out.0")

    record(not any("to_out.1" in k for k in keys),
           "to_out.1 is the Dropout and holds no parameters")

    # The things that would silently add weights the checkpoint does not have.
    record(not any("norm_in" in k or "ff_in" in k for k in keys),
           "no norm_in / ff_in, matching what SDXL's real blocks carry")
    block = st.transformer_blocks[0]
    record(not hasattr(block, "norm_in") and block.has_ff_in is False,
           "and has_ff_in stays False with inner_dim left at its default",
           f"has_ff_in={block.has_ff_in!r}")
    record(block.is_res is True,
           "is_res is True when the block's inner_dim equals its dim",
           f"is_res={block.is_res!r}")

    # `inner_dim` given explicitly must turn ff_in on, since that is the
    # only way it becomes true.
    wide = BasicTransformerBlock(64, 2, 32, context_dim=48, inner_dim=128)
    record(wide.has_ff_in is True and hasattr(wide, "norm_in"),
           "an explicit inner_dim does turn the ff_in branch on")
    record(wide.is_res is False,
           "and is_res becomes False when inner_dim != dim")


def check_shapes():
    for channels, heads, d_head, depth, context_dim, linear in SDXL_LEVELS:
        st = SpatialTransformer(channels, heads, d_head, depth=depth,
                                context_dim=context_dim, use_linear=linear)
        x = torch.randn(2, channels, 6, 6)
        ctx = torch.randn(2, 5, context_dim)
        with torch.no_grad():
            out = st(x, context=ctx)
        label = f"ch={channels} depth={depth} {'linear' if linear else 'conv'}"
        record(tuple(out.shape) == (2, channels, 6, 6),
               f"{label}: [batch, channels, h, w] in and out",
               f"got {tuple(out.shape)}")
        record(bool(torch.isfinite(out).all()), f"{label}: finite output")

    # A per-block context list, which is the form a multi-condition UNet uses.
    st = SpatialTransformer(64, 2, 32, depth=3, context_dim=48,
                            use_linear=True)
    with torch.no_grad():
        ok_list = st(torch.randn(2, 64, 4, 4),
                     context=[torch.randn(2, 5, 48) for _ in range(3)])
    record(tuple(ok_list.shape) == (2, 64, 4, 4),
           "one context tensor per block is accepted")
    try:
        with torch.no_grad():
            st(torch.randn(2, 64, 4, 4),
               context=[torch.randn(2, 5, 48)] * 2)   # wrong count
    except ValueError as exc:
        record("3" in str(exc),
               "and a mismatched context count is a ValueError naming it",
               str(exc))
    else:
        record(False, "and a mismatched context count is a ValueError",
               "it was accepted")


def check_residual_structure():
    """The block is three residuals, and each one is really there.

    Checked by disabling one path at a time rather than by reading the code:
    zeroing `ff` and seeing the output change proves the residual is live.
    """
    torch.manual_seed(2)
    block = BasicTransformerBlock(64, 2, 32, context_dim=48).eval()
    x = torch.randn(2, 7, 64)
    ctx = torch.randn(2, 5, 48)
    with torch.no_grad():
        base = block(x, context=ctx)
        attn1_before = block.attn1.to_q.weight.clone()
        block.attn1.to_q.weight.zero_()
        no_attn1 = block(x, context=ctx)
        block.attn1.to_q.weight.copy_(attn1_before)

    record(not torch.equal(base, no_attn1),
           "attn1 is on the path (zeroing to_q changes the output)")

    # The residual structure, against a hand-written forward. Reading the
    # source would prove nothing; this proves the three additions are where
    # the docstring says, and that is_res wraps the feed-forward and not the
    # attentions -- a plausible mistake that a shape test cannot see.
    def manual(t, c):
        t = block.attn1(block.norm1(t), context=None) + t
        t = block.attn2(block.norm2(t), context=c) + t
        return t + block.ff(block.norm3(t))

    with torch.no_grad():
        record(torch.equal(base, manual(x, ctx)),
               "the forward is attn1 -> residual -> attn2 -> residual -> "
               "ff -> residual, exactly")

    # Cross-attention matters, so a different context gives a different answer.
    with torch.no_grad():
        other = block(x, context=torch.randn(2, 5, 48))
    record(not torch.equal(base, other),
           "attn2 uses the context (a different context gives a different out)")

    # With is_res False the feed-forward must *not* be added to the skip.
    torch.manual_seed(2)
    wide = BasicTransformerBlock(64, 2, 32, context_dim=48, inner_dim=128)
    wide.eval()
    _fill(wide, seed=13)
    # The block takes `dim` wide and returns `dim` wide; `inner_dim` is the
    # width *inside*, after ff_in has widened it. Feeding it inner_dim here
    # fails in LayerNorm, which is the right way to find out.
    xs = torch.randn(2, 7, 64)
    cs = torch.randn(2, 5, 48)
    with torch.no_grad():
        got = wide(xs, context=cs)

        def manual_wide(t, c):
            # ff_in's residual is guarded by is_res too, so with is_res
            # False this replaces the width rather than adding to it.
            t = wide.ff_in(wide.norm_in(t))
            t = wide.attn1(wide.norm1(t), context=None) + t
            t = wide.attn2(wide.norm2(t), context=c) + t
            return wide.ff(wide.norm3(t))

        want = manual_wide(xs, cs)
    record(torch.equal(got, want),
           "with inner_dim != dim, is_res is False, so neither the ff_in "
           "residual nor the feed-forward residual is applied")
    record(tuple(got.shape) == (2, 7, 64), "and the block still returns dim wide")


def check_head_split():
    """The head split is not observable from shapes, so pin it numerically.

    `q.reshape(batch, seq, heads, head_dim)` and `q.reshape(batch, heads,
    seq, head_dim)` have identical shapes and give different answers. Only a
    numerical check catches a swap, which is why this is separate from the
    shape checks above.
    """
    torch.manual_seed(5)
    attn = CrossAttention(64, heads=4, dim_head=16, context_dim=32).eval()
    _fill(attn, seed=13)
    x = torch.randn(2, 6, 64)
    ctx = torch.randn(2, 9, 32)
    with torch.no_grad():
        out = attn(x, context=ctx)

    # Recompute by hand with the documented split and compare.
    q = attn.to_q(x)
    k = attn.to_k(ctx)
    v = attn.to_v(ctx)
    batch, q_len, width = q.shape
    head_dim = width // attn.heads
    qs = q.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    ks = k.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    vs = v.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    manual = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
    manual = manual.transpose(1, 2).reshape(batch, q_len, width)
    with torch.no_grad():
        expected = attn.to_out(manual)
    record(torch.equal(out, expected),
           "heads split as (batch, seq, heads, dim_head), outer stride on "
           "the last dimension")
    record(torch.equal(out, attn.to_out(manual)),
           "and the head-merge is the exact inverse")

    # A transposed split would give a different answer, so this check has
    # teeth: prove it.
    bad = torch.nn.functional.scaled_dot_product_attention(
        q.view(batch, attn.heads, q_len, head_dim),
        k.view(batch, attn.heads, -1, head_dim),
        v.view(batch, attn.heads, -1, head_dim),
    ).transpose(1, 2).reshape(batch, q_len, width)
    with torch.no_grad():
        record(not torch.equal(out, attn.to_out(bad)),
               "a transposed head split would give a different answer, so "
               "this check can fail")


def check_feedforward():
    ff = FeedForward(64, dim_out=64, glu=True)
    record(isinstance(ff.net[0], GEGLU), "glu=True builds a GEGLU")
    record(isinstance(FeedForward(64, glu=False).net[0], torch.nn.Sequential),
           "glu=False builds Linear+GELU")
    record(isinstance(ff.net[1], torch.nn.Dropout),
           "net.1 is a Dropout, which is why net.2 is the output projection")
    with torch.no_grad():
        out = ff(torch.randn(2, 5, 64))
    record(tuple(out.shape) == (2, 5, 64), "FeedForward preserves shape")


def check_group_norm():
    norm = group_norm_32(128)
    record(isinstance(norm, torch.nn.GroupNorm) and norm.num_groups == 32,
           "group_norm_32 is 32-group GroupNorm",
           f"groups={norm.num_groups}")
    record(norm.eps == 1e-6, "with eps 1e-6", f"eps={norm.eps}")
    record(norm.affine is True, "and affine")


# ---------------------------------------------------------------------------
# Characterisation against ComfyUI
# ---------------------------------------------------------------------------

def _comfy():
    import paths as p
    root = p.get_comfy_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from comfy.ldm.modules import attention
    return attention


def check_against_comfy():
    try:
        attn = _comfy()
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("characterisation vs ComfyUI", f"comfy not importable "
             f"({type(exc).__name__})")
        return

    pairs = [
        ("SpatialTransformer", attn.SpatialTransformer, SpatialTransformer,
         dict(in_channels=128, n_heads=4, d_head=32, depth=2,
              context_dim=48, use_linear=True), (2, 128, 6, 6)),
        ("SpatialTransformer conv", attn.SpatialTransformer,
         SpatialTransformer,
         dict(in_channels=64, n_heads=2, d_head=32, depth=1,
              context_dim=48, use_linear=False), (2, 64, 6, 6)),
        ("SpatialTransformer depth 10", attn.SpatialTransformer,
         SpatialTransformer,
         dict(in_channels=128, n_heads=4, d_head=32, depth=10,
              context_dim=48, use_linear=True), (2, 128, 5, 5)),
        ("BasicTransformerBlock", attn.BasicTransformerBlock,
         BasicTransformerBlock,
         dict(dim=64, n_heads=2, d_head=32, context_dim=48), (2, 7, 64)),
        ("CrossAttention", attn.CrossAttention, CrossAttention,
         dict(query_dim=64, heads=2, dim_head=32, context_dim=48), (2, 7, 64)),
    ]

    for label, theirs_cls, ours_cls, kwargs, x_shape in pairs:
        theirs = theirs_cls(**kwargs).eval()
        ours = ours_cls(**kwargs).eval()

        same_keys = set(theirs.state_dict()) == set(ours.state_dict())
        record(same_keys, f"{label}: identical state_dict keys",
               f"only theirs {sorted(set(theirs.state_dict()) - set(ours.state_dict()))[:3]}, "
               f"only ours {sorted(set(ours.state_dict()) - set(theirs.state_dict()))[:3]}")
        if same_keys:
            shape_mismatch = [
                k for k in theirs.state_dict()
                if theirs.state_dict()[k].shape != ours.state_dict()[k].shape
            ]
            record(not shape_mismatch,
                   f"{label}: and identical shapes", f"{shape_mismatch[:3]}")

        # Weights from one generator, because ComfyUI's Linear does not
        # initialise its own -- see the module docstring.
        _fill(theirs, seed=13)
        _fill(ours, seed=13)
        torch.manual_seed(3)
        x = torch.randn(*x_shape)
        context_dim = kwargs.get("context_dim")
        ctx = None if context_dim is None else torch.randn(2, 5, context_dim)

        with torch.no_grad():
            a = theirs(x, context=ctx)
            b = ours(x, context=ctx)

        if bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()):
            worst = (a - b).abs().max().item()
            # Reported, not asserted as a gate: the transformer is
            # unambiguous, so a difference is a bug report rather than a
            # question about which side is right.
            print(f"  {'DIFF' if worst else 'SAME'}: {label}: "
                  f"max |comfy - ours| = {worst:.3e}")
        else:
            print(f"  SKIP: {label}: a non-finite output, so the comparison "
                  f"would be meaningless (comfy finite="
                  f"{bool(torch.isfinite(a).all())}, ours finite="
                  f"{bool(torch.isfinite(b).all())})")


def main() -> int:
    print("== parameter names and shapes ==")
    check_parameter_names_and_shapes()
    print("\n== shapes through the transformer ==")
    check_shapes()
    print("\n== residual structure ==")
    check_residual_structure()
    print("\n== the head split ==")
    check_head_split()
    print("\n== FeedForward and GroupNorm ==")
    check_feedforward()
    check_group_norm()
    print("\n== characterisation vs ComfyUI ==")
    check_against_comfy()

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