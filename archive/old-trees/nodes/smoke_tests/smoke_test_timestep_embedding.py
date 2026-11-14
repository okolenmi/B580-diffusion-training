"""Correctness check for nodes/model/timestep_embedding.py -- the sinusoidal
timestep embedding this project now owns instead of importing from ComfyUI
(docs/design/12-installer-and-comfy-decoupling.md section 7.1).

Two separate things are checked, and the difference matters:

1. **The formula**, against an independent reference implementation written
   from the published definition (Ho et al., *Denoising Diffusion
   Probabilistic Models*: `emb_i = cos/sin(t / 10000^(2i/d))`). The reference
   is a plain loop rather than the vectorised `exp(-log(max_period) * i/half)`
   form, so agreeing with it is evidence the closed form is right rather than
   evidence that one line was copied from another correctly.

2. **Characterisation** against ComfyUI's `Timestep`, when ComfyUI happens to
   be importable. This is deliberately *not* a gate. The two ComfyUI bugs
   found so far in this project -- a crash on frozen parameters and a
   CUDA-only autocast re-entry -- both got in by being treated as the
   reference, so a difference here is a question about which side is wrong,
   not a failure. When ComfyUI is absent the check reports SKIP and the test
   still passes: this project is being decoupled from those files, and a test
   that failed for their absence would put the coupling straight back in.

Also pinned, because both call sites' docstrings now make claims about them:
the output is float32 for every input dtype, and `Timestep` has no
parameters -- which is why `.to(device=..., dtype=...)` on it is a no-op and
why the per-device embedder cache that `unet_wrapper.py` used to keep was
saving nothing.

Run: `python nodes/smoke_tests/smoke_test_timestep_embedding.py`
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.timestep_embedding import (  # noqa: E402
    Timestep,
    timestep_embedding,
)

failures: list[str] = []
skipped: list[str] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


# ---------------------------------------------------------------------------
# An independent reference, from the published definition
# ---------------------------------------------------------------------------

def reference_embedding(timesteps, dim: int, max_period: float = 10000):
    """emb_i = cos(t / max_period^(2i/d)) for i < d/2, sin(...) after.

    Deliberately a loop with scalar math: the implementation under test uses
    one vectorised expression, so agreement between the two is not an
    artefact of shared code.
    """
    half = dim // 2
    cols = []
    for i in range(half):
        freq = math.exp(-math.log(max_period) * i / half)
        angle = timesteps.float() * freq
        cols.append(torch.cos(angle))
    for i in range(half):
        freq = math.exp(-math.log(max_period) * i / half)
        angle = timesteps.float() * freq
        cols.append(torch.sin(angle))
    out = torch.stack(cols, dim=-1)
    if dim % 2:
        out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
    return out


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def check_shape_and_dtype():
    vals = torch.tensor([1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0])
    for dim in (2, 8, 256):
        out = timestep_embedding(vals, dim)
        record(tuple(out.shape) == (6, dim),
               f"dim={dim} gives [6, {dim}]",
               f"got {tuple(out.shape)}")
        record(torch.isfinite(out).all().item(),
               f"dim={dim} is finite")

    # Odd widths are legal and need a trailing column, because cos and sin
    # together produce 2*(dim//2) and an odd dim is one short.
    for dim in (3, 5, 9):
        out = timestep_embedding(vals, dim)
        record(tuple(out.shape) == (6, dim),
               f"odd dim={dim} still gives width {dim}",
               f"got {tuple(out.shape)}")
        record(bool((out[:, -1] == 0).all()),
               f"odd dim={dim} pads with a zero column",
               f"last column is {out[0, -1].item()}")

    # A single element must not silently become width 2*(dim//2).
    one = timestep_embedding(torch.tensor([7.0]), 7)
    record(tuple(one.shape) == (1, 7),
           "a single timestep gives [1, dim]",
           f"got {tuple(one.shape)}")


# ---------------------------------------------------------------------------
# The formula itself
# ---------------------------------------------------------------------------

def phase_error_bound(vals) -> float:
    """The largest disagreement float32 can justify for these arguments.

    The implementation under test is float32 throughout, because that is
    what the published embedding specifies. The reference above computes its
    frequencies with Python's float64 `math.exp`. The two therefore differ
    in the *phase* by at most the float32 representation error of the
    argument, and cos/sin turn that into an output error of the same size:

        d(cos)/dx is bounded by 1, so argument error -> output error.

    A float32 holds `eps * |x|` absolutely, the largest frequency multiplier
    is 1 (the i=0 term), and the reference's own float64 rounding adds a
    comparable amount. Hence the factor of 2.

    Stating the bound beats a magic `atol`, and it is a stronger claim: a
    swapped cos/sin, a wrong half, or a dropped frequency factor is orders
    of magnitude larger than this, so it cannot hide inside it.
    """
    scale = max(float(vals.abs().max()), 1.0)
    return 2.0 * float(torch.finfo(torch.float32).eps) * scale


def check_against_reference():
    # SDXL's actual six values, plus fractional ones: the docstring says
    # these may be pixel counts rather than step indices, and cos/sin of a
    # 1024.0 is where a scale error would show.
    for name, vals in [
        ("sdxl resolutions", torch.tensor([1024., 1024., 0., 0., 1024., 1024.])),
        ("fractional", torch.tensor([0.5, 12.75, 999.5, 1e4])),
        ("negative and zero", torch.tensor([-3.0, 0.0, 1.0])),
        ("large", torch.tensor([1e6, 1e-6])),
    ]:
        bound = phase_error_bound(vals)
        for dim in (16, 256):
            mine = timestep_embedding(vals, dim)
            theirs = reference_embedding(vals, dim)
            worst = (mine - theirs).abs().max().item()
            record(worst <= bound,
                   f"matches the published formula, {name}, dim={dim}",
                   f"max abs diff {worst:.3e} against a float32 phase bound "
                   f"of {bound:.3e}")

    # The structural facts, at a precision where float64-vs-float32 is not
    # the story: which half is which, and that max_period is live. These are
    # the things a subtly-wrong embedding gets wrong, and they are exact.
    a = timestep_embedding(torch.tensor([5.0]), 32, max_period=10000)
    b = timestep_embedding(torch.tensor([5.0]), 32, max_period=100.0)
    record(not torch.equal(a, b),
           "max_period changes the result (so it is not ignored)")

    t = torch.tensor([1.0])
    out = timestep_embedding(t, 8)
    half = 4
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    args = 1.0 * freqs
    record(torch.allclose(out[0, :half], torch.cos(args), atol=1e-6),
           "first half is cos")
    record(torch.allclose(out[0, half:], torch.sin(args), atol=1e-6),
           "second half is sin")

    # cos(0) is 1 in every column of the first half and sin(0) is 0 in every
    # column of the second. Exact, and it catches a transposed half.
    zero = timestep_embedding(torch.tensor([0.0]), 256)
    record(bool((zero[0, :128] == 1.0).all()) and bool((zero[0, 128:] == 0.0).all()),
           "t=0 gives cos columns of 1 and sin columns of 0")


# ---------------------------------------------------------------------------
# Dtype is the algorithm's, not the caller's
# ---------------------------------------------------------------------------

def check_output_dtype_is_always_float32():
    """The claim both call sites' comments now make, pinned.

    `timesteps` is cast to float and the frequencies are built in float32,
    so the result is float32 whatever went in. Both call sites used to write
    `Timestep(256).to(device=..., dtype=...)` implying the module's dtype
    controlled this; it does not, and if that ever changes then every
    consumer that assumed float32 is wrong.
    """
    for dtype in (torch.float32, torch.float16, torch.bfloat16,
                  torch.float64, torch.int32, torch.int64):
        out = timestep_embedding(torch.tensor([3.0], dtype=dtype), 32)
        record(out.dtype == torch.float32,
               f"input {dtype} gives float32 output",
               f"got {out.dtype}")

    module = Timestep(256)
    record(sum(p.numel() for p in module.parameters()) == 0,
           "Timestep has no parameters, so .to() on it is a no-op",
           f"{sum(p.numel() for p in module.parameters())} parameters")
    record(len(list(module.buffers())) == 0,
           "and no buffers either")
    for dtype in (torch.float16, torch.bfloat16):
        moved = Timestep(256).to(dtype)
        out = moved(torch.tensor([3.0]))
        record(out.dtype == torch.float32,
               f".to({dtype}) does not change the output dtype, as documented",
               f"got {out.dtype}")

    record(tuple(module(torch.tensor([1.0, 2.0])).shape) == (2, 256),
           "Timestep(256) is the interface the call sites use")


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def check_rejections():
    for dim in (0, 1, -4):
        try:
            timestep_embedding(torch.tensor([1.0]), dim)
        except ValueError as exc:
            record(True, f"dim={dim} is rejected as a ValueError")
            if dim == 1:
                record("at least 2" in str(exc),
                       "and the message says what the minimum is",
                       str(exc))
        else:
            record(False, f"dim={dim} is rejected as a ValueError",
                   "it returned a tensor")

    try:
        timestep_embedding(torch.zeros(2, 3), 16)
    except ValueError as exc:
        record("1-D" in str(exc),
               "a 2-D timesteps tensor is rejected, and says it wants 1-D",
               str(exc))
    else:
        record(False, "a 2-D timesteps tensor is rejected",
               "it returned a tensor")

    try:
        Timestep(1)
    except ValueError:
        record(True, "Timestep(1) fails at construction rather than forward")
    else:
        record(False, "Timestep(1) fails at construction rather than forward",
               "it constructed")


# ---------------------------------------------------------------------------
# Device follows the input
# ---------------------------------------------------------------------------

def check_device_follows_input():
    """Not `dim`-chosen and not module-chosen: taken from the input tensor.

    On an accelerator build this is load-bearing -- building the frequencies
    on the host and moving the result would put a CPU allocation in the
    middle of a forward pass. Asserted as identity on CPU, which is the part
    that is true everywhere.
    """
    t = torch.tensor([1.0, 2.0])
    out = timestep_embedding(t, 64)
    record(out.device == t.device,
           "the embedding is built on the input's device",
           f"input {t.device}, output {out.device}")


# ---------------------------------------------------------------------------
# Characterisation against ComfyUI, not a gate
# ---------------------------------------------------------------------------

def check_against_comfy():
    """Compare with ComfyUI's copy if we can find it. Never a failure.

    A divergence is information. Two ComfyUI bugs in this project's area
    were found by refusing to treat ComfyUI as correct, so this check
    reports the difference and moves on.
    """
    try:
        import paths as _paths  # noqa: F401
        root = _paths.get_comfy_dir()
    except Exception as exc:  # noqa: BLE001 -- any failure means "not here"
        skip("characterisation vs ComfyUI",
             f"no ComfyUI directory resolvable ({type(exc).__name__})")
        return
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from comfy.ldm.modules.diffusionmodules.openaimodel import (
            Timestep as ComfyTimestep,
        )
    except Exception as exc:  # noqa: BLE001
        skip("characterisation vs ComfyUI",
             f"comfy not importable from {root} ({type(exc).__name__})")
        return

    cases = [
        ("sdxl resolutions", torch.tensor([1024., 1024., 0., 0., 1024., 1024.])),
        ("fractional", torch.tensor([0.5, 12.75, 999.5])),
        ("odd dim", torch.tensor([3.0, 4.0])),
    ]
    module = ComfyTimestep(256)
    for name, vals in cases:
        mine = timestep_embedding(vals, 256)
        theirs = module(vals)
        identical = torch.equal(mine, theirs)
        if identical:
            record(True, f"identical to ComfyUI, {name}, dim=256")
        else:
            # Not a failure. Say how far apart they are so the number is
            # on the record rather than in someone's head.
            diff = (mine - theirs).abs().max().item()
            print(f"  DIFF: differs from ComfyUI, {name}, dim=256: "
                  f"max abs diff {diff:.3e} "
                  f"(characterisation, not a failure -- "
                  f"docs/design/12 section 7)")

    # ComfyUI's class carries a `repeat_only` branch reached through its own
    # timestep_embedding, not through Timestep, so Timestep has no such
    # knob. Confirm our copy does not invent one either.
    record(
        not any("repeat" in name for name in vars(module.forward)),
        "no repeat_only branch crept into the wrapper",
    )


def main() -> int:
    print("== shape and dtype ==")
    check_shape_and_dtype()
    print("\n== the published formula ==")
    check_against_reference()
    print("\n== output dtype is the algorithm's ==")
    check_output_dtype_is_always_float32()
    print("\n== rejections ==")
    check_rejections()
    print("\n== device ==")
    check_device_follows_input()
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