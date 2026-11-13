"""The Anima family: its modules against what the real files contain.

    python -m next.tests.check_anima

The fixture holds the tensor names and shapes of the owner's headers (``fixtures/anima_headers.json.gz``). Networks are built
without weights, so the full-size ones cost nothing. Numbers against ComfyUI's own code are checked by
``reference/equivalence_anima.py`` (needs a ComfyUI checkout).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from ..families.anima.dit import AnimaDiT, DiTConfig, RotaryEmbedding3D, rotate_split_half
from .support import check, run

FIXTURE = Path(__file__).parent / "fixtures" / "anima_headers.json.gz"
FILE_ONLY = ("net.pos_embedder.",)  # buffers some files carry; ours are computed


def headers() -> dict:
    with gzip.open(FIXTURE, "rt") as handle:
        return json.load(handle)


def dit_matches_files():
    """Every key and shape of both real Anima files is in our network, and nothing else is."""
    for name, blocks in (("anima_baseV10", 28), ("anima29B_v10", 40)):
        theirs = {k[len("net."):]: v[1] for k, v in headers()[name].items() if not k.startswith(FILE_ONLY)}
        cfg = DiTConfig.from_shapes(blocks, theirs["final_layer.adaln_modulation.1.weight"][1], theirs["x_embedder.proj.1.weight"][1])
        with torch.device("meta"):
            net = AnimaDiT(cfg)
        ours = {k: list(v.shape) for k, v in net.state_dict().items()}
        check(ours == theirs, f"{name}: {len(theirs)} tensors, same names and shapes "
                              f"(missing {sorted(set(theirs) - set(ours))[:2]}, extra {sorted(set(ours) - set(theirs))[:2]})")


def small_forward():
    """A small network runs on one latent grid of any even size, deterministically, and keeps the shapes."""
    torch.manual_seed(0)
    net = AnimaDiT(DiTConfig(model_channels=96, num_blocks=2, num_heads=2, adapter_blocks=1)).eval()
    x, t, ctx = torch.randn(2, 16, 12, 10), torch.tensor([0.2, 0.9]), torch.randn(2, 7, 1024)
    with torch.no_grad():
        a, b = net(x, t, ctx), net(x, t, ctx)
    check(a.shape == x.shape and torch.equal(a, b), "latents in, velocity of the same shape out, repeatable")
    with torch.no_grad():
        one = net(x[:1], t[:1], ctx[:1])
    check(torch.allclose(a[:1], one, atol=1e-5), "a sample's output does not depend on its batch neighbours")
    try:
        net(torch.randn(1, 16, 11, 10), t[:1], ctx[:1])
    except ValueError as exc:
        check("patch" in str(exc), "an odd latent size is refused with the reason")
    else:
        check(False, "an odd latent size is refused")


def rotation():
    """Rotary angles turn pairs without changing their length; the three axes share the head dimension."""
    rope = RotaryEmbedding3D(128, (4.0, 4.0, 1.0))
    angles = rope.angles(1, 5, 7, "cpu")
    check(angles.shape == (35, 64), f"35 tokens, 64 angles per head dimension of 128 ({tuple(angles.shape)})")
    x = torch.randn(1, 35, 3, 128)
    check(torch.allclose(rotate_split_half(x, angles).norm(dim=-1), x.norm(dim=-1), atol=1e-4), "a rotation keeps lengths")
    check(torch.equal(rotate_split_half(x, torch.zeros_like(angles)), x), "zero angles change nothing")
    check(torch.equal(angles[:, :22], torch.zeros(35, 22)), "the time axis does not move on a single frame")


def checkpointing_flags():
    net = AnimaDiT(DiTConfig(model_channels=96, num_blocks=4, num_heads=2, adapter_blocks=1))
    calls = []
    net.set_checkpointer(lambda fn, args, params: (calls.append(1), fn(*args))[1], 0.5)
    x, t, ctx = torch.randn(1, 16, 4, 4), torch.tensor([0.5]), torch.randn(1, 5, 1024)
    net(x, t, ctx).sum().backward()
    check(len(calls) == 2, f"half the blocks run through the checkpointer ({len(calls)} of 4)")
    check(len(net.checkpoint_blocks()) == 4, "every block can be chosen one by one")


if __name__ == "__main__":
    run(dit_matches_files, small_forward, rotation, checkpointing_flags)
