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

from ..families.anima.conditioning import AnimaConditioning, AnimaConditioningKit, AnimaTextEncoder, TextEncoding
from ..families.anima.dit import AnimaDiT, DiTConfig, RotaryEmbedding3D, rotate_split_half
from ..families.anima.text import QwenConfig, QwenEncoder
from ..families.anima.tokens import AnimaTokenizer, QWEN_PAD, vocabulary_dir
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


def qwen_matches_file():
    """Our Qwen3-0.6B has the file's tensors: same names, same shapes, nothing else."""
    theirs = {k: v[1] for k, v in headers()["qwen_3_06b_base"].items()}
    with torch.device("meta"):
        net = QwenEncoder()
    ours = {k: list(v.shape) for k, v in net.state_dict().items()}
    check(ours == theirs, f"{len(theirs)} tensors, same names and shapes "
                          f"(missing {sorted(set(theirs) - set(ours))[:2]}, extra {sorted(set(ours) - set(theirs))[:2]})")


class _Words:
    """A stand-in tokenizer: one Qwen id per character, T5 ids are the character codes mod 100 plus ``</s>``."""

    def tokenize(self, text):
        return [10 + ord(c) % 50 for c in text] or [0], [2 + ord(c) % 90 for c in text] + [1]


def _encoder(batch_size=16):
    torch.manual_seed(0)
    cfg = QwenConfig(vocab=100, width=1024, layers=1, heads=2, kv_heads=1, head_dim=64, mlp_width=64)
    return AnimaTextEncoder(QwenEncoder(cfg), _Words(), out_dtype=torch.float32, tokens=32, batch_size=batch_size)


def conditioning_batches():
    """Prompts of different lengths share a batch of fixed width; blending ends are the pure sides; the adapter output of a
    padded prompt equals its output when run alone."""
    enc = _encoder()
    kit = AnimaConditioningKit(enc)
    a, b = kit.encode(["a cat", "a very long prompt about a cat"])
    batch = kit.stack([a, b])
    check(batch.hidden.shape == (2, 32, 1024) and batch.ids.shape == (2, 32), "every batch has the same width")
    check(batch.hidden_mask.sum(1).tolist() == [5, 30] and batch.ids_mask.sum(1).tolist() == [6, 31], "masks count the real tokens")
    check(not batch.hidden[0, 5:].any(), "positions past the prompt are zero")
    cut = enc.encode_text(["x" * 100])[0]
    check(cut.hidden.shape[1] == 32 and cut.ids.shape[1] == 32 and cut.ids[0, -1].item() == 1, "a long prompt is cut and still ends in </s>")
    check(kit.blank().hidden_mask.sum().item() == 1, "the empty prompt is one pad token")
    # blending
    mid = a.blend(b, torch.tensor([0.0]))
    check(torch.equal(mid.hidden, a.hidden) and torch.equal(mid.hidden_mask, a.hidden_mask) and torch.equal(mid.ids, a.ids), "p=0 is the caption")
    end = a.blend(b, 1.0)
    check(torch.equal(end.hidden, b.hidden) and torch.equal(end.hidden_mask, b.hidden_mask) and torch.equal(end.ids, b.ids), "p=1 is the partner")
    half = a.blend(b, 0.4, ["qwen"])
    check(torch.allclose(half.hidden, 0.6 * a.hidden + 0.4 * b.hidden, atol=1e-6) and torch.equal(half.ids, a.ids), "segments limit the blend")
    check(a.index(0).hidden.shape == a.hidden.shape and batch.index(1).ids.equal(b.ids), "index picks samples")
    # batching changes nothing in float32
    one = [enc.encode_text([p])[0] for p in ("a cat", "a very long prompt about a cat")]
    many = enc.encode_text(["a cat", "a very long prompt about a cat"])
    check(all(torch.allclose(x.hidden, y.hidden, atol=1e-5) and torch.equal(x.ids, y.ids) for x, y in zip(one, many)),
          "batched encoding agrees with one at a time")
    # the adapter ignores padding
    torch.manual_seed(1)
    net = AnimaDiT(DiTConfig(model_channels=96, num_blocks=1, num_heads=2, adapter_blocks=2)).eval()
    with torch.no_grad():
        alone = net.encode_text(a.hidden[:, :5], a.ids[:, :6])
        padded = net.encode_text(batch.hidden, batch.ids, batch.hidden_mask, batch.ids_mask)[:1]
    check(alone.shape == padded.shape == (1, 512, 1024) and torch.allclose(alone, padded, atol=1e-5), "padding does not change the context")


def tokenizer_files():
    """The real tokenizers, when their vocabulary is at hand (``reference/equivalence_tokens.py`` compares them to Hugging Face)."""
    try:
        vocabulary_dir("qwen"), vocabulary_dir("t5")
    except FileNotFoundError:
        print("  skipped: no tokenizer vocabulary (set QWEN_TOKENIZER_DIR and T5_TOKENIZER_DIR)")
        return
    tok = AnimaTokenizer()
    qwen, t5 = tok.tokenize("1girl, miku (vocaloid)")
    check(t5[-1] == 1 and len(qwen) > 5, "booru tags tokenize as plain text")
    check(tok.tokenize("")[0] == [QWEN_PAD] and tok.tokenize("")[1] == [1], "the empty prompt: one pad token, and </s>")


if __name__ == "__main__":
    run(dit_matches_files, small_forward, rotation, checkpointing_flags, qwen_matches_file, conditioning_batches, tokenizer_files)
