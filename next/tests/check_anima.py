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

import tempfile

from safetensors.torch import save_file

from ..families.anima import adapters as anima_adapters
from ..families.anima.conditioning import AnimaConditioning, AnimaConditioningKit, AnimaTextEncoder, TextEncoding
from ..families.anima.dit import AnimaDiT, DiTConfig, RotaryEmbedding3D, rotate_split_half
from ..families.anima.denoiser import load_denoiser
from ..families.anima.latents import LATENTS
from ..families.anima.objective import AnimaObjective
from ..families.anima.text import QwenConfig, QwenEncoder
from ..families.anima.tokens import AnimaTokenizer, QWEN_PAD, vocabulary_dir
from ..data.ids import spec_fields
from ..families.api.adapters import TrainableSelection, descriptors_from_shapes, plan_adapter

from ..families.api.parts import ModelPart, PartInfo, PartKind, PartSource
from ..families.general.adapters.inject import descriptors_from_model, inject
from ..families.general.vae.codec import LatentSpec, VAECodec
from ..families.general.vae.wan import WanImageVAE, to_image_state
from ..families.sdxl.latents import LATENTS as SDXL_LATENTS
from .support import check, raises, run

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


def vae_matches_file():
    """The file's video tensors become exactly the single-image network's tensors (time convolutions dropped)."""
    theirs = {k: tuple(v[1]) for k, v in headers()["qwen_image_vae"].items()}
    folded = to_image_state({k: torch.empty(shape, device="meta") for k, shape in theirs.items()})
    with torch.device("meta"):
        net = WanImageVAE(dim=96)
    ours = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    check({k: tuple(v.shape) for k, v in folded.items()} == ours, f"{len(ours)} tensors of {len(theirs)} in the file (the rest are time convolutions)")


def latent_codec():
    """Latents are normalised per channel and come back; the cache identity sees the statistics and SDXL's stays as it was."""
    torch.manual_seed(0)
    net = WanImageVAE(dim=8, mult=(1, 2, 2, 2), blocks=1)
    codec = VAECodec(net, LATENTS, "cpu", torch.float32)
    image = torch.rand(1, 3, 32, 32)
    z = codec.encode(image)
    raw = net.encode(image * 2 - 1)
    mean, std = torch.tensor(LATENTS.mean).view(1, -1, 1, 1), torch.tensor(LATENTS.std).view(1, -1, 1, 1)
    check(z.shape == (1, 16, 4, 4) and torch.allclose(z, (raw - mean) / std, atol=1e-6), "encode subtracts the mean and divides by the spread")
    check(torch.allclose(codec.model.decode(z * std + mean), codec.model.decode(raw), atol=1e-5) and codec.decode(z).shape == (1, 3, 32, 32),
          "decode undoes it")
    check(set(spec_fields(LATENTS)) >= {"mean", "std"} and set(spec_fields(SDXL_LATENTS)) == {"channels", "spatial_factor", "scale", "shift"},
          "the cache identity holds the statistics; SDXL's did not change")
    check(LATENTS.pixel_multiple == 16 and SDXL_LATENTS.pixel_multiple == 8, "Anima images are cut to multiples of 16 px")


def flow_objective():
    """Samples lie in the asked range, the velocity target inverts to the clean latent, sigma is noise over signal."""
    objective = AnimaObjective()
    g = torch.Generator().manual_seed(0)
    for mode in objective.timestep_modes:
        t = objective.sample_t(500, mode, 0.2, 0.6, g)
        check(0.2 <= t.min() and t.max() <= 0.6, f"{mode}: draws stay inside [0.2, 0.6]")
    x0, noise, t = torch.randn(3, 16, 4, 4), torch.randn(3, 16, 4, 4), torch.tensor([0.1, 0.5, 0.9])
    x_t, target, sigma = objective.noisy(x0, noise, t)
    check(torch.equal(target, noise - x0) and torch.allclose(sigma, t / (1 - t)), "target is noise - x0; sigma is s / (1 - s)")
    check(torch.allclose(objective.to_x0(target, x_t, sigma), x0, atol=1e-5), "the true velocity gives back x0")
    check(AnimaObjective().sample_t(2000, "sigmoid", 0.0, 1.0, g).mean() > AnimaObjective(shift=1.0).sample_t(2000, "sigmoid", 0.0, 1.0, g).mean(),
          "a larger shift moves the draws toward noise")
    raises(ValueError, lambda: AnimaObjective(shift=0), "a zero shift is refused")


def adapter_plan_and_loading():
    """The plan from the real file's key names equals the plan from the model; LoRA keys are the kohya ones ComfyUI reads; a file
    written from a small network loads back strictly and predicts the same."""
    theirs = {k: tuple(v[1]) for k, v in headers()["anima_baseV10"].items()}
    with torch.device("meta"):
        full = AnimaDiT(DiTConfig())
    layout = anima_adapters.LAYOUT
    from_file = plan_adapter(layout, descriptors_from_shapes(theirs, "net."), TrainableSelection(), 16, 16)
    from_model = plan_adapter(layout, descriptors_from_model(full), TrainableSelection(), 16, 16)
    by_path = lambda plan: sorted(plan.modules, key=lambda m: m.path)  # noqa: E731  (a file lists its keys alphabetically)
    check(by_path(from_file) == by_path(from_model) and len(from_model.modules) == 280, f"default plan: 280 modules in 28 blocks ({len(from_model.modules)})")
    check(not any(m.path.startswith("llm_adapter") for m in from_model.modules), "the text adapter stays untrained by default")
    everything = plan_adapter(layout, descriptors_from_model(full), TrainableSelection(groups=tuple(g.name for g in layout.groups())), 16, 16)
    check(len(everything.modules) > len(from_model.modules) + 6 * 10, "every group can be chosen")

    torch.manual_seed(0)
    cfg = DiTConfig(model_channels=96, num_blocks=2, num_heads=2, adapter_blocks=2)
    source = AnimaDiT(cfg).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/dit.safetensors"
        save_file({f"net.{k}": v.contiguous() for k, v in source.state_dict().items()} | {"net.pos_embedder.seq": torch.arange(4.0)}, path)
        info = PartInfo(PartKind.DENOISER, "denoiser", "net.", 1, "F32", 0, {"blocks": 2, "width": 96, "adapter_blocks": 2})
        part = ModelPart("anima", "denoiser", PartSource(path, "net."), info, "auto")
        denoiser = load_denoiser(part, dtype=torch.float32, config=cfg)
    kit = AnimaConditioningKit(_encoder())
    cond = kit.stack(kit.encode(["a cat", "a dog on a long road"]))
    x, t = torch.randn(2, 16, 8, 8), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        check(torch.allclose(denoiser.predict(x, t, cond), source(x, t, source.encode_text(cond.hidden, cond.ids, cond.hidden_mask, cond.ids_mask)), atol=1e-6),
              "a loaded DiT predicts like the one it was saved from")
    part_ = anima_adapters.inject_denoiser(denoiser, TrainableSelection(), rank=4, alpha=4)
    keys = set(anima_adapters.SCHEMES["kohya"].export(part_))
    check("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight" in keys and "lora_unet_blocks_1_mlp_layer2.alpha" in keys,
          "LoRA keys are lora_unet_<path with underscores>")
    check(not any("llm_adapter" in k for k in keys), "nothing of the text adapter is written")


if __name__ == "__main__":
    run(dit_matches_files, small_forward, rotation, checkpointing_flags, qwen_matches_file, conditioning_batches, tokenizer_files,
        vae_matches_file, latent_codec, flow_objective, adapter_plan_and_loading)
