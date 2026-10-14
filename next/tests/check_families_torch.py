"""Torch-based self-checks for next/families/general and the family kits (CPU, tiny random models).

    python -m next.tests.check_families_torch
"""

from __future__ import annotations

import random

import torch

import json
import tempfile

from ..families.general.clip import tokenizer as tok
from ..families.general.clip.checkpoint import to_tower_state
from ..families.general.clip.model import CLIPConfig, CLIPTextTower
from ..families.general.schedules import flow, timesteps
from ..families.general.tensors import read_prefixed
from ..families.general.schedules.ddpm import (DiffusionProcess, DiscreteSchedule, EpsParameterization,
                                               VPredParameterization, ZeroTerminalSNRSchedule, scale_input)
from ..families.general.vae.autoencoder import SDXL_DDCONFIG, AutoencoderKL
from ..families.general.vae.codec import LatentSpec, VAECodec
from .support import check, raises, run


def schedules():
    s = DiscreteSchedule()
    alpha, sigma = s.alpha_sigma(torch.tensor([0, 999]))
    check(abs(sigma[1].item() - 14.6146) < 1e-3 and abs(alpha[0].item() - 0.9996) < 1e-3, "SDXL sigma_max is 14.6146")
    check(s.alpha_sigma(5)[0].ndim == 0, "int timestep gives scalars")
    zt = ZeroTerminalSNRSchedule()
    a_last, s_last = zt.alpha_sigma(999)
    check(a_last.item() == 0.0 and torch.isinf(s_last), "zero terminal SNR ends at pure noise")
    raises(ValueError, lambda: DiffusionProcess(zt, EpsParameterization()), "eps with zero-terminal-SNR is refused", "v-prediction")
    x0, noise, t = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8), torch.tensor([10, 500, 990])
    for param in (EpsParameterization(), VPredParameterization()):
        xt, target, sig = DiffusionProcess(s, param).noisy(x0, noise, t)
        check((param.to_x0(target, xt, sig) - x0).abs().max() < 1e-4, f"{param.name}: target inverts to x0")
        check((param.to_eps(target, xt, sig) - noise).abs().max() < 1e-4, f"{param.name}: target inverts to eps")
    scaled = scale_input(xt, sig.flatten())
    check(scaled.shape == xt.shape and scale_input(xt, 2.0, torch.bfloat16).dtype == torch.bfloat16, "input scaling shapes and dtype")


def flow_and_timesteps():
    x0, noise = torch.randn(2, 16, 1, 4, 4), torch.randn(2, 16, 1, 4, 4)
    s = torch.tensor([0.25, 0.75])
    xs, v = flow.noisy(x0, noise, s)
    check((flow.to_x0(v, xs, s) - x0).abs().max() < 1e-5 and torch.equal(v, noise - x0), "flow velocity inverts to x0 (5-D latents)")
    g = torch.Generator().manual_seed(0)
    draws = flow.sample_sigma(2000, "sigmoid", shift=3.0, generator=g)
    check(0 < draws.min() and draws.max() < 1 and draws.mean() > 0.5, "shifted sigmoid sampling favours noise")
    check(all(0 < flow.sample_sigma(50, m).min() for m in flow.SAMPLERS), "every flow sampler stays in (0,1)")
    check(flow.shift_sigma(torch.tensor(0.5), 1.0).item() == 0.5, "shift 1 is the identity")
    check(all(torch.isfinite(flow.loss_weight(torch.tensor([0.1, 0.9]), w)).all() for w in flow.WEIGHTINGS), "flow weightings are finite")
    raises(ValueError, lambda: flow.sample_sigma(2, "nope"), "unknown flow sampler", "nope")
    rng = random.Random(0)
    check(all(0 <= timesteps.sample_timestep(rng, m, 0, 999) <= 999 for m in timesteps.MODES for _ in range(50)), "scalar timesteps stay in range")
    batch = timesteps.sample_timesteps(500, "low", 0, 999, torch.Generator().manual_seed(1))
    check(batch.dtype == torch.long and batch.float().mean() < 400, "low mode is biased toward small t")
    raises(ValueError, lambda: timesteps.sample_timestep(rng, "adaptive", 0, 9), "train-time modes are not static modes", "adaptive")


def vae():
    cfg = {**SDXL_DDCONFIG, "ch": 32, "ch_mult": (1, 2, 4), "num_res_blocks": 1}
    model = AutoencoderKL(4, cfg)
    spec = LatentSpec(4, 4, 0.13025)
    check(spec.noise_shape(64, 96, 2) == (2, 4, 16, 24), "latent spec gives the noise shape")
    codec = VAECodec.kl_from_state_dict(model.state_dict(), cfg, 4, spec)
    images = torch.rand(2, 3, 32, 32)
    z = codec.encode(images)
    check(z.shape == (2, 4, 8, 8), f"encode downsamples by the family's factor ({tuple(z.shape)})")
    out = codec.decode(z)
    check(out.dtype == torch.uint8 and out.shape == (2, 3, 32, 32), "decode returns uint8 images")
    check(torch.equal(codec.encode((images * 255).to(torch.uint8)).round(decimals=2), codec.encode((images * 255).to(torch.uint8)).round(decimals=2)),
          "uint8 input is accepted")
    check(not any(p.requires_grad for p in codec.model.parameters()), "the codec freezes its model")
    broken = dict(model.state_dict())
    broken.pop("quant_conv.weight")
    raises(ValueError, lambda: VAECodec.kl_from_state_dict(broken, cfg, 4, spec), "weights that do not fit are refused", "missing")


def to_openclip(state: dict, layers: int) -> dict:
    """Tower names rewritten as an openclip file would spell them."""
    out = {"positional_embedding": state["text_model.embeddings.position_embedding.weight"],
           "token_embedding.weight": state["text_model.embeddings.token_embedding.weight"],
           "ln_final.weight": state["text_model.final_layer_norm.weight"],
           "ln_final.bias": state["text_model.final_layer_norm.bias"],
           "text_projection": state["text_projection.weight"].t().contiguous()}
    for i in range(layers):
        b, d = f"transformer.resblocks.{i}.", f"text_model.encoder.layers.{i}."
        for old, new in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"), ("mlp.c_fc", "mlp.fc1"),
                         ("mlp.c_proj", "mlp.fc2"), ("attn.out_proj", "self_attn.out_proj")):
            for suf in ("weight", "bias"):
                out[f"{b}{old}.{suf}"] = state[f"{d}{new}.{suf}"]
        for suf in ("weight", "bias"):
            out[f"{b}attn.in_proj_{suf}"] = torch.cat([state[f"{d}self_attn.{n}.{suf}"] for n in ("q_proj", "k_proj", "v_proj")])
    return out


def clip_tower():
    cfg = CLIPConfig(hidden=32, intermediate=64, heads=4, layers=3, activation="quick_gelu", vocab=100, eos_id=99)
    tower = CLIPTextTower(cfg).eval()
    tokens = torch.randint(0, 90, (3, 77))
    tokens[:, 0] = 98
    tokens[0, 4:] = 99
    tokens[1, 9] = 99
    tokens[2, 30] = 99
    with torch.no_grad():
        hidden, pooled = tower(tokens)
        h_last, _ = tower(tokens, layer=-1)
        h_normed, _ = tower(tokens, layer=-2, norm_hidden=True)
    check(hidden.shape == (3, 77, 32) and pooled.shape == (3, 32), "tower outputs hidden state and pooled vector")
    check(not torch.allclose(hidden, h_last) and not torch.allclose(hidden, h_normed), "layer and final-norm choices change the output")
    tokens2 = tokens.clone()
    tokens2[0, 60] = 7
    with torch.no_grad():
        _, pooled2 = tower(tokens2)
    check(torch.allclose(pooled, pooled2) and pooled.shape[0] == 3, "pooling reads the first EOS position, not the padding after it")
    with torch.no_grad():
        later, _ = tower(tokens2)
    check(torch.allclose(hidden[0, :60], later[0, :60], atol=1e-5), "attention is causal")
    raises(ValueError, lambda: tower(tokens, layer=-9), "layer out of range", "out of range")
    state = tower.state_dict()
    converted = to_tower_state(to_openclip(state, 3), "openclip", 3)
    check(set(converted) == set(state) and all(torch.equal(converted[k], state[k]) for k in state),
          "openclip key translation round-trips (fused qkv, transposed projection)")
    nested = to_tower_state({"transformer." + k: v for k, v in state.items()}, "hf_nested", 3)
    check(set(nested) == set(state), "nested layout drops the transformer prefix")
    raises(ValueError, lambda: to_tower_state({}, "nope", 1), "unknown layout", "nope")
    with tempfile.TemporaryDirectory() as tmp:
        from safetensors.torch import save_file
        save_file({"te.a.weight": torch.ones(2), "te.b.weight": torch.zeros(2), "other.x": torch.ones(1)}, f"{tmp}/f.safetensors")
        got = read_prefixed(f"{tmp}/f.safetensors", "te.", exclude=("b.",), dtype=torch.float16)
        check(list(got) == ["a.weight"] and got["a.weight"].dtype == torch.float16, "read_prefixed slices, excludes and casts")


def clip_tokenizer():
    with tempfile.TemporaryDirectory() as tmp:
        chars = sorted(set(tok.bytes_to_unicode().values()))
        vocab: dict[str, int] = {}
        for c in chars:
            vocab[c] = len(vocab)
            vocab[c + "</w>"] = len(vocab)
        merges = ["#version: 0.2", "c a", "ca t</w>"]
        for m in merges[1:]:
            a, b = m.split(" ")
            vocab[a + b] = len(vocab)
        with open(f"{tmp}/vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(f"{tmp}/merges.txt", "w") as f:
            f.write("\n".join(merges) + "\n")
        t = tok.CLIPTokenizer(tmp)
        out = t.tokenize("cat")
        check(len(out["l"]) == 1 and len(out["l"][0]) == 77 and out["l"][0][0] == tok.CLIP_BOS, "one section of 77 starting with BOS")
        check(out["l"][0][2] == tok.CLIP_EOS and out["l"][0][-1] == tok.CLIP_EOS, "CLIP-L pads with EOS")
        check(out["g"][0][2] == tok.CLIP_EOS and out["g"][0][-1] == 0, "CLIP-G pads with 0")
        check(out["l"][0][1] == vocab["cat</w>"], "merges apply")
        check(len(t.tokenize("cat " * 100)["l"]) > 1, "long prompts become several sections")
        check(all(len(s) == 77 for s in t.tokenize("cat " * 100)["g"]), "every section is full length")
        check(t.tokenize("")["l"][0][1] == tok.CLIP_EOS, "empty prompt is BOS then EOS")
        check(t.tokenize("(cat)") != t.tokenize("cat"), "parentheses are text, not weights")
        raises(FileNotFoundError, lambda: tok.CLIPTokenizer(tmp + "/nowhere"), "missing vocabulary is a clear error", "expected")
        check(sorted(tok.bytes_to_unicode()) == list(range(256)) and len(set(tok.bytes_to_unicode().values())) == 256,
              "byte alphabet is a bijection")


if __name__ == "__main__":
    run(schedules, flow_and_timesteps, vae, clip_tower, clip_tokenizer)
