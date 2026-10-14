"""Self-checks for next/families/sdxl (CPU, tiny random models).

    python -m next.tests.check_sdxl
"""

from __future__ import annotations

import json
import tempfile

import torch
from safetensors.torch import save_file

from ..families.api.parts import PartInfo, PartKind, make_part
from ..families.general.clip import tokenizer as tok
from ..families.general.clip.model import CLIPConfig, CLIPTextTower
from ..families.general.embeddings import timestep_embedding
from ..families.sdxl.conditioning import (SDXLConditioning, SDXLConditioningKit, SDXLTextEncoder, load_text_encoder,
                                          size_embedding)
from .check_families_torch import to_openclip
from .support import check, raises, run

L_CFG = CLIPConfig(hidden=768, intermediate=48, heads=12, layers=2, activation="quick_gelu", projection=False)
G_CFG = CLIPConfig(hidden=1280, intermediate=40, heads=20, layers=2, activation="gelu")


def make_tokenizer(tmp: str) -> tok.CLIPTokenizer:
    chars = sorted(set(tok.bytes_to_unicode().values()))
    vocab: dict[str, int] = {}
    for c in chars:
        vocab[c] = len(vocab)
        vocab[c + "</w>"] = len(vocab)
    vocab["ca"], vocab["cat</w>"] = len(vocab), len(vocab) + 1
    with open(f"{tmp}/vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(f"{tmp}/merges.txt", "w") as f:
        f.write("#version: 0.2\nc a\nca t</w>\n")
    return tok.CLIPTokenizer(tmp)


def towers(dtype=torch.float32):
    torch.manual_seed(0)
    return CLIPTextTower(L_CFG).to(dtype), CLIPTextTower(G_CFG).to(dtype)


def conditioning_ops():
    ctx = torch.randn(2, 77, 2048)
    y = torch.randn(2, 2816)
    a, b = SDXLConditioning(ctx, y), SDXLConditioning(torch.randn(2, 77, 2048), torch.randn(2, 2816))
    check(torch.equal(a.blend(b, 0.0).ctx, a.ctx) and torch.allclose(a.blend(b, 1.0).ctx, b.ctx, atol=1e-6), "blend 0 and 1 give the ends")
    half = a.blend(b, 0.5, segments=["clip_g"])
    check(torch.equal(half.ctx[..., :768], a.ctx[..., :768]) and not torch.equal(half.ctx[..., 768:], a.ctx[..., 768:]),
          "blending clip_g leaves clip_l alone")
    check(torch.equal(half.y, a.y), "and leaves y alone")
    per_sample = a.blend(b, torch.tensor([0.0, 1.0]))
    check(torch.equal(per_sample.ctx[0], a.ctx[0]) and torch.allclose(per_sample.ctx[1], b.ctx[1], atol=1e-6), "per-sample blend weights")
    raises(ValueError, lambda: a.blend(b, 0.5, ["nope"]), "unknown segment", "nope")
    raises(ValueError, lambda: a.blend(SDXLConditioning(torch.randn(2, 154, 2048), y), 0.5), "mismatched lengths", "cannot blend")
    seg = a.segments()
    check(seg["clip_l"].shape[-1] == 768 and seg["clip_g"].shape[-1] == 1280 and seg["pooled"].shape[-1] == 1280
          and seg["size"].shape[-1] == 1536, "segments cover ctx and y")
    check(SDXLConditioning.stack([a.index(0), a.index(1)]).ctx.equal(ctx) and a.index(slice(0, 1)).y.shape == (1, 2816), "stack and index")
    check(a.nbytes() == ctx.numel() * 4 + y.numel() * 4 and a.to(dtype=torch.bfloat16).ctx.dtype == torch.bfloat16, "nbytes and dtype")
    emb = size_embedding(1024, 768, crop=(5, 6), target=(1000, 900))
    check(emb.shape == (1, 1536) and torch.equal(emb[:, 256:512], timestep_embedding(torch.tensor([768.0]), 256)), "size embedding order is h, w, crop, target")
    check(torch.equal(emb[:, 1280:], timestep_embedding(torch.tensor([900.0]), 256)), "target width is last")


def encoder_cases():
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer = make_tokenizer(tmp)
        l32, g32 = towers()
        enc = SDXLTextEncoder(l32, g32, tokenizer, out_dtype=torch.float32, batch_size=8)
        out = enc.encode_text(["cat", "cat cat", ""])
        check(out[0].ctx.shape == (1, 77, 2048) and out[0].pooled.shape == (1, 1280), "one section: (1, 77, 2048) and pooled (1, 1280)")
        long = enc.encode_text(["cat " * 100])[0]
        check(long.ctx.shape[1] == 154 or long.ctx.shape[1] % 77 == 0 and long.ctx.shape[1] > 77, f"long prompt joins sections ({long.ctx.shape[1]})")
        check(enc.batching_available(), "float32 towers batch")
        serial = SDXLTextEncoder(l32, g32, tokenizer, out_dtype=torch.float32, batch_size=1).encode_text(["cat", "cat cat", ""])
        check(all(torch.allclose(a.ctx, b.ctx, atol=1e-5) and torch.allclose(a.pooled, b.pooled, atol=1e-5) for a, b in zip(out, serial)),
              "batched encoding agrees with one-at-a-time")
        mixed = enc.encode_text(["cat " * 100, "cat"])
        alone = enc.encode_text(["cat"])[0]
        check(torch.allclose(mixed[1].ctx, alone.ctx, atol=1e-5), "a long prompt in the batch does not disturb its neighbour")
        check(enc.encode_text([]) == [], "no prompts, no work")
        half_l, half_g = towers(torch.float16)
        check(not SDXLTextEncoder(half_l, half_g, tokenizer).batching_available(), "float16 towers do not batch")
        kit = SDXLConditioningKit(enc, default_size=(1024, 1024))
        cond = kit.encode(["cat", "cat cat"], [(1024, 1024), (832, 1216)])
        check(cond[0].y.shape == (1, 2816) and cond[0].y.dtype == torch.float32, "y is (1, 2816) float32")
        check(not torch.equal(cond[0].y[:, 1280:], cond[1].y[:, 1280:]) and torch.allclose(cond[0].y[:, :1280], enc.encode_text(["cat"])[0].pooled.float(), atol=1e-5),
              "sizes change only the size half of y")
        check(kit.cache_key("cat", (832, 1216)) != kit.cache_key("cat", (1024, 1024)) and kit.cache_key("cat") == kit.cache_key("cat", (1024, 1024)),
              "cache key includes the size, defaulting to the kit's")
        check(kit.blank().ctx.shape == (1, 77, 2048) and kit.bytes_per_prompt() == 77 * 2048 * 4 + 2816 * 4, "blank and bytes per prompt")
        check(torch.equal(enc.encode_text([""])[0].ctx, kit.blank().ctx), "blank is the empty prompt")


def loading_cases():
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer = make_tokenizer(tmp)
        l, g = towers()
        state = {}
        state.update({f"conditioner.embedders.0.transformer.{k}": v for k, v in l.state_dict().items()})
        state.update({f"conditioner.embedders.1.model.{k}": v for k, v in to_openclip(g.state_dict(), G_CFG.layers).items()})
        state["conditioner.embedders.0.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand(1, -1)
        path = f"{tmp}/sdxl.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, path)
        parts = {
            "clip_l": make_part("sdxl", path, PartInfo(PartKind.TEXT_ENCODER, "clip_l", "conditioner.embedders.0.", 0, "F32", 0,
                                                       {"layout": "hf_nested", "layers": 2})),
            "clip_g": make_part("sdxl", path, PartInfo(PartKind.TEXT_ENCODER, "clip_g", "conditioner.embedders.1.model.", 0, "F32", 0,
                                                       {"layout": "openclip", "layers": 2})),
        }
        parts["clip_l"] = make_part("sdxl", path, PartInfo(PartKind.TEXT_ENCODER, "clip_l", "conditioner.embedders.0.", 0, "F32", 0,
                                                           {"layout": "hf_nested", "layers": 2}))
        loaded = load_text_encoder(parts, tokenizer, dtype=torch.float32, out_dtype=torch.float32, configs=(L_CFG, G_CFG))
        direct = SDXLTextEncoder(l, g, tokenizer, out_dtype=torch.float32)
        a, b = loaded.encode_text(["cat"])[0], direct.encode_text(["cat"])[0]
        check(torch.allclose(a.ctx, b.ctx, atol=1e-6) and torch.allclose(a.pooled, b.pooled, atol=1e-6),
              "towers loaded from a checkpoint-shaped file match the originals (hf-nested L, openclip G)")
        raises(KeyError, lambda: load_text_encoder({"clip_l": parts["clip_l"]}, tokenizer), "a missing tower is named", "clip_g")
        bad = dict(parts)
        bad["clip_g"] = make_part("sdxl", path, PartInfo(PartKind.TEXT_ENCODER, "clip_g", "conditioner.embedders.1.model.", 0, "F32", 0,
                                                         {"layout": "openclip", "layers": 7}))
        raises(ValueError, lambda: load_text_encoder(bad, tokenizer, configs=(L_CFG, G_CFG)), "layer count mismatch is refused", "layers")


if __name__ == "__main__":
    run(conditioning_ops, encoder_cases, loading_cases)
