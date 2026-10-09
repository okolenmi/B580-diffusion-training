"""Project SDClipModel (+ OpenCLIP key converter) vs HuggingFace CLIPTextModelWithProjection."""
import sys
import os; sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection
from nodes.model.clip import SDClipModel
from nodes.model.clip_state_dict import clip_text_transformers_convert

def rand_init(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() > 1: p.copy_(torch.randn_like(p) / (p.shape[-1]) ** 0.5)
            elif n.endswith("weight"): p.copy_(1 + 0.1 * torch.randn_like(p))
            else: p.copy_(0.1 * torch.randn_like(p))

def make_tokens(pad):
    rows = []
    for L in (5, 20, 76):
        ids = [49406] + torch.randint(1000, 40000, (L - 1,)).tolist() + [49407]
        rows.append(ids + [pad] * (77 - len(ids)))
    return torch.tensor(rows)

for name, act, pad in (("CLIP-L-like (quick_gelu, pad=EOS)", "quick_gelu", 49407), ("CLIP-G-like (gelu, pad=0)", "gelu", 0)):
    torch.manual_seed(3)
    cfg = CLIPTextConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_hidden_layers=4,
                         max_position_embeddings=77, hidden_act=act, projection_dim=64, vocab_size=49408,
                         bos_token_id=49406, eos_token_id=49407, pad_token_id=pad)
    hf = CLIPTextModelWithProjection(cfg).eval(); rand_init(hf)
    mine_cfg = {"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_hidden_layers": 4,
                "max_position_embeddings": 77, "hidden_act": act, "eos_token_id": 49407, "vocab_size": 49408}
    mine = SDClipModel(mine_cfg, layer="hidden", layer_idx=-2, layer_norm_hidden_state=False,
                       special_tokens={"start": 49406, "end": 49407, "pad": pad}, return_projected_pooled=True)
    hsd = hf.state_dict()
    if "gelu" == act:     # build an OpenCLIP-format dict by hand, then use the project's converter
        o = {"p.token_embedding.weight": hsd["text_model.embeddings.token_embedding.weight"],
             "p.positional_embedding": hsd["text_model.embeddings.position_embedding.weight"],
             "p.ln_final.weight": hsd["text_model.final_layer_norm.weight"], "p.ln_final.bias": hsd["text_model.final_layer_norm.bias"],
             "p.text_projection": hsd["text_projection.weight"].t().contiguous()}   # OpenCLIP: x @ proj
        for i in range(4):
            h = f"text_model.encoder.layers.{i}."; r = f"p.transformer.resblocks.{i}."
            for s in ("weight", "bias"):
                o[r + f"attn.in_proj_{s}"] = torch.cat([hsd[h + f"self_attn.{q}_proj.{s}"] for q in "qkv"], 0)
                o[r + f"attn.out_proj.{s}"] = hsd[h + f"self_attn.out_proj.{s}"]
                o[r + f"ln_1.{s}"] = hsd[h + f"layer_norm1.{s}"]; o[r + f"ln_2.{s}"] = hsd[h + f"layer_norm2.{s}"]
                o[r + f"mlp.c_fc.{s}"] = hsd[h + f"mlp.fc1.{s}"]; o[r + f"mlp.c_proj.{s}"] = hsd[h + f"mlp.fc2.{s}"]
        conv = clip_text_transformers_convert({k: v.clone() for k, v in o.items()}, "p.", "transformer.")
        sd = {k: v for k, v in conv.items()}
    else:
        sd = {"transformer." + k: v for k, v in hsd.items()}
    res = mine.load_state_dict(sd, strict=False)
    print(f"[{name}] load: missing={[k for k in res.missing_keys if 'logit_scale' not in k]} unexpected={res.unexpected_keys[:3]}")
    ids = make_tokens(pad)
    with torch.no_grad():
        z, pooled = mine(ids)
        out = hf(ids, output_hidden_states=True)
    zh, ph = out.hidden_states[-2], out.text_embeds
    print(f"   ctx(penultimate) max|diff|={(z - zh).abs().max():.2e} (scale {zh.abs().max():.2f}) | pooled max|diff|={(pooled - ph).abs().max():.2e} (scale {ph.abs().max():.2f})")
