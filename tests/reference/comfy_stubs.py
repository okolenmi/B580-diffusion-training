"""Loads ComfyUI's Anima code (predict2.py, anima/model.py) with stubs for everything but the maths, so ours can be compared.

Not part of the test run: it needs a ComfyUI checkout (``COMFY_REF``, the folder holding ``ldm/``), ``einops``, and is used by
``equivalence_anima.py``. The one stubbed op that carries maths is ``rms_rope_split_half`` (RMS norm, then rotary rotation of the
pairs ``(x[i], x[i + D/2])``), written here from the rotation matrices ComfyUI builds.
"""
import sys, types, torch, torch.nn as nn, torch.nn.functional as F
import os
REF = os.environ["COMFY_REF"]

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

class Ops:
    Linear = nn.Linear; LayerNorm = nn.LayerNorm; RMSNorm = nn.RMSNorm
    class Embedding(nn.Embedding):
        def forward(self, x, out_dtype=None):
            return super().forward(x)
mod("torchvision"); mod("torchvision.transforms", functional=None, InterpolationMode=None); sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
mod("comfy"); mod("comfy.ldm"); mod("comfy.ldm.modules")
mod("comfy.patcher_extension")
class AttentionTensorContainer:
    def __init__(self, t): self.t = t
class ComfyAttention(nn.Module): pass
mod("comfy.ldm.modules.attention", AttentionTensorContainer=AttentionTensorContainer, ComfyAttention=ComfyAttention, optimized_attention=None)
mod("comfy.ldm.common_dit", pad_to_patch_size=lambda x, p: x)
mod("comfy.ops", cast_bias_weight=lambda m, t, offloadable=True: (m.weight, None, None), uncast_bias_weight=lambda *a: None)

def rms_rope_split_half(q, k, rope, qs, ks, eps):
    # q,k: (B, L, H, D); rope: (1, L, 1, 1, D/2, 2, 2) rotation matrices over split-half pairs
    def one(t, scale):
        t = t.float(); t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * scale.float()
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)   # (...,D/2,1,2)
        r = rope.reshape(rope.shape[1], 1, rope.shape[3], 2, 2)[None]         # (1,L,1,D/2,2,2)
        out = (r[..., 0] * t_[..., 0] + r[..., 1] * t_[..., 1])               # (...,D/2,2)
        return out.movedim(-1, -2).reshape(t.shape)
    return one(q, qs), one(k, ks)
mod("comfy.quant_ops", ck=types.SimpleNamespace(rms_rope_split_half=rms_rope_split_half))

import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=None)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
mod("comfy.ldm.cosmos")
load("comfy.ldm.cosmos.position_embedding", REF + "/ldm/cosmos/position_embedding.py")
pre = load("comfy.ldm.cosmos.predict2", REF + "/ldm/cosmos/predict2.py")
# attention op: plain sdpa on [B,S,H,D]
def sdpa(q, k, v, transformer_options=None, preferred_attention=None):
    q, k, v = (x.t.transpose(1, 2) if hasattr(x, "t") else x.transpose(1, 2) for x in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
pre.torch_attention_op = sdpa
pre.Attention.__init__.__globals__["torch_attention_op"] = sdpa
anima = load("comfy.ldm.anima.model", REF + "/ldm/anima/model.py")

sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]; sys.modules["comfy.ldm"].common_dit = sys.modules["comfy.ldm.common_dit"]

for n in ("ops","quant_ops","patcher_extension"):
    setattr(sys.modules["comfy"], n, sys.modules["comfy."+n])
