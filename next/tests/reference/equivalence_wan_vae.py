"""Our single-image Wan VAE against ComfyUI's real 3-D causal one: same random weights, one-frame videos.

    COMFY_REF=<ComfyUI>/comfy python -m next.tests.reference.equivalence_wan_vae

ComfyUI's ``Conv3d`` is stubbed as a true 3-D convolution that zero-pads two frames before the single frame (its fast path
skips the padding by truncating the kernel; here the padding is really done, so the slicing in ``to_image_state`` is tested).
"""
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REF = os.environ["COMFY_REF"]


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class Conv3d(nn.Conv3d):
    def forward(self, x, autopad=None):
        if autopad == "causal_zero":
            x = F.pad(x, (0, 0, 0, 0, self.kernel_size[0] - 1, 0))
        return super().forward(x)


def vae_attention():
    def attend(q, k, v):
        b, c, h, w = q.shape
        q, k, v = (t.reshape(b, 1, c, h * w).transpose(2, 3) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(b, c, h, w)
    return attend


mod("comfy")
mod("comfy.ops", disable_weight_init=types.SimpleNamespace(Conv2d=nn.Conv2d, Conv3d=Conv3d))
mod("comfy.ldm"); mod("comfy.ldm.modules"); mod("comfy.ldm.modules.diffusionmodules")
mod("comfy.ldm.modules.diffusionmodules.model", vae_attention=vae_attention,
    torch_cat_if_needed=lambda xs, dim: torch.cat([x for x in xs if x is not None], dim=dim))
sys.modules["comfy"].ops = sys.modules["comfy.ops"]
spec = importlib.util.spec_from_file_location("wan_vae_ref", REF + "/ldm/wan/vae.py")
ref_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_module)

from ...families.general.vae.wan import WanImageVAE, to_image_state  # noqa: E402

torch.manual_seed(0)
ref = ref_module.WanVAE(dim=24, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                        temperal_downsample=[False, True, True]).eval()
for p in ref.parameters():
    p.data.normal_(0, 0.1)
for name, p in ref.named_parameters():
    if name.endswith("gamma"):
        p.data.fill_(1.0).add_(torch.randn_like(p) * 0.1)
mine = WanImageVAE(dim=24).eval()
print(mine.load_state_dict(to_image_state(ref.state_dict())))
image = torch.rand(2, 3, 64, 48) * 2 - 1
with torch.no_grad():
    a = ref.encode(image[:, :, None])[:, :, 0]
    b = mine.encode(image)
    print("encode max abs diff", (a - b).abs().max().item(), "scale", a.abs().mean().item())
    z = torch.randn(2, 16, 8, 6)
    a = ref.decode(z[:, :, None])[:, :, 0]
    b = mine.decode(z)
    print("decode max abs diff", (a - b).abs().max().item(), "scale", a.abs().mean().item())
