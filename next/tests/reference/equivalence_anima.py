"""Our Anima transformer and adapter against ComfyUI's, same weights, random inputs: the differences must be 0.

    COMFY_REF=<ComfyUI>/comfy python -m next.tests.reference.equivalence_anima
"""
import torch

from . import comfy_stubs as ref_dit
from ...families.anima.dit import AnimaDiT, DiTConfig
torch.manual_seed(0)
ops = ref_dit.Ops
cfg = DiTConfig(model_channels=192, num_blocks=2, num_heads=2, adapter_blocks=1)   # head_dim 96 -> dim_h 32, dim_t 32
ref = ref_dit.anima.Anima(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    concat_padding_mask=True, model_channels=192, num_blocks=2, num_heads=2, crossattn_emb_channels=1024, pos_emb_cls="rope3d",
    pos_emb_learnable=True, use_adaln_lora=True, adaln_lora_dim=256, rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False, operations=ops, dtype=torch.float32, device="cpu")
mine = AnimaDiT(cfg)
# the reference adapter has 6 blocks; make mine match by config
mine = AnimaDiT(DiTConfig(model_channels=192, num_blocks=2, num_heads=2, adapter_blocks=6))
sd = ref.state_dict()
res = mine.load_state_dict(sd, strict=False)
print("missing", res.missing_keys[:5], "unexpected", res.unexpected_keys[:5])
for p in ref.parameters(): p.data.normal_(0, 0.05)
mine.load_state_dict(ref.state_dict(), strict=False)
x = torch.randn(2, 16, 1, 12, 10); t = torch.tensor([0.3, 0.8]); ctx = torch.randn(2, 20, 1024)
a = ref._forward(x, t, ctx)                       # (B,C,1,H,W)
b = mine(x[:, :, 0], t, ctx)
print("dit max abs diff", (a[:, :, 0] - b).abs().max().item(), "scale", a.abs().mean().item())
# adapter
hid = torch.randn(2, 17, 1024); ids = torch.randint(0, 32128, (2, 9))
ra = ref.llm_adapter(hid, ids); mb = mine.llm_adapter(hid, ids)
print("adapter max abs diff", (ra - mb).abs().max().item(), "scale", ra.abs().mean().item())
mine2 = AnimaDiT(DiTConfig(model_channels=192, num_blocks=2, num_heads=2, adapter_blocks=6, rope_extrapolation=(1.0, 1.0, 1.0)))
mine2.load_state_dict(ref.state_dict(), strict=False)
print("control (other rope) diff", (a[:, :, 0] - mine2(x[:, :, 0], t, ctx)).abs().max().item())
x2 = torch.randn(1, 16, 1, 8, 14); a2 = ref._forward(x2, t[:1], ctx[:1]); print("other shape diff", (a2[:, :, 0] - mine(x2[:, :, 0], t[:1], ctx[:1])).abs().max().item())
