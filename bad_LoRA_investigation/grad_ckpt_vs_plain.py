"""Gradients of LoRA params: project's custom checkpointing vs plain autograd."""
import sys
import os; sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from nodes.model.unet import UNetModel
from nodes.model.lora import LoRAConfig, inject_lora_into_unet
from nodes.model.gradient_checkpointing import FrozenParamSafeCheckpointing, NoCheckpointing

CFG = dict(image_size=16, in_channels=4, out_channels=4, model_channels=32,
           num_res_blocks=[2, 2, 2], channel_mult=[1, 2, 4], num_head_channels=16,
           use_spatial_transformer=True, transformer_depth=[0, 0, 1, 1, 2, 2],
           transformer_depth_middle=2, transformer_depth_output=[0, 0, 0, 1, 1, 1, 2, 2, 2],
           context_dim=64, use_linear_in_transformer=True, num_classes="sequential", adm_in_channels=64)

def build(use_ckpt, strategy):
    torch.manual_seed(5)
    strategy.apply()
    m = UNetModel(**CFG, use_checkpoint=use_ckpt).train()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() > 1: p.copy_(torch.randn_like(p) / (p.numel() // p.shape[0]) ** 0.5)
            elif n.endswith("weight"): p.copy_(1 + 0.1 * torch.randn_like(p))
            else: p.copy_(0.1 * torch.randn_like(p))
    reg = inject_lora_into_unet(m, LoRAConfig(rank=4, alpha=2.0))
    for p in m.parameters(): p.requires_grad_(False)
    torch.manual_seed(9)
    with torch.no_grad():
        for _, _, _, l in reg:
            l.lora_A.copy_(torch.randn_like(l.lora_A) * 0.2); l.lora_B.copy_(torch.randn_like(l.lora_B) * 0.2)
            l.lora_A.requires_grad_(True); l.lora_B.requires_grad_(True)
    return m, reg

def grads(m, reg):
    torch.manual_seed(11)
    B = 2; x = torch.randn(B, 4, 16, 16); ctx = torch.randn(B, 77, 64); y = torch.randn(B, 64)
    tgt = torch.randn(B, 4, 16, 16); t = torch.tensor([30., 700.])
    loss = ((m(x, t, context=ctx, y=y) - tgt) ** 2).mean(); loss.backward()
    return loss.item(), {n: (l.lora_A.grad.clone() if l.lora_A.grad is not None else None,
                             l.lora_B.grad.clone() if l.lora_B.grad is not None else None) for n, _, _, l in reg}

mp, rp = build(False, NoCheckpointing());           lp, gp = grads(mp, rp)
mc, rc = build(True, FrozenParamSafeCheckpointing()); lc, gc = grads(mc, rc)
print("loss plain", lp, "ckpt", lc)
worst, none_plain, none_ckpt = 0.0, [], []
for k in gp:
    for i, nm in enumerate("AB"):
        a, b = gp[k][i], gc[k][i]
        if a is None: none_plain.append(k + nm)
        if b is None: none_ckpt.append(k + nm)
        if a is not None and b is not None:
            worst = max(worst, ((a - b).abs().max() / (a.abs().max() + 1e-12)).item())
print("modules:", len(gp), "| missing grad plain:", len(none_plain), "ckpt:", len(none_ckpt), "| worst rel grad diff:", worst)
print("cond-path grads (plain, ckpt) |gradB| :", {k: (float(gp[k][1].norm()), float(gc[k][1].norm())) for k in gp if "time_embed" in k or "label_emb" in k})
