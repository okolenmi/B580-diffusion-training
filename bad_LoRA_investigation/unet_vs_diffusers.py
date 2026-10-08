"""Independent check: project UNetModel vs diffusers UNet2DConditionModel (scaled-down SDXL shape).

Weights are random, copied across with diffusers' OWN ldm->diffusers converter,
so neither side's key logic is trusted. Both run in fp32 on CPU.
"""
import sys, copy
import os; sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from diffusers import UNet2DConditionModel
from diffusers.loaders import single_file_utils as S
from nodes.model.unet import UNetModel

torch.manual_seed(0)
CFG = dict(image_size=16, in_channels=4, out_channels=4, model_channels=32,
           num_res_blocks=[2, 2, 2], channel_mult=[1, 2, 4], num_head_channels=16,
           use_spatial_transformer=True, transformer_depth=[0, 0, 1, 1, 2, 2],
           transformer_depth_middle=2, transformer_depth_output=[0, 0, 0, 1, 1, 1, 2, 2, 2],
           context_dim=64, use_linear_in_transformer=True, num_classes="sequential",
           adm_in_channels=64, use_checkpoint=False)
ours = UNetModel(**CFG).eval()
# make every parameter non-trivial (zero-init convs would hide bugs)
with torch.no_grad():
    for n, p in ours.named_parameters():
        if p.dim() > 1:
            fan_in = p.numel() // p.shape[0]
            p.copy_(torch.randn_like(p) / fan_in ** 0.5)
        elif n.endswith("weight"):           # norm scales
            p.copy_(1 + 0.1 * torch.randn_like(p))
        else:                                # biases
            p.copy_(0.1 * torch.randn_like(p))

sd = {"model.diffusion_model." + k: v.clone() for k, v in ours.state_dict().items()}
dcfg = dict(
    sample_size=16, in_channels=4, out_channels=4, center_input_sample=False, flip_sin_to_cos=True, freq_shift=0,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    block_out_channels=[32, 64, 128], layers_per_block=2, downsample_padding=1, mid_block_scale_factor=1,
    act_fn="silu", norm_num_groups=32, norm_eps=1e-5, cross_attention_dim=64,
    transformer_layers_per_block=[1, 1, 2], attention_head_dim=[2, 4, 8], use_linear_projection=True,
    addition_embed_type="text_time", addition_time_embed_dim=8, projection_class_embeddings_input_dim=64,
    upcast_attention=False, resnet_time_scale_shift="default", only_cross_attention=False,
)
ref = UNet2DConditionModel(**dcfg).eval()
conv = S.convert_ldm_unet_checkpoint(sd, dcfg)
miss, unexp = ref.load_state_dict(conv, strict=False)
print("diffusers load: missing", len(miss), "unexpected", len(unexp), miss[:3], unexp[:3])

B = 2
x = torch.randn(B, 4, 16, 16); ctx = torch.randn(B, 77, 64)
pooled = torch.randn(B, 16)
time_ids = torch.tensor([[512., 768., 0., 0., 512., 768.], [1024., 1024., 0., 0., 1024., 1024.]])
from diffusers.models.embeddings import Timesteps
ids_emb = Timesteps(8, True, 0)(time_ids.flatten()).reshape(B, -1)
y = torch.cat([pooled, ids_emb], dim=1)

worst = 0
for t in (0, 1, 100, 500, 900, 999):
    ts = torch.full((B,), float(t))
    with torch.no_grad():
        a = ours(x, ts, context=ctx, y=y)
        b = ref(x, ts, encoder_hidden_states=ctx,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
    err = (a - b).abs().max().item(); rel = err / b.abs().max().item(); worst = max(worst, rel)
    print(f"t={t:4d} max|diff|={err:.3e} rel={rel:.3e}")
print("WORST rel", worst)

# ---- non-vacuity checks ----
with torch.no_grad():
    o0 = ours(x, torch.full((B,), 10.), context=ctx, y=y); o1 = ours(x, torch.full((B,), 900.), context=ctx, y=y)
    print("output std", o0.std().item(), "| t-sensitivity |o(10)-o(900)|", (o0 - o1).abs().mean().item())
    # y and context sensitivity
    o2 = ours(x, torch.full((B,), 10.), context=ctx * 0, y=y); o3 = ours(x, torch.full((B,), 10.), context=ctx, y=y * 0)
    print("ctx sensitivity", (o0 - o2).abs().mean().item(), "| y sensitivity", (o0 - o3).abs().mean().item())
    # perturb one cross-attn weight in the reference -> must now differ
    name = [n for n, _ in ref.named_parameters() if "attn2.to_out.0.weight" in n][3]
    dict(ref.named_parameters())[name].add_(0.01)
    b = ref(x, torch.full((B,), 10.), encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
    print("after perturbing", name, "-> diff", (o0 - b).abs().max().item())
