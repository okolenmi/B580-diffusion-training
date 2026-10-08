"""LoRA forward + export vs an INDEPENDENT merge (ComfyUI key rule) into a diffusers model."""
import sys, copy
import os; sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from diffusers import UNet2DConditionModel
from diffusers.loaders import single_file_utils as S
from diffusers.models.embeddings import Timesteps
from nodes.model.unet import UNetModel
from nodes.model.lora import LoRAConfig, inject_lora_into_unet, extract_lora_weights

CFG = dict(image_size=16, in_channels=4, out_channels=4, model_channels=32, num_res_blocks=[2, 2, 2], channel_mult=[1, 2, 4],
           num_head_channels=16, use_spatial_transformer=True, transformer_depth=[0, 0, 1, 1, 2, 2], transformer_depth_middle=2,
           transformer_depth_output=[0, 0, 0, 1, 1, 1, 2, 2, 2], context_dim=64, use_linear_in_transformer=True,
           num_classes="sequential", adm_in_channels=64, use_checkpoint=False)
torch.manual_seed(1)
ours = UNetModel(**CFG).eval()
with torch.no_grad():
    for n, p in ours.named_parameters():
        if p.dim() > 1: p.copy_(torch.randn_like(p) / (p.numel() // p.shape[0]) ** 0.5)
        elif n.endswith("weight"): p.copy_(1 + 0.1 * torch.randn_like(p))
        else: p.copy_(0.1 * torch.randn_like(p))
base_sd = {"model.diffusion_model." + k: v.clone() for k, v in ours.state_dict().items()}

RANK, ALPHA = 4, 2.0
reg = inject_lora_into_unet(ours, LoRAConfig(rank=RANK, alpha=ALPHA))
print("injected modules:", len(reg), "| cond-path:", sum(("time_embed" in r[0] or "label_emb" in r[0]) for r in reg))
with torch.no_grad():
    for _, _, _, layer in reg:
        layer.lora_B.copy_(torch.randn_like(layer.lora_B) * 0.3)      # make LoRA strong
        layer.lora_A.copy_(torch.randn_like(layer.lora_A) * 0.3)
exported = extract_lora_weights(reg)

# ---- independent merge: ComfyUI's rule  weight += (alpha/rank) * up @ down
lookup = {}
for k in base_sd:
    if k.startswith("model.diffusion_model.") and k.endswith(".weight"):
        lookup["lora_unet_" + k[len("model.diffusion_model."):-len(".weight")].replace(".", "_")] = k
merged = {k: v.clone() for k, v in base_sd.items()}
n_ok = 0
for key in [k for k in exported if k.endswith(".lora_down.weight")]:
    mk = key[:-len(".lora_down.weight")]
    assert mk in lookup, f"exported LoRA key has no base weight under ComfyUI's mapping: {mk}"
    down, up, alpha = exported[key], exported[mk + ".lora_up.weight"], float(exported[mk + ".alpha"][0])
    assert up.shape[0] == merged[lookup[mk]].shape[0] and down.shape[1] == merged[lookup[mk]].shape[1], mk
    merged[lookup[mk]] += (alpha / down.shape[0]) * (up @ down); n_ok += 1
print("merged modules:", n_ok, "of", len(reg))

dcfg = dict(sample_size=16, in_channels=4, out_channels=4, center_input_sample=False, flip_sin_to_cos=True, freq_shift=0,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    block_out_channels=[32, 64, 128], layers_per_block=2, downsample_padding=1, mid_block_scale_factor=1,
    act_fn="silu", norm_num_groups=32, norm_eps=1e-5, cross_attention_dim=64,
    transformer_layers_per_block=[1, 1, 2], attention_head_dim=[2, 4, 8], use_linear_projection=True,
    addition_embed_type="text_time", addition_time_embed_dim=8, projection_class_embeddings_input_dim=64,
    upcast_attention=False, resnet_time_scale_shift="default", only_cross_attention=False)
ref = UNet2DConditionModel(**dcfg).eval()
ref.load_state_dict(S.convert_ldm_unet_checkpoint(merged, dcfg))
ref0 = UNet2DConditionModel(**dcfg).eval(); ref0.load_state_dict(S.convert_ldm_unet_checkpoint(base_sd, dcfg))

B = 2; x = torch.randn(B, 4, 16, 16); ctx = torch.randn(B, 77, 64); pooled = torch.randn(B, 16)
tid = torch.tensor([[512., 768., 0., 0., 512., 768.], [1024., 1024., 0., 0., 1024., 1024.]])
y = torch.cat([pooled, Timesteps(8, True, 0)(tid.flatten()).reshape(B, -1)], 1)
for t in (0, 100, 500, 999):
    ts = torch.full((B,), float(t))
    with torch.no_grad():
        live = ours(x, ts, context=ctx, y=y)
        mrg = ref(x, ts, encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}).sample
        base = ref0(x, ts, encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}).sample
    print(f"t={t:4d} |live-merged|max={(live-mrg).abs().max():.3e}  (LoRA effect size |live-base|max={(live-base).abs().max():.3e})")
