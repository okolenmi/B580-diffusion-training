"""Project AutoencoderKL (encode mode + decode) vs diffusers AutoencoderKL, scaled-down SDXL VAE."""
import sys
import os; sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from diffusers import AutoencoderKL
from diffusers.loaders import single_file_utils as S
from nodes.model.vae import AutoencoderKL as MyVAE

torch.manual_seed(0)
dd = dict(double_z=True, z_channels=4, resolution=64, in_channels=3, out_ch=3, ch=32, ch_mult=[1, 2, 4, 4],
          num_res_blocks=2, attn_resolutions=[], dropout=0.0)
mine = MyVAE(4, dd).eval()
with torch.no_grad():
    for n, p in mine.named_parameters():
        if p.dim() > 1: p.copy_(torch.randn_like(p) / (p.numel() // p.shape[0]) ** 0.5)
        elif n.endswith("weight"): p.copy_(1 + 0.1 * torch.randn_like(p))
        else: p.copy_(0.1 * torch.randn_like(p))
sd = {"first_stage_model." + k: v.clone() for k, v in mine.state_dict().items()}
vcfg = dict(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4, block_out_channels=[32, 64, 128, 128], layers_per_block=2,
            act_fn="silu", latent_channels=4, norm_num_groups=32, sample_size=64, scaling_factor=0.13025,
            force_upcast=False)
ref = AutoencoderKL(**vcfg).eval()
conv = S.convert_ldm_vae_checkpoint(sd, vcfg)
print("load:", ref.load_state_dict(conv, strict=False))
x = torch.randn(2, 3, 64, 64).clamp(-1, 1)
with torch.no_grad():
    a = mine.encode(x)                              # project: posterior mode
    b = ref.encode(x).latent_dist.mode()
    print("encode: shape", tuple(a.shape), "max|diff|", (a - b).abs().max().item(), "scale", b.abs().max().item())
    da = mine.decode(a); db = ref.decode(b).sample
    print("decode: max|diff|", (da - db).abs().max().item(), "scale", db.abs().max().item())
