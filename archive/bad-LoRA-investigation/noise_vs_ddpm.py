"""Project noising + UNet-input scaling vs diffusers DDPMScheduler.add_noise (scaled_linear, SDXL betas)."""
import os, sys
sys.path.insert(0, os.environ.get("REPO", "."))
import torch
from diffusers import DDPMScheduler
from nodes.components.diffusion import DiscreteLinearNoiseSchedule, KarrasInputScaler
sch = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
mine, scal = DiscreteLinearNoiseSchedule(), KarrasInputScaler()
torch.manual_seed(0); x0 = torch.randn(8, 4, 16, 16); eps = torch.randn_like(x0)
worst = 0
for t in (0, 1, 50, 200, 500, 800, 999):
    tt = torch.full((8,), t, dtype=torch.long); ref = sch.add_noise(x0, eps, tt)
    a, s = mine.alpha_sigma(tt)
    xin = scal.scale_input(x0 + s.view(-1, 1, 1, 1) * eps, s.view(-1, 1, 1, 1)).float()
    worst = max(worst, ((xin - ref).abs().max() / ref.abs().max()).item())
print("worst relative error (bf16-limited, expect ~3e-3):", worst)
