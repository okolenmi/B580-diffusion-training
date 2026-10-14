"""Torch-based self-checks for next/families/general and the family kits (CPU, tiny random models).

    python -m next.tests.check_families_torch
"""

from __future__ import annotations

import random

import torch

from ..families.general.schedules import flow, timesteps
from ..families.general.schedules.ddpm import (DiffusionProcess, DiscreteSchedule, EpsParameterization,
                                               VPredParameterization, ZeroTerminalSNRSchedule, scale_input)
from ..families.general.vae.autoencoder import SDXL_DDCONFIG, AutoencoderKL
from ..families.general.vae.codec import LatentSpec, VAECodec
from .support import check, raises, run


def schedules():
    s = DiscreteSchedule()
    alpha, sigma = s.alpha_sigma(torch.tensor([0, 999]))
    check(abs(sigma[1].item() - 14.6146) < 1e-3 and abs(alpha[0].item() - 0.9996) < 1e-3, "SDXL sigma_max is 14.6146")
    check(s.alpha_sigma(5)[0].ndim == 0, "int timestep gives scalars")
    zt = ZeroTerminalSNRSchedule()
    a_last, s_last = zt.alpha_sigma(999)
    check(a_last.item() == 0.0 and torch.isinf(s_last), "zero terminal SNR ends at pure noise")
    raises(ValueError, lambda: DiffusionProcess(zt, EpsParameterization()), "eps with zero-terminal-SNR is refused", "v-prediction")
    x0, noise, t = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8), torch.tensor([10, 500, 990])
    for param in (EpsParameterization(), VPredParameterization()):
        xt, target, sig = DiffusionProcess(s, param).noisy(x0, noise, t)
        check((param.to_x0(target, xt, sig) - x0).abs().max() < 1e-4, f"{param.name}: target inverts to x0")
        check((param.to_eps(target, xt, sig) - noise).abs().max() < 1e-4, f"{param.name}: target inverts to eps")
    scaled = scale_input(xt, sig.flatten())
    check(scaled.shape == xt.shape and scale_input(xt, 2.0, torch.bfloat16).dtype == torch.bfloat16, "input scaling shapes and dtype")


def flow_and_timesteps():
    x0, noise = torch.randn(2, 16, 1, 4, 4), torch.randn(2, 16, 1, 4, 4)
    s = torch.tensor([0.25, 0.75])
    xs, v = flow.noisy(x0, noise, s)
    check((flow.to_x0(v, xs, s) - x0).abs().max() < 1e-5 and torch.equal(v, noise - x0), "flow velocity inverts to x0 (5-D latents)")
    g = torch.Generator().manual_seed(0)
    draws = flow.sample_sigma(2000, "sigmoid", shift=3.0, generator=g)
    check(0 < draws.min() and draws.max() < 1 and draws.mean() > 0.5, "shifted sigmoid sampling favours noise")
    check(all(0 < flow.sample_sigma(50, m).min() for m in flow.SAMPLERS), "every flow sampler stays in (0,1)")
    check(flow.shift_sigma(torch.tensor(0.5), 1.0).item() == 0.5, "shift 1 is the identity")
    check(all(torch.isfinite(flow.loss_weight(torch.tensor([0.1, 0.9]), w)).all() for w in flow.WEIGHTINGS), "flow weightings are finite")
    raises(ValueError, lambda: flow.sample_sigma(2, "nope"), "unknown flow sampler", "nope")
    rng = random.Random(0)
    check(all(0 <= timesteps.sample_timestep(rng, m, 0, 999) <= 999 for m in timesteps.MODES for _ in range(50)), "scalar timesteps stay in range")
    batch = timesteps.sample_timesteps(500, "low", 0, 999, torch.Generator().manual_seed(1))
    check(batch.dtype == torch.long and batch.float().mean() < 400, "low mode is biased toward small t")
    raises(ValueError, lambda: timesteps.sample_timestep(rng, "adaptive", 0, 9), "train-time modes are not static modes", "adaptive")


def vae():
    cfg = {**SDXL_DDCONFIG, "ch": 32, "ch_mult": (1, 2, 4), "num_res_blocks": 1}
    model = AutoencoderKL(4, cfg)
    spec = LatentSpec(4, 4, 0.13025)
    check(spec.noise_shape(64, 96, 2) == (2, 4, 16, 24), "latent spec gives the noise shape")
    codec = VAECodec.kl_from_state_dict(model.state_dict(), cfg, 4, spec)
    images = torch.rand(2, 3, 32, 32)
    z = codec.encode(images)
    check(z.shape == (2, 4, 8, 8), f"encode downsamples by the family's factor ({tuple(z.shape)})")
    out = codec.decode(z)
    check(out.dtype == torch.uint8 and out.shape == (2, 3, 32, 32), "decode returns uint8 images")
    check(torch.equal(codec.encode((images * 255).to(torch.uint8)).round(decimals=2), codec.encode((images * 255).to(torch.uint8)).round(decimals=2)),
          "uint8 input is accepted")
    check(not any(p.requires_grad for p in codec.model.parameters()), "the codec freezes its model")
    broken = dict(model.state_dict())
    broken.pop("quant_conv.weight")
    raises(ValueError, lambda: VAECodec.kl_from_state_dict(broken, cfg, 4, spec), "weights that do not fit are refused", "missing")


if __name__ == "__main__":
    run(schedules, flow_and_timesteps, vae)
