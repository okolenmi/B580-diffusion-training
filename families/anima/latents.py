"""Anima's latent space (the Wan 2.1 / Qwen-Image VAE) and VAE loading."""

from __future__ import annotations

import torch

from ..api.parts import ModelPart
from ..general.tensors import read_prefixed
from ..general.vae.codec import LatentSpec, VAECodec
from ..general.vae.wan import WanImageVAE, to_image_state
from .dit import DiTConfig

# Per-channel statistics of the Wan 2.1 latent space; the network is trained on (latent - mean) / std.
MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160)

# The same kind of linear colour guess as SDXL's, for the normalised Wan latents (ComfyUI's Wan21 latent format).
RGB = ((-0.1299, -0.1692, 0.2932), (0.0671, 0.0406, 0.0442), (0.3568, 0.2548, 0.1747), (0.0372, 0.2344, 0.1420),
       (0.0313, 0.0189, -0.0328), (0.0296, -0.0956, -0.0665), (-0.3477, -0.4059, -0.2925), (0.0166, 0.1902, 0.1975),
       (-0.0412, 0.0267, -0.1364), (-0.1293, 0.0740, 0.1636), (0.0680, 0.3019, 0.1128), (0.0032, 0.0581, 0.0639),
       (-0.1251, 0.0927, 0.1699), (0.0060, -0.0633, 0.0005), (0.3477, 0.2275, 0.2950), (0.1984, 0.0913, 0.1861))
RGB_BIAS = (-0.1835, -0.0868, -0.3360)

LATENTS = LatentSpec(channels=16, spatial_factor=8, scale=1.0, mean=MEAN, std=STD, grid=DiTConfig.patch, rgb=RGB, rgb_bias=RGB_BIAS)


def load_vae(part: ModelPart, *, device="cpu", dtype: torch.dtype = torch.float32) -> VAECodec:
    """Build the codec from the ``vae`` part of a Load Model node (strict: a VAE that does not fit is an error)."""
    state = to_image_state(read_prefixed(part.source.path, part.source.prefix, dtype=dtype))
    if "encoder.conv1.weight" not in state:
        raise ValueError("the VAE file has no encoder.conv1.weight; Anima needs the Wan 2.1 / Qwen-Image autoencoder")
    with torch.device("meta"):
        model = WanImageVAE(dim=state["encoder.conv1.weight"].shape[0], z_dim=state["conv2.weight"].shape[0])
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"VAE weights do not fit: {len(result.missing_keys)} missing (e.g. {result.missing_keys[:2]}), "
                         f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[:2]})")
    return VAECodec(model, LATENTS, device, dtype)
