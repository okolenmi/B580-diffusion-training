"""SDXL's latent space and VAE loading."""

from __future__ import annotations

import torch

from ..api.parts import ModelPart
from ..general.tensors import read_prefixed
from ..general.vae.autoencoder import SDXL_DDCONFIG
from ..general.vae.codec import LatentSpec, VAECodec

# Where each channel of the scaled latents leans in colour, from ComfyUI's latent formats; ``(latents @ RGB + BIAS + 1) / 2`` is a blurry picture without a decoder.
RGB = ((0.3651, 0.4232, 0.4341), (-0.2533, -0.0042, 0.1068), (0.1076, 0.1111, -0.0362), (-0.3165, -0.2492, -0.2188))
RGB_BIAS = (0.1084, -0.0175, -0.0011)

LATENTS = LatentSpec(channels=4, spatial_factor=8, scale=0.13025, rgb=RGB, rgb_bias=RGB_BIAS)


def load_vae(part: ModelPart, *, device="cpu", dtype: torch.dtype = torch.float32) -> VAECodec:
    """Build the codec from the ``vae`` part of a Load Model node (strict: a VAE that does not fit is an error)."""
    state = read_prefixed(part.source.path, part.source.prefix, dtype=dtype)
    return VAECodec.kl_from_state_dict(state, SDXL_DDCONFIG, embed_dim=4, spec=LATENTS, device=device, dtype=dtype)
