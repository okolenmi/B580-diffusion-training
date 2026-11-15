"""SDXL's latent space and VAE loading."""

from __future__ import annotations

import torch

from ..api.parts import ModelPart
from ..general.tensors import read_prefixed
from ..general.vae.autoencoder import SDXL_DDCONFIG
from ..general.vae.codec import LatentSpec, VAECodec

LATENTS = LatentSpec(channels=4, spatial_factor=8, scale=0.13025)


def load_vae(part: ModelPart, *, device="cpu", dtype: torch.dtype = torch.float32) -> VAECodec:
    """Build the codec from the ``vae`` part of a Load Model node (strict: a VAE that does not fit is an error)."""
    state = read_prefixed(part.source.path, part.source.prefix, dtype=dtype)
    return VAECodec.kl_from_state_dict(state, SDXL_DDCONFIG, embed_dim=4, spec=LATENTS, device=device, dtype=dtype)
