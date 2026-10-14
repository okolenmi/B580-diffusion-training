"""The SDXL family: one object that wires its areas together for the nodes."""

from __future__ import annotations

from . import adapters, conditioning, denoiser, latents, objective

FAMILY_ID = "sdxl"
LABEL = "SDXL"


class SDXLFamily:
    id = FAMILY_ID
    label = LABEL
    layout = adapters.LAYOUT
    schemes = adapters.SCHEMES
    latent_spec = latents.LATENTS
    objective = objective.SDXLObjective
    load_text_encoder = staticmethod(conditioning.load_text_encoder)
    conditioning_kit = conditioning.SDXLConditioningKit
    load_denoiser = staticmethod(denoiser.load_denoiser)
    load_vae = staticmethod(latents.load_vae)
    plan_from_header = staticmethod(adapters.plan_from_header)
    inject_denoiser = staticmethod(adapters.inject_denoiser)


FAMILY = SDXLFamily()
