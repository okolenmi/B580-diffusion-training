"""The SDXL family: one object that wires its areas together for the nodes."""

from __future__ import annotations

from ..general.adapters import merge as merge_io
from ..general.adapters import schemes as scheme_io
from ..general.quantize import int8 as int8_io
from ..general.quantize import nf4 as nf4_io
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
    save_adapter = staticmethod(scheme_io.save_adapter)
    read_adapter = staticmethod(scheme_io.read_adapter)
    load_checked = staticmethod(scheme_io.load_checked)
    merge_adapter = staticmethod(merge_io.merge_adapter)
    quantize_frozen = staticmethod(int8_io.quantize_frozen)
    quantize_frozen_nf4 = staticmethod(nf4_io.quantize_frozen)


FAMILY = SDXLFamily()
