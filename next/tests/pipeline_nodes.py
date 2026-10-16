"""Test-only pieces for running a training graph on the CPU: a source node that fabricates model parts and a tiny family.

The family keeps SDXL's real objective, adapter layout and file schemes and swaps only what needs real weights
(UNet, text encoders, VAE) for tiny stand-ins.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from ..families.api.parts import ClipGPart, ClipLPart, DenoiserPart, PartInfo, PartKind, PartSource, VAEPart
from ..families.general.ldm_unet.unet import UNetModel
from ..families.sdxl.denoiser import SDXLDenoiser
from ..families.sdxl.family import SDXLFamily
from ..graph import Node, Port
from .check_data import FakeEncoder
from .check_engine import FakeKit
from .check_sdxl import TINY_UNET

VAE_PREFIX = "first_stage_model."
FILE_ENV = "pipeline_nodes_file"


class TinyCodec(FakeEncoder):
    def release(self) -> None:
        pass


class TinyText:
    """Stands in for the loaded text encoders; remembers where it was moved."""

    def __init__(self):
        self.device = "cpu"

    def modules(self):
        return ()

    def to(self, device):
        self.device = str(device)
        return self


class TinyFamily(SDXLFamily):
    @staticmethod
    def load_denoiser(part, *, device="cpu", dtype=torch.float32, config=None):
        torch.manual_seed(0)
        return SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)

    @staticmethod
    def load_text_encoder(parts, *args, **kwargs):
        return TinyText()

    @staticmethod
    def conditioning_kit(encoder):
        return FakeKit()

    @staticmethod
    def load_vae(part, *, device="cpu", dtype=torch.float32):
        return TinyCodec()


def _part(cls, role: str, path: str, prefix: str, kind: PartKind):
    return cls("sdxl", role, PartSource(path, prefix), PartInfo(kind, role, prefix, 1, "float32", 4))


class TinyModelSource(Node):
    """Stands in for Load Model: hands out the four parts of an SDXL file that holds one tiny tensor under the VAE prefix."""

    CATEGORY = "Test"
    INPUTS = {"file": Port("file", str)}
    OUTPUTS = {"denoiser": Port("denoiser", DenoiserPart), "clip_l": Port("clip_l", ClipLPart),
               "clip_g": Port("clip_g", ClipGPart), "vae": Port("vae", VAEPart)}

    def build(self, **inputs):
        path = inputs["file"]
        if not Path(path).exists():
            save_file({VAE_PREFIX + "encoder.weight": torch.arange(8.0)}, path)
        return {"denoiser": _part(DenoiserPart, "denoiser", path, "model.diffusion_model.", PartKind.DENOISER),
                "clip_l": _part(ClipLPart, "clip_l", path, "cond_stage_model.", PartKind.TEXT_ENCODER),
                "clip_g": _part(ClipGPart, "clip_g", path, "conditioner.", PartKind.TEXT_ENCODER),
                "vae": _part(VAEPart, "vae", path, VAE_PREFIX, PartKind.VAE)}
