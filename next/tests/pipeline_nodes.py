"""Test-only pieces for running a training graph on the CPU: a source node that fabricates model parts and a tiny family.

The family keeps SDXL's real objective, adapter layout and file schemes and swaps only what needs real weights
(UNet, text encoders, VAE) for tiny stand-ins.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from ..families.anima.conditioning import AnimaTextEncoder
from ..families.anima.denoiser import AnimaDenoiser
from ..families.anima.dit import AnimaDiT, DiTConfig
from ..families.anima.family import AnimaFamily
from ..families.anima.latents import LATENTS as ANIMA_LATENTS
from ..families.anima.text import QwenConfig, QwenEncoder
from ..families.api.parts import ClipGPart, ClipLPart, DenoiserPart, PartInfo, PartKind, PartSource, Qwen3Part, VAEPart
from ..families.general.vae.codec import VAECodec
from ..families.general.vae.wan import WanImageVAE
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


class _Words:
    """A tokenizer stand-in: one Qwen id per character, T5 ids from the character codes, then ``</s>``."""

    def tokenize(self, text):
        return [10 + ord(c) % 50 for c in text] or [0], [2 + ord(c) % 90 for c in text] + [1]


class TinyAnimaFamily(AnimaFamily):
    """The real Anima objective, kit, layout and file schemes; tiny networks in place of the three files."""

    @staticmethod
    def load_denoiser(part, *, device="cpu", dtype=torch.float32, config=None):
        torch.manual_seed(0)
        return AnimaDenoiser(AnimaDiT(DiTConfig(model_channels=96, num_blocks=2, num_heads=2, adapter_blocks=1)), dtype=torch.float32)

    @staticmethod
    def load_text_encoder(parts, *args, **kwargs):
        torch.manual_seed(0)
        cfg = QwenConfig(vocab=100, width=1024, layers=1, heads=2, kv_heads=1, head_dim=64, mlp_width=64)
        return AnimaTextEncoder(QwenEncoder(cfg), _Words(), out_dtype=torch.float32, tokens=32)

    @staticmethod
    def load_vae(part, *, device="cpu", dtype=torch.float32):
        torch.manual_seed(0)
        return VAECodec(WanImageVAE(dim=8, mult=(1, 2, 2, 2), blocks=1), ANIMA_LATENTS, "cpu", torch.float32)


class TinyAnimaSource(Node):
    """Stands in for three Load Model nodes: the DiT, Qwen3 and VAE parts of one tiny file."""

    CATEGORY = "Test"
    INPUTS = {"file": Port("file", str)}
    OUTPUTS = {"denoiser": Port("denoiser", DenoiserPart), "qwen3": Port("qwen3", Qwen3Part), "vae": Port("vae", VAEPart)}

    def build(self, **inputs):
        path = inputs["file"]
        if not Path(path).exists():
            save_file({VAE_PREFIX + "encoder.weight": torch.arange(8.0)}, path)
        make = lambda cls, role, prefix, kind: cls("anima", role, PartSource(path, prefix), PartInfo(kind, role, prefix, 1, "float32", 4))  # noqa: E731
        return {"denoiser": make(DenoiserPart, "denoiser", "net.", PartKind.DENOISER),
                "qwen3": make(Qwen3Part, "qwen3", "model.", PartKind.TEXT_ENCODER),
                "vae": make(VAEPart, "vae", VAE_PREFIX, PartKind.VAE)}
