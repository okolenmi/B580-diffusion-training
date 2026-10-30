"""SDXL as a rendering backend: a checkpoint, optionally with a saved adapter, whose strength is set per image."""

from __future__ import annotations

import gc

import torch
from torch import Tensor

from ..accept.loading import load_parts
from ..engine.runtime.memory import is_out_of_memory
from ..families.api.adapters import TrainableSelection
from ..families.general.adapters.schemes import read_adapter
from ..families.sdxl.adapters import inject_denoiser
from ..families.sdxl.conditioning import SDXLConditioning
from .tiling import TiledDecoder


class SDXLBackend:
    """Keeps on the device only what the current stage needs: the text encoders, the network, or the decoder."""

    def __init__(self, models: str, checkpoint: str, device: str, lora: str | None = None, decode_dtype: torch.dtype = torch.float32):
        family, parts = load_parts(models, checkpoint)
        self.family, self.device = family, torch.device(device)
        self.denoiser = family.load_denoiser(parts["denoiser"], device="cpu", dtype=torch.bfloat16)
        self.adapted = self._adapt(lora) if lora else None
        self.encoder = family.load_text_encoder(parts, device="cpu", dtype=torch.float16)
        self.vae = family.load_vae(parts["vae"], device="cpu", dtype=decode_dtype)

    def _adapt(self, path: str):
        state, header = read_adapter(path)
        rank = int(header["rank"])
        adapted = inject_denoiser(self.denoiser, TrainableSelection(), rank=rank, alpha=float(header.get("alpha") or rank),
                                  kind=header.get("kind") or "lora", compute="activation")
        scheme = self.family.schemes.get(header.get("scheme") or "kohya")
        report = self.family.load_checked(scheme, adapted, state)
        if report.missing or report.unexpected:
            raise ValueError(f"{path}: {len(report.missing)} adapter layers missing, {len(report.unexpected)} file keys unused")
        return adapted

    def _move(self, **where: str) -> None:
        for name, target in where.items():
            if name == "vae":
                self.vae.model.to(target)
                self.vae.device = torch.device(target)
            else:
                getattr(self, name).to(target)
        gc.collect()
        backend = getattr(torch, self.device.type, None)
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()

    def stage(self, name: str) -> None:
        device, cpu = str(self.device), "cpu"
        if name == "encode":
            self._move(denoiser=cpu, vae=cpu, encoder=device)
        elif name == "sample":
            self._move(encoder=cpu, vae=cpu, denoiser=device)
            self.denoiser.eval()
        elif name == "decode":
            self._move(denoiser=cpu, encoder=cpu, vae=device)
        else:
            raise ValueError(f"unknown stage {name!r}")

    def condition(self, prompt: str, size: tuple[int, int]) -> SDXLConditioning:
        text = self.encoder.encode_text([prompt])[0]
        return self.encoder.assemble(text, size).to(self.device)

    def predict(self, x_in: Tensor, t: Tensor, conditioning: SDXLConditioning, strength: float) -> Tensor:
        x_in, t = x_in.to(self.device), t.to(self.device)
        if self.adapted is None:
            return self.denoiser.predict(x_in, t, conditioning)
        with self.adapted.gate.override(torch.full((x_in.shape[0],), float(strength), device=self.device)):
            return self.denoiser.predict(x_in, t, conditioning)

    def encode_image(self, pixels: Tensor) -> Tensor:
        """(B, 3, H, W) uint8 -> scaled latents (the autoencoder's stage)."""
        return self.vae.encode(pixels)

    def decode(self, latents: Tensor) -> Tensor:
        try:
            return self.vae.decode(latents)
        except Exception as error:
            if not is_out_of_memory(error):
                raise
            gc.collect()
            return TiledDecoder(self.vae.decode)(latents)
