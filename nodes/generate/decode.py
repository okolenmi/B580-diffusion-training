"""Decode Image: latents to pixels with the VAE, in tiles when the whole image does not fit."""

from __future__ import annotations

import gc
from typing import Any

import torch

from engine.devices import Accelerator, part_dtype, resolve_device
from engine.generate.tiling import TiledDecoder
from engine.handles import Images, Latents
from engine.runtime.memory import is_out_of_memory
from families.api.parts import VAEPart
from ..family_lookup import family_of
from graph import Node, Port
from .shelf_access import shelf_of


class DecodeImageNode(Node):
    DISPLAY_NAME = "Decode Image"
    CATEGORY = "Generation"
    READS_SETTINGS = ("device.default", "precision.vae")
    INPUTS = {
        "vae": Port("vae", VAEPart),
        "latents": Port("latents", Latents),
    }
    OUTPUTS = {"images": Port("images", Images)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: VAEPart = inputs["vae"]
        latents: Latents = inputs["latents"]
        family = family_of(self.context, part.family_id)
        device = resolve_device(self.context.settings.get("device.default"))
        with self.context.task("Loading the VAE"):
            vae = family.load_vae(part, device=device, dtype=part_dtype(part.dtype, self.context.settings.get("precision.vae"), torch.float32))
        shelf = shelf_of(self.context, device)
        pixels = []
        try:
            with torch.no_grad(), self.context.task("Decoding", total=len(latents.recipes)) as task:
                for index in range(len(latents.recipes)):
                    one = latents.tensor[index:index + 1]
                    image = self._decode(vae, one, shelf)
                    pixels.append(image[0].cpu())
                    task.update(index + 1)
        finally:
            del vae
            gc.collect()
            Accelerator(device).empty_cache()
        return {"images": Images(torch.stack(pixels), latents.recipes, latents.meta)}

    @staticmethod
    def _decode(vae: Any, one: torch.Tensor, shelf: Any) -> torch.Tensor:
        """Whole image first; when it runs out of memory the kept denoisers go to the host and it is tried once more, and only
        then done in tiles."""
        try:
            return vae.decode(one)
        except Exception as error:
            if not is_out_of_memory(error):
                raise
        gc.collect()
        if shelf.yield_room():
            try:
                return vae.decode(one)
            except Exception as error:
                if not is_out_of_memory(error):
                    raise
            gc.collect()
        return TiledDecoder(vae.decode)(one)
