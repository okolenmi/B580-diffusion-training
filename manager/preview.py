"""VAE-based preview generator for visual pruning."""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core.vae_decode import VAEDecoder
from core.comfy_setup import xpu_empty_cache


class PreviewGenerator:
    """Decodes latents to small thumbnails for the dataset manager UI."""

    def __init__(self, device: str, vae_sd: dict):
        self.device = device
        self.vae = VAEDecoder.from_vae_sd(vae_sd, device=device)

    def generate_preview(self, latent: torch.Tensor, output_path: Path, max_size: int = 256):
        """Decode a single latent and save as a WebP thumbnail, preserving aspect ratio.

        The longest side is resized to max_size, so non-square latents produce
        non-square previews that reflect the actual training data shape.
        """
        if self.vae is None:
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            # Shape (1, 4, H, W)
            decoded = self.vae.decode(latent.to(self.device))
            
            img_np = decoded[0].cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype('uint8').transpose(1, 2, 0)
            pil_img = Image.fromarray(img_np)
            
            w, h = pil_img.size
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = round(w * scale), round(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            pil_img.save(str(output_path), "WEBP", quality=80)
            
            del decoded, img_np, pil_img
            xpu_empty_cache()

    def free(self):
        if self.vae:
            self.vae.free()
            self.vae = None
        gc.collect()
        xpu_empty_cache()
