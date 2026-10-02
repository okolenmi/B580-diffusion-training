"""VAE decode utility — load VAE from checkpoint and decode latents to images.

Usage:
    from .vae_decode import VAEDecoder

    # From teacher checkpoint
    vae = VAEDecoder.from_checkpoint(teacher_sd, device="xpu")

    # Decode latents
    images = vae.decode(latents)  # (B, 3, H, W) float32 [0, 255]

    # Or save directly
    vae.save_images(latents, output_dir="samples/")

    # Clean up when done
    vae.free()
"""

import gc
from pathlib import Path

import torch
from PIL import Image


class VAEDecoder:
    """Wrapper around ComfyUI's VAE for decoding latents to images."""

    # Default scale factor for SDXL VAE (can be overridden if VAE has different config)
    DEFAULT_SCALE_FACTOR = 0.13025

    def __init__(self, device="xpu", scale_factor=None):
        self.device = device
        self.vae = None
        self.scale_factor = scale_factor or self.DEFAULT_SCALE_FACTOR

    @classmethod
    def from_checkpoint(cls, checkpoint_sd, device="xpu"):
        """
        Build a VAE decoder from a full model checkpoint.

        Parameters
        ----------
        checkpoint_sd : dict
            Full state dict from a .safetensors file.
        device : str
            Target device ("xpu", "cuda", or "cpu").

        Returns
        -------
        VAEDecoder or None if no VAE weights found.
        """
        vae_sd = {k.replace("first_stage_model.", ""): v
                  for k, v in checkpoint_sd.items()
                  if k.startswith("first_stage_model")}
        if not vae_sd:
            return None

        decoder = cls(device=device)
        decoder._load_vae(vae_sd)
        return decoder

    @classmethod
    def from_vae_sd(cls, vae_sd, device="xpu"):
        """
        Build a VAE decoder from pre-extracted VAE state dict.

        Parameters
        ----------
        vae_sd : dict
            State dict with keys like "encoder.", "decoder.", etc.
            (without "first_stage_model." prefix).
        device : str

        Returns
        -------
        VAEDecoder
        """
        decoder = cls(device=device)
        decoder._load_vae(vae_sd)
        return decoder

    def _load_vae(self, vae_sd):
        from comfy.ldm.models.autoencoder import AutoencoderKL

        embed_dim = 4
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }

        self.vae = AutoencoderKL(embed_dim=embed_dim, ddconfig=ddconfig)
        missing, unexpected = self.vae.load_state_dict(vae_sd, strict=False)
        if missing:
            print(f"    VAE: {len(missing)} missing keys, {len(unexpected)} unexpected")
        self.vae.eval()
        self.vae.to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents to RGB images.

        Parameters
        ----------
        latents : torch.Tensor
            Shape (B, 4, H, W), float32 on any device.
            Expected to be unscaled latents (raw diffusion space).
            H×W = 64×64 for 512px, 96×96 for 768px, 128×128 for 1024px (SDXL 8× downscale).

        Returns
        -------
        torch.Tensor
            Shape (B, 3, H_img, W_img), dtype uint8, values [0, 255].
        """
        latents = latents.to(device=self.device, dtype=torch.float32)
        # SDXL VAE expects latents divided by scale_factor
        # process_out: latent / scale_factor = latent / 0.13025
        latents = latents / self.scale_factor
        decoded = self.vae.decode(latents)
        decoded = decoded.float().clamp(-1, 1)
        images = ((decoded + 1.0) / 2.0 * 255.0).to(dtype=torch.uint8)
        return images.cpu()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB images to latents.
        
        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 3, H, W), float32 [0, 1] or uint8 [0, 255].
            
        Returns
        -------
        torch.Tensor
            Shape (B, 4, H/8, W/8), float32.
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            
        # Normalize to [-1, 1]
        images = images.to(self.device) * 2.0 - 1.0
        
        # Encode
        # AutoencoderKL.encode returns a distribution or the latent directly
        # In ComfyUI, it returns the latent directly for SDXL
        latents = self.vae.encode(images)
        
        # SDXL VAE requires multiplying by scale_factor after encoding
        # process_in: latent * scale_factor = latent * 0.13025
        return (latents * self.scale_factor).cpu()

    @torch.no_grad()
    def save_images(self, latents, output_dir, prefix="sample", start_idx=0, batch_size=1):
        """
        Decode latents and save as PNG files.

        Parameters
        ----------
        latents : torch.Tensor or list
            Shape (B, 4, H, W) or list of (1, 4, H, W) tensors.
        output_dir : str or Path
            Directory to save images.
        prefix : str
            Filename prefix.
        start_idx : int
            Starting index for filenames.
        batch_size : int
            Decode this many latents at a time to avoid OOM.
            Default 1 (safest for large VAEs).

        Returns
        -------
        int : number of images saved.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Handle list of single latents
        if isinstance(latents, list):
            latents = torch.cat(latents)

        n_saved = 0
        total = latents.shape[0]
        for start in range(0, total, batch_size):
            batch = latents[start:start + batch_size]
            images = self.decode(batch)
            for i in range(images.shape[0]):
                img_np = images[i].permute(1, 2, 0).numpy()
                pil_img = Image.fromarray(img_np)
                path = out_dir / f"{prefix}_{start_idx + n_saved:03d}.png"
                pil_img.save(str(path))
                n_saved += 1

        return n_saved

    def free(self):
        """Release VAE from VRAM."""
        if self.vae is not None:
            self.vae.to("cpu")
            del self.vae
            self.vae = None
            gc.collect()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
