"""ComfyUI UNet wrapper and random conditioning generation."""

import gc

import torch

from .lora import (
    LoRAConfig,
    extract_lora_weights,
    inject_lora_into_unet,
    lora_param_count,
    load_lora_into_model,
    merge_lora_into_unet,
)
from .seed import derive_seed


class ComfyUNetWrapper:
    """Wraps ComfyUI's UNetModel with SDXL config for distillation/LoRA training."""

    SDXL_CONFIG = {
        "image_size":                32,
        "in_channels":               4,
        "out_channels":              4,
        "model_channels":            320,
        "num_res_blocks":            [2, 2, 2],
        "channel_mult":              [1, 2, 4],
        "num_head_channels":         64,
        "use_spatial_transformer":   True,
        "transformer_depth":         [0, 0, 2, 2, 10, 10],
        "transformer_depth_middle":  10,
        "transformer_depth_output":  [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "context_dim":               2048,
        "use_linear_in_transformer": True,
        "num_classes":               "sequential",
        "adm_in_channels":           2816,
        "legacy":                    False,
        "use_checkpoint":            True,
        "use_temporal_attention":    False,
        "use_temporal_resblock":     False,
    }

    def __init__(self, unet_sd: dict, device: str, dtype: torch.dtype,
                 use_checkpoint=True, adm_in_channels=2816,
                 lora_config: LoRAConfig | None = None):
        self.device = device
        self.dtype = dtype
        self.lora_config = lora_config
        self.lora_registry = None

        sd = {k.replace("model.diffusion_model.", ""): v for k, v in unet_sd.items()}
        import comfy.ldm.modules.diffusionmodules.openaimodel as om
        cfg = dict(self.SDXL_CONFIG)
        cfg["dtype"] = dtype
        cfg["use_checkpoint"] = use_checkpoint
        cfg["adm_in_channels"] = adm_in_channels
        self.model = om.UNetModel(**cfg)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"    Warning: {len(missing)} missing keys (first: {missing[0]})")
        self.model = self.model.to(device=device, dtype=dtype)

        if lora_config is not None:
            self._init_lora()

    def _init_lora(self):
        self.lora_registry = inject_lora_into_unet(self.model, self.lora_config)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for _, _, _, layer in self.lora_registry:
            if hasattr(layer, "lora_A"):
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)
        n_lora = lora_param_count(self.lora_registry)
        n_actual = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    LoRA injected: rank={self.lora_config.rank}, "
              f"alpha={self.lora_config.alpha}, "
              f"params={n_lora:,} ({n_lora/1024:.1f}K) "
              f"trainable={n_actual:,}")

    def inject_lora(self, config: LoRAConfig):
        self.lora_config = config
        self._init_lora()

    def merge_lora(self):
        if self.lora_registry:
            merge_lora_into_unet(self.lora_registry)

    def has_lora(self) -> bool:
        return self.lora_registry is not None

    def get_lora_weights(self):
        if self.lora_registry:
            return extract_lora_weights(self.lora_registry)
        return {}

    def load_lora_weights(self, state_dict):
        if self.lora_registry:
            load_lora_into_model(self.lora_registry, state_dict)

    def lora_parameters(self):
        if not self.lora_registry:
            return []
        params = []
        for _, _, _, layer in self.lora_registry:
            if hasattr(layer, "lora_A") and isinstance(layer.lora_A, torch.nn.Parameter):
                params.append(layer.lora_A)
                params.append(layer.lora_B)
        return params

    def forward(self, x_t, timestep, context, y):
        """Unified forward pass. 
        Always pass as keywords to avoid positional mismatches in patched models.
        """
        x_t = x_t.to(dtype=self.dtype)
        timestep = timestep.to(dtype=torch.float32)
        context = context.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        
        return self.model(x=x_t, timesteps=timestep, context=context, y=y)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def state_dict(self):
        return self.model.state_dict()

    def to(self, device=None, **kwargs):
        self.model.to(device=device, **kwargs)
        if device is not None:
            self.device = str(device)
        return self

    def enable_gradient_checkpointing(self):
        """Reduce VRAM by discarding activations (recomputed on backward)."""
        for module in self.model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = True


# ---------------------------------------------------------------------------
# Random conditioning generation
# ---------------------------------------------------------------------------

_EMBEDDER_CACHE = {}

def make_rand_cond(batch: int, device: str, dtype: torch.dtype,
                   base_seed: int, step: int, latent_size: int = 64):
    """
    Random conditioning for distillation.
    Seeds are derived via derive_seed so teacher and student always get the
    same tensors for the same (base_seed, step) pair.
    """
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(derive_seed(base_seed, step, "cond_ctx"))
    ctx = torch.randn(batch, 77, 2048, generator=cpu_gen).to(device=device, dtype=dtype)
    
    cpu_gen.manual_seed(derive_seed(base_seed, step, "cond_y_pooled"))
    pooled = torch.randn(batch, 1280, generator=cpu_gen).to(device=device, dtype=dtype)

    # Resolution embeddings (SDXL VAE has 8x downscale)
    px = (latent_size if latent_size > 0 else 64) * 8

    # Reuse embedder to save VRAM and time
    cache_key = (device, dtype)
    if cache_key not in _EMBEDDER_CACHE:
        from comfy.model_base import Timestep
        _EMBEDDER_CACHE[cache_key] = Timestep(256).to(device=device, dtype=dtype)
    
    embedder = _EMBEDDER_CACHE[cache_key]

    # original_h, original_w, crop_h, crop_w, target_h, target_w
    vals = torch.tensor([px, px, 0, 0, px, px], device=device, dtype=dtype)
    time_embs = embedder(vals)  # (6, 256)
    time_emb_flat = time_embs.view(1, -1).repeat(batch, 1)

    y = torch.cat([pooled, time_emb_flat], dim=-1)
    return ctx, y

def clear_embedder_cache():
    """Move all cached Timestep embedders to CPU and clear the cache.

    Call this between cyclic training cycles, after models have been moved to
    CPU and before building the next cache.  The Timestep model is small but
    it sits on the XPU/CUDA device and prevents full GPU memory reclamation
    during the CLIP encoding phase.
    """
    global _EMBEDDER_CACHE
    for model in _EMBEDDER_CACHE.values():
        try:
            model.to("cpu")
        except Exception:
            pass
    _EMBEDDER_CACHE.clear()
    gc.collect()
