"""SDXL UNet construction, and random conditioning generation.

The model is `nodes/model/unet.py`, this project's own implementation of the
published SDXL UNet (design doc 12, section 7.3, section A). It used to be
`comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel`.

**Moved here from `core/unet_wrapper.py` on 2026-10-02.** Unchanged apart
from its imports. `nodes/model/lora_injector.py:228` constructs this
class for every LoRA path in the graph, so the model the node route
trains on was living in `core/`; owning it here is what lets the
`layer_classes` parameter on `inject_lora_into_unet` replace the
module-global patching that `adapter_injection.py` used to do.

`core/unet_wrapper.py` remains as a re-export shim for `core/`'s and
`manager/`'s own use.

One known gap, unchanged by the move and worth stating plainly since it
is now in the file where a reader would look for it: `_init_lora()`
freezes every parameter and then re-enables `requires_grad` on each
layer's `lora_A`/`lora_B`, gated by `hasattr(layer, "lora_A")`. That
gate is False for a DoRA layer, which holds its `lora_A` nested one
level down by composition, so a DoRA build ends up with no trainable
adapter parameters at all -- silently, since loss can still move from
whatever else is trainable. `adapter_injection.py`'s
`reenable_dora_requires_grad()` and `dora_trainable_parameters()` exist
to correct this from the outside and are called by `lora_injector.py`.
"""
import gc

import torch

from ..components.seed import derive_seed
from .lora import (
    LoRAConfig,
    extract_lora_weights,
    inject_lora_into_unet,
    lora_param_count,
    load_lora_into_model,
    merge_lora_into_unet,
)
from .timestep_embedding import Timestep
from .unet import UNetModel


class ComfyUNetWrapper:
    """Builds the SDXL UNet for distillation/LoRA training.

    The class name is ComfyUI-era and no longer accurate -- it wraps nothing
    from ComfyUI any more. It is kept because `lora_injector.py` names it, and
    renaming it is a mechanical follow-up rather than something to fold into
    the change that makes the name true.
    """

    #: The published SDXL UNet configuration. Every key here is passed to
    #: `nodes.model.unet.UNetModel`; the parameters that configuration used
    #: to carry for ComfyUI's benefit and no longer exist are gone:
    #:
    #: * `legacy` -- only ever changed how `dim_head` was derived, and SDXL
    #:   passes false, so there is nothing to reproduce.
    #: * `use_temporal_attention`, `use_temporal_resblock` -- video. SDXL
    #:   passes false for both.
    #:
    #: The LoRA block-weight paths this project's configs use are written in
    #: terms of the module names this builds (`input_blocks.3.1....attn1.to_q`),
    #: which is why the names, not just the shapes, are a contract.
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
        "use_checkpoint":            True,
    }

    def __init__(self, unet_sd: dict, device: str, dtype: torch.dtype,
                 use_checkpoint=True, adm_in_channels=2816,
                 lora_config: LoRAConfig | None = None, layer_classes=None):
        self.device = device
        self.dtype = dtype
        self.lora_config = lora_config
        # Kept so every helper that inspects the registry can gate on the
        # classes actually injected, rather than on module globals that
        # used to be patched underneath this object. See lora.py's
        # _is_adapter_layer.
        self.layer_classes = layer_classes
        self.lora_registry = None

        sd = {k.replace("model.diffusion_model.", ""): v for k, v in unet_sd.items()}
        cfg = dict(self.SDXL_CONFIG)
        cfg["use_checkpoint"] = use_checkpoint
        cfg["adm_in_channels"] = adm_in_channels
        self.model = UNetModel(**cfg)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"    Warning: {len(missing)} missing keys (first: {missing[0]})")
        self.model = self.model.to(device=device, dtype=dtype)

        if lora_config is not None:
            self._init_lora()

    def _init_lora(self):
        self.lora_registry = inject_lora_into_unet(
            self.model, self.lora_config, layer_classes=self.layer_classes)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for _, _, _, layer in self.lora_registry:
            if hasattr(layer, "lora_A"):
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)
        n_lora = lora_param_count(self.lora_registry, layer_classes=self.layer_classes)
        n_actual = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    LoRA injected: rank={self.lora_config.rank}, "
              f"alpha={self.lora_config.alpha}, "
              f"params={n_lora:,} ({n_lora/1024:.1f}K) "
              f"trainable={n_actual:,}")

    def inject_lora(self, config: LoRAConfig, layer_classes=None):
        self.lora_config = config
        if layer_classes is not None:
            self.layer_classes = layer_classes
        self._init_lora()

    def merge_lora(self):
        if self.lora_registry:
            merge_lora_into_unet(self.lora_registry, layer_classes=self.layer_classes)

    def has_lora(self) -> bool:
        return self.lora_registry is not None

    def get_lora_weights(self):
        if self.lora_registry:
            return extract_lora_weights(self.lora_registry, layer_classes=self.layer_classes)
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

    # `enable_gradient_checkpointing()` used to live here: it walked
    # `self.model.modules()` and set `module.use_checkpoint = True` on
    # anything that had the attribute. It had no call sites, and
    # checkpointing no longer works that way -- it is a strategy object
    # applied around the forward (`nodes/model/gradient_checkpointing.py`),
    # chosen per call site rather than by mutating the model once. Setting
    # a bool on modules was the monkeypatch that strategy replaced.


# ---------------------------------------------------------------------------
# Random conditioning generation
# ---------------------------------------------------------------------------

#: One embedder for the process, built on first use. This was a dict keyed
#: by (device, dtype) "to save VRAM and time", and it saved nothing:
#: Timestep has no parameters and no buffers, so there was nothing to hold
#: per device and nothing to reclaim. The device of the input tensor is
#: what decides where the embedding is computed. Owned rather than imported
#: -- see nodes/model/timestep_embedding.py.
_EMBEDDER: Timestep | None = None


def _embedder() -> Timestep:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Timestep(256)
    return _EMBEDDER

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

    embedder = _embedder()

    # original_h, original_w, crop_h, crop_w, target_h, target_w
    vals = torch.tensor([px, px, 0, 0, px, px], device=device, dtype=dtype)
    time_embs = embedder(vals)  # (6, 256)
    time_emb_flat = time_embs.view(1, -1).repeat(batch, 1)

    y = torch.cat([pooled, time_emb_flat], dim=-1)
    return ctx, y

def clear_embedder_cache():
    """Drop the process-wide Timestep embedder.

    Nothing on the device is reclaimed, and the previous version of this
    docstring was wrong about that: it said the embedder "sits on the
    XPU/CUDA device and prevents full GPU memory reclamation", and it does
    not and never could. `Timestep` has no parameters and no buffers, so
    `model.to("cpu")` moved zero bytes -- it was a no-op in a try/except
    that existed to hide the no-op. What this does is drop the reference so
    the next call rebuilds it, and collect.
    """
    global _EMBEDDER
    _EMBEDDER = None
    gc.collect()
