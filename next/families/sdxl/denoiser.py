"""The SDXL UNet: its configuration, loading from a checkpoint slice, and a callable that casts inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..api.parts import ModelPart
from ..general.ldm_unet.attention_modes import MODES as ATTENTION_MODES
from ..general.ldm_unet.fused import FusedGeglu
from ..general.ldm_unet.unet import Checkpointer, UNetModel
from ..general.tensors import read_prefixed

SDXL_UNET_CONFIG = {
    "in_channels": 4, "out_channels": 4, "model_channels": 320,
    "num_res_blocks": [2, 2, 2], "channel_mult": [1, 2, 4], "num_head_channels": 64,
    "transformer_depth": [0, 0, 2, 2, 10, 10], "transformer_depth_middle": 10,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10], "context_dim": 2048,
    "use_linear_in_transformer": True, "num_classes": "sequential", "adm_in_channels": 2816,
}


class SDXLDenoiser:
    """Wraps the UNet so callers pass latents, a float timestep, ``ctx`` and ``y`` in any dtype."""

    def __init__(self, model: UNetModel, device="cpu", dtype: torch.dtype = torch.bfloat16):
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.channels_last = False
        self.model.to(device=self.device, dtype=dtype)

    def __call__(self, x_t: Tensor, timestep: Tensor, context: Tensor, y: Tensor) -> Tensor:
        x = x_t.to(self.dtype)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.model(x, timestep.to(torch.float32), context.to(self.dtype), y.to(self.dtype))

    def predict(self, x_t: Tensor, timestep: Tensor, conditioning) -> Tensor:
        """The family-neutral entry point: noisy input, timesteps and an SDXL conditioning in, prediction out."""
        return self(x_t, timestep, conditioning.ctx, conditioning.y)

    def parameters(self):
        return self.model.parameters()

    def module(self) -> nn.Module:
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def train(self, mode: bool = True) -> "SDXLDenoiser":
        self.model.train(mode)
        return self

    def eval(self) -> "SDXLDenoiser":
        return self.train(False)

    def to(self, device=None, dtype: torch.dtype | None = None) -> "SDXLDenoiser":
        self.model.to(device=device, dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def attention_scopes(self) -> dict:
        """Named ways to run attention inside a captured graph: name -> a function returning a context manager."""
        return dict(ATTENTION_MODES)

    def enable_channels_last(self) -> None:
        """Keep the frozen convolutions' weights and the input in channels-last layout, which oneDNN runs without reordering them on
        every call. Trainable convolutions stay as they are (a saved adapter must be contiguous)."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and not any(p.requires_grad for p in module.parameters()):
                module.to(memory_format=torch.channels_last)
        self.channels_last = True

    def enable_fused_geglu(self, device) -> str:
        """Run the feed-forward activation as one fused kernel. Returns "" when on, otherwise why it stays off."""
        return FusedGeglu.enable(device)

    def set_checkpointer(self, checkpointer: Checkpointer | None, attention_fraction: float = 1.0) -> None:
        self.model.set_checkpointer(checkpointer, attention_fraction)

    def checkpoint_blocks(self) -> list[tuple[str, nn.Module]]:
        return self.model.checkpoint_blocks()

    def set_checkpointed(self, checkpointer: Checkpointer | None, flags) -> None:
        self.model.set_checkpointed(checkpointer, flags)


def build_unet(config: dict | None = None) -> UNetModel:
    with torch.device("meta"):
        return UNetModel(**(config or SDXL_UNET_CONFIG))


def load_denoiser(part: ModelPart, *, device="cpu", dtype: torch.dtype = torch.bfloat16,
                  config: dict | None = None) -> SDXLDenoiser:
    """Build the UNet from the ``denoiser`` part of a Load Model node. Missing or unexpected keys are an error."""
    state = read_prefixed(part.source.path, part.source.prefix, dtype=dtype)
    model: nn.Module = build_unet(config)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"UNet weights do not fit: {len(result.missing_keys)} missing (e.g. {result.missing_keys[:2]}), "
                         f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[:2]})")
    return SDXLDenoiser(model, device, dtype)
