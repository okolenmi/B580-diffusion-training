"""The Anima transformer: loading from a file slice and a callable that runs the text adapter and casts inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..api.parts import ModelPart
from ..general.tensors import read_prefixed
from .dit import AnimaDiT, DiTConfig

_FILE_ONLY = ("pos_embedder.",)  # buffers some files carry; the rotary angles are computed


class AnimaDenoiser:
    """Wraps the DiT so callers pass latents, flow positions and an ``AnimaConditioning`` in any dtype.

    The text adapter lives in the DiT file and runs here on every call: the conditioning holds its inputs (Qwen3 states and
    T5 ids), which keeps the cache independent of the DiT file.
    """

    def __init__(self, model: AnimaDiT, device="cpu", dtype: torch.dtype = torch.bfloat16):
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.model.to(device=self.device, dtype=dtype)

    def predict(self, x_t: Tensor, timestep: Tensor, conditioning) -> Tensor:
        """Noisy latents (B, 16, H, W) and flow positions (B,) in, velocity out."""
        dtype = self.dtype
        context = self.model.encode_text(conditioning.hidden.to(dtype), conditioning.ids, conditioning.hidden_mask,
                                         conditioning.ids_mask)
        return self.model(x_t.to(dtype), timestep.to(torch.float32), context)

    def mid_features(self, *_args) -> Tensor:
        raise NotImplementedError("the self-perceptual loss reads a UNet's middle block; Anima has no such tap yet")

    def parameters(self):
        return self.model.parameters()

    def module(self) -> nn.Module:
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def train(self, mode: bool = True) -> AnimaDenoiser:
        self.model.train(mode)
        return self

    def eval(self) -> AnimaDenoiser:
        return self.train(False)

    def to(self, device=None, dtype: torch.dtype | None = None) -> AnimaDenoiser:
        self.model.to(device=device, dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def set_checkpointer(self, checkpointer, attention_fraction: float = 1.0) -> None:
        self.model.set_checkpointer(checkpointer, attention_fraction)

    def checkpoint_blocks(self) -> list[tuple[str, nn.Module]]:
        return self.model.checkpoint_blocks()

    def set_checkpointed(self, checkpointer, flags) -> None:
        self.model.set_checkpointed(checkpointer, flags)


def load_denoiser(part: ModelPart, *, device="cpu", dtype: torch.dtype = torch.bfloat16,
                  config: DiTConfig | None = None) -> AnimaDenoiser:
    """Build the DiT (and its text adapter) from the ``denoiser`` part of a Load Model node. Missing or unexpected keys are an error."""
    state = {k: v for k, v in read_prefixed(part.source.path, part.source.prefix, dtype=dtype).items() if not k.startswith(_FILE_ONLY)}
    if config is None:
        info = part.info.config
        config = DiTConfig.from_shapes(info["blocks"], info["width"], state["x_embedder.proj.1.weight"].shape[1],
                                       info.get("adapter_blocks") or 6)
    with torch.device("meta"):
        model = AnimaDiT(config)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"DiT weights do not fit: {len(result.missing_keys)} missing (e.g. {result.missing_keys[:2]}), "
                         f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[:2]})")
    return AnimaDenoiser(model, device, dtype)
