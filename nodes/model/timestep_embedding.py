"""Sinusoidal timestep embedding -- owned here rather than imported.

Design doc 12, section 7.1. Both call sites, `clip_encoder.py` and
`unet_wrapper.py`, need one thing from ComfyUI's diffusion modules: a
`Timestep(256)` to sinusoidal-encode the six resolution values in an SDXL
conditioning vector, 6 x 256 = 1536 dims. Nothing else.

Why own it, having already fixed the import path. ComfyUI defines the
class in `comfy.ldm.modules.diffusionmodules.openaimodel` and re-exports it
from `comfy.model_base`; both call sites imported from the re-export, which
costs an extra 1.27 s and 901 modules per process on this machine
(measured 4.32 s / 3323 modules through `model_base`, against 3.05 s / 2422
through `openaimodel`). Importing from `openaimodel` fixed that. Owning the
seven lines removes the last edge, and with it a question that is not ours
to answer: whether a seven-line class in someone else's 3,000-line module
is still there in the version that gets installed.

Provenance: the algorithm is not ComfyUI's. It is the sinusoidal embedding
from Ho et al., *Denoising Diffusion Probabilistic Models*, and the same
closed form appears in guided-diffusion and in diffusers as
`get_timestep_embedding`. `smoke_test_timestep_embedding.py` pins it to the
published formula independently, and separately characterises it against
ComfyUI's copy.

**The output dtype is always float32**, for any input dtype, and that is a
property of the algorithm rather than of the caller. `timesteps` is cast to
float and the frequencies are built in float32, so cos and sin of a float32
argument come back float32. `Timestep` has no parameters and no buffers, so
`.to(device=..., dtype=...)` on it does nothing at all -- not even the
device half. Both call sites used to write that, implying a control they did
not have. The device of the *input* tensor is what decides where the
embedding is computed, and the consumer casts the result: `torch.cat`
promotes, so an SDXL `y` assembled from a half-precision pooled output and
these float32 embeddings is float32 either way. That is what ComfyUI does
too, so this is characterisation rather than a fix -- see the design doc's
section 7 on why a divergence here would be a question about which side is
wrong, not a gate.
"""

from __future__ import annotations

import math

import torch

__all__ = ["Timestep", "timestep_embedding"]


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000,
) -> torch.Tensor:
    """Encode `timesteps` as sinusoidal features, shape ``[N, dim]``.

    :param timesteps: 1-D tensor of N indices, one per batch element. May be
        fractional -- SDXL passes pixel counts, not step indices.
    :param dim: output width. Must be at least 2; an odd `dim` is allowed
        and gets a trailing zero column, because cos and sin together
        produce ``2 * (dim // 2)`` columns and an odd width needs one more.
    :param max_period: the lowest frequency, i.e. the period of the
        slowest sinusoid. 10000 is the value every published
        implementation of this embedding uses.
    :return: ``[N, dim]``, float32.

    ComfyUI's copy carries a `repeat_only` flag that, when set, returns
    ``repeat(timesteps, 'b -> b d')`` -- not an embedding at all, just the
    timestep numbers tiled. It is dropped here: neither call site wants it,
    and a caller who reaches for it is asking for something whose name does
    not describe what they get. `UNetModel` uses that branch in ComfyUI, so
    this module is not a drop-in for the whole of `openaimodel` and is not
    trying to be -- it is the one class both of our call sites want.
    """
    if dim < 2:
        # ComfyUI divides by `dim // 2` without checking, so `dim=1` fails
        # as a broadcast error between (N, 1) and (1, 0). Say what is
        # wrong instead.
        raise ValueError(f"timestep embedding dim must be at least 2, "
                         f"got {dim}")
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be 1-D, got shape "
                         f"{tuple(timesteps.shape)}")

    half = dim // 2
    # float32 explicitly, and on the input's device. The device is load
    # -bearing on an accelerator build: building this on the CPU and moving
    # it would put a host allocation in the middle of a forward pass.
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Timestep(torch.nn.Module):
    """A fixed-width sinusoidal embedding of timesteps.

    Parameterless by construction: this is an algorithm, not a learned
    thing, so it holds no weights, has no state dict entry, and `.to()` on
    it is a no-op. See the module docstring on the output dtype.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)