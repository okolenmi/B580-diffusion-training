"""The ComfyUI model I/O contract, as free functions.

    input:  xc = x_t / sqrt(sigma^2 + 1)
    output: raw eps or v-prediction
    x0:     obtained from the raw output by the model's own parameterization

`nodes/components/diffusion.py` owns that maths as objects. This module is
the free-function façade over them, for the call sites in `manager/` that
predate injection and call per step. It adds no arithmetic of its own --
every body delegates -- so there is one implementation and
`smoke_test_diffusion_equivalence.py` covers it.

The predecessor was `core/model_io.py`, whose docstring listed the four
callers it served (`cache_trajectory`, `cache_random`, `train_step`,
`train`). Only `manager/builder.py` still calls these; the other three
were retired with the old trainer, so the module is much smaller than it
was.
"""

from __future__ import annotations



import torch

from .diffusion import EpsParameterization, KarrasInputScaler, VPredParameterization
from .noise_schedule import eps_to_vpred, vpred_to_eps

__all__ = [
    "comfy_input_transform", "raw_to_denoised", "raw_to_target", "make_init_noise",
]

_SCALER = KarrasInputScaler()
_EPS = EpsParameterization()
_VPRED = VPredParameterization()


def comfy_input_transform(x_t: torch.Tensor, sigma) -> torch.Tensor:
    """ComfyUI's ``calculate_input``: ``xc = x_t / sqrt(sigma^2 + 1)``.

    Works with a scalar sigma and with a per-sample sigma tensor, and
    returns bf16 because that is what the UNet expects.
    """
    return _SCALER.scale_input(x_t, sigma)


def raw_to_denoised(raw: torch.Tensor, x_t: torch.Tensor,
                    alpha, sigma, model_type: str) -> torch.Tensor:
    """The model's raw output as clean x0 -- ComfyUI's ``calculate_denoised``."""
    return _VPRED.to_x0(raw, x_t, alpha, sigma) if model_type == "vpred" \
        else _EPS.to_x0(raw, x_t, alpha, sigma)


def raw_to_target(raw: torch.Tensor, x_t: torch.Tensor, alpha, sigma,
                  teacher_type: str, student_type: str) -> torch.Tensor:
    """The teacher's output, expressed as whatever the student predicts.

    Same-parameterisation on both sides is the identity, which is why this
    is a dispatch and not a conversion.
    """
    if teacher_type == "vpred" and student_type == "eps":
        return vpred_to_eps(raw, x_t, alpha, sigma)
    if teacher_type == "eps" and student_type == "vpred":
        return eps_to_vpred(raw, x_t, alpha, sigma)
    return raw


def make_init_noise(shape, device, dtype, sigma, generator=None) -> torch.Tensor:
    """Initial noise scaled to ComfyUI's txt2img ``noise_scaling``.

    With no ``latent_image``, ComfyUI computes ``x_t = noise * sigma``.

    Drawn on CPU whatever the target device: the generator a caller hands
    in is CPU-seeded, and drawing on the accelerator would silently ignore
    that seed. This is why the parameter is named ``cpu_gen`` at the one
    other call site rather than ``generator``.
    """
    cpu_gen = generator if generator is not None else torch.Generator(device="cpu")
    noise = torch.randn(shape, generator=cpu_gen, device="cpu").to(
        device=device, dtype=dtype
    )
    s = sigma if not isinstance(sigma, torch.Tensor) else sigma.item()
    return noise * s