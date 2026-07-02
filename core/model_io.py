"""ComfyUI model interaction layer.

Central helpers for calling the UNet and converting its output, eliminating
repeated logic across cache_trajectory.py, cache_random.py, train_step.py,
and train.py.

Every ComfyUI diffusion model expects this I/O contract:
  - Input:  xc = x_t / sqrt(sigma^2 + 1)    (calculate_input)
  - Output: raw eps or v-prediction
  - The sampler needs denoised (x0), obtained via calculate_denoised
"""

import math

import torch

from .noise_schedule import eps_to_x0, eps_to_vpred, vpred_to_eps, vpred_to_x0


# ---------------------------------------------------------------------------
# Input transformation  (ComfyUI's calculate_input)
# ---------------------------------------------------------------------------

def comfy_input_transform(x_t: torch.Tensor, sigma) -> torch.Tensor:
    """Apply ComfyUI's calculate_input: xc = x_t / sqrt(sigma^2 + 1).

    Works with both scalar sigma (float) and per-sample sigma tensors.
    Returns bf16 to match the UNet's expected dtype.
    """
    if isinstance(sigma, torch.Tensor) and sigma.ndim > 0:
        sigma_for_input = (sigma.float() ** 2 + 1.0) ** 0.5
        if sigma_for_input.ndim < 4:
            sigma_for_input = sigma_for_input.view(-1, 1, 1, 1)
        return (x_t / sigma_for_input).to(torch.bfloat16)
    else:
        # Scalar case — use math.sqrt to avoid torch.sqrt on Python float
        s = sigma if not isinstance(sigma, torch.Tensor) else sigma.item()
        return (x_t / math.sqrt(s ** 2 + 1.0)).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Output conversion  (ComfyUI's calculate_denoised + target conversion)
# ---------------------------------------------------------------------------

def raw_to_denoised(raw: torch.Tensor, x_t: torch.Tensor,
                    alpha, sigma, model_type: str) -> torch.Tensor:
    """Convert raw model output to denoised x0, matching calculate_denoised."""
    if model_type == "vpred":
        return vpred_to_x0(raw, x_t, alpha, sigma)
    else:
        return eps_to_x0(raw, x_t, alpha, sigma)


def raw_to_target(raw: torch.Tensor, x_t: torch.Tensor,
                  alpha, sigma, teacher_type: str, student_type: str) -> torch.Tensor:
    """Convert teacher output to the prediction target the student should learn."""
    if teacher_type == "vpred" and student_type == "eps":
        return vpred_to_eps(raw, x_t, alpha, sigma)
    elif teacher_type == "eps" and student_type == "vpred":
        return eps_to_vpred(raw, x_t, alpha, sigma)
    else:
        return raw


# ---------------------------------------------------------------------------
# Noise initialization  (ComfyUI's noise_scaling for txt2img)
# ---------------------------------------------------------------------------

def make_init_noise(shape, device, dtype, sigma, generator=None) -> torch.Tensor:
    """Create initial noise scaled to match ComfyUI's txt2img noise_scaling.

    In ComfyUI for txt2img (no latent_image): x_t = noise * sigma
    """
    cpu_gen = generator
    if cpu_gen is None:
        cpu_gen = torch.Generator(device="cpu")
    noise = torch.randn(shape, generator=cpu_gen, device="cpu").to(device=device, dtype=dtype)
    s = sigma if not isinstance(sigma, torch.Tensor) else sigma.item()
    return noise * s
