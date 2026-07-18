"""Teacher cache builder — random mode (pure Gaussian noise x0)."""

import gc
import random
import sys
import contextlib
from typing import Optional

import torch
from tqdm import tqdm

COND_SEED_MULTIPLIER = 10007

from .cache_utils import resolve_gen_batch_size, warn_batch_mismatch, pin_tensors_parallel
from .comfy_setup import xpu_empty_cache
from .model_io import comfy_input_transform, raw_to_target
from .noise_schedule import get_alpha_sigma, sample_timestep
from .seed import derive_seed
from .unet_wrapper import ComfyUNetWrapper, make_rand_cond
from .lora import lora_gate_override


def build_teacher_cache(teacher_unet_sd: dict, teacher_type: str,
                        student_type: str,
                        n_samples: int, batch_size: int,
                        device: str, seed: int,
                        cache_batch: int = 8,
                        cache_batch_size: int = None,
                        latent_size: int = 0,
                        t_mode: str = "uniform",
                        t_low: int = 0, t_high: int = 999,
                        teacher_model=None,
                        teacher_lora_gate: Optional[float] = None) -> list:
    """
    Pre-compute n_samples teacher outputs on GPU (bf16), then discard teacher.
    Returns list of (x_t, target_epsilon, alpha, sigma, t_val) stored on CPU.
    
    cache_batch_size controls the batch size for teacher forward passes during
    cache generation. If None or 0, falls back to batch_size.

    teacher_lora_gate: see build_teacher_cache_trajectory's docstring in
    cache_trajectory.py -- same mechanism, forces the LoRA gate to this value
    around teacher.forward() when teacher_model is the live student object.
    """
    _gate_zero = (torch.tensor(teacher_lora_gate) if teacher_lora_gate is not None else None)
    def _teacher_ctx():
        return lora_gate_override(_gate_zero) if _gate_zero is not None else contextlib.nullcontext()

    gen_batch_size = resolve_gen_batch_size(cache_batch_size, batch_size)
    warn_batch_mismatch(gen_batch_size, batch_size)

    # Resolve latent spatial size
    latent_dim = latent_size if (latent_size and latent_size > 0) else 64
    if latent_dim != 64:
        print(f"    Latent size: {latent_dim}×{latent_dim} ({latent_dim*8}×{latent_dim*8} px)")
    
    if teacher_model is not None:
        print(f"    Reusing existing teacher model for random cache...")
        teacher = teacher_model
    else:
        print(f"    Loading teacher on {device} (bf16) for cache generation...")
        teacher = ComfyUNetWrapper(teacher_unet_sd, device=device, dtype=torch.bfloat16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Enforce a minimum safe t_low — same reasoning as cache_trajectory:
    # near-clean timesteps (t < 20, sigma < 0.01) produce garbage teacher
    # outputs that corrupt the last sampler step during inference.
    _SAFE_T_LOW = max(t_low, 20)
    if _SAFE_T_LOW != t_low:
        print(f"    Note: t_low raised from {t_low} to {_SAFE_T_LOW} to avoid near-clean timesteps.")
    t_low = _SAFE_T_LOW

    rng = random.Random(seed)

    meta = []
    for _ in range(n_samples):
        t_val = sample_timestep(rng, t_mode, t_low, t_high)
        alpha, sigma = get_alpha_sigma(t_val)
        meta.append((t_val, alpha.item(), sigma.item()))

    cache = []
    print(f"    Generating {n_samples} samples (cache_batch={cache_batch})...")

    with torch.no_grad():
        idx = 0
        is_tty = sys.stdout.isatty()
        pbar_cache = tqdm(total=n_samples, desc="  Caching", unit="sample", leave=False) if is_tty else None
        while idx < n_samples:
            chunk = meta[idx: idx + cache_batch]
            B = len(chunk)

            t_vals = [c[0] for c in chunk]
            alphas = [c[1] for c in chunk]
            sigmas = [c[2] for c in chunk]

            _gen = torch.Generator(device="cpu")
            _gen.manual_seed(derive_seed(seed, idx, "x0"))
            x0_big = torch.randn(B * gen_batch_size, 4, latent_dim, latent_dim,
                                 generator=_gen).to(device=device, dtype=torch.bfloat16)
            _gen.manual_seed(derive_seed(seed, idx, "noise"))
            noise_big = torch.randn(B * gen_batch_size, 4, latent_dim, latent_dim,
                                    generator=_gen).to(device=device, dtype=torch.bfloat16)

            alpha_vec = torch.tensor(
                [a for a in alphas for _ in range(gen_batch_size)],
                dtype=torch.bfloat16, device=device).view(-1, 1, 1, 1)
            sigma_vec = torch.tensor(
                [s for s in sigmas for _ in range(gen_batch_size)],
                dtype=torch.bfloat16, device=device).view(-1, 1, 1, 1)
            # ComfyUI forward process: x_t = x0 + sigma * eps
            # (sigma already encodes the alpha ratio; do NOT multiply by alpha_vec)
            x_t_big = x0_big + sigma_vec * noise_big
            del x0_big, noise_big

            t_big = torch.tensor(
                [t for t in t_vals for _ in range(gen_batch_size)],
                dtype=torch.long, device=device)

            x_t_transformed = comfy_input_transform(x_t_big, sigma_vec)

            ctx_list, y_list = [], []
            for j in range(B):
                # Per-sample conditioning: seed encodes (batch_meta_idx, within_batch_k).
                ctx_batch, y_batch = [], []
                for k in range(gen_batch_size):
                    _per_sample_seed = (idx + j) * COND_SEED_MULTIPLIER + k
                    _ctx_k, _y_k = make_rand_cond(1, device, torch.bfloat16, seed, _per_sample_seed,
                                                  latent_size=latent_dim)
                    ctx_batch.append(_ctx_k)
                    y_batch.append(_y_k)
                ctx_list.append(torch.cat(ctx_batch))
                y_list.append(torch.cat(y_batch))
                del ctx_batch, y_batch
            ctx_big = torch.cat(ctx_list)
            y_big = torch.cat(y_list)
            del ctx_list, y_list

            with _teacher_ctx():
                raw_big = teacher.forward(x_t_transformed, t_big, ctx_big, y_big)

            # Convert teacher output to student's prediction target type
            target_big = raw_to_target(raw_big, x_t_big, alpha_vec, sigma_vec,
                                       teacher_type, student_type)
            del raw_big, alpha_vec, sigma_vec, x_t_transformed

            # Split everything per-entry including ctx/y for the new cache format v2
            # Cache stores: (x_t, target, ctx, y, ctx_u, y_u, alpha, sigma, t_val)
            for j in range(B):
                ctx_j = ctx_big[j*gen_batch_size:(j+1)*gen_batch_size]
                y_j = y_big[j*gen_batch_size:(j+1)*gen_batch_size]
                x_t_j = x_t_big[j*gen_batch_size:(j+1)*gen_batch_size]
                target_j = target_big[j*gen_batch_size:(j+1)*gen_batch_size]
                
                # For random mode, unc is always zero-cond
                ctx_u_j = torch.zeros_like(ctx_j)
                y_u_j = torch.zeros_like(y_j)

                # Move to CPU first, then pin all 6 tensors concurrently.
                cpu_tensors = [
                    t.to("cpu", non_blocking=True).float().contiguous()
                    for t in (x_t_j, target_j, ctx_j, y_j, ctx_u_j, y_u_j)
                ]
                pinned = pin_tensors_parallel(*cpu_tensors)

                cache.append((
                    pinned[0], pinned[1], pinned[2], pinned[3], pinned[4], pinned[5],
                    chunk[j][1],
                    chunk[j][2],
                    chunk[j][0],
                ))
            del ctx_big, y_big, x_t_big, target_big, ctx_j, y_j, x_t_j, target_j, ctx_u_j, y_u_j
            idx += B
            if pbar_cache:
                pbar_cache.update(B)

        if pbar_cache:
            pbar_cache.close()

    if teacher_model is None:
        del teacher
        xpu_empty_cache()
        gc.collect()
    print(f"    Cache ready ({n_samples} samples).")
    return cache
