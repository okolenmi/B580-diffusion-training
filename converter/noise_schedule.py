"""Noise schedule and prediction type conversions."""

import math

import torch


# ---------------------------------------------------------------------------
# Noise schedule — matches ComfyUI's ModelSamplingDiscrete
# ---------------------------------------------------------------------------
# ComfyUI uses:
#   alpha = sqrt(alphas_cumprod)
#   sigma = sqrt((1 - alphas_cumprod) / alphas_cumprod)
#
# Forward process (from EPS.noise_scaling):
#   x_t = x0 + sigma * eps
#
# This means sigma is the NOISE LEVEL, not sqrt(1-ab).
# The relationship: sigma = sqrt(1-ab)/sqrt(ab) = sqrt(1-ab)/alpha

def make_schedule(n=1000, beta_start=0.00085, beta_end=0.012):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, n) ** 2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alpha_t = alphas_cumprod.sqrt()
    sigma_t = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return alpha_t, sigma_t

ALPHA_T, SIGMA_T = make_schedule()


def get_alpha_sigma(t):
    """Return (alpha, sigma) for a given timestep index.

    Accepts either a plain int or a torch.Tensor.  ALPHA_T and SIGMA_T stay on
    CPU; the returned scalars are moved to the caller's device only when t is a
    Tensor, avoiding mutations of the module-level globals that would break
    concurrent use across devices.
    """
    # ALPHA_T / SIGMA_T live on CPU.  If t is a device tensor, indexing directly
    # raises "indices should be on the same device as the indexed tensor".
    # Move t to CPU for the lookup, then send the result to t's original device.
    if isinstance(t, torch.Tensor) and t.device.type != "cpu":
        t_cpu = t.cpu()
        a = ALPHA_T[t_cpu].to(t.device)
        s = SIGMA_T[t_cpu].to(t.device)
    else:
        a = ALPHA_T[t]
        s = SIGMA_T[t]
    return a, s


def sample_timestep(rng, mode: str, t_low: int, t_high: int) -> int:
    """
    Sample a timestep in [t_low, t_high] according to mode:
      uniform -- equal probability across the range (default)
      low     -- Beta(1, 3): biased toward low t (fine details, late denoising)
      mid     -- Beta(2, 2): biased toward middle t
      high    -- Beta(3, 1): biased toward high t (coarse structure)
      logit   -- logit-normal: concentrates around middle with heavier tails
    """
    lo, hi = t_low, t_high
    if mode == "uniform":
        return rng.randint(lo, hi)
    if mode == "logit":
        u = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-u))
        return max(lo, min(hi, int(round(lo + p * (hi - lo)))))
    alpha_beta = {"low": (1.0, 3.0), "mid": (2.0, 2.0), "high": (3.0, 1.0)}
    a, b = alpha_beta.get(mode, (1.0, 1.0))
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    p = x / (x + y)
    return max(lo, min(hi, int(round(lo + p * (hi - lo)))))


# ---------------------------------------------------------------------------
# Prediction type conversions
# ---------------------------------------------------------------------------
# ComfyUI forward process (from EPS.noise_scaling):
#   x_t = x0 + sigma * eps
#
# V_PREDICTION.calculate_denoised (with sigma_data=1.0):
#   x0 = x_t / (sigma^2 + 1) - v * sigma / sqrt(sigma^2 + 1)
#
# From x_t = x0 + sigma * eps:
#   x0 = x_t - sigma * eps
#
# Equating the two expressions for x0:
#   x_t - sigma * eps = x_t / (sigma^2 + 1) - v * sigma / sqrt(sigma^2 + 1)
#   sigma * eps = x_t * sigma^2 / (sigma^2 + 1) + v * sigma / sqrt(sigma^2 + 1)
#   eps = x_t * sigma / (sigma^2 + 1) + v / sqrt(sigma^2 + 1)
#
# And: v = eps / sqrt(sigma^2 + 1) - x_t * sigma / (sigma^2 + 1)
#     = (eps - sigma * x0) / sqrt(sigma^2 + 1)

def vpred_to_x0(v, x_t, alpha, sigma):
    """Convert v-prediction to clean x0 (matches ComfyUI V_PREDICTION.calculate_denoised)."""
    denom = sigma ** 2 + 1.0
    return x_t / denom - v * sigma / torch.sqrt(denom)


def eps_to_x0(eps, x_t, alpha, sigma):
    """Convert epsilon to clean x0."""
    return x_t - sigma * eps


def vpred_to_eps(v, x_t, alpha, sigma):
    """Convert v-prediction output to epsilon target."""
    denom = sigma ** 2 + 1.0
    return x_t * sigma / denom + v / torch.sqrt(denom)


def eps_to_vpred(eps, x_t, alpha, sigma):
    """Convert epsilon output to v-prediction target.
    
    Derivation from x0 = x_t - sigma * eps and x0 = x_t/(sigma^2+1) - v*sigma/sqrt(sigma^2+1):
    v = eps * sqrt(sigma^2 + 1) - x_t * sigma / sqrt(sigma^2 + 1)
    """
    denom_sqrt = torch.sqrt(sigma ** 2 + 1.0)
    return (eps * (sigma ** 2 + 1.0) - x_t * sigma) / denom_sqrt
