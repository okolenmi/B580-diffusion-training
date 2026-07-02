"""Distillation Converter package.

Supports distillation, cyclic, LoRA, and full fine-tune training.
Uses ComfyUI's UNet directly — LDM key format throughout.
"""

# Config model (new OOP structure with discriminated unions)
from .config_model import (
    TrainingConfig,
    CommonSettings,
    ModelPaths,
    TuningMethod,
    LoRATuning,
    CyclicTuning,
    DistillationTuning,
    FullTuning,
    CacheConfig,
    TrajectoryCache,
    RandomCache,
)

# Config I/O
from .config_io import (
    read_config,
    write_config,
    write_default_config,
    config_to_toml_string,
    config_from_toml_string,
    upgrade_config_file,
)

# Legacy compatibility re-exports
from .config_io import read_config as load_config
from .config_model import TrainingConfig

# Cache utilities
from .cache_utils import (
    resolve_gen_batch_size,
    shuffle_and_rebatch_cache,
    warn_batch_mismatch,
)

# Model I/O
from .model_io import (
    comfy_input_transform,
    make_init_noise,
    raw_to_denoised,
    raw_to_target,
)

# Noise schedule
from .noise_schedule import (
    ALPHA_T,
    SIGMA_T,
    eps_to_vpred,
    eps_to_x0,
    get_alpha_sigma,
    make_schedule,
    sample_timestep,
    vpred_to_eps,
    vpred_to_x0,
)

# LR schedules
from .schedules import make_cosine_lr, make_lr_schedule, make_poly_lr

# Optimizers
from .optimizers import CPUAdamW, ChunkedXPUAdafactor, FusedXPUAdafactor

# UNet wrapper
from .unet_wrapper import ComfyUNetWrapper, make_rand_cond

# LoRA
from .lora import LoRAConfig, LoRALinear, inject_lora_into_unet, extract_lora_weights

__all__ = [
    # Config model
    "TrainingConfig", "CommonSettings", "ModelPaths",
    "TuningMethod", "LoRATuning", "CyclicTuning", "DistillationTuning", "FullTuning",
    "CacheConfig", "TrajectoryCache", "RandomCache",
    # Config I/O
    "read_config", "write_config", "write_default_config",
    "config_to_toml_string", "config_from_toml_string", "upgrade_config_file",
    # Legacy
    "load_config",
    # Cache utilities
    "resolve_gen_batch_size", "shuffle_and_rebatch_cache", "warn_batch_mismatch",
    # Model I/O
    "comfy_input_transform", "make_init_noise", "raw_to_denoised", "raw_to_target",
    # Noise schedule
    "ALPHA_T", "SIGMA_T",
    "make_schedule", "get_alpha_sigma", "sample_timestep",
    "vpred_to_x0", "eps_to_x0", "vpred_to_eps", "eps_to_vpred",
    # LR schedules
    "make_cosine_lr", "make_poly_lr", "make_lr_schedule",
    # Optimizers
    "CPUAdamW", "ChunkedXPUAdafactor", "FusedXPUAdafactor",
    # UNet wrapper
    "ComfyUNetWrapper", "make_rand_cond",
    # LoRA
    "LoRAConfig", "LoRALinear", "inject_lora_into_unet", "extract_lora_weights",
]
