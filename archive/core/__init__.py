"""Core training package.

Supports distillation, cyclic, LoRA, and full fine-tune training.
Uses ComfyUI's UNet directly — LDM key format throughout.

**This module imports nothing eagerly** (PEP 562 lazy attribute access).
That is not a micro-optimisation, it is a correctness contract this
package depends on, in two places:

* ``backend/cli.py`` sets the XPU/SYCL performance environment variables
  with the comment "safe before any child", and its stated invariant is
  that the call touches no torch. Importing ``core.xpu_env`` runs this
  file first — with eager re-exports that pulled in ``torch`` (via
  ``optimizers``/``unet_wrapper``) *before* the environment variables
  were set, quietly invalidating the invariant (verified 2026-10-01:
  ``torch`` appeared in ``sys.modules`` across that single import).
  SYCL reads those variables at its own runtime init rather than at
  ``import torch``, so the result was probably harmless in practice —
  "probably" is not a contract, and the fix is three lines of laziness.
* ``nodes/`` keeps ``core.*`` imports out of module scope so importing
  ``nodes.dataset`` stays torch-free (see
  ``nodes/dataset/timestep_modes.py`` and
  ``docs/design/resources-controller/08-consolidation.md``). A package
  ``__init__`` that eagerly imports torch would defeat that no matter how
  carefully the call sites are written.

The names below are still reachable exactly as before -- ``core.LoRALinear``
and ``from core import config_io`` both behave identically -- they are just
resolved on first use. Every re-export here is unused outside this
package; they exist for the convenience of code that treats ``core`` as a
flat namespace.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

#: Public name -> the submodule that defines it. Kept explicit rather
#: than derived, so the mapping is greppable and reviewable.
_EXPORTS: dict[str, str] = {
    # config model
    "TrainingConfig": "config_model",
    "CommonSettings": "config_model",
    "ModelPaths": "config_model",
    "TuningMethod": "config_model",
    "LoRATuning": "config_model",
    "CyclicTuning": "config_model",
    "DistillationTuning": "config_model",
    "FullTuning": "config_model",
    "CacheConfig": "config_model",
    "TrajectoryCache": "config_model",
    "RandomCache": "config_model",
    # config I/O
    "read_config": "config_io",
    "write_config": "config_io",
    "write_default_config": "config_io",
    "config_to_toml_string": "config_io",
    "config_from_toml_string": "config_io",
    "upgrade_config_file": "config_io",
    "load_config": "config_io",  # legacy alias of read_config
    # cache utilities
    "resolve_gen_batch_size": "cache_utils",
    "shuffle_and_rebatch_cache": "cache_utils",
    "warn_batch_mismatch": "cache_utils",
    # model I/O
    "comfy_input_transform": "model_io",
    "make_init_noise": "model_io",
    "raw_to_denoised": "model_io",
    "raw_to_target": "model_io",
    # noise schedule
    "ALPHA_T": "noise_schedule",
    "SIGMA_T": "noise_schedule",
    "eps_to_vpred": "noise_schedule",
    "eps_to_x0": "noise_schedule",
    "get_alpha_sigma": "noise_schedule",
    "make_schedule": "noise_schedule",
    "sample_timestep": "noise_schedule",
    "vpred_to_eps": "noise_schedule",
    "vpred_to_x0": "noise_schedule",
    # LR schedules
    "make_cosine_lr": "schedules",
    "make_lr_schedule": "schedules",
    "make_poly_lr": "schedules",
    # optimizers
    "CPUAdamW": "optimizers",
    "ChunkedXPUAdafactor": "optimizers",
    "FusedXPUAdafactor": "optimizers",
    # UNet wrapper
    "ComfyUNetWrapper": "unet_wrapper",
    "make_rand_cond": "unet_wrapper",
    # LoRA
    "LoRAConfig": "lora",
    "LoRALinear": "lora",
    "inject_lora_into_unet": "lora",
    "extract_lora_weights": "lora",
}

#: Re-exports that are aliases for another name rather than real
#: attributes of their module. ``load_config`` has been ``read_config``
#: for a long time and is kept so old callers keep working.
_ALIASES: dict[str, str] = {"load_config": "read_config"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resolve a re-exported name on first use (PEP 562)."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, _ALIASES.get(name, name))
    globals()[name] = value  # cache it: subsequent lookups skip this
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover -- import-time cost is the point
    from .cache_utils import (
        resolve_gen_batch_size,
        shuffle_and_rebatch_cache,
        warn_batch_mismatch,
    )
    from nodes.config_io import (
        config_from_toml_string,
        config_to_toml_string,
        read_config,
        read_config as load_config,
        upgrade_config_file,
        write_config,
        write_default_config,
    )
    from nodes.config_model import (
        CacheConfig,
        CommonSettings,
        CyclicTuning,
        DistillationTuning,
        FullTuning,
        LoRATuning,
        ModelPaths,
        RandomCache,
        TrainingConfig,
        TrajectoryCache,
        TuningMethod,
    )
    from .lora import (
        LoRAConfig,
        LoRALinear,
        extract_lora_weights,
        inject_lora_into_unet,
    )
    from .model_io import (
        comfy_input_transform,
        make_init_noise,
        raw_to_denoised,
        raw_to_target,
    )
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
    from .optimizers import CPUAdamW, ChunkedXPUAdafactor, FusedXPUAdafactor
    from .schedules import make_cosine_lr, make_lr_schedule, make_poly_lr
    from .unet_wrapper import ComfyUNetWrapper, make_rand_cond