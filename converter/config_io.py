"""TOML I/O for TrainingConfig with flat-to-section migration."""

import copy
import shutil
import tomllib
import tomli_w
from pathlib import Path
from typing import Any

from .config_model import TrainingConfig


# ─── Flat-to-Section Key Mapping ───
# Migrates old flat-format keys to section-based format.

FLAT_TO_SECTION_MAP: dict[str, str] = {
    # mode -> tuning.method
    "mode": "tuning.method",

    # lora keys -> tuning.*
    "lora_rank": "tuning.rank",
    "lora_alpha": "tuning.alpha",
    "lora_dropout": "tuning.dropout",
    "lora_output": "tuning.lora_output",
    "lora_target_all": "tuning.target_all",

    # cyclic keys -> tuning.*
    "cycle_steps": "tuning.cycle_steps",
    "cycle_state_decay": "tuning.cycle_state_decay",

    # cache keys -> cache.*
    "cache_mode": "cache.mode",
    "cache_batch": "cache.batch",
    "cache_batch_size": "cache.batch_size",
    "cache_traj_steps_min": "cache.traj_steps_min",
    "cache_traj_steps_max": "cache.traj_steps_max",
    "cache_traj_skip_steps": "cache.traj_skip_steps",
    "cache_sequence_size": "cache.sequence_size",
    "cache_sequence_mode": "cache.sequence_mode",
    "cache_cfg": "cache.cfg",
    "cache_cond_mode": "cache.cond_mode",
    "cache_positive_prompt": "cache.positive_prompt",
    "cache_negative_prompt": "cache.negative_prompt",
    "cache_student_mix": "cache.student_mix",
    "cache_student_anchor_steps": "cache.student_anchor_steps",
    "cache_student_chain_len": "cache.student_chain_len",
    "cache_student_chain_noise": "cache.student_chain_noise",

    # path keys -> paths.*
    "base_model": "paths.base_model",
    "base": "paths.base_model",
    "teacher": "paths.base_model",
    "output": "paths.checkpoint_output",
    "student": "paths.student",
    "dataset_name": "paths.dataset_name",
    "resume_checkpoint": "paths.resume_checkpoint",
    "resume_optimizer": "paths.resume_optimizer",
    "comfy_dir": "paths.comfy_dir",

    # common keys -> common.*
    "steps": "common.steps",
    "batch_size": "common.batch_size",
    "grad_accum": "common.grad_accum",
    "seed": "common.seed",
    "lr": "common.lr",
    "optimizer": "common.optimizer",
    "adafactor_scale_param": "common.adafactor_scale_param",
    "save_every": "common.save_every",
    "lr_schedule": "common.lr_schedule",
    "lr_end": "common.lr_end",
    "lr_power": "common.lr_power",
    "lr_warmup_steps": "common.lr_warmup_steps",
    "lr_warmup_start": "common.lr_warmup_start",
    "lr_strategy": "common.lr_strategy",
    "center_mult": "common.center_mult",
    "side_mult": "common.side_mult",
    "time_mult": "common.time_mult",
    "snr_weighting": "common.snr_weighting",
    "t_mode": "common.t_mode",
    "t_low": "common.t_low",
    "t_high": "common.t_high",
    "device": "common.device",
    "no_compile": "common.no_compile",
    "no_checkpoint": "common.no_checkpoint",
    "dump_cache_samples": "common.dump_cache_samples",
    "save_on_crash": "common.save_on_crash",
    "teacher_type": "common.teacher_type",
    "student_type": "common.student_type",
    "resume_step": "common.resume_step",
    "cache_latent_size": "common.latent_size",
    "cfg_aware": "common.cfg_aware",
    "training_cfg_min": "common.training_cfg_min",
    "training_cfg_max": "common.training_cfg_max",
    "training_positive_prompt": "common.training_positive_prompt",
    "training_negative_prompt": "common.training_negative_prompt",
}

_FLAT_KEYS = set(FLAT_TO_SECTION_MAP.keys())

# Fields that belong to cache section (so we know which section-specific keys
# end up in the cache discriminator)
_CACHE_MODE_KEYS_TRAJECTORY = {
    "cache.batch_size", "cache.traj_steps_min", "cache.traj_steps_max",
    "cache.traj_skip_steps", "cache.sequence_size", "cache.sequence_mode",
    "cache.cfg", "cache.cond_mode", "cache.positive_prompt", "cache.negative_prompt",
    "cache.student_mix", "cache.student_anchor_steps", "cache.student_chain_len",
    "cache.student_chain_noise",
}
_CACHE_MODE_KEYS_RANDOM = {
    "cache.batch",
}


def _migrate_flat_to_nested(data: dict) -> dict:
    """Migrate a flat legacy dict to the new section-based structure."""
    migrated = {
        "paths": {},
        "common": {},
        "tuning": {},
        "cache": {},
    }

    # First, handle tuning method (formerly 'mode')
    mode = data.get("mode", "distillation")
    migrated["tuning"]["method"] = mode

    # Then iterate over all keys and use the mapping
    for k, v in data.items():
        if k in FLAT_TO_SECTION_MAP:
            new_path = FLAT_TO_SECTION_MAP[k]
            parts = new_path.split(".")
            current = migrated
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = v
        elif k in ["start_from", "reset_optimizer"]:
            migrated[k] = v
        # Sections that might already be partially present
        elif k in ["paths", "common", "tuning", "cache"] and isinstance(v, dict):
            # Shallow merge
            migrated[k].update(v)

    return migrated


def _is_flat_format(data: dict) -> bool:
    """Detect if the TOML data uses the old flat format."""
    return bool(_FLAT_KEYS & data.keys())


def _migrate_dict(data: dict) -> dict:
    """Recursively migrate old keys to new names and structure."""
    
    # 1. If it's a legacy flat dict, use the full mapper
    if _is_flat_format(data):
        return _migrate_flat_to_nested(data)

    # 2. Even if it looks sectioned, some keys might be at the root
    # or need renaming inside sections.
    #
    # Work on a deep copy from here on. Everything below mutates `data`
    # in place and returns the same object -- if we didn't copy first,
    # `migrated == data` in upgrade_config_file() would be comparing the
    # object to itself (always True), so a real migration (e.g.
    # paths.teacher -> paths.base_model) would never be detected as a
    # change and would silently never get written back to disk.
    data = copy.deepcopy(data)

    # Check for root-level keys that should be in common/paths/tuning
    keys_to_migrate = [k for k in data.keys() if k in FLAT_TO_SECTION_MAP]
    for k in keys_to_migrate:
        new_path = FLAT_TO_SECTION_MAP[k]
        parts = new_path.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        # If it's a flat key, it should probably override the section key
        # to ensure that if someone added a flat key to an existing config,
        # it is respected.
        current[parts[-1]] = data.pop(k)

    # 3. Check for old keys inside sections
    if "paths" in data and isinstance(data["paths"], dict):
        p = data["paths"]
        if "teacher" in p and "base_model" not in p:
            p["base_model"] = p.pop("teacher")
        if "base" in p and "base_model" not in p:
            p["base_model"] = p.pop("base")
        if "output" in p and "checkpoint_output" not in p:
            p["checkpoint_output"] = p.pop("output")
        if "training_set_name" in p:
            p.pop("training_set_name")

    if "tuning" in data and isinstance(data["tuning"], dict):
        t = data["tuning"]
        if "output" in t and "lora_output" not in t:
            t["lora_output"] = t.pop("output")

    return data


def upgrade_config_file(path: Path, backup: bool = True) -> bool:
    """Upgrade a config file to the latest format in-place.
    
    Returns True if any migration was performed.
    """
    if not path.exists():
        return False

    with open(path, "rb") as f:
        data = tomllib.load(f)

    migrated = _migrate_dict(data)
    
    # Simple way to check if anything changed: compare keys or structure
    # For now, we'll just assume if it was flat it changed, or if it has 'teacher'
    if migrated == data:
        return False

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
        print(f"  Backup created: {backup_path}")

    with open(path, "wb") as f:
        tomli_w.dump(migrated, f)
    print(f"  Config migrated to latest format: {path}")
    return True


def read_config(path: str | Path) -> TrainingConfig:
    """Read and validate a TOML config file.
    
    Auto-migrates from old format to new section-based format.
    """
    p = Path(path).resolve()
    if not p.exists():
        return TrainingConfig()

    with open(p, "rb") as f:
        data = tomllib.load(f)

    # Perform migration
    migrated_data = _migrate_dict(data)
    
    return TrainingConfig.model_validate(migrated_data)


def write_config(path: str | Path, config: TrainingConfig):
    """Write a TrainingConfig to a TOML file atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="python", exclude_none=True)

    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        tomli_w.dump(data, f)
    tmp_path.replace(p)


def write_default_config(path: str | Path):
    """Write a config file with all defaults."""
    write_config(path, TrainingConfig())
    print(f"Default config written to: {path}")


def config_to_toml_string(config: TrainingConfig) -> str:
    """Serialize config to TOML string (for raw text editing)."""
    data = config.model_dump(mode="python", exclude_none=True)
    return tomli_w.dumps(data)


def config_from_toml_string(text: str) -> TrainingConfig:
    """Parse TOML string into TrainingConfig."""
    import tomllib
    data = tomllib.loads(text)
    if _is_flat_format(data):
        data = _migrate_flat_to_nested(data)
    return TrainingConfig.model_validate(data)
