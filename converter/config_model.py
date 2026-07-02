"""Training configuration models with OOP structure and discriminated unions.

Each training method (LoRA, Cyclic, Distillation, Full) has its own config object.
Cache modes (Trajectory, Random) also use discriminated unions.
Common settings are shared across all methods.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


def _derive_path(base: str, suffix: str) -> str:
    """Derive a sibling path from the output path by replacing the suffix."""
    p = Path(base)
    return str(p.parent / (p.stem + suffix))


# ─── Common Settings (shared across all training methods) ───

class CommonSettings(BaseModel):
    """Settings that apply regardless of training mode."""

    steps: int = Field(default=1000, ge=100, le=200000)
    batch_size: int = Field(default=1, ge=1, le=16)
    grad_accum: int = Field(default=1, ge=1, le=32)

    lr: float = Field(default=1e-5, ge=1e-7, le=1e-2)
    optimizer: Literal["fused-adafactor", "xpu-adafactor", "adamw"] = "fused-adafactor"
    adafactor_scale_param: bool = False

    seed: int = 42
    save_every: int = Field(default=100, ge=0, le=10000)

    lr_schedule: Literal["cosine", "poly"] = "cosine"
    lr_end: float = Field(default=1e-5, ge=0.0)
    lr_power: float = Field(default=2.0, ge=0.1, le=5.0)
    lr_warmup_steps: int = 0
    lr_warmup_start: float = Field(default=0.0, ge=0.0)

    lr_strategy: Literal["uniform", "radial"] = "uniform"
    center_mult: float = 0.67
    side_mult: float = 1.5
    time_mult: float = 1.8

    snr_weighting: Literal["uniform", "snr", "min_snr_5", "inverse_snr", "decay_snr"] = "snr"

    t_mode: Literal["uniform", "low", "mid", "high", "logit"] = "uniform"
    t_low: int = Field(default=0, ge=0, le=999)
    t_high: int = Field(default=999, ge=0, le=999)

    device: str = "xpu"
    no_compile: bool = True
    no_checkpoint: bool = True
    dump_cache_samples: bool = False
    save_on_crash: bool = True

    data_source: Literal["teacher", "dataset"] = Field(
        default="teacher",
        description="Where training data comes from: 'teacher' generates on-the-fly, 'dataset' uses a managed dataset"
    )
    teacher_type: Literal["vpred", "eps"] = "vpred"
    student_type: Literal["vpred", "eps"] = "eps"
    resume_step: int = 0

    latent_size: int = Field(default=0, ge=0, le=256, description="Spatial size (e.g. 64). 0 = auto (64).")

    cfg_aware: bool = False
    training_cfg_min: float = Field(default=1.0, ge=1.0, le=20.0)
    training_cfg_max: float = Field(default=1.0, ge=1.0, le=20.0)

    training_positive_prompt: str = ""
    training_negative_prompt: str = ""

    # Experimental: adversarial pre-conditioning
    # Each cond/uncond pass gets a low-power draft from its opponent before
    # the main forward, giving each signal situational awareness of what it
    # is working against.  A fraction of steps run clean (no draft) to ensure
    # the model never loses its standard inference behaviour.
    pre_cond_enable: bool = False
    pre_cond_power_min: float = Field(default=0.1, ge=0.0, le=1.0)
    pre_cond_power_max: float = Field(default=0.5, ge=0.0, le=1.0)
    pre_cond_clean_ratio: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_fused_grad_accum(self) -> "CommonSettings":
        if self.optimizer == "fused-adafactor" and self.grad_accum > 1:
            import warnings
            warnings.warn(
                f"fused-adafactor does not support gradient accumulation "
                f"(grad_accum={self.grad_accum} will be ignored — effective value is 1). "
                f"Use xpu-adafactor or adamw if you need grad_accum > 1.",
                stacklevel=2,
            )
        return self


# ─── Model Paths ───

class ModelPaths(BaseModel):
    """File paths for models, datasets, and checkpoints."""

    base_model: str = Field(default="", description="Source model to distill from or use as base for LoRA")
    checkpoint_output: str = Field(default="", description="Where to save the full trained model checkpoint")
    student: Optional[str] = Field(default=None, description="Optional starting weights for student")
    dataset_name: Optional[str] = Field(default=None, description="Managed dataset name")
    resume_checkpoint: Optional[str] = Field(default=None, description="Path to resume checkpoint")
    resume_optimizer: Optional[str] = Field(default=None, description="Path to resume optimizer state")
    comfy_dir: Optional[str] = Field(default=None, description="ComfyUI directory override")


# ─── Training Method Configs (discriminated by `method`) ───

class LoRATuning(BaseModel):
    """Settings specific to LoRA (Low-Rank Adaptation) training."""

    method: Literal["lora"] = "lora"
    rank: int = Field(default=64, ge=1, le=256)
    alpha: float = Field(default=1.0, ge=0.01, le=128.0)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    lora_output: str = Field(default="", description="Path to save trained LoRA weights")
    target_all: bool = Field(default=False, description="If True, inject LoRA into every Linear/Conv2d in weighted blocks (Full LoRA).")
    lora_continue_from: Optional[str] = Field(default=None, description="Path to existing LoRA adapter to resume from")
    block_weighting: str = Field(
        default="",
        description="Optional comma-separated list of block multipliers (e.g. 'input_7:1.0,middle:0.5,output_0:0.2')"
    )


class CyclicTuning(BaseModel):
    """Settings specific to Cyclic distillation."""

    method: Literal["cyclic"] = "cyclic"
    cycle_steps: int = Field(default=500, ge=50, le=5000, description="Steps between cache rebuilds")
    cycle_state_decay: float = Field(default=1.0, ge=0.0, le=1.0, description="Optimizer state decay between cycles")


class DistillationTuning(BaseModel):
    """Settings specific to single-pass Distillation."""

    method: Literal["distillation"] = "distillation"


class FullTuning(BaseModel):
    """Settings specific to Full Fine-Tune."""

    method: Literal["full"] = "full"


TuningMethod = Annotated[
    Union[LoRATuning, CyclicTuning, DistillationTuning, FullTuning],
    Field(discriminator="method"),
]


# ─── Cache Configs (discriminated by `mode`) ───

class TrajectoryCache(BaseModel):
    """Settings for trajectory-based cache generation."""

    mode: Literal["trajectory"] = "trajectory"
    batch_size: Optional[int] = Field(default=None, ge=0, le=16, description="0/None = use common.batch_size")
    traj_steps_min: int = Field(default=20, ge=5, le=100)
    traj_steps_max: int = Field(default=30, ge=5, le=100)
    traj_skip_steps: int = Field(default=4, ge=0, le=20, description="Skip first N steps of each trajectory")
    sequence_size: int = Field(default=0, ge=0, le=50, description="Samples per trajectory. 0 = all.")
    sequence_mode: Literal["random", "span", "span_high", "span_mid", "span_low"] = "random"
    cfg: float = Field(default=1.0, ge=1.0, le=20.0, description="CFG scale during generation")
    cond_mode: Literal["random", "zero", "prompt"] = "random"
    positive_prompt: str = ""
    negative_prompt: str = ""

    student_mix: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of student latents (DAgger)")
    student_anchor_steps: int = Field(default=5, ge=1, le=20)
    student_chain_len: int = Field(default=3, ge=1, le=20)
    student_chain_noise: float = Field(default=0.02, ge=0.0, le=0.2)


class RandomCache(BaseModel):
    """Settings for random cache generation."""

    mode: Literal["random"] = "random"
    batch: int = Field(default=8, ge=1, le=64, description="Batch size for random cache generation")


CacheConfig = Annotated[
    Union[TrajectoryCache, RandomCache],
    Field(discriminator="mode"),
]


# ─── Top-Level Training Config ───

class TrainingConfig(BaseModel):
    """Complete training configuration — composition of focused sub-models."""

    paths: ModelPaths = Field(default_factory=ModelPaths)
    common: CommonSettings = Field(default_factory=CommonSettings)
    tuning: TuningMethod = Field(default_factory=DistillationTuning)
    cache: CacheConfig = Field(default_factory=TrajectoryCache)
    # Only "resume" is actually branched on anywhere in Trainer -- any other
    # value behaves identically (fresh run using paths.student / teacher
    # weights / tuning.lora_continue_from as normal init sources). "teacher"
    # and "student" exist purely as descriptive defaults for config files;
    # they're kept as accepted values so existing configs don't fail
    # validation, but they carry no separate behavior of their own.
    start_from: Literal["teacher", "student", "resume"] = Field(
        default="teacher",
        description="'resume' restores optimizer state + step count from "
                     "paths.resume_optimizer/resume_checkpoint. Any other "
                     "value just means: fresh run, step 0.",
    )
    reset_optimizer: bool = Field(default=False, description="Discard saved optimizer states on resume")

    @model_validator(mode="after")
    def _fill_resume_paths(self) -> "TrainingConfig":
        if self.paths.checkpoint_output and not self.paths.resume_checkpoint:
            self.paths.resume_checkpoint = _derive_path(self.paths.checkpoint_output, ".resume.safetensors")
        if self.paths.checkpoint_output and not self.paths.resume_optimizer:
            self.paths.resume_optimizer = _derive_path(self.paths.checkpoint_output, ".resume.optstate")

        # For LoRA, we derive from lora_output
        if self.tuning.method == "lora" and getattr(self.tuning, "lora_output", None):
            if not self.paths.resume_checkpoint:
                self.paths.resume_checkpoint = _derive_path(self.tuning.lora_output, ".resume.safetensors")
            if not self.paths.resume_optimizer:
                self.paths.resume_optimizer = _derive_path(self.tuning.lora_output, ".resume.optstate")
        return self
