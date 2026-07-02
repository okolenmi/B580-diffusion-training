"""UI option metadata — separate from config models.

Each option is keyed by its dotted config path.
Visibility rules use the same dotted paths so the frontend can evaluate them.
Fields are grouped into four logical sections:
  1. MODE       — structural choices (training method, cache) + their params
  2. DATA       — what to train from (models, dataset, architecture)
  3. TRAINING   — universal training parameters (steps, lr, loss, system)
  4. LAUNCH     — per-launch options (continue from, reset optimizer)

Groups are numbered so the frontend sorts them alphabetically.
"""

from __future__ import annotations

from typing import Any

# Each option entry mirrors what the frontend expects:
#   id (optional, defaults to key), label, type, default, choices,
#   min, max, step, placeholder, help, group, visible_when
OptionDef = dict[str, Any]

# Convenience: tuning.method condition used by many fields
_MODE_DISTILL = ("distillation", "cyclic", "lora")
_MODE_ALL = ("distillation", "cyclic", "lora", "full")
_MODE_TRAJ = {"tuning.method": ["distillation", "cyclic", "lora"], "cache.mode": "trajectory", "common.data_source": "teacher"}
_MODE_RANDOM = {"tuning.method": ["distillation", "cyclic", "lora"], "cache.mode": "random", "common.data_source": "teacher"}

OPTION_TREE: dict[str, OptionDef] = {
    # ─── 1. MODE — the structural choice that determines everything ───
    "tuning.method": {
        "label": "Training Method", "type": "select",
        "choices": [
            {"value": "distillation", "label": "Distillation (single pass)"},
            {"value": "cyclic",       "label": "Cyclic (recommended)"},
            {"value": "lora",         "label": "LoRA (Low-Rank Adaptation)"},
            {"value": "full",         "label": "Full Fine-Tune"},
        ],
        "group": "1. MODE",
        "help": "Which training algorithm to use — this determines which other options are relevant.",
    },

    # LoRA-specific
    "tuning.rank": {
        "label": "LoRA Rank", "type": "number",
        "min": 1, "max": 256, "step": 1,
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "Inner dimension of the low-rank decomposition (default 64, range 1-256). Higher = more trainable params, stronger adaptation, larger output file. Filesize scales roughly linearly with rank.",
    },
    "tuning.alpha": {
        "label": "LoRA Alpha", "type": "number",
        "min": 0.01, "max": 128.0, "step": 0.01,
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "Scales the LoRA update before adding: W_new = W + (alpha/rank) * BA (default 1.0). Higher alpha = stronger influence from the LoRA weights.",
    },
    "tuning.dropout": {
        "label": "LoRA Dropout", "type": "number",
        "min": 0.0, "max": 0.9, "step": 0.05,
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "Drop chance for LoRA parameters during training (default 0.0 = off). Each LoRA weight has this probability of being zeroed per step. 0.1 = 10% chance. Helps prevent overfitting on small datasets.",
    },
    "tuning.lora_output": {
        "label": "LoRA Output Path", "type": "text",
        "placeholder": "models/loras/my_lora.safetensors",
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "Where to save the trained LoRA weights.",
    },
    "tuning.target_all": {
        "label": "Full LoRA (All Layers)", "type": "checkbox",
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "If enabled, inject LoRA into every Linear and Conv2d layer in the weighted blocks. Warning: This is much slower and requires more VRAM.",
    },
    "tuning.lora_continue_from": {
        "label": "LoRA Continue From", "type": "text",
        "placeholder": "path/to/existing_lora.safetensors",
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora", "start_from": "lora_checkpoint"},
        "help": "Optional: Path to an existing LoRA adapter file to resume training from. If empty or invalid, training starts from scratch.",
    },
    "tuning.block_weighting": {
        "label": "Block Weighting", "type": "text",
        "placeholder": "input_blocks.7:1.0, middle_block:0.5, output_blocks.0:0.0",
        "group": "1. MODE",
        "visible_when": {"tuning.method": "lora"},
        "help": "Optional comma-separated list of block multipliers (0.0 to 1.0). Format: 'block_id:weight'. Names: input_blocks.N, middle_block, output_blocks.N. Weight 0.0 disables a block entirely.",
    },

    # Cyclic-specific
    "tuning.cycle_steps": {
        "label": "Steps Per Cycle", "type": "number",
        "min": 50, "max": 5000, "step": 50,
        "group": "1. MODE",
        "visible_when": {"tuning.method": "cyclic"},
        "help": "Steps between cache rebuilds in cyclic mode.",
    },
    "tuning.cycle_state_decay": {
        "label": "Cycle State Decay", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.1,
        "group": "1. MODE",
        "visible_when": {"tuning.method": "cyclic"},
        "help": "Decay optimizer states between cycles. 1.0 = keep states, 0.0 = full reset.",
    },

    # Cache mode — only relevant for distillation/cyclic with teacher data
    "cache.mode": {
        "label": "Cache Mode", "type": "select",
        "choices": [
            {"value": "trajectory", "label": "Trajectory"},
            {"value": "random",     "label": "Random"},
        ],
        "group": "1. MODE",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora"], "common.data_source": "teacher"},
    },

    # Trajectory cache
    "cache.batch_size": {
        "label": "Traj Cache Batch", "type": "number",
        "min": 0, "max": 16,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
        "help": "0 or empty = use training batch size.",
    },
    "cache.traj_steps_min": {
        "label": "Traj Min Steps", "type": "number",
        "min": 5, "max": 100,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
    },
    "cache.traj_steps_max": {
        "label": "Traj Max Steps", "type": "number",
        "min": 5, "max": 100,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
    },
    "cache.traj_skip_steps": {
        "label": "Traj Skip Steps", "type": "number",
        "min": 0, "max": 20,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
        "help": "Skip first N steps of each trajectory.",
    },
    "cache.sequence_size": {
        "label": "Traj Seq Size", "type": "number",
        "min": 0, "max": 50,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
        "help": "Samples per trajectory. 0 = all.",
    },
    "cache.sequence_mode": {
        "label": "Seq Selection", "type": "select",
        "choices": [
            {"value": "random",     "label": "Random"},
            {"value": "span",       "label": "Span"},
            {"value": "span_high",  "label": "Span High-t"},
            {"value": "span_mid",   "label": "Span Mid-t"},
            {"value": "span_low",   "label": "Span Low-t"},
        ],
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
    },
    "cache.cfg": {
        "label": "Cache CFG", "type": "number",
        "min": 1.0, "max": 20.0, "step": 0.1,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
        "help": "CFG scale during trajectory generation.",
    },
    "cache.cond_mode": {
        "label": "Cond Mode", "type": "select",
        "choices": [
            {"value": "random", "label": "Random"},
            {"value": "zero",   "label": "Zero (uncond)"},
            {"value": "prompt", "label": "Prompt-based"},
        ],
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
    },
    "cache.positive_prompt": {
        "label": "Pos Prompt", "type": "text",
        "group": "1. MODE",
        "visible_when": {"cache.mode": "trajectory", "cache.cond_mode": "prompt"},
    },
    "cache.negative_prompt": {
        "label": "Neg Prompt", "type": "text",
        "group": "1. MODE",
        "visible_when": {"cache.mode": "trajectory", "cache.cond_mode": "prompt"},
    },
    "cache.student_mix": {
        "label": "Student Mix (DAgger)", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.05,
        "group": "1. MODE",
        "visible_when": _MODE_TRAJ,
        "help": "Fraction of student latents in cache.",
    },
    "cache.student_anchor_steps": {
        "label": "Anchor Steps", "type": "number",
        "min": 1, "max": 20,
        "group": "1. MODE",
        "visible_when": {"cache.mode": "trajectory", "cache.student_mix": "__truthy__"},
    },
    "cache.student_chain_len": {
        "label": "Chain Length", "type": "number",
        "min": 1, "max": 20,
        "group": "1. MODE",
        "visible_when": {"cache.mode": "trajectory", "cache.student_mix": "__truthy__"},
    },
    "cache.student_chain_noise": {
        "label": "Chain Noise", "type": "number",
        "min": 0.0, "max": 0.2, "step": 0.01,
        "group": "1. MODE",
        "visible_when": {"cache.mode": "trajectory", "cache.student_mix": "__truthy__"},
    },

    # Random cache
    "cache.batch": {
        "label": "Random Cache Batch", "type": "number",
        "min": 1, "max": 64,
        "group": "1. MODE",
        "visible_when": _MODE_RANDOM,
    },

    # ─── 2. DATA — what source to train from ───
    "common.data_source": {
        "label": "Data Source", "type": "select",
        "choices": [
            {"value": "teacher", "label": "Teacher Model (generate on-the-fly)"},
            {"value": "dataset", "label": "Managed Dataset (pre-existing)"},
        ],
        "group": "2. DATA",
        "help": "Where training data comes from. Teacher generates trajectories via cache; dataset uses pre-curated samples.",
    },
    "paths.base_model": {
        "label": "Base Model", "type": "text",
        "placeholder": "models/checkpoints/...",
        "group": "2. DATA",
        "help": "Source model. Used as teacher (distillation/cyclic) or base (LoRA/full). Required unless starting from a full student checkpoint.",
    },
    "paths.checkpoint_output": {
        "label": "Output Checkpoint", "type": "text",
        "placeholder": "models/checkpoints/...",
        "group": "2. DATA",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "full"]},
        "help": "Where to save the full trained model checkpoint.",
    },
    "paths.student": {
        "label": "Student Init", "type": "text",
        "placeholder": "Optional starting weights",
        "group": "2. DATA",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora"], "common.data_source": "teacher"},
        "help": "Start student from this checkpoint instead of copying teacher.",
    },
    "paths.dataset_name": {
        "label": "Dataset Name", "type": "text",
        "group": "2. DATA",
        "visible_when": {"common.data_source": "dataset"},
        "help": "Name of the managed dataset to use for training.",
    },
    "paths.resume_checkpoint": {
        "label": "Resume Checkpoint", "type": "text",
        "group": "2. DATA",
        "visible_when": {"start_from": "resume"},
        "help": "Internal path for mid-run weight checkpoints. Auto-derived if empty.",
    },
    "paths.resume_optimizer": {
        "label": "Resume Optimizer", "type": "text",
        "group": "2. DATA",
        "visible_when": {"start_from": "resume"},
        "help": "Internal path for mid-run optimizer states. Auto-derived if empty.",
    },
    "paths.comfy_dir": {
        "label": "ComfyUI Directory", "type": "text",
        "group": "2. DATA",
        "help": "Path to ComfyUI (default: auto-detect from current dir or COMFY_DIR env).",
    },
    "common.teacher_type": {
        "label": "Teacher Type", "type": "select",
        "choices": [
            {"value": "vpred", "label": "v-prediction"},
            {"value": "eps",   "label": "epsilon"},
        ],
        "group": "2. DATA",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora"], "common.data_source": "teacher"},
    },
    "common.student_type": {
        "label": "Student Type", "type": "select",
        "choices": [
            {"value": "vpred", "label": "v-prediction"},
            {"value": "eps",   "label": "epsilon"},
        ],
        "group": "2. DATA",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora", "full"]},
    },
    "common.latent_size": {
        "label": "Latent Size (8x = PX)", "type": "number",
        "min": 0, "max": 256, "step": 8,
        "group": "2. DATA",
        "help": "Spatial size (e.g. 64). 0 = auto (64).",
    },
    "common.cfg_aware": {
        "label": "CFG-Aware Tuning", "type": "checkbox",
        "group": "2. DATA",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora"]},
        "help": "Train student to internalize CFG via explicit scale input (adm_in 2816→3072).",
    },
    "common.training_cfg_min": {
        "label": "Training CFG Min", "type": "number",
        "min": 1.0, "max": 20.0, "step": 0.1,
        "group": "2. DATA",
        "visible_when": {"common.cfg_aware": True},
    },
    "common.training_cfg_max": {
        "label": "Training CFG Max", "type": "number",
        "min": 1.0, "max": 20.0, "step": 0.1,
        "group": "2. DATA",
        "visible_when": {"common.cfg_aware": True},
    },
    "common.training_positive_prompt": {
        "label": "Training Pos Prompt", "type": "text",
        "group": "2. DATA",
        "visible_when": {"cache.mode": "trajectory", "cache.cond_mode": "prompt"},
        "help": "Positive prompt for student forward passes. Overrides cache conditioning.",
    },
    "common.training_negative_prompt": {
        "label": "Training Neg Prompt", "type": "text",
        "group": "2. DATA",
        "visible_when": {"cache.mode": "trajectory", "cache.cond_mode": "prompt"},
        "help": "Negative prompt for student forward passes.",
    },

    # ─── 3. TRAINING — universal parameters that apply regardless of mode ───
    "common.steps": {
        "label": "Total Steps", "type": "number",
        "min": 100, "max": 200000, "step": 100,
        "group": "3. TRAINING",
        "help": "Total training steps for this session.",
    },
    "common.batch_size": {
        "label": "Training Batch Size", "type": "number",
        "min": 1, "max": 16,
        "group": "3. TRAINING",
    },
    "common.grad_accum": {
        "label": "Grad Accumulation", "type": "number",
        "min": 1, "max": 32,
        "group": "3. TRAINING",
    },
    "common.seed": {
        "label": "Global Seed", "type": "number",
        "group": "3. TRAINING",
    },
    "common.save_every": {
        "label": "Save Every N Steps", "type": "number",
        "min": 0, "max": 10000, "step": 50,
        "group": "3. TRAINING",
        "help": "Save checkpoint every N steps. 0 = only at end.",
    },
    "common.lr": {
        "label": "Learning Rate", "type": "number",
        "min": 1e-7, "max": 1e-2, "step": 1e-7,
        "placeholder": "1e-5",
        "group": "3. TRAINING",
    },
    "common.optimizer": {
        "label": "Optimizer", "type": "select",
        "choices": [
            {"value": "fused-adafactor", "label": "Fused Adafactor"},
            {"value": "xpu-adafactor",   "label": "XPU Adafactor"},
            {"value": "adamw",           "label": "AdamW"},
        ],
        "group": "3. TRAINING",
    },
    "common.adafactor_scale_param": {
        "label": "Adafactor Scale Param", "type": "checkbox",
        "group": "3. TRAINING",
        "visible_when": {"common.optimizer": ["fused-adafactor", "xpu-adafactor"]},
    },
    "common.lr_schedule": {
        "label": "LR Schedule", "type": "select",
        "choices": [
            {"value": "cosine", "label": "Cosine"},
            {"value": "poly",   "label": "Polynomial"},
        ],
        "group": "3. TRAINING",
    },
    "common.lr_end": {
        "label": "LR End", "type": "number",
        "min": 0.0, "step": 1e-7,
        "group": "3. TRAINING",
        "visible_when": {"common.lr_schedule": "poly"},
    },
    "common.lr_power": {
        "label": "LR Power", "type": "number",
        "min": 0.1, "max": 5.0, "step": 0.1,
        "group": "3. TRAINING",
        "visible_when": {"common.lr_schedule": "poly"},
    },
    "common.lr_warmup_steps": {
        "label": "Warmup Steps", "type": "number",
        "min": 0,
        "group": "3. TRAINING",
    },
    "common.lr_warmup_start": {
        "label": "Warmup Start LR", "type": "number",
        "min": 0.0,
        "group": "3. TRAINING",
    },
    "common.lr_strategy": {
        "label": "Param LR Strategy", "type": "select",
        "choices": [
            {"value": "uniform", "label": "Uniform"},
            {"value": "radial",  "label": "Radial"},
        ],
        "group": "3. TRAINING",
    },
    "common.center_mult": {
        "label": "Center Mult", "type": "number",
        "group": "3. TRAINING",
        "visible_when": {"common.lr_strategy": "radial"},
    },
    "common.side_mult": {
        "label": "Side Mult", "type": "number",
        "group": "3. TRAINING",
        "visible_when": {"common.lr_strategy": "radial"},
    },
    "common.time_mult": {
        "label": "Time Mult", "type": "number",
        "group": "3. TRAINING",
        "visible_when": {"common.lr_strategy": "radial"},
    },
    "common.snr_weighting": {
        "label": "SNR Weighting", "type": "select",
        "choices": [
            {"value": "uniform",     "label": "Uniform"},
            {"value": "snr",         "label": "Min-SNR"},
            {"value": "inverse_snr", "label": "Inverse SNR"},
            {"value": "decay_snr",   "label": "Decay SNR (Hybrid)"},
        ],
        "group": "3. TRAINING",
    },
    "common.t_mode": {
        "label": "T-Sampling Mode", "type": "select",
        "choices": [
            {"value": "uniform", "label": "Uniform"},
            {"value": "low",     "label": "Low-t (Details)"},
            {"value": "mid",     "label": "Mid-t (Balance)"},
            {"value": "high",    "label": "High-t (Structure)"},
            {"value": "logit",   "label": "Logit-normal"},
        ],
        "group": "3. TRAINING",
    },
    "common.t_low": {
        "label": "T-Low", "type": "number",
        "min": 0, "max": 999,
        "group": "3. TRAINING",
    },
    "common.t_high": {
        "label": "T-High", "type": "number",
        "min": 0, "max": 999,
        "group": "3. TRAINING",
    },
    "common.device": {
        "label": "Device", "type": "text",
        "group": "3. TRAINING",
    },
    "common.no_compile": {
        "label": "No Compile", "type": "checkbox",
        "group": "3. TRAINING",
    },
    "common.no_checkpoint": {
        "label": "No Checkpoint", "type": "checkbox",
        "group": "3. TRAINING",
    },
    "common.dump_cache_samples": {
        "label": "Dump Cache Samples", "type": "checkbox",
        "group": "3. TRAINING",
    },
    "common.save_on_crash": {
        "label": "Save on Crash", "type": "checkbox",
        "group": "3. TRAINING",
        "help": "Save checkpoint when pressing Ctrl+\\ (stop button).",
    },
    "common.pre_cond_enable": {
        "label": "Adversarial Pre-cond", "type": "checkbox",
        "group": "3. TRAINING",
        "help": "Enable adversarial pre-conditioning (drafts from opponent). Requires cond AND uncond targets.",
    },
    "common.pre_cond_power_min": {
        "label": "Pre-cond Power Min", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.05,
        "group": "3. TRAINING",
        "visible_when": {"common.pre_cond_enable": True},
    },
    "common.pre_cond_power_max": {
        "label": "Pre-cond Power Max", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.05,
        "group": "3. TRAINING",
        "visible_when": {"common.pre_cond_enable": True},
    },
    "common.pre_cond_clean_ratio": {
        "label": "Pre-cond Clean Ratio", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.05,
        "group": "3. TRAINING",
        "visible_when": {"common.pre_cond_enable": True},
        "help": "Fraction of steps to run without pre-conditioning.",
    },
    "common.resume_step": {
        "label": "Resume Step Override", "type": "number",
        "group": "3. TRAINING",
        "help": "Override starting step counter (0 = auto).",
    },
}

# Per-launch options — not part of the config model, shown at the bottom.
SYNTHETIC_OPTIONS: list[OptionDef] = [
    {
        "id": "start_from",
        "label": "Continue From",
        "type": "select",
        "default": "teacher",
        "choices": [
            {"value": "teacher",        "label": "Teacher Checkpoint"},
            {"value": "student",        "label": "Student Checkpoint"},
            {"value": "resume",         "label": "Resume (Auto-pick)"},
        ],
        "group": "4. LAUNCH",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "full"]},
        "help": "Starting point for this training run.",
    },
    {
        "id": "start_from",
        "label": "Continue From",
        "type": "select",
        "default": "teacher",
        "choices": [
            {"value": "teacher",        "label": "Base Model (New LoRA)"},
            {"value": "resume",         "label": "Resume (Auto-pick)"},
            {"value": "lora_checkpoint","label": "LoRA Checkpoint (Manual)"},
        ],
        "group": "4. LAUNCH",
        "visible_when": {"tuning.method": "lora"},
        "help": "Starting point for this LoRA training run.",
    },
    {
        "id": "reset_optimizer",
        "label": "Reset Optimizer",
        "type": "checkbox",
        "default": False,
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora", "full"], "start_from": "resume"},
        "group": "4. LAUNCH",
        "help": "Discard saved optimizer states, start fresh.",
    },
]
