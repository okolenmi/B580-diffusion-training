"""UI-specific metadata that cannot be derived from the config schema.

This used to be a 558-line file that hand-duplicated every field's type,
default, min/max, and choices alongside core/config_model.py -- the two were
never enforced to agree, and in fact had already drifted (common.snr_weighting
was missing the "min_snr_5" choice that core/config_model.py has always had).

All of that -- path, type, default, min/max, raw choice values, and a base
visible_when for fields that only exist on one variant of a discriminated
union -- is now derived directly from the Pydantic model by config_schema.py.
It cannot drift, because it IS the schema.

What's left here is only what a schema genuinely cannot know:
  - label: display name (falls back to the raw field name if omitted)
  - help: falls back to the Pydantic field's own `description=...` if omitted
  - group: which UI section this field belongs to
  - placeholder / step: pure form cosmetics
  - choice_labels: friendly display names for enum values (e.g.
    "fused-adafactor" -> "Fused Adafactor"); falls back to a naive
    title-cased version of the raw value if omitted
  - extra_visible_when: additional visibility conditions *beyond* "which
    union variant is this field on" -- e.g. cache fields also depend on
    common.data_source == "teacher", which is cross-cutting business logic
    no schema can infer on its own

See options.py for how this gets merged with config_schema.py's output.
"""

from __future__ import annotations

from typing import Any

ExtraDef = dict[str, Any]

# Keyed by the same dotted config path as config_schema.py's output.
# Groups: "1. MODE", "2. DATA", "3. TRAINING" (numbered so the frontend
# sorts them in this order; "0. LAUNCH" is reserved for SYNTHETIC_OPTIONS -- sorted
# first since deciding what to continue from is naturally the first choice,
# not the last
# below, which aren't real config fields).
EXTRAS: dict[str, ExtraDef] = {
    'tuning.method': {'label': 'Training Method',
 'help': 'Which training algorithm to use — this determines which other options are '
         'relevant.',
 'group': '0. LAUNCH',
 'order': 0,
 'choice_order': ['distillation', 'cyclic', 'lora', 'full'],
 'choice_labels': {'distillation': 'Distillation (single pass)',
                   'cyclic': 'Cyclic (recommended)',
                   'lora': 'LoRA (Low-Rank Adaptation)',
                   'full': 'Full Fine-Tune'}},
    'tuning.rank': {'label': 'LoRA Rank',
 'help': 'Inner dimension of the low-rank decomposition (default 64, range 1-256). '
         'Higher = more trainable params, stronger adaptation, larger output file. '
         'Filesize scales roughly linearly with rank.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'step': 1},
    'tuning.alpha': {'label': 'LoRA Alpha',
 'help': 'Scales the LoRA update before adding: W_new = W + (alpha/rank) * BA (default '
         '1.0). Higher alpha = stronger influence from the LoRA weights.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'step': 0.01},
    'tuning.dropout': {'label': 'LoRA Dropout',
 'help': 'Drop chance for LoRA parameters during training (default 0.0 = off). Each '
         'LoRA weight has this probability of being zeroed per step. 0.1 = 10% chance. '
         'Helps prevent overfitting on small datasets.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'step': 0.05},
    'tuning.lora_output': {'label': 'LoRA Output Path',
 'help': 'Where to save the trained LoRA weights. A filename is saved relative to '
         'the LoRAs directory; a full path is used as-is.',
 'placeholder': 'my_character_v1.safetensors',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'file_kind': 'lora'},
    'tuning.target_all': {'label': 'Full LoRA (All Layers)',
 'help': 'If enabled, inject LoRA into every Linear and Conv2d layer in the weighted '
         'blocks. Warning: This is much slower and requires more VRAM.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings'},
    'tuning.lora_continue_from': {'label': 'LoRA Continue From',
 'help': 'Optional: Path to an existing LoRA adapter file to resume training from. If '
         'empty or invalid, training starts from scratch. Pick from the dropdown, or '
         'type a filename (relative to the LoRAs directory) or a full absolute path.',
 'placeholder': 'existing_lora.safetensors',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'file_kind': 'lora',
 'extra_visible_when': {'start_from': 'lora_checkpoint'}},
    'tuning.block_weighting': {'label': 'Block Weighting',
 'help': 'Optional comma-separated list of block multipliers (0.0 to 1.0). Format: '
         "'block_id:weight'. Names: input_blocks.N, middle_block, output_blocks.N. "
         'Weight 0.0 disables a block entirely.',
 'placeholder': 'input_blocks.7:1.0, middle_block:0.5, output_blocks.0:0.0',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings'},
    'tuning.gate_enabled': {'label': 'Enable Timestep Gating',
 'help': 'Off by default -- LoRA applies uniformly across all timesteps. Turn this on '
         'to scale the LoRA delta toward ~1 inside [Gate: Train Range Low, Gate: Train '
         'Range High] (your dataset\'s actual t range) and toward ~0 outside it.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings'},
    'tuning.gate_train_low': {'label': 'Gate: Train Range Low (t)',
 'help': 'Lower bound of the timestep range where the LoRA trains at full strength.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'extra_visible_when': {'tuning.gate_enabled': True}},
    'tuning.gate_train_high': {'label': 'Gate: Train Range High (t)',
 'help': 'Upper bound of the timestep range where the LoRA trains at full strength.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'extra_visible_when': {'tuning.gate_enabled': True}},
    'tuning.gate_width': {'label': 'Gate: Transition Width',
 'help': 'Controls how sharp the fade is at each edge of the training range above -- '
         'smaller = sharper cutoff, larger = more gradual. If comparable to or larger '
         'than the training range itself, the middle of that range won\'t reach full '
         'LoRA strength; use a smaller width for a narrow range.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'extra_visible_when': {'tuning.gate_enabled': True}},
    'tuning.cycle_steps': {'label': 'Steps Per Cycle',
 'help': 'Steps between cache rebuilds in cyclic mode.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'step': 50},
    'tuning.cycle_state_decay': {'label': 'Cycle State Decay',
 'help': 'Decay optimizer states between cycles. 1.0 = keep states, 0.0 = full reset.',
 'group': '0. LAUNCH',
 'subgroup': 'Method-Specific Settings',
 'step': 0.1},
    'cache.mode': {'label': 'Cache Mode',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.batch_size': {'label': 'Traj Cache Batch',
 'help': '0 or empty = use training batch size.',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.traj_steps_min': {'label': 'Traj Min Steps',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.traj_steps_max': {'label': 'Traj Max Steps',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.traj_skip_steps': {'label': 'Traj Skip Steps',
 'help': 'Skip first N steps of each trajectory.',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.sequence_size': {'label': 'Traj Seq Size',
 'help': 'Samples per trajectory. 0 = all.',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.sequence_mode': {'label': 'Seq Selection',
 'group': '6. ADVANCED',
 'choice_labels': {'span_high': 'Span High-t',
                   'span_mid': 'Span Mid-t',
                   'span_low': 'Span Low-t'},
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.cfg': {'label': 'Cache CFG',
 'help': 'CFG scale during trajectory generation (used when Cache CFG Random is off).',
 'group': '6. ADVANCED',
 'step': 0.1,
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.cfg_random': {'label': 'Cache CFG Random',
 'help': 'Distillation with CFG: draw a random CFG scale per trajectory from '
         '[min, max] instead of the fixed value above, to diversify training '
         'targets across guidance strengths. More expensive.',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.cfg_min': {'label': 'Cache CFG Min',
 'group': '6. ADVANCED',
 'step': 0.1,
 'extra_visible_when': {'cache.cfg_random': True}},
    'cache.cfg_max': {'label': 'Cache CFG Max',
 'group': '6. ADVANCED',
 'step': 0.1,
 'extra_visible_when': {'cache.cfg_random': True}},
    'cache.cond_mode': {'label': 'Cond Mode',
 'group': '6. ADVANCED',
 'choice_labels': {'zero': 'Zero (uncond)', 'prompt': 'Prompt-based'},
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.positive_prompt': {'label': 'Pos Prompt',
 'group': '6. ADVANCED',
 'extra_visible_when': {'cache.cond_mode': 'prompt'}},
    'cache.negative_prompt': {'label': 'Neg Prompt',
 'group': '6. ADVANCED',
 'extra_visible_when': {'cache.cond_mode': 'prompt'}},
    'cache.student_mix': {'label': 'Student Mix (DAgger)',
 'help': 'Fraction of student latents in cache.',
 'group': '6. ADVANCED',
 'step': 0.05,
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'cache.student_anchor_steps': {'label': 'Anchor Steps',
 'group': '6. ADVANCED',
 'extra_visible_when': {'cache.student_mix': '__truthy__'}},
    'cache.student_chain_len': {'label': 'Chain Length',
 'group': '6. ADVANCED',
 'extra_visible_when': {'cache.student_mix': '__truthy__'}},
    'cache.student_chain_noise': {'label': 'Chain Noise',
 'group': '6. ADVANCED',
 'step': 0.01,
 'extra_visible_when': {'cache.student_mix': '__truthy__'}},
    'cache.batch': {'label': 'Random Cache Batch',
 'group': '6. ADVANCED',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'common.data_source': {'label': 'Data Source',
 'help': 'Where training data comes from. Teacher generates trajectories via cache; '
         'dataset uses pre-curated samples.',
 'group': '3. MODEL & DATA',
 'order': 0,
 'choice_labels': {'teacher': 'Teacher Model (generate on-the-fly)',
                   'dataset': 'Managed Dataset (pre-existing)'}},
    'paths.base_model': {'label': 'Base Model',
 'help': 'Source model. Used as teacher (distillation/cyclic) or base (LoRA/full). '
         'Required unless starting from a full student checkpoint. Pick from the '
         'dropdown, or type a filename (relative to the checkpoints directory) or a '
         'full absolute path.',
 'placeholder': 'sdxl_base.safetensors',
 'group': '3. MODEL & DATA',
 'file_kind': 'checkpoint'},
    'paths.checkpoint_output': {'label': 'Output Checkpoint',
 'help': 'Where to save the full trained model checkpoint. A filename is saved '
         'relative to the checkpoints directory; a full path is used as-is.',
 'placeholder': 'my_run/checkpoint.safetensors',
 'group': '3. MODEL & DATA',
 'file_kind': 'checkpoint',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'full']}},
    'paths.student': {'label': 'Student Init',
 'help': 'Start student from this checkpoint instead of copying teacher. Pick from '
         'the dropdown, or type a filename (relative to the checkpoints directory) '
         'or a full absolute path.',
 'placeholder': 'Optional starting weights',
 'group': '3. MODEL & DATA',
 'file_kind': 'checkpoint',
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'paths.dataset_name': {'label': 'Dataset Name',
 'help': 'Name of the managed dataset to use for training.',
 'group': '3. MODEL & DATA',
 'order': 1,
 'extra_visible_when': {'common.data_source': 'dataset'}},
    'paths.resume_checkpoint': {'label': 'Resume Checkpoint',
 'help': 'Internal path for mid-run weight checkpoints. Auto-derived if empty.',
 'group': '5. SYSTEM',
 'extra_visible_when': {'start_from': 'resume'}},
    'paths.resume_optimizer': {'label': 'Resume Optimizer',
 'help': 'Internal path for mid-run optimizer states. Auto-derived if empty.',
 'group': '5. SYSTEM',
 'extra_visible_when': {'start_from': 'resume'}},
    'paths.comfy_dir': {'label': 'ComfyUI Directory',
 'help': 'Path to ComfyUI (default: auto-detect from current dir or COMFY_DIR env).',
 'group': '3. MODEL & DATA'},
    'common.teacher_type': {'label': 'Teacher Type',
 'group': '3. MODEL & DATA',
 'choice_labels': {'vpred': 'v-prediction', 'eps': 'epsilon'},
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora'],
                        'common.data_source': 'teacher'}},
    'common.student_type': {'label': 'Student Type',
 'group': '3. MODEL & DATA',
 'choice_labels': {'vpred': 'v-prediction', 'eps': 'epsilon'},
 'extra_visible_when': {'tuning.method': ['distillation', 'cyclic', 'lora', 'full']}},
    'common.latent_size': {'label': 'Latent Size (8x = PX)',
 'help': 'Spatial size (e.g. 64). 0 = auto (64).',
 'group': '3. MODEL & DATA',
 'step': 8},
    'common.use_dataset_cfg': {'label': 'Use Dataset CFG',
 'help': "Mix each sample's target using its own stored per-trajectory CFG "
         'metadata. Off ignores the stored CFG entirely (equivalent to CFG=1).',
 'group': '4. SAMPLING',
 'extra_visible_when': {'paths.dataset_name': '__truthy__'}},
    'common.training_positive_prompt': {'label': 'Training Pos Prompt',
 'help': 'Positive prompt for student forward passes. Overrides cache conditioning.',
 'group': '4. SAMPLING',
 'extra_visible_when': {'cache.mode': 'trajectory', 'cache.cond_mode': 'prompt'}},
    'common.training_negative_prompt': {'label': 'Training Neg Prompt',
 'help': 'Negative prompt for student forward passes.',
 'group': '4. SAMPLING',
 'extra_visible_when': {'cache.mode': 'trajectory', 'cache.cond_mode': 'prompt'}},
    'common.steps': {'label': 'Total Steps',
 'help': 'Total training steps for this session.',
 'group': '1. TRAINING',
 'step': 100},
    'common.batch_size': {'label': 'Training Batch Size', 'group': '1. TRAINING'},
    'common.grad_accum': {'label': 'Grad Accumulation', 'group': '1. TRAINING'},
    'common.seed': {'label': 'Global Seed', 'group': '1. TRAINING'},
    'common.save_every': {'label': 'Save Every N Steps',
 'help': 'Save checkpoint every N steps. 0 = only at end.',
 'group': '1. TRAINING',
 'step': 50},
    'common.lr': {'label': 'Learning Rate', 'placeholder': '1e-5', 'group': '1. TRAINING', 'step': 1e-07},
    'common.optimizer': {'label': 'Optimizer',
 'group': '1. TRAINING',
 'choice_labels': {'xpu-adafactor': 'XPU Adafactor', 'adamw': 'AdamW'}},
    'common.adafactor_scale_param': {'label': 'Adafactor Scale Param',
 'group': '1. TRAINING',
 'extra_visible_when': {'common.optimizer': ['fused-adafactor', 'xpu-adafactor']}},
    'common.lr_schedule': {'label': 'LR Schedule',
 'group': '1. TRAINING',
 'choice_labels': {'poly': 'Polynomial'}},
    'common.lr_end': {'label': 'LR End',
 'group': '1. TRAINING',
 'step': 1e-07,
 'extra_visible_when': {'common.lr_schedule': 'poly'}},
    'common.lr_power': {'label': 'LR Power',
 'group': '1. TRAINING',
 'step': 0.1,
 'extra_visible_when': {'common.lr_schedule': 'poly'}},
    'common.lr_warmup_steps': {'label': 'Warmup Steps', 'group': '1. TRAINING'},
    'common.lr_warmup_start': {'label': 'Warmup Start LR', 'group': '1. TRAINING'},
    'common.lr_strategy': {'label': 'Param LR Strategy', 'group': '1. TRAINING'},
    'common.center_mult': {'label': 'Center Mult',
 'group': '1. TRAINING',
 'extra_visible_when': {'common.lr_strategy': 'radial'}},
    'common.side_mult': {'label': 'Side Mult',
 'group': '1. TRAINING',
 'extra_visible_when': {'common.lr_strategy': 'radial'}},
    'common.time_mult': {'label': 'Time Mult',
 'group': '1. TRAINING',
 'extra_visible_when': {'common.lr_strategy': 'radial'}},
    'common.snr_weighting': {'label': 'SNR Weighting',
 'group': '4. SAMPLING',
 'choice_labels': {'snr': 'Min-SNR',
                   'inverse_snr': 'Inverse SNR',
                   'decay_snr': 'Decay SNR (Hybrid)'}},
    'common.t_mode': {'label': 'T-Sampling Mode',
 'group': '4. SAMPLING',
 'choice_labels': {'low': 'Low-t (Details)',
                   'mid': 'Mid-t (Balance)',
                   'high': 'High-t (Structure)',
                   'logit': 'Logit-normal'}},
    'common.t_low': {'label': 'T-Low', 'group': '4. SAMPLING'},
    'common.t_high': {'label': 'T-High', 'group': '4. SAMPLING'},
    'common.device': {'label': 'Device', 'group': '1. TRAINING'},
    'common.no_compile': {'label': 'No Compile', 'group': '5. SYSTEM'},
    'common.no_checkpoint': {'label': 'No Checkpoint', 'group': '5. SYSTEM'},
    'common.dump_cache_samples': {'label': 'Dump Cache Samples', 'group': '5. SYSTEM'},
    'common.save_on_crash': {'label': 'Save on Crash',
 'help': 'Save checkpoint when pressing Ctrl+\\ (stop button).',
 'group': '5. SYSTEM'},
    'common.pre_cond_enable': {'label': 'Adversarial Pre-cond',
 'help': 'Enable adversarial pre-conditioning (drafts from opponent). Requires cond '
         'AND uncond targets.',
 'group': '6. ADVANCED'},
    'common.pre_cond_power_min': {'label': 'Pre-cond Power Min',
 'group': '6. ADVANCED',
 'step': 0.05,
 'extra_visible_when': {'common.pre_cond_enable': True}},
    'common.pre_cond_power_max': {'label': 'Pre-cond Power Max',
 'group': '6. ADVANCED',
 'step': 0.05,
 'extra_visible_when': {'common.pre_cond_enable': True}},
    'common.pre_cond_clean_ratio': {'label': 'Pre-cond Clean Ratio',
 'help': 'Fraction of steps to run without pre-conditioning.',
 'group': '6. ADVANCED',
 'step': 0.05,
 'extra_visible_when': {'common.pre_cond_enable': True}},
    'common.resume_step': {'label': 'Resume Step Override',
 'help': 'Override starting step counter (0 = auto).',
 'group': '5. SYSTEM'},
}


# Per-launch options -- NOT real TrainingConfig fields (well, start_from and
# reset_optimizer technically exist as persisted fields too, but that's a
# *different*, launch-transient concept -- see server/routes_training.py's
# docstring and PROGRESS.md for the full story). Deliberately excluded from
# config_schema.py's auto-generated output in options.py so only this
# per-launch version is ever shown.
SYNTHETIC_OPTIONS: list[ExtraDef] = [
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
        "group": "0. LAUNCH",
        "visible_when": {"tuning.method": ["distillation", "cyclic", "full"]},
        "help": "Starting point for this training run.",
        # This field is deliberately never written to the saved config file (see
        # module docstring above) -- it's meant to be picked fresh each launch,
        # not silently inherited by future runs. But that also means it has no
        # server-side "current value" to restore on a new session, so the
        # frontend remembers your last choice in localStorage instead, purely
        # as a UI convenience -- distinct from, and never written into, the
        # actual persisted config.
        "persist_locally": True,
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
        "group": "0. LAUNCH",
        "visible_when": {"tuning.method": "lora"},
        "help": "Starting point for this LoRA training run.",
        "persist_locally": True,
    },
    {
        "id": "reset_optimizer",
        "label": "Reset Optimizer",
        "type": "checkbox",
        "default": False,
        "visible_when": {"tuning.method": ["distillation", "cyclic", "lora", "full"], "start_from": "resume"},
        "group": "0. LAUNCH",
        "help": "Discard saved optimizer states, start fresh.",
        "persist_locally": True,
    },
]
