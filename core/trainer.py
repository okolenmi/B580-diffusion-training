"""Distillation Trainer — orchestrates cyclic training and cache building.

Accepts typed TrainingConfig instead of argparse.Namespace.
No getattr() calls — all config access is direct and type-safe.
"""

import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from safetensors.torch import load_file

from .cache_random import build_teacher_cache as build_random_cache
from .cache_trajectory import build_teacher_cache_trajectory as build_trajectory_cache
from .comfy_setup import xpu_empty_cache
from .config_model import TrainingConfig, LoRATuning, CyclicTuning
from .lora import LoRAConfig
from .noise_schedule import get_alpha_sigma
from .optimizer_builder import build_optimizer
from .optimizers import FusedXPUAdafactor
from .progress_writer import ProgressWriter
from .save import (
    load_lora_checkpoint,
    save_checkpoint,
    save_lora_checkpoint,
    save_midrun,
)
from .schedules import make_lr_schedule
from .timer import StepTimer
from .train_step import _make_weight_track, run_training_loop
from .unet_wrapper import ComfyUNetWrapper, clear_embedder_cache


class Trainer:
    """Main training coordinator — works with typed TrainingConfig."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.common.device)

        # Extract run_id from CLI if set (stored on model copy)
        cli_run_id = getattr(config.common, "_cli_run_id", None)
        self.progress_writer = ProgressWriter(cli_run_id) if cli_run_id else None

        self.tuning = config.tuning
        self._is_lora = isinstance(self.tuning, LoRATuning)
        self._is_cyclic = isinstance(self.tuning, CyclicTuning)

        # State
        self.teacher_unet_sd = None
        self.student_unet_sd = None
        self.non_unet = None
        self.student_non_unet = None
        self.student: Optional[ComfyUNetWrapper] = None
        self.teacher: Optional[ComfyUNetWrapper] = None
        self.optimizer = None
        self.student_encoder = None
        self.prompt_cache = None

        # Conditioning overrides
        self.training_ctx = None
        self.training_y = None
        self.training_ctx_u = None
        self.training_y_u = None

        # Managed dataset
        self.dataset_loader = None
        paths = config.paths
        if paths.dataset_name:
            from manager.loader import ManagedDatasetLoader
            from paths import get_datasets_dir
            dataset_root = get_datasets_dir() / paths.dataset_name
            self.dataset_loader = ManagedDatasetLoader(
                dataset_root,
                batch_size=config.common.batch_size,
            )
            print(f"  Training from managed dataset: {paths.dataset_name} (all trajectories)")
            print(f"  Total samples: {len(self.dataset_loader)}")

        # Steps
        # common.resume_step is a manual override. If it's left at 0 and this
        # run is actually configured to resume, auto-detect the step from the
        # saved optimizer-state file instead of requiring the user to keep it
        # in sync by hand. This has to happen here (before total_steps is
        # computed), not later inside build_optimizer() -- there is no
        # optimizer object yet at that point to hang a "resolved step" off of.
        self.start_step = config.common.resume_step or 0
        if (self.start_step == 0
                and config.start_from == "resume"
                and not config.reset_optimizer
                and paths.resume_optimizer
                and Path(paths.resume_optimizer).exists()):
            from .save import peek_resume_step
            self.start_step = peek_resume_step(paths.resume_optimizer)

        self.total_steps = self.start_step + config.common.steps
        self.run_steps = config.common.steps

        if self.start_step > 0:
            print(f"  Resuming from step {self.start_step}")

    def load_models(self, teacher_only: bool = False):
        """Load checkpoints to CPU memory."""
        print("[1/4] Loading checkpoint(s) to RAM...")
        
        teacher_sd = None
        if self.config.paths.base_model:
            teacher_sd = load_file(self.config.paths.base_model)
            self.teacher_unet_sd = {k: v for k, v in teacher_sd.items() if self._is_unet(k)}
            self.non_unet = {k: v for k, v in teacher_sd.items() if not self._is_unet(k)}
        elif self.config.common.data_source == "teacher":
            raise ValueError("No base model path provided in config (paths.base_model), but data_source='teacher' requires it.")

        # Encode student training prompts
        _t_pos = self.config.common.training_positive_prompt.strip()
        _t_neg = self.config.common.training_negative_prompt.strip()
        if (_t_pos or _t_neg) and not teacher_only:
            from .clip_encode import SDXLClipEncoder
            _clip_sd = teacher_sd
            if self.config.paths.student:
                try:
                    _stu_full = load_file(self.config.paths.student)
                    _stu_non_unet = {k: v for k, v in _stu_full.items() if not self._is_unet(k)}
                    if _stu_non_unet:
                        _clip_sd = _stu_full
                        self.student_non_unet = _stu_non_unet
                    del _stu_full
                except Exception as e:
                    print(f"  [WARN] Could not load student checkpoint '{self.config.paths.student}' "
                          f"for CLIP encoding ({e}). Falling back to teacher CLIP weights.")

            if _clip_sd is None:
                print("  [WARN] No model weights available for CLIP encoding (no teacher and no student non-unet).")
            else:
                _enc = SDXLClipEncoder(_clip_sd, self.device)
                _latent_dim = self.config.common.latent_size or 64
                _px = _latent_dim * 8
                if _t_pos:
                    _ctx, _y = _enc.encode_for_unet(_t_pos, 1, height=_px, width=_px)
                    self.training_ctx = _ctx.cpu().pin_memory()
                    self.training_y = _y.cpu().pin_memory()
                if _t_neg:
                    _ctx_u, _y_u = _enc.encode_for_unet(_t_neg, 1, height=_px, width=_px)
                    self.training_ctx_u = _ctx_u.cpu().pin_memory()
                    self.training_y_u = _y_u.cpu().pin_memory()
                _enc.unload()
                del _enc

        if teacher_sd:
            del teacher_sd
            
        if teacher_only:
            return self

        if self.config.paths.student:
            student_sd = load_file(self.config.paths.student)
            self.student_unet_sd = {k: v for k, v in student_sd.items() if self._is_unet(k)}
            self.student_non_unet = {k: v for k, v in student_sd.items() if not self._is_unet(k)}
            del student_sd

        # In LoRA mode, student base = teacher weights (if available)
        if self._is_lora:
            has_base = any("model.diffusion_model." in k for k in (self.student_unet_sd or {}))
            if not has_base:
                # If we have something in student_unet_sd but no base, it must be LoRA adapters
                if self.student_unet_sd:
                    self._resume_adapter_sd = self.student_unet_sd
                
                if self.teacher_unet_sd:
                    self.student_unet_sd = dict(self.teacher_unet_sd)
                    self.student_non_unet = dict(self.non_unet) if self.non_unet else None
                elif not self.student_unet_sd:
                    # This is a problem for LoRA if we don't have ANY base weights
                    raise ValueError("LoRA training requires a base model (paths.teacher or paths.student) to initialize weights.")

        gc.collect()
        return self

    def setup(self):
        return self

    def _is_unet(self, key: str) -> bool:
        return "model.diffusion_model." in key or "lora_unet_" in key

    def _build_lora_config(self) -> Optional[LoRAConfig]:
        if not self._is_lora:
            return None
        t = self.tuning
        
        block_weights = None
        if hasattr(t, "block_weighting") and t.block_weighting:
            block_weights = {}
            for part in t.block_weighting.split(","):
                if ":" in part:
                    name, val = part.split(":", 1)
                    try:
                        block_weights[name.strip()] = float(val.strip())
                    except ValueError:
                        pass

        return LoRAConfig(
            rank=t.rank,
            alpha=t.alpha,
            dropout=t.dropout,
            block_weights=block_weights,
            target_all=t.target_all,
        )

    def build_optimizer(self):
        comm = self.config.common
        if self.student is None:
            sd = self.student_unet_sd if self.student_unet_sd is not None else self.teacher_unet_sd
            lora_cfg = self._build_lora_config()
            self.student = ComfyUNetWrapper(
                sd, self.device, torch.bfloat16,
                use_checkpoint=not comm.no_checkpoint,
                lora_config=lora_cfg,
                adm_in_channels=3072 if comm.cfg_aware else 2816,
            )

            # 1. Check for manual LoRA adapter weights (e.g. from --student CLI)
            if self._is_lora and hasattr(self, "_resume_adapter_sd"):
                # If this is the same file as resume_checkpoint, we'll let the resume
                # logic handle it below to avoid double loading.
                if self.config.paths.student != self.config.paths.resume_checkpoint:
                    print(f"  Loading LoRA adapter from student path")
                    self.student.load_lora_weights(self._resume_adapter_sd)
                del self._resume_adapter_sd

            if self._is_lora and self.tuning.lora_continue_from:
                p = Path(self.tuning.lora_continue_from)
                if p.exists():
                    print(f"  Continuing LoRA from adapter: {p}")
                    load_lora_checkpoint(self.student, str(p))
                else:
                    print(f"  [WARN] lora_continue_from path does not exist: {p}")

            # 2. In LoRA mode, resume_checkpoint holds only adapter weights saved by
            # save_midrun.  Only load if explicitly requested via start_from='resume'.
            paths = self.config.paths
            if (self._is_lora
                    and self.config.start_from == "resume"
                    and paths.resume_checkpoint
                    and Path(paths.resume_checkpoint).exists()):
                print(f"  Resuming LoRA adapters from: {paths.resume_checkpoint}")
                load_lora_checkpoint(self.student, paths.resume_checkpoint)

        self.student.train()

        if self._is_lora:
            self.optimizer = build_optimizer(
                self.student, comm, self.device,
                params=self.student.lora_parameters(),
            )
        else:
            self.optimizer = build_optimizer(self.student, comm, self.device)

        # 3. Load optimizer state if resuming
        paths = self.config.paths
        if (self.config.start_from == "resume"
                and paths.resume_optimizer 
                and Path(paths.resume_optimizer).exists() 
                and not self.config.reset_optimizer):
            from .save import load_optstate
            print(f"  Resuming optimizer state from: {paths.resume_optimizer}")
            saved_step = load_optstate(self.optimizer, paths.resume_optimizer)
            if saved_step != self.start_step:
                print(f"  [WARN] optimizer file step ({saved_step}) does not match "
                      f"resolved start_step ({self.start_step}). Momentum/t are "
                      f"still restored from the file; the training range uses "
                      f"start_step as resolved in __init__.")

        if isinstance(self.optimizer, FusedXPUAdafactor):
            self.optimizer.register_hooks()
        return self

    def train(self):
        comm = self.config.common
        paths = self.config.paths
        cycle_steps = self.tuning.cycle_steps if self._is_cyclic else self.run_steps
        total_cycles = (self.run_steps + cycle_steps - 1) // cycle_steps
        lr_fn = make_lr_schedule(comm, self.run_steps, self.start_step)
        t0 = time.time()
        global_step = self.start_step
        weight_track = _make_weight_track()
        timer = StepTimer(report_every=100)

        print("[3/4] Initializing models and encoder...")
        full_teacher_sd = {}
        if self.teacher_unet_sd:
            full_teacher_sd.update(self.teacher_unet_sd)
        if self.non_unet:
            full_teacher_sd.update(self.non_unet)
        
        if full_teacher_sd:
            self.teacher = ComfyUNetWrapper(full_teacher_sd, device=self.device, dtype=torch.bfloat16)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            
            # If using a dataset, we don't need the teacher on device during training
            if self.dataset_loader is not None:
                self.teacher.to("cpu")
                xpu_empty_cache()
        else:
            print("  Teacher model not loaded (using dataset for training data).")

        _clip_sd = {}
        if self.non_unet:
            _clip_sd.update(self.non_unet)
        if self.student_non_unet:
            _clip_sd.update(self.student_non_unet)

        from .clip_encode import SDXLClipEncoder
        self.student_encoder = SDXLClipEncoder(_clip_sd, self.device)

        # Pre-calculate prompt cache for managed datasets
        if self.dataset_loader is not None:
            print("  Pre-calculating unique CLIP embeddings for dataset...")
            unique_prompts = set()
            for t in self.dataset_loader.trajectories:
                unique_prompts.add(t["prompt"])
                if t.get("metadata"):
                    try:
                        meta = json.loads(t["metadata"])
                        if isinstance(meta, dict) and meta.get("neg"):
                            unique_prompts.add(meta["neg"])
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        print(f"    [WARN] Skipping malformed metadata for prompt "
                              f"'{t.get('prompt', '')[:40]}': {e}")

            self.prompt_cache = {}
            for prompt in tqdm(list(unique_prompts), desc="Encoding Prompts"):
                ctx, pooled = self.student_encoder.encode_prompt(prompt)
                # Ensure at least 77 tokens
                if ctx.shape[1] < 77:
                    padding = torch.zeros((ctx.shape[0], 77 - ctx.shape[1], ctx.shape[2]), 
                                         device=ctx.device, dtype=ctx.dtype)
                    ctx = torch.cat([ctx, padding], dim=1)
                
                self.prompt_cache[prompt] = (ctx.cpu().float().contiguous(), pooled.cpu().float().contiguous())

            print(f"  Prompt cache built: {len(self.prompt_cache)} unique prompts.")
            self.student_encoder.unload()

        del full_teacher_sd, _clip_sd
        xpu_empty_cache()
        gc.collect()

        print(f"[4/4] Training {self.run_steps} steps in {total_cycles} cycles...")
        cycle_idx = 0
        try:
            while global_step < self.total_steps:
                this_cycle_steps = min(cycle_steps, self.run_steps - (global_step - self.start_step))
                if this_cycle_steps <= 0:
                    break

                if self.optimizer is not None and cycle_idx > 0:
                    self._archive_optimizer()

                if self.dataset_loader:
                    cache = self.dataset_loader
                elif self.teacher:
                    cache = self._get_cache(run_steps=this_cycle_steps, start_step=global_step)
                else:
                    raise RuntimeError("No data source available: neither dataset_loader nor teacher model is present.")

                if self.teacher:
                    self.teacher = self.teacher.to("cpu")
                xpu_empty_cache()
                gc.collect()

                if not self.dataset_loader and (self.training_ctx is not None or self.training_ctx_u is not None):
                    cache = self._apply_conditioning_override(cache)

                if cycle_idx == 0 and not self.dataset_loader and comm.dump_cache_samples:
                    self._dump_cache_samples(cache)

                if self.optimizer is None:
                    self.build_optimizer()
                else:
                    if self.student:
                        self.student = self.student.to(self.device)
                        self.student.train()
                    self._restore_optimizer()

                def save_callback(step):
                    if self._is_lora:
                        if self.tuning.lora_output:
                            p = Path(self.tuning.lora_output)
                            save_path = p.parent / f"{p.stem}_{step:08d}{p.suffix}"
                            save_lora_checkpoint(self.student, str(save_path),
                                                 optimizer=self.optimizer)
                    else:
                        if self.config.paths.checkpoint_output:
                            _non_unet = self.student_non_unet or self.non_unet
                            p = Path(self.config.paths.checkpoint_output)
                            save_path = p.parent / f"{p.stem}_{step:08d}{p.suffix}"
                            save_checkpoint(self.student, _non_unet, str(save_path), torch.float16)

                steps_done, stopped, weight_track = run_training_loop(
                    student=self.student, optimizer=self.optimizer, cache=cache,
                    config=self.config, device=self.device, start_step=global_step,
                    run_steps=this_cycle_steps, total_steps=self.total_steps,
                    expected_seed=42, lr_fn=lr_fn, save_dtype=torch.float16,
                    timer=timer, t0=t0, weight_track=weight_track,
                    progress_writer=self.progress_writer,
                    student_encoder=self.student_encoder,
                    prompt_cache=self.prompt_cache,
                    save_callback=save_callback,
                )

                global_step += steps_done
                if stopped:
                    break

                # Intermediate save
                _non_unet = self.student_non_unet or self.non_unet
                if paths.resume_checkpoint:
                    save_midrun(self.student, self.optimizer, paths.resume_checkpoint,
                                paths.resume_optimizer, global_step, torch.bfloat16,
                                non_unet=_non_unet)

                cycle_idx += 1
                if not self.dataset_loader:
                    del cache
                
                xpu_empty_cache()
                gc.collect()

        finally:
            if self.optimizer:
                if self._is_lora and self.tuning.lora_output:
                    save_lora_checkpoint(self.student, self.tuning.lora_output,
                                         optimizer=self.optimizer)
                elif self._is_lora and not self.tuning.lora_output:
                    print("    [WARN] LoRA mode but no output path set — skipped save.")
                else:
                    _non_unet = self.student_non_unet or self.non_unet
                    save_checkpoint(self.student, _non_unet, paths.checkpoint_output, torch.float16)

            if self.progress_writer:
                self.progress_writer.done()
            print(f"Training session complete at step {global_step}.")

    def _get_cache(self, run_steps, start_step):
        comm = self.config.common
        cache_cfg = self.config.cache

        seed = comm.seed + start_step
        no_compile = comm.no_compile

        if hasattr(cache_cfg, "traj_steps_min"):
            # Trajectory cache
            from .config_model import TrajectoryCache
            assert isinstance(cache_cfg, TrajectoryCache)

            return build_trajectory_cache(
                self.teacher_unet_sd, comm.teacher_type, comm.student_type,
                run_steps, comm.batch_size, str(self.device), seed,
                gen_batch_size=cache_cfg.batch_size,
                traj_steps_min=cache_cfg.traj_steps_min,
                traj_steps_max=cache_cfg.traj_steps_max,
                sequence_size=cache_cfg.sequence_size,
                sequence_mode=cache_cfg.sequence_mode,
                cfg_scale=cache_cfg.cfg,
                cond_mode=cache_cfg.cond_mode,
                positive_prompt=cache_cfg.positive_prompt,
                negative_prompt=cache_cfg.negative_prompt,
                t_low=comm.t_low, t_high=comm.t_high,
                traj_skip_steps=cache_cfg.traj_skip_steps,
                cache_batch_size=cache_cfg.batch_size,
                latent_size=comm.latent_size,
                teacher_model=self.teacher,
                student_model=None,
                student_unet_sd=self.student_unet_sd,
                student_mix_frac=cache_cfg.student_mix,
                student_anchor_steps=cache_cfg.student_anchor_steps,
                student_chain_len=cache_cfg.student_chain_len,
                student_chain_noise=cache_cfg.student_chain_noise,
                no_compile=no_compile,
                progress_writer=self.progress_writer,
                cfg_aware=comm.cfg_aware,
                cfg_min=comm.training_cfg_min,
                cfg_max=comm.training_cfg_max,
                student_positive_prompt=comm.training_positive_prompt,
                student_negative_prompt=comm.training_negative_prompt,
            )
        else:
            # Random cache
            from .config_model import RandomCache
            assert isinstance(cache_cfg, RandomCache)
            return build_random_cache(
                self.teacher_unet_sd, comm.teacher_type, comm.student_type,
                run_steps, comm.batch_size, str(self.device), seed,
                cache_batch=cache_cfg.batch,
                cache_batch_size=None,
                latent_size=comm.latent_size,
                t_mode=comm.t_mode, t_low=comm.t_low, t_high=comm.t_high,
                teacher_model=self.teacher,
            )

    def _archive_optimizer(self):
        """Free device memory before rebuilding the teacher cache for the next
        cyclic-training cycle.

        None of our custom optimizers (CPUAdamW, ChunkedXPUAdafactor,
        FusedXPUAdafactor, ForeachXPUAdafactor) implement state_dict() /
        load_state_dict() -- they weren't torch.optim.Optimizer subclasses to
        begin with, so the old "archive to RAM via state_dict()" approach
        either silently no-op'd (for the types that were skipped) or crashed
        with AttributeError (for ForeachXPUAdafactor, which wasn't skipped).

        Instead, we move each optimizer's own state tensors between device
        and CPU in place via offload_states_to_cpu()/reload_states_to_device(),
        which every optimizer class now implements (CPUAdamW's are no-ops
        since its state already lives on CPU).
        """
        if self.optimizer is None:
            return

        if hasattr(self.optimizer, "offload_states_to_cpu"):
            print("    Offloading optimizer states to CPU...")
            self.optimizer.offload_states_to_cpu()

        if self.student:
            self.student = self.student.to("cpu")
        xpu_empty_cache()
        gc.collect()

    def _restore_optimizer(self):
        """Move optimizer state back onto the training device, and apply the
        configured inter-cycle decay (CyclicTuning.cycle_state_decay), which
        previously existed as a config field but was never actually applied
        anywhere."""
        if self.optimizer is None:
            return

        if hasattr(self.optimizer, "reload_states_to_device"):
            print("    Reloading optimizer states onto device...")
            self.optimizer.reload_states_to_device(self.device)

        if self._is_cyclic and hasattr(self.optimizer, "decay_states"):
            decay = self.tuning.cycle_state_decay
            if decay < 1.0:
                print(f"    Applying cycle_state_decay={decay}...")
                self.optimizer.decay_states(decay)

    def _apply_conditioning_override(self, cache):
        print("    Applying student conditioning override...")
        new_cache = []
        for entry in cache:
            # Cache entries have 9 elements when CFG is off, 10 when CFG is on
            # (the 10th element is target_uncond).  Unpack defensively so this
            # method works for both formats without crashing.
            x, tc, c, y, cu, yu, at, st, t = entry[:9]
            tu = entry[9] if len(entry) >= 10 else None
            if self.training_ctx is not None:
                c = self.training_ctx.repeat(x.shape[0], 1, 1)
                y = self.training_y.repeat(x.shape[0], 1)
            if self.training_ctx_u is not None:
                cu = self.training_ctx_u.repeat(x.shape[0], 1, 1)
                yu = self.training_y_u.repeat(x.shape[0], 1)
            entry_out = (x, tc, c, y, cu, yu, at, st, t)
            if tu is not None:
                entry_out = entry_out + (tu,)
            new_cache.append(entry_out)
        return new_cache

    def _dump_cache_samples(self, cache):
        print("    Dumping 20 cache samples for visual inspection...")
        # checkpoint_output may be empty (e.g. LoRA-only run); fall back to cwd.
        base = self.config.paths.checkpoint_output
        if base:
            out_dir = Path(base).parent / "previews"
        else:
            out_dir = Path("previews")
        out_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for e in cache:
            if isinstance(e, dict):
                samples.append(e)
            else:
                # Tuple cache: (x_t, target, ctx, y, ctx_u, y_u, at, st, t_val, ...)
                samples.append({"x_t": e[0], "t": e[8]})

        samples.sort(key=lambda x: x["t"])
        to_decode = samples[:20]

        from .vae_decode import VAEDecoder
        from PIL import Image
        vae = VAEDecoder.from_vae_sd(self.non_unet, self.device)

        for i, s in enumerate(to_decode):
            # vae.decode() returns a (B, 3, H, W) uint8 tensor, not a PIL image.
            tensor = vae.decode(s["x_t"].to(self.device, dtype=torch.bfloat16))
            img_np = tensor[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(img_np).save(out_dir / f"sample_{i}_t{s['t']}.png")

        vae.free()
        del vae
        xpu_empty_cache()
        gc.collect()
