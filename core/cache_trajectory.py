"""Teacher cache builder — trajectory mode (Euler denoising sampler)."""

import gc
import math
import random
import sys

import torch
from tqdm import tqdm

from .cache_utils import resolve_gen_batch_size, warn_batch_mismatch, pin_tensors_parallel
from .comfy_setup import xpu_empty_cache
from .model_io import comfy_input_transform, make_init_noise, raw_to_denoised, raw_to_target
from .noise_schedule import get_alpha_sigma
from .progress_writer import ProgressWriter
from .seed import derive_seed
from .unet_wrapper import ComfyUNetWrapper, make_rand_cond


def _euler_step(x_t, denoised, sigma_t, sigma_next):
    """
    One Euler denoising step given a denoised (x0) prediction.

    Matches ComfyUI's sample_euler in k_diffusion/sampling.py:
      d = (x - denoised) / sigma
      x_next = x + d * (sigma_next - sigma)

    No clamping — ComfyUI's Euler sampler does not clamp denoised outputs.
    """
    d = (x_t - denoised) / sigma_t
    return x_t + d * (sigma_next - sigma_t)


def build_teacher_cache_trajectory(
        teacher_unet_sd, teacher_type, student_type,
        n_samples, batch_size, device, seed,
        gen_batch_size=None,
        precomputed_ctx=None,
        precomputed_ctx_unc=None,
        precomputed_y=None,
        precomputed_y_unc=None,
        traj_steps_min=20, traj_steps_max=30,
        sequence_size=0,
        sequence_mode="random",
        cfg_scale=1.0,
        cond_mode="random",
        positive_prompt="",
        negative_prompt="",
        t_low=0, t_high=999,
        traj_skip_steps=4,
        cache_batch_size=None,
        latent_size=0,
        teacher_model=None,
        student_model=None,
        student_unet_sd=None,
        student_mix_frac=0.0,
        student_anchor_steps=5,
        student_chain_len=3,
        student_chain_noise=0.02,
        no_compile=False,
        progress_writer: ProgressWriter | None = None,
        cfg_random: bool = False,
        cfg_min: float = 1.0,
        cfg_max: float = 1.0,
        student_positive_prompt: str = "",
        student_negative_prompt: str = "",
        student_unet_sd_for_clip=None,
):
    """
    Build cache using real teacher trajectories (Euler sampler).

    student_positive_prompt / student_negative_prompt: if set, the ctx/y
    stored in the cache (used for the student forward during training) are
    encoded from these prompts via the student CLIP rather than reusing the
    teacher conditioning.  This lets you teach the student to map a different
    prompt vocabulary (e.g. realistic) onto the teacher's output style.

    student_unet_sd_for_clip: full state dict containing conditioner keys
    for the student checkpoint.  If None, falls back to teacher_unet_sd
    (correct when both checkpoints share CLIP weights, which is typical).
    """

    # Resolve spatial size — 0 means "use default 64" (512px with 8x VAE downscale)
    latent_dim = latent_size if (latent_size and latent_size > 0) else 64
    if latent_dim != 64:
        print(f"    Latent size: {latent_dim}×{latent_dim} ({latent_dim*8}×{latent_dim*8} px)")

    rng = random.Random(seed)

    # Resolve batch size for cache generation
    gen_batch_size = resolve_gen_batch_size(cache_batch_size, batch_size)
    warn_batch_mismatch(gen_batch_size, batch_size)

    # --- 1. CLIP encoding (MUST happen before UNets are on VRAM) ---
    # Two sets of conditioning may be needed:
    #   teacher_ctx / teacher_ctx_unc  — used for trajectory Euler steps and target passes
    #   student_ctx / student_ctx_unc  — stored in the cache, used for student forward during training
    # When no student prompts are given, both sets are identical (normal behaviour).
    precomputed_student_ctx     = None
    precomputed_student_ctx_unc = None
    precomputed_student_y       = None
    precomputed_student_y_unc   = None

    if cond_mode == "prompt":
        print(f"    Encoding teacher conditioning prompts via CLIP...")
        from .clip_encode import SDXLClipEncoder
        _px = latent_dim * 8
        _clip = SDXLClipEncoder(teacher_unet_sd, device=device)

        # Ensure we have valid prompts (handle None)
        _pos = positive_prompt or ""
        _neg = negative_prompt or ""

        _ctx, _y = _clip.encode_for_unet(_pos, batch_size=gen_batch_size,
                                          height=_px, width=_px)
        precomputed_ctx = _ctx.cpu().float().contiguous()
        precomputed_y   = _y.cpu().float().contiguous()

        _ctx_u, _y_u = _clip.encode_for_unet(_neg, batch_size=gen_batch_size,
                                               height=_px, width=_px)
        precomputed_ctx_unc = _ctx_u.cpu().float().contiguous()
        precomputed_y_unc   = _y_u.cpu().float().contiguous()

        _clip.unload()
        del _clip
        gc.collect()
        xpu_empty_cache()
        print(f"    Teacher prompts encoded (pos='{_pos[:40]}', neg='{_neg[:40]}').")

        # Encode student conditioning if different prompts are requested.
        # student_unet_sd_for_clip is currently always the teacher's non_unet
        # (CLIP weights are identical in almost all fine-tune setups), so we
        # always use teacher CLIP here. A truly different student CLIP would
        # require separate student checkpoint CLIP weights.
        _s_pos = student_positive_prompt.strip() if student_positive_prompt else ""
        _s_neg = student_negative_prompt.strip() if student_negative_prompt else ""
        _has_student_prompts = bool(_s_pos or _s_neg)

        if _has_student_prompts:
            # Fall back to teacher prompts for whichever side is not overridden
            _eff_s_pos = _s_pos if _s_pos else _pos
            _eff_s_neg = _s_neg if _s_neg else _neg
            print(f"    Encoding student conditioning via CLIP...")
            print(f"      pos: '{_eff_s_pos[:60]}'")
            print(f"      neg: '{_eff_s_neg[:60]}'")
            _clip_sd = student_unet_sd_for_clip if student_unet_sd_for_clip is not None else teacher_unet_sd
            _sclip = SDXLClipEncoder(_clip_sd, device=device)

            _s_ctx, _s_y = _sclip.encode_for_unet(_eff_s_pos, batch_size=gen_batch_size,
                                                    height=_px, width=_px)
            precomputed_student_ctx = _s_ctx.cpu().float().contiguous()
            precomputed_student_y   = _s_y.cpu().float().contiguous()

            _s_ctx_u, _s_y_u = _sclip.encode_for_unet(_eff_s_neg, batch_size=gen_batch_size,
                                                        height=_px, width=_px)
            precomputed_student_ctx_unc = _s_ctx_u.cpu().float().contiguous()
            precomputed_student_y_unc   = _s_y_u.cpu().float().contiguous()

            _sclip.unload()
            del _sclip
            gc.collect()
            xpu_empty_cache()
            print(f"    Student prompts encoded.")
        else:
            # No separate student prompts — cache will store teacher conditioning.
            # Student will train on the same prompts the teacher used.
            print(f"    Student conditioning: reusing teacher prompts (no override set).")

    # --- 2. Load/Activate UNet models (Teacher and optionally Student) ---
    if teacher_model is not None:
        print(f"    Activating existing teacher model on {device}...")
        teacher = teacher_model
        teacher.to(device)
    else:
        print(f"    Loading teacher on {device} (bf16) for trajectory cache...")
        teacher = ComfyUNetWrapper(teacher_unet_sd, device=device, dtype=torch.bfloat16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Compile teacher for faster cache generation
        if hasattr(torch, "compile") and not no_compile:
            try:
                torch._dynamo.config.recompile_limit = 64
                teacher.model = torch.compile(teacher.model, mode="default",
                                              dynamic=False, backend="inductor")
                print("    Teacher compiled.")
            except Exception as e:
                print(f"    Teacher compile failed ({e}), running eager.")

    # Load student for chain generation if mixing is requested
    student_cache = None
    use_student_mix = (student_mix_frac > 1e-6 and 
                       (student_model is not None or student_unet_sd is not None))
    if use_student_mix:
        if student_model is not None:
            # Move existing student model to device now that CLIP is gone
            print(f"    Activating existing student model on {device} for chain generation "
                  f"(mix={student_mix_frac:.0%}, chain={student_chain_len}, "
                  f"noise={student_chain_noise:.3f})...")
            student_cache = student_model
            student_cache.to(device)
        else:
            print(f"    Loading student on {device} (bf16) for chain generation "
                  f"(mix={student_mix_frac:.0%}, chain={student_chain_len}, "
                  f"noise={student_chain_noise:.3f})...")
            student_cache = ComfyUNetWrapper(student_unet_sd, device=device,
                                             dtype=torch.bfloat16)
            student_cache.eval()
            for p in student_cache.parameters():
                p.requires_grad_(False)
    elif student_mix_frac > 1e-6:
        print("    Warning: student_mix_frac > 0 but no student weights available. "
              "Falling back to teacher-only trajectories.")

    # --- 3. Trajectory configuration ---
    # Enforce a minimum safe t_low for training samples.
    # Timesteps below ~20 have sigma < 0.01 — the latent is nearly clean and
    # the teacher's output at those steps is unreliable (near-zero noise level
    # means tiny floating-point errors dominate the gradient signal).
    _SAFE_T_LOW = max(t_low, 20)
    if _SAFE_T_LOW != t_low:
        print(f"    Note: t_low raised from {t_low} to {_SAFE_T_LOW} to avoid near-clean timesteps.")
    t_low = _SAFE_T_LOW

    if cfg_random:
        use_cfg = cfg_max > 1.0 + 1e-6
        _cfg_min = max(1.0, cfg_min)
        _cfg_max = max(_cfg_min, cfg_max)
    else:
        use_cfg = cfg_scale > 1.0 + 1e-6
        _cfg_min = _cfg_max = cfg_scale

    avg_steps = (traj_steps_min + traj_steps_max) / 2
    seq_eff = sequence_size if (0 < sequence_size < avg_steps) else avg_steps
    if cfg_random and use_cfg:
        cfg_str = f"CFG-random={_cfg_min:.1f}\u2013{_cfg_max:.1f}"
    elif use_cfg:
        cfg_str = f"CFG={cfg_scale:.1f}"
    else:
        cfg_str = "CFG=off"
    mix_str = f"mix={student_mix_frac:.0%}" if use_student_mix else "mix=off"
    est_trajs = math.ceil(n_samples / seq_eff)
    print(f"    Mode: trajectory | steps/traj={traj_steps_min}–{traj_steps_max} | "
          f"seq={int(seq_eff)} | {cfg_str} | {mix_str}")
    print(f"    Est. trajectories needed: ~{est_trajs} "
          f"({est_trajs * avg_steps:.0f} teacher fwds for traj, "
          f"{n_samples} for targets, each fwd covers batch_size={gen_batch_size} latents)")

    def make_traj_timesteps(n_steps):
        """Evenly-spaced integer timesteps from t_high down to t_low."""
        hi, lo = t_high, t_low
        return [round(hi - i * (hi - lo) / n_steps) for i in range(n_steps + 1)]

    def run_model_denoised(model, x_t, t_val_int, ctx, y, model_type, w=None):
        """Run model and return denoised (x0) prediction.

        Matches ComfyUI's BaseModel._apply_model:
          1. Transform input via calculate_input
          2. Run model to get raw output (v or eps)
          3. Convert raw to denoised via calculate_denoised
        """
        t_tensor = torch.tensor([t_val_int] * x_t.shape[0],
                                dtype=torch.long, device=device)
        at, st = get_alpha_sigma(t_val_int)
        at_t = torch.tensor(at.item(), device=device, dtype=torch.bfloat16)
        st_t = torch.tensor(st.item(), device=device, dtype=torch.bfloat16)

        xc = comfy_input_transform(x_t, st_t)
        raw = model.forward(xc, t_tensor, ctx, y)
        denoised = raw_to_denoised(raw, x_t, at_t, st_t, model_type)
        del t_tensor, raw, at_t, st_t, xc
        return denoised, at.item(), st.item()

    cache = []
    traj_idx = 0
    n_trajectories = 0
    n_student_trajs = 0

    if progress_writer:
        progress_writer.cache_start(target=n_samples, est_trajs=est_trajs)

    is_tty = sys.stdout.isatty()
    pbar = tqdm(total=n_samples, desc="  Caching (traj)", unit="sample", leave=False) if is_tty else None

    with torch.no_grad():
        while len(cache) < n_samples:
            use_chain = use_student_mix and (rng.random() < student_mix_frac)

            n_steps = rng.randint(traj_steps_min, traj_steps_max)
            t_grid = make_traj_timesteps(n_steps)

            valid_start = min(traj_skip_steps, n_steps)
            # Exclude the final step (index n_steps) which lands at t_low (usually t=0).
            # At t=0 sigma=0, so comfy_input_transform divides by 1 (fine), but the
            # teacher forward at timestep=0 on an already-clean latent produces garbage
            # output that has no meaningful distillation signal. Training on these samples
            # teaches the LoRA to corrupt its output at the last sampler step, which is
            # exactly the "last step trashes the image" symptom.
            valid_range = range(valid_start, n_steps)
            n_valid = len(valid_range)

            seq = sequence_size if (0 < sequence_size < n_valid) else n_valid
            if seq >= n_valid:
                keep_indices = set(valid_range)
            elif sequence_mode in ("span", "span_low", "span_mid", "span_high"):
                max_start = n_valid - seq
                if sequence_mode == "span_low":
                    x = rng.gammavariate(3.0, 1.0)
                    y_b = rng.gammavariate(1.0, 1.0)
                elif sequence_mode == "span_mid":
                    x = rng.gammavariate(2.0, 1.0)
                    y_b = rng.gammavariate(2.0, 1.0)
                elif sequence_mode == "span_high":
                    x = rng.gammavariate(1.0, 1.0)
                    y_b = rng.gammavariate(3.0, 1.0)
                else:
                    x = 1.0
                    y_b = 1.0
                p = rng.random() if (x == 1.0 and y_b == 1.0) else x / (x + y_b)
                start = valid_start + max(0, min(max_start, int(round(p * max_start))))
                keep_indices = set(range(start, start + seq))
            else:
                keep_indices = set(rng.sample(list(valid_range), seq))

            # Initial noise at t_high — scale by sigma to match ComfyUI's noise_scaling
            _, st_first = get_alpha_sigma(t_grid[0])
            cpu_gen = torch.Generator(device="cpu")
            cpu_gen.manual_seed(derive_seed(seed, traj_idx, "traj_init"))
            x_t = make_init_noise(
                (gen_batch_size, 4, latent_dim, latent_dim), device, torch.bfloat16, st_first, cpu_gen)

            if cond_mode == "zero":
                ctx = torch.zeros(gen_batch_size, 77, 2048, device=device, dtype=torch.bfloat16)
                y = torch.zeros(gen_batch_size, 2816, device=device, dtype=torch.bfloat16)
            elif precomputed_ctx is not None:
                # Use precomputed conditioning (CLIP-encoded, stored on CPU pinned)
                ctx = precomputed_ctx.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                y = precomputed_y.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            else:
                ctx, y = make_rand_cond(gen_batch_size, device, torch.bfloat16, seed, traj_idx,
                                        latent_size=latent_dim)
            if use_cfg:
                if precomputed_ctx_unc is not None:
                    ctx_unc = precomputed_ctx_unc.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    y_unc = precomputed_y_unc.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                else:
                    # CFG is enabled but no negative prompt provided: use zero-conditioning
                    ctx_unc = torch.zeros(gen_batch_size, 77, 2048, device=device, dtype=torch.bfloat16)
                    y_unc = torch.zeros(gen_batch_size, 2816, device=device, dtype=torch.bfloat16)

            # Sample CFG scale for this trajectory.
            # cfg_random: w is drawn from [cfg_min, cfg_max] and stays fixed for the
            # entire trajectory — denoising, target pass, and stored w are all the same.
            # Standard: traj_w = cfg_scale (fixed or 1.0).
            traj_w = rng.uniform(_cfg_min, _cfg_max) if (cfg_random and use_cfg) else cfg_scale

            collected = []
            in_student_chain = False
            student_steps_done = 0
            max_needed_step = max(keep_indices)

            for step_i in range(n_steps):
                t_val = t_grid[step_i]
                t_next = t_grid[step_i + 1]
                at_cur, st_cur = get_alpha_sigma(t_val)
                at_next, st_next = get_alpha_sigma(t_next)
                atc = at_cur.item()
                stc = st_cur.item()
                atn = at_next.item()
                stn = st_next.item()

                if use_chain and step_i == student_anchor_steps:
                    in_student_chain = True

                # Track whether this step uses the student, before the else branch
                # can reset in_student_chain, so noise is applied correctly below.
                _this_step_is_student = in_student_chain and student_steps_done < student_chain_len

                if _this_step_is_student:
                    denoised_step, _, _ = run_model_denoised(
                        student_cache, x_t, t_val, ctx, y, student_type, w=traj_w)
                    student_steps_done += 1
                else:
                    in_student_chain = False
                    xc = comfy_input_transform(x_t, stc)
                    t_tensor = torch.tensor([t_val] * gen_batch_size, dtype=torch.long, device=device)

                    raw = teacher.forward(xc, t_tensor, ctx, y)
                    if use_cfg:
                        raw_unc = teacher.forward(xc, t_tensor, ctx_unc, y_unc)
                        # We used to combine them here, but now we keep them separate 
                        # for the teacher Euler step. For the Euler step itself, 
                        # we still need a combined result to move x_t forward.
                        raw_combined = raw_unc + traj_w * (raw - raw_unc)
                        at_t = torch.tensor(atc, device=device, dtype=torch.bfloat16)
                        st_t = torch.tensor(stc, device=device, dtype=torch.bfloat16)
                        denoised_step = raw_to_denoised(raw_combined, x_t, at_t, st_t, teacher_type)
                        del raw_unc, raw_combined
                    else:
                        at_t = torch.tensor(atc, device=device, dtype=torch.bfloat16)
                        st_t = torch.tensor(stc, device=device, dtype=torch.bfloat16)
                        denoised_step = raw_to_denoised(raw, x_t, at_t, st_t, teacher_type)
                    del raw, at_t, st_t, t_tensor, xc

                if step_i in keep_indices:
                    collected.append((
                        x_t.to("cpu", non_blocking=False).float().contiguous(),
                        atc, stc, t_val, traj_w,
                    ))

                if step_i == max_needed_step and max_needed_step < n_steps:
                    break

                x_t = _euler_step(x_t, denoised_step, stc, stn)
                # Add noise after every student step, including the last one.
                # Uses _this_step_is_student (captured before the else branch
                # could reset in_student_chain) so the final student step is
                # also covered.
                if _this_step_is_student and student_chain_noise > 0:
                    cpu_gen.manual_seed(derive_seed(seed, traj_idx * 100 + step_i, "chain_noise"))
                    noise = torch.randn(x_t.shape, generator=cpu_gen
                                        ).to(device=device, dtype=torch.bfloat16)
                    # With ComfyUI's schedule, sigma already represents the noise level.
                    # Use sigma_next as the noise scale (proportional to noise at next step).
                    x_t = x_t + student_chain_noise * stn * noise
                    del noise

                del denoised_step

            # Save the trajectory conditioning to CPU before freeing GPU memory.
            # TEACHER conditioning (traj_ctx / traj_ctx_unc) is used for the target
            # pass below.  STUDENT conditioning (cache ctx/y) is what gets stored in
            # the cache entry and is used for the student forward during training.
            # When no student prompts are set they are identical (normal behaviour).
            if precomputed_ctx is not None:
                # Prompt mode: precomputed tensors are already on CPU.
                traj_ctx_cpu     = precomputed_ctx.float().contiguous()       # teacher, for target pass
                traj_y_cpu       = precomputed_y.float().contiguous()
                traj_ctx_unc_cpu = (precomputed_ctx_unc.float().contiguous()
                                    if precomputed_ctx_unc is not None
                                    else torch.zeros_like(traj_ctx_cpu))
                traj_y_unc_cpu   = (precomputed_y_unc.float().contiguous()
                                    if precomputed_y_unc is not None
                                    else torch.zeros_like(traj_y_cpu))

                # Student conditioning for cache storage
                cache_ctx_cpu     = (precomputed_student_ctx.float().contiguous()
                                     if precomputed_student_ctx is not None
                                     else traj_ctx_cpu)
                cache_y_cpu       = (precomputed_student_y.float().contiguous()
                                     if precomputed_student_y is not None
                                     else traj_y_cpu)
                cache_ctx_unc_cpu = (precomputed_student_ctx_unc.float().contiguous()
                                     if precomputed_student_ctx_unc is not None
                                     else traj_ctx_unc_cpu)
                cache_y_unc_cpu   = (precomputed_student_y_unc.float().contiguous()
                                     if precomputed_student_y_unc is not None
                                     else traj_y_unc_cpu)
            else:
                # random / zero mode: save the live GPU tensors to CPU now.
                # No student prompt override is possible in these modes — student
                # conditioning is identical to teacher (random or zero).
                traj_ctx_cpu = ctx.to("cpu", non_blocking=False).float().contiguous()
                traj_y_cpu   = y.to("cpu", non_blocking=False).float().contiguous()
                if use_cfg:
                    traj_ctx_unc_cpu = ctx_unc.to("cpu", non_blocking=False).float().contiguous()
                    traj_y_unc_cpu   = y_unc.to("cpu", non_blocking=False).float().contiguous()
                else:
                    traj_ctx_unc_cpu = torch.zeros_like(traj_ctx_cpu)
                    traj_y_unc_cpu   = torch.zeros_like(traj_y_cpu)
                # In random/zero mode teacher == student conditioning
                cache_ctx_cpu     = traj_ctx_cpu
                cache_y_cpu       = traj_y_cpu
                cache_ctx_unc_cpu = traj_ctx_unc_cpu
                cache_y_unc_cpu   = traj_y_unc_cpu

            del x_t, ctx, y
            if use_cfg:
                del ctx_unc, y_unc

            # Pin all four conditioning tensors concurrently.
            # pin_memory() is a kernel mlock syscall that releases the GIL,
            # so threading gives real speedup here.
            ctx_cpu_p, y_cpu_p, ctx_u_cpu_p, y_u_cpu_p = pin_tensors_parallel(
                cache_ctx_cpu, cache_y_cpu, cache_ctx_unc_cpu, cache_y_unc_cpu
            )

            n_added = 0
            for (x_t_cpu, at, st, t_val, w) in collected:
                if len(cache) >= n_samples:
                    break

                x_t_gpu = x_t_cpu.to(device=device, dtype=torch.bfloat16)
                alpha_v = torch.tensor(at, device=device, dtype=torch.bfloat16)
                sigma_v = torch.tensor(st, device=device, dtype=torch.bfloat16)

                ctx_t   = traj_ctx_cpu.to(device=device, dtype=torch.bfloat16)
                y_t     = traj_y_cpu.to(device=device, dtype=torch.bfloat16)

                t_tensor = torch.tensor([t_val] * gen_batch_size,
                                        dtype=torch.long, device=device)
                xc = comfy_input_transform(x_t_gpu, st)

                raw_cond = teacher.forward(xc, t_tensor, ctx_t, y_t)
                del ctx_t, y_t

                if use_cfg:
                    # In sequential dual-pass mode, we store BOTH conditioned and unconditioned targets.
                    # Optimization: convert raw_cond now to free its VRAM before the next forward.
                    tgt_cond = raw_to_target(raw_cond.float(), x_t_gpu.float(),
                                             alpha_v.float(), sigma_v.float(),
                                             teacher_type, student_type)
                    del raw_cond

                    ctx_u_t = traj_ctx_unc_cpu.to(device=device, dtype=torch.bfloat16)
                    y_u_t   = traj_y_unc_cpu.to(device=device, dtype=torch.bfloat16)
                    raw_uncond = teacher.forward(xc, t_tensor, ctx_u_t, y_u_t)
                    del ctx_u_t, y_u_t
                    
                    tgt_uncond = raw_to_target(raw_uncond.float(), x_t_gpu.float(),
                                               alpha_v.float(), sigma_v.float(),
                                               teacher_type, student_type)
                    del raw_uncond
                    
                    # Safety clamp per-sample RMS for targets
                    TARGET_RMS_MAX = 10.0
                    for _tgt in (tgt_cond, tgt_uncond):
                        for _b in range(_tgt.shape[0]):
                            rms = _tgt[_b].pow(2).mean().sqrt()
                            if rms > TARGET_RMS_MAX:
                                _tgt[_b].mul_(TARGET_RMS_MAX / rms.clamp(min=1e-6))
                    
                    tgt_cond_cpu   = tgt_cond.to("cpu", non_blocking=True).float().contiguous()
                    tgt_uncond_cpu = tgt_uncond.to("cpu", non_blocking=True).float().contiguous()
                    del tgt_cond, tgt_uncond
                else:
                    raw_target = raw_cond.float()
                    del raw_cond
                    
                    tgt_combined = raw_to_target(raw_target, x_t_gpu.float(),
                                                 alpha_v.float(), sigma_v.float(),
                                                 teacher_type, student_type)
                    # Safety clamp per-sample RMS for targets
                    TARGET_RMS_MAX = 10.0
                    for _b in range(tgt_combined.shape[0]):
                        rms = tgt_combined[_b].pow(2).mean().sqrt()
                        if rms > TARGET_RMS_MAX:
                            tgt_combined[_b].mul_(TARGET_RMS_MAX / rms.clamp(min=1e-6))

                    tgt_cond_cpu   = tgt_combined.to("cpu", non_blocking=True).float().contiguous()
                    tgt_uncond_cpu = None
                    del raw_target, tgt_combined

                del xc, t_tensor

                # Pin x_t and target tensors concurrently — these are the
                # largest per-entry tensors and benefit most from parallel mlock.
                x_t_p, tgt_c_p, tgt_u_p = pin_tensors_parallel(
                    x_t_cpu, tgt_cond_cpu, tgt_uncond_cpu
                )

                # Cache entry:
                #   Standard: (x_t, target_cond, ctx, y, ctx_u, y_u, alpha, sigma, t_val, target_uncond)
                #   If no CFG: target_uncond is None
                entry = (
                    x_t_p, tgt_c_p,
                    ctx_cpu_p, y_cpu_p, ctx_u_cpu_p, y_u_cpu_p,
                    at, st, t_val,
                    tgt_u_p,
                )

                cache.append(entry)
                n_added += 1
                del x_t_gpu, tgt_cond_cpu, tgt_uncond_cpu, alpha_v, sigma_v

            del ctx_cpu_p, y_cpu_p, ctx_u_cpu_p, y_u_cpu_p
            # cache_* may alias traj_* when prompts are identical.
            # Capture identity BEFORE any del so the name is still bound.
            _cache_ctx_distinct     = cache_ctx_cpu     is not traj_ctx_cpu
            _cache_y_distinct       = cache_y_cpu       is not traj_y_cpu
            _cache_ctx_unc_distinct = cache_ctx_unc_cpu is not traj_ctx_unc_cpu
            _cache_y_unc_distinct   = cache_y_unc_cpu   is not traj_y_unc_cpu
            del traj_ctx_cpu, traj_y_cpu, traj_ctx_unc_cpu, traj_y_unc_cpu
            if _cache_ctx_distinct:     del cache_ctx_cpu
            if _cache_y_distinct:       del cache_y_cpu
            if _cache_ctx_unc_distinct: del cache_ctx_unc_cpu
            if _cache_y_unc_distinct:   del cache_y_unc_cpu

            if use_chain:
                n_student_trajs += 1
            traj_idx += 1
            n_trajectories += 1
            if pbar:
                pbar.update(n_added)

            # Progress update for server monitor (every iteration)
            if progress_writer:
                pct = min(100, len(cache) / n_samples * 100)
                progress_writer.cache_progress(
                    done=len(cache), total=n_samples, pct=pct)

    if pbar:
        pbar.close()
    cache = cache[:n_samples]

    # Only delete models if we created them (not if they were passed in)
    if teacher_model is None:
        del teacher
        xpu_empty_cache()
        gc.collect()

    if student_cache is not None and student_model is None:
        del student_cache
        xpu_empty_cache()
        gc.collect()

    if progress_writer:
        progress_writer.cache_done(n_samples=len(cache), n_trajs=n_trajectories)
    print(f"    Cache ready: {len(cache)} samples from {n_trajectories} trajectories "
          f"({n_student_trajs} student-chain, "
          f"{n_trajectories - n_student_trajs} teacher). Teacher freed.")
    return cache
