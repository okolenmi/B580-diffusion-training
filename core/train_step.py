"""Core training step implementation — accepts typed config."""

from __future__ import annotations

import gc as _gc
import queue
import sys
import threading
import time
import json
from collections import deque
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .comfy_setup import xpu_empty_cache
from .config_model import CommonSettings
from .lora import compute_lora_gate, set_lora_gate
from .model_io import comfy_input_transform, raw_to_denoised, raw_to_target
from .noise_schedule import get_alpha_sigma
from .optimizer_builder import build_optimizer, update_lr
from .optimizers import FusedXPUAdafactor
from .timer import StepTimer
from .save import save_midrun
from .unet_wrapper import ComfyUNetWrapper, clear_embedder_cache


import gc as _gc
_gc.disable()

# ---------------------------------------------------------------------------
# Background GC worker — keeps gc.collect() off the main thread so it can't
# fight with autograd backward hooks for the GIL and stall training.
# One pending collection at a time is enough; extras are silently dropped.
# ---------------------------------------------------------------------------
_gc_queue: queue.Queue = queue.Queue(maxsize=1)

def _gc_worker():
    while True:
        _gc_queue.get()
        _gc.collect()

_gc_thread = threading.Thread(target=_gc_worker, daemon=True, name="bg-gc")
_gc_thread.start()

def _request_gc():
    """Ask the background thread to run gc.collect(). Non-blocking."""
    try:
        _gc_queue.put_nowait(1)
    except queue.Full:
        pass   # a collection is already queued, skip

class ThreadedPrefetcher:
    """Asynchronously fetches batches from a generator using a thread."""
    def __init__(self, loader, device, buffer_size=8):
        self.loader = loader
        self.device = device
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._fill_queue, daemon=True)
        self.thread.start()

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.stream = torch.xpu.Stream()
        elif torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def _fill_queue(self):
        try:
            while not self.stopped:
                # Loop infinitely over the loader
                for batch in self.loader:
                    if self.stopped: break
                    self.queue.put(batch)
        except Exception as e:
            print(f"[Prefetcher Error] {e}")
            import traceback
            traceback.print_exc()
            self.queue.put(None)

    def _move(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        return obj

    def get_next(self):
        batch = self.queue.get()
        if batch is None: return None
        
        if self.stream:
            with torch.xpu.stream(self.stream) if hasattr(torch, "xpu") else torch.cuda.stream(self.stream):
                batch = self._move(batch)
            
            if hasattr(torch, "xpu"):
                torch.xpu.current_stream().wait_stream(self.stream)
            else:
                torch.cuda.current_stream().wait_stream(self.stream)
        else:
            batch = self._move(batch)
        
        return batch

    def close(self):
        self.stopped = True
        # Keep draining until the thread exits.  The thread may be blocked inside
        # queue.put() (queue full) or have just passed the self.stopped check but
        # not yet called put().  Draining in a loop unblocks it each time so it
        # can re-check self.stopped and exit cleanly.
        import time as _time
        deadline = _time.monotonic() + 5.0
        while self.thread.is_alive() and _time.monotonic() < deadline:
            while not self.queue.empty():
                try: self.queue.get_nowait()
                except queue.Empty: break
            self.thread.join(timeout=0.1)
        # Final drain so callers don't see stale batches after close()
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break

    def sync(self):
        """Synchronize the prefetch stream with the default stream.

        Call this before operations (like preview generation) that will
        perform heavy GPU work and memory management outside the normal
        training loop. This ensures no async H2D transfers are in flight,
        preventing xpu_empty_cache() calls from corrupting prefetch state
        and causing subsequent xpu_synchronize() calls to deadlock.
        """
        if self.stream is not None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.current_stream().wait_stream(self.stream)
                self.stream.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.current_stream().wait_stream(self.stream)
                self.stream.synchronize()


class CachePrefetcher:
    """Handles async transfer of cache batches to device."""
    def __init__(self, cache: List[Tuple], device: torch.device):
        self.cache = cache
        self.device = device
        self.stream = None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.stream = torch.xpu.Stream()
        elif torch.cuda.is_available():
            self.stream = torch.cuda.Stream()

        self.next_batch = None
        self.idx = 0
        self._prefetch()

    def _prefetch(self):
        if not self.cache:
            self.next_batch = None
            return

        if self.idx >= len(self.cache):
            print("\n[WARNING] Cache have less elements than cycle needs. Cache will be re-used to finish cycle.")
            self.idx = 0

        entry = self.cache[self.idx]
        self.idx += 1

        def _move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device, non_blocking=True)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            return obj

        if self.stream:
            with torch.xpu.stream(self.stream) if hasattr(torch, "xpu") else torch.cuda.stream(self.stream):
                self.next_batch = _move(entry)
        else:
            self.next_batch = _move(entry)

    def get_next(self):
        if self.stream:
            if hasattr(torch, "xpu"): torch.xpu.current_stream().wait_stream(self.stream)
            else: torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch
        self._prefetch()
        return batch

    def close(self):
        if self.stream is not None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self.stream.synchronize()
            elif torch.cuda.is_available():
                self.stream.synchronize()
        self.next_batch = None
        self.cache = None
        self.stream = None
        _gc.collect()

    def sync(self):
        """Synchronize the prefetch stream with the default stream.

        Call this before operations (like preview generation) that will
        perform heavy GPU work and memory management outside the normal
        training loop. This ensures no async H2D transfers are in flight,
        preventing xpu_empty_cache() calls from corrupting prefetch state
        and causing subsequent xpu_synchronize() calls to deadlock.
        """
        if self.stream is not None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.current_stream().wait_stream(self.stream)
                self.stream.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.current_stream().wait_stream(self.stream)
                self.stream.synchronize()


def _make_weight_track() -> Dict[str, Any]:
    return {
        # deque(maxlen=100): O(1) append/pop, Welford running mean+variance
        "loss_win": deque(maxlen=100),
        "loss_mean": 0.0,   # running mean (Welford)
        "loss_M2":   0.0,   # running sum of squared deviations
        "loss_n":    0,     # count of samples seen
        "last_W": None,
        "delta_total": 0.0,
        "delta_count": 0,
        "_tracked_param": None,
        "_last_lr": None,   # cache to skip redundant update_lr calls
    }


def _find_tracked_param(student: nn.Module) -> torch.Tensor | None:
    """Return the first 2D-or-higher trainable parameter for weight tracking.

    Prefer trainable params (LoRA A/B) over frozen ones so the metric
    actually reflects learning progress.  Falls back to any 2D param.
    Result is cached in weight_track to avoid repeated scans.
    """
    # Prefer trainable params (most meaningful for LoRA)
    for p in student.parameters():
        if p.requires_grad and p.ndim >= 2:
            return p
    # Fall back to any 2D param (full fine-tune with all params frozen would be odd)
    for p in student.parameters():
        if p.ndim >= 2:
            return p
    return None


def _generate_drafts(
        student: nn.Module,
        xc: torch.Tensor,
        t_tensor: torch.Tensor,
        ctx: torch.Tensor,
        y: torch.Tensor,
        ctx_u: torch.Tensor,
        y_u: torch.Tensor,
        device: torch.device,
        power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate low-power opponent drafts for adversarial pre-conditioning.

    Returns (xc_for_cond, xc_for_uncond):
      xc_for_cond   = xc + power * uncond_draft   (cond pass sees what uncond would do)
      xc_for_uncond = xc + power * cond_draft     (uncond pass sees what cond would do)

    Both forwards run under no_grad — drafts influence the main passes as input
    perturbations only.  Gradients flow through the main passes, not the drafts.
    """
    with torch.no_grad():
        c_bf  = ctx.to(device, dtype=torch.bfloat16)
        y_bf  = y.to(device, dtype=torch.bfloat16)
        cu_bf = ctx_u.to(device, dtype=torch.bfloat16)
        yu_bf = y_u.to(device, dtype=torch.bfloat16)

        # Draft for cond pass: opponent = uncond
        uncond_draft = student.forward(xc, t_tensor, cu_bf, yu_bf)
        xc_for_cond  = xc + power * uncond_draft.to(xc.dtype)
        del uncond_draft

        # Draft for uncond pass: opponent = cond
        cond_draft    = student.forward(xc, t_tensor, c_bf, y_bf)
        xc_for_uncond = xc + power * cond_draft.to(xc.dtype)
        del cond_draft, c_bf, y_bf, cu_bf, yu_bf

    return xc_for_cond, xc_for_uncond


_RESOLUTION_CACHE = {}

def _get_resolution_embeddings(width: int, height: int, device: torch.device, dtype: torch.dtype):
    """Get or create cached SDXL resolution embeddings (1536-dim) for a given shape."""
    key = (width, height, str(device), dtype)
    if key in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[key]
    
    from comfy.model_base import Timestep
    embedder = Timestep(256).to(device=device, dtype=dtype)
    
    # original_h, original_w, crop_h, crop_w, target_h, target_w
    # For training we assume crop=0 and target=original
    vals = [height, width, 0, 0, height, width]
    time_embs = []
    for val in vals:
        time_embs.append(embedder(torch.tensor([val], device=device, dtype=dtype)))
    
    res_emb = torch.cat(time_embs, dim=-1) # (1, 1536)
    _RESOLUTION_CACHE[key] = res_emb
    return res_emb


def _run_one_step(
        student: nn.Module,
        optimizer: Any,
        prefetcher: CachePrefetcher,
        config: CommonSettings,
        device: torch.device,
        global_step: int,
        start_step: int,
        total_steps: int,
        expected_seed: int,
        lr_fn: Callable[[int], float],
        snr_weighting: str,
        is_fused: bool,
        timer: StepTimer,
        t0: float,
        weight_track: Dict[str, Any],
        pbar: tqdm | None = None,
        save_dtype: torch.dtype = torch.float16,
        stop_flag: Callable[[], bool] | None = None,
        progress_writer: Any = None,
        student_encoder: Any = None,
        prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None = None,
        gate_train_low: float | None = None,
        gate_train_high: float | None = None,
        gate_width: float = 100.0,
        trace: bool = False,
) -> bool:
    """Run a single optimization step."""
    if trace:
        print(f"  [trace] step {global_step}: calling prefetcher.get_next()", flush=True)
    _entry = prefetcher.get_next()
    if trace:
        print(f"  [trace] step {global_step}: get_next() returned", flush=True)
    if _entry is None:
        return True

    repulsive_alpha = 0.0  # experimental, not exposed in config model
    traj_type = _entry.get("traj_type", "good") if isinstance(_entry, dict) else "good"

    if isinstance(_entry, dict):
        x_t = _entry["x_t"]
        t_val = _entry["t"]
        at_t, st_t = get_alpha_sigma(t_val)
        
        # Resolve shape-aware conditioning
        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        res_emb = _get_resolution_embeddings(batch_w, batch_h, device, torch.bfloat16)

        if _entry.get("target_p") is not None and _entry.get("target_n") is not None:
            target_c = _entry["target_p"]
            target_u = _entry["target_n"]
        else:
            target_c = _entry["target"]
            target_u = None

        if prompt_cache is not None:
            prompt = _entry.get("prompt", "")
            neg_prompt = _entry.get("neg_prompt", "")

            if prompt in prompt_cache:
                ctx_base, pooled_base = prompt_cache[prompt]
                ctx = ctx_base.to(device, dtype=torch.bfloat16).repeat(x_t.shape[0], 1, 1)
                pooled = pooled_base.to(device, dtype=torch.bfloat16).repeat(x_t.shape[0], 1)

                # Assemble 'y' vector: [pooled (1280), resolution (1536)] = 2816
                y = torch.cat([pooled, res_emb.repeat(x_t.shape[0], 1)], dim=-1)
            else:
                ctx, y = student_encoder.encode_for_unet(prompt, batch_size=x_t.shape[0],
                                                          height=batch_h, width=batch_w)

            if neg_prompt in prompt_cache:
                ctx_u_base, pooled_u_base = prompt_cache[neg_prompt]
                ctx_u = ctx_u_base.to(device, dtype=torch.bfloat16).repeat(x_t.shape[0], 1, 1)
                pooled_u = pooled_u_base.to(device, dtype=torch.bfloat16).repeat(x_t.shape[0], 1)
                y_u = torch.cat([pooled_u, res_emb.repeat(x_t.shape[0], 1)], dim=-1)
            else:
                ctx_u, y_u = student_encoder.encode_for_unet(neg_prompt, batch_size=x_t.shape[0],
                                                              height=batch_h, width=batch_w)
        elif student_encoder is not None:
            prompt = _entry.get("prompt", "")
            neg_prompt = _entry.get("neg_prompt", "")
            ctx, y = student_encoder.encode_for_unet(prompt, batch_size=x_t.shape[0],
                                                      height=batch_h, width=batch_w)
            ctx_u, y_u = student_encoder.encode_for_unet(neg_prompt, batch_size=x_t.shape[0],
                                                          height=batch_h, width=batch_w)
        else:
            ctx = _entry.get("ctx")
            y = _entry.get("y")
            ctx_u = _entry.get("ctx_u")
            y_u = _entry.get("y_u")
    else:
        x_t, target_c, ctx, y, ctx_u, y_u, at_t, st_t, t_val = _entry[:9]
        target_u = _entry[9] if len(_entry) >= 10 else None

    timer.start("1_transform")
    xc = comfy_input_transform(x_t, st_t)
    if isinstance(t_val, torch.Tensor):
        # Already a tensor (e.g. from ManagedDatasetLoader batch); ensure correct
        # device, dtype, and shape — flatten to (B,) in case it came in as (B,1).
        t_tensor = t_val.to(device=device, dtype=torch.long).view(-1)
    else:
        t_tensor = torch.tensor([t_val] * x_t.shape[0], dtype=torch.long, device=device)
    timer.stop("1_transform")

    if gate_train_low is not None:
        set_lora_gate(compute_lora_gate(t_tensor, gate_train_low, gate_train_high, gate_width))
    else:
        set_lora_gate(None)

    # grad_accum is meaningful only for ChunkedXPUAdafactor / CPUAdamW.
    # FusedXPUAdafactor applies updates inside backward hooks immediately,
    # so accumulation across multiple backward() calls is not supported.
    effective_accum = 1 if is_fused else config.grad_accum

    # Zero gradients only at the start of each accumulation cycle -- not
    # every step. Zeroing every step would wipe out gradients accumulated
    # from prior steps in the same cycle before optimizer.step() (gated on
    # the cycle boundary below) ever sees them, silently defeating grad
    # accumulation entirely regardless of what grad_accum is set to.
    # FusedXPUAdafactor frees grads inside its backward hooks — zero_grad
    # is a no-op there regardless of this gating (effective_accum is
    # always 1 for it, so the condition is always true anyway).
    if global_step % effective_accum == 0:
        optimizer.zero_grad()

    # Update LR only when the schedule value actually changed (saves a linear scan
    # over param_lr on every step for uniform LR with ChunkedXPUAdafactor).
    new_lr = lr_fn(global_step)
    if new_lr != weight_track.get("_last_lr"):
        update_lr(optimizer, new_lr)
        weight_track["_last_lr"] = new_lr

    # Determine how many backward passes will run this step so we can scale
    # each pass's loss correctly.  Both cond and uncond contribute equally to
    # the parameter gradient; dividing by n_passes keeps the total gradient
    # magnitude equivalent to a single-pass step regardless of whether we have
    # separate cond/uncond targets.
    n_passes = 2 if target_u is not None else 1
    loss_scale = effective_accum * n_passes

    # For FusedXPUAdafactor: reset the per-backward flag so self.t increments
    # exactly once for the first backward pass this step.
    # We inform it that n_passes are required for a full update.
    if is_fused:
        optimizer.begin_step(sub_steps=n_passes)

    # Adversarial pre-conditioning (experimental, opt-in via config).
    # Only active when both cond and uncond targets exist — single-pass steps
    # have no opponent to generate a draft from.
    xc_cond  = xc  # default: unmodified input for cond pass
    xc_uncond = xc  # default: unmodified input for uncond pass
    _using_pre_cond = False

    if (config.pre_cond_enable
            and target_u is not None
            and ctx_u is not None
            and y_u is not None):
        import random as _random
        if _random.random() >= config.pre_cond_clean_ratio:
            power = _random.uniform(config.pre_cond_power_min,
                                    config.pre_cond_power_max)
            timer.start("1b_pre_cond_drafts")
            xc_cond, xc_uncond = _generate_drafts(
                student, xc, t_tensor,
                ctx, y, ctx_u, y_u,
                device, power,
            )
            timer.stop("1b_pre_cond_drafts")
            _using_pre_cond = True

    # Pre-cast text embeddings to bfloat16 once per step.
    # This avoids doing it 2-4 times inside _run_pass.
    c_bf  = ctx.to(device, dtype=torch.bfloat16)
    y_bf  = y.to(device, dtype=torch.bfloat16)
    cu_bf = ctx_u.to(device, dtype=torch.bfloat16) if ctx_u is not None else None
    yu_bf = y_u.to(device, dtype=torch.bfloat16) if y_u is not None else None

    # Pre-calculate SNR weighting factor if applicable.
    # This factor is constant for both passes in a single step.
    snr_w = 1.0
    if snr_weighting in ("snr", "min_snr_5", "inverse_snr", "decay_snr"):
        def _to_scalar(v):
            if isinstance(v, torch.Tensor):
                return v.float().mean().item()
            return float(v)
        st_f = _to_scalar(st_t)
        # SNR = (1/sigma)^2 for the x_t = x0 + sigma*eps formulation
        snr = 1.0 / (st_f**2 + 1e-8)
        
        if snr_weighting == "snr":
            snr_w = snr / (snr + 1.0)
        elif snr_weighting == "min_snr_5":
            snr_w = min(snr, 5.0) / snr
        elif snr_weighting == "decay_snr":
            # Hybrid: Uniform (1.0) at high noise, SNR at low noise.
            # Smoothly transition based on sigma.
            snr_val = snr / (snr + 1.0)
            # sigma > 1.0 is high noise; sigma < 0.1 is low noise.
            # We use a sigmoid-like blend.
            weight_mix = torch.sigmoid(torch.tensor(st_f * 2.0 - 1.0)).item()
            snr_w = weight_mix + (1.0 - weight_mix) * snr_val
        else:  # inverse_snr
            snr_w = 1.0 / (snr + 1.0)

    def _run_pass(xc_in, t_in, c_in, y_in_v, target_in, timer_label):
        if trace:
            print(f"  [trace] step {global_step}: {timer_label} forward starting", flush=True)
        timer.start(timer_label)
        student_out = student.forward(xc_in, t_in, c_in, y_in_v)
        if trace:
            print(f"  [trace] step {global_step}: {timer_label} forward done, computing loss/backward", flush=True)
        # Keep per-sample loss for SNR weighting; reduce to scalar afterward.
        per_sample = (student_out.float() - target_in.to(device).float()).pow(2)
        # per_sample shape: (B, C, H, W) — mean over all dims except batch
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)  # (B,)

        loss = (per_sample * snr_w).mean()

        if traj_type == "bad" and repulsive_alpha > 0.0:
            loss = repulsive_alpha / (loss + 1.0)

        loss = loss / loss_scale
        loss.backward()
        if trace:
            print(f"  [trace] step {global_step}: {timer_label} backward done", flush=True)

        del student_out
        timer.stop(timer_label)
        return loss

    loss_c = _run_pass(xc_cond, t_tensor, c_bf, y_bf, target_c, "2_student_cond")
    del target_c, ctx, y, c_bf, y_bf

    if target_u is not None:
        if is_fused:
            optimizer.prepare_next_pass()
        loss_u = _run_pass(xc_uncond, t_tensor, cu_bf, yu_bf, target_u, "3_student_uncond")
        del target_u, ctx_u, y_u, cu_bf, yu_bf
        loss_display_tensor = (loss_c + loss_u) * (loss_scale / 2)
    else:
        if ctx_u is not None: del ctx_u
        if y_u is not None: del y_u
        if cu_bf is not None: del cu_bf
        if yu_bf is not None: del yu_bf
        loss_display_tensor = loss_c * loss_scale

    if not is_fused and (global_step + 1) % effective_accum == 0:
        if trace:
            print(f"  [trace] step {global_step}: optimizer.step() starting", flush=True)
        timer.start("5_opt_step")
        optimizer.step(n_steps=effective_accum)
        timer.stop("5_opt_step")
        if trace:
            print(f"  [trace] step {global_step}: optimizer.step() done", flush=True)

    del xc, t_tensor
    if _using_pre_cond:
        del xc_cond, xc_uncond

    # Retrieve the loss value. We do this every step for real-time feedback.
    # On XPU, this causes a host-device synchronization, but at ~1 step/sec
    # the impact is negligible.
    if trace:
        print(f"  [trace] step {global_step}: calling loss_display_tensor.item() (XPU sync point)", flush=True)
    lv = loss_display_tensor.item()
    if trace:
        print(f"  [trace] step {global_step}: loss_display_tensor.item() returned ({lv:.5f})", flush=True)
    weight_track["_last_lv"] = lv
    del loss_display_tensor

    # Welford-style window stats using deque(maxlen=100).
    # deque handles eviction automatically; we recompute avg/std over the window.
    # For maxlen=100 this is 100 additions — cheap and numerically stable.
    loss_win = weight_track["loss_win"]
    loss_win.append(lv)
    n = len(loss_win)
    avg = sum(loss_win) / n
    std = (sum((x - avg) ** 2 for x in loss_win) / n) ** 0.5

    # Weight tracking — only every 100 steps, uses a cached param reference
    # so we never scan the entire parameter list at runtime.
    if (global_step + 1) % 100 == 0:
        if trace:
            print(f"  [trace] step {global_step}: weight-tracking block (XPU sync point)", flush=True)
        tracked_p = weight_track.get("_tracked_param")
        if tracked_p is None:
            tracked_p = _find_tracked_param(student)
            weight_track["_tracked_param"] = tracked_p

        if tracked_p is not None:
            # .abs().mean() is a single reduction — one sync, unavoidable here
            curr_W = tracked_p.data.detach().float().abs().mean().item()
            if weight_track["last_W"] is not None:
                delta = abs(curr_W - weight_track["last_W"])
                weight_track["delta_total"] += delta
                weight_track["delta_count"] += 1
            weight_track["last_W"] = curr_W
        if trace:
            print(f"  [trace] step {global_step}: weight-tracking block done", flush=True)

    _wd = (weight_track["delta_total"] / weight_track["delta_count"]
           if weight_track["delta_count"] > 0 and (global_step + 1) >= 100 else 0.0)

    elapsed = time.time() - t0
    steps_done = (global_step - start_step) + 1
    eta = (elapsed / steps_done) * (total_steps - (global_step + 1)) if steps_done > 0 else 0

    if pbar is not None:
        pbar.set_postfix(loss=f"{lv:.5f}", avg=f"{avg:.5f}", std=f"{std:.5f}",
                         lr=f"{optimizer.lr:.1e}",
                         dW=f"{_wd:.3f}",
                         ETA=f"{eta/60:.0f}m")

    if not sys.stdout.isatty() and (global_step + 1) % 100 == 0:
        print(f"Step {global_step+1:5d}/{total_steps}: loss={lv:.5f} avg={avg:.5f} lr={optimizer.lr:.2e}")

    if progress_writer is not None:
        if trace:
            print(f"  [trace] step {global_step}: progress_writer.step() starting (file I/O)", flush=True)
        progress_writer.step(global_step=global_step, total_steps=total_steps,
                             loss=lv, avg=avg, std=std, lr=optimizer.lr,
                             eta_sec=eta if eta > 0 else None)
        if trace:
            print(f"  [trace] step {global_step}: progress_writer.step() done", flush=True)

    timer.tick()
    if trace:
        print(f"  [trace] step {global_step}: end of _run_one_step", flush=True)
    if stop_flag and stop_flag():
        print(f"\n  Stopping at step {global_step + 1}.")
        return True
    return False


def run_training_loop(
        student, optimizer, cache, config,
        device,
        start_step, run_steps, total_steps,
        expected_seed, lr_fn,
        save_dtype,
        timer=None, t0=None,
        stop_flag=None,
        weight_track=None,
        progress_writer=None,
        student_encoder=None,
        prompt_cache=None,
        save_callback=None,
        preview_callback=None,
):
    if timer is None: timer = StepTimer(report_every=100)
    if t0 is None: t0 = time.time()
    if stop_flag is None: stop_flag = lambda: False
    if weight_track is None: weight_track = _make_weight_track()

    _snr_w = config.common.snr_weighting
    _is_fused = isinstance(optimizer, FusedXPUAdafactor)
    _save_on_crash = config.common.save_on_crash
    # Timestep-gated LoRA: only meaningful for LoRA tuning, and only if the
    # user actually configured a protected interval (None/None keeps current
    # behavior -- LoRA applies uniformly across all timesteps, no gating at all).
    _gate_train_low = getattr(config.tuning, "gate_train_low", None)
    _gate_train_high = getattr(config.tuning, "gate_train_high", None)
    _gate_width = getattr(config.tuning, "gate_width", 100.0)
    # loss_win is a deque(maxlen=100) created by _make_weight_track().
    # If weight_track was passed in from a previous cycle it already has the deque.
    if "loss_win" not in weight_track or not isinstance(weight_track["loss_win"], deque):
        weight_track["loss_win"] = deque(maxlen=100)

    is_tty = sys.stdout.isatty()
    pbar = None
    if is_tty:
        pbar = tqdm(range(run_steps), desc="Training", unit="step", ncols=0, mininterval=5.0)

    if progress_writer is not None:
        _cycles = (run_steps // max(getattr(config.tuning, "cycle_steps", run_steps), 1)) or 1
        progress_writer.training_start(run_steps=run_steps, total_steps=total_steps, cycles=_cycles)

    steps_done = 0
    stopped = False
    crashed = False

    if isinstance(cache, list):
        prefetcher = CachePrefetcher(cache, device)
    else:
        prefetcher = ThreadedPrefetcher(cache, device)

    try:
        trace_remaining = 0
        for i in range(run_steps):
            global_step = start_step + i
            trace_this_step = trace_remaining > 0
            if trace_this_step:
                trace_remaining -= 1
                print(f"  [trace] step {global_step}: begin (post-preview tracing active)", flush=True)

            stopped = _run_one_step(
                student=student, optimizer=optimizer, prefetcher=prefetcher,
                config=config.common, device=device, global_step=global_step,
                start_step=start_step, total_steps=total_steps,
                expected_seed=expected_seed, lr_fn=lr_fn,
                snr_weighting=_snr_w, is_fused=_is_fused,
                timer=timer, t0=t0,
                weight_track=weight_track, pbar=pbar,
                save_dtype=save_dtype, stop_flag=stop_flag,
                progress_writer=progress_writer,
                student_encoder=student_encoder,
                prompt_cache=prompt_cache,
                gate_train_low=_gate_train_low,
                gate_train_high=_gate_train_high,
                gate_width=_gate_width,
                trace=trace_this_step,
            )

            if pbar is not None: pbar.update(1)
            steps_done += 1

            # Maintenance: clear XPU cache and trigger background GC.
            # Increased interval (250 steps) to minimize periodic stalls.
            if (global_step + 1) % 250 == 0:
                xpu_empty_cache()
                _request_gc()

            if config.common.save_every > 0 and (global_step + 1) % config.common.save_every == 0:
                if save_callback and (global_step + 1) > 0:
                    save_callback(global_step + 1)

            if (config.preview.enabled and config.preview.every_n_steps > 0
                    and (global_step + 1) % config.preview.every_n_steps == 0):
                if preview_callback:
                    # CRITICAL: Sync the prefetch stream before preview generation.
                    # The CachePrefetcher has a separate GPU stream for async H2D
                    # transfers. If preview's heavy GPU work and xpu_empty_cache()
                    # calls run while a prefetch transfer is in flight, the XPU
                    # driver can corrupt stream state. After 2-4 previews this
                    # causes xpu_synchronize() to deadlock with 0 GPU usage.
                    prefetcher.sync()
                    preview_callback(global_step + 1)
                    print(f"  [preview] step {global_step + 1}: callback returned, "
                          f"loop continuing to next step", flush=True)
                    trace_remaining = 3

            if stopped: break
    except Exception:
        crashed = True
        stopped = True
        raise
    finally:
        prefetcher.close()
        _gc.collect()
        if pbar is not None: pbar.close()
        # Emergency mid-run save on crash so progress is not completely lost.
        # Only fires when save_on_crash=true and we have a save_callback to call.
        if crashed and _save_on_crash and save_callback is not None:
            try:
                step_n = start_step + steps_done
                if step_n > 0:
                    print(f"  [save_on_crash] Saving emergency checkpoint at step {step_n}...")
                    save_callback(step_n)
                    print("  [save_on_crash] Done.")
                else:
                    print("  [save_on_crash] Step 0, skipping save.")
            except Exception as e:
                print(f"  [save_on_crash] Failed: {e}")

    return steps_done, stopped, weight_track
