"""Save infrastructure — checkpoints, atomic saves, optimizer state persistence."""

import gc
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from .lora import extract_lora_weights, load_lora_into_model
from .optimizers import CPUAdamW, FusedXPUAdafactor
from .unet_wrapper import ComfyUNetWrapper


def _derive_path(base: str, suffix: str) -> str:
    """Derive a sibling path from the output path by replacing the suffix."""
    p = Path(base)
    return str(p.parent / (p.stem + suffix))


def _atomic_save(tensors: dict, path: str):
    """Save via temp file then atomic rename to prevent corruption."""
    tmp = path + ".tmp"
    save_file(tensors, tmp)
    Path(tmp).replace(Path(path))


def save_checkpoint(student: ComfyUNetWrapper, non_unet: dict,
                    path: str, save_dtype: torch.dtype):
    """Final save -- streams weights key-by-key to minimise peak RAM.

    Builds the output dict incrementally so we never hold both the live
    BF16 model tensors and the full FP16 output copies in RAM at the same
    time.  non_unet (VAE, CLIP) is merged last; the caller must have already
    freed the optimizer and teacher state dicts before calling here.
    """
    student.eval()
    out = {}
    with torch.no_grad():
        for k, v in student.state_dict().items():
            k = k.replace("_orig_mod.", "")
            # Cast one tensor at a time and immediately drop the source
            # reference so the BF16 page can be reclaimed before the next
            # iteration allocates its FP16 copy.
            out[f"model.diffusion_model.{k}"] = v.to("cpu", dtype=save_dtype).contiguous()
            del v
    gc.collect()
    if non_unet:
        out.update(non_unet)
    print(f"    Writing {len(out)} keys...")
    _atomic_save(out, path)
    del out
    gc.collect()
    student.train()
    print(f"    Saved: {path}")


def xpu_empty_cache():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def save_midrun(student, optimizer,
                resume_checkpoint, resume_optimizer, step, save_dtype,
                non_unet: dict = None):
    """Mid-training checkpoint: weights + optimizer state."""
    # Create resume directory if needed
    resume_dir = Path(resume_checkpoint).parent
    resume_dir.mkdir(parents=True, exist_ok=True)
    
    tqdm.write(f"\n  [checkpoint] step {step} - saving to:\n      {resume_checkpoint}\n      {resume_optimizer}")
    xpu_empty_cache()

    # Remove fused hooks first — must happen before reading optimizer state
    # to guarantee no hook fires while we are iterating vr/vc/vs.
    is_fused = isinstance(optimizer, FusedXPUAdafactor)
    if is_fused:
        optimizer.remove_hooks()

    # Save optimizer states (step is saved in optstate file)
    tqdm.write("    Optimizer states...")
    opt_out = {
        "__step__": torch.tensor(step, dtype=torch.int32),
        "__t__": torch.tensor(optimizer.t, dtype=torch.int32),
    }
    if isinstance(optimizer, CPUAdamW):
        opt_out["__type__"] = torch.tensor(0, dtype=torch.int32)
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            opt_out[f"m_{i}"] = m.to(dtype=torch.bfloat16).contiguous()
            opt_out[f"v_{i}"] = v.to(dtype=torch.bfloat16).contiguous()
    else:  # ChunkedXPUAdafactor or FusedXPUAdafactor
        opt_out["__type__"] = torch.tensor(1, dtype=torch.int32)
        for i in range(len(optimizer.vr)):
            if optimizer.vr[i] is not None:
                opt_out[f"vr_{i}"] = optimizer.vr[i].cpu().float().contiguous()
            if optimizer.vc[i] is not None:
                opt_out[f"vc_{i}"] = optimizer.vc[i].cpu().float().contiguous()
            if optimizer.vs[i] is not None:
                opt_out[f"vs_{i}"] = optimizer.vs[i].cpu().float().flatten().contiguous()
            if optimizer.exp_avg[i] is not None:
                opt_out[f"ea_{i}"] = optimizer.exp_avg[i].cpu().float().contiguous()
        # Chunked uses a shared _tiny_vs flat tensor; Fused uses per-param _tiny_vs_map.
        if isinstance(optimizer, FusedXPUAdafactor):
            for idx, v in getattr(optimizer, "_tiny_vs_map", {}).items():
                opt_out[f"__tiny_vs_{idx}__"] = v.cpu().float().contiguous()
        elif getattr(optimizer, "_tiny_vs", None) is not None:
            opt_out["__tiny_vs__"] = optimizer._tiny_vs.cpu().float().contiguous()
    _atomic_save(opt_out, resume_optimizer)
    del opt_out
    gc.collect()

    # Save weights — LoRA mode: only adapter weights; full mode: entire UNet + non_unet
    student.eval()
    if student.has_lora():
        # LoRA resume: store only the trained adapter weights (lora_down / lora_up / alpha).
        # Saving the full state_dict would embed base_weight buffers under non-standard
        # keys that cannot be loaded back by the normal UNet or LoRA loaders.
        tqdm.write("    Weights (LoRA adapters)...")
        lora_weights = student.get_lora_weights()
        if not lora_weights:
            tqdm.write("    [WARN] get_lora_weights() returned empty dict — skipping weight save.")
        else:
            _atomic_save(lora_weights, resume_checkpoint)
            tqdm.write(f"    LoRA resume saved: {len(lora_weights)} tensors.")
        del lora_weights
    else:
        tqdm.write("    Weights (UNet" + (" + CLIP/VAE" if non_unet else "") + ")...")
        weights_out = {}
        with torch.no_grad():
            for k, v in student.state_dict().items():
                k = k.replace("_orig_mod.", "")
                weights_out[f"model.diffusion_model.{k}"] = v.to("cpu", dtype=torch.bfloat16).contiguous()

        if non_unet:
            # Filter and cast non_unet weights
            for k, v in non_unet.items():
                if isinstance(v, torch.Tensor):
                    weights_out[k] = v.to("cpu", dtype=torch.bfloat16).contiguous()
                else:
                    weights_out[k] = v

        _atomic_save(weights_out, resume_checkpoint)
        del weights_out

    gc.collect()
    student.train()
    if is_fused:
        optimizer.register_hooks()

    tqdm.write("    Done.\n")


def peek_resume_step(opt_path) -> int:
    """Read just the saved step number out of an optimizer-state file.

    Used by Trainer at construction time to resolve start_step *before* the
    optimizer object even exists, so we don't need a full load_optstate()
    round-trip (or a fragile "set this attribute after the fact" pattern)
    just to know where to resume from. Returns 0 if the file is missing,
    unreadable, or has no __step__ entry.
    """
    p = Path(opt_path)
    if not p.exists():
        return 0
    try:
        from safetensors import safe_open
        with safe_open(str(p), framework="pt", device="cpu") as f:
            if "__step__" in f.keys():
                return int(f.get_tensor("__step__").item())
    except Exception as e:
        print(f"    [WARN] Could not peek resume step from {opt_path}: {e}")
    return 0


def load_optstate(optimizer, opt_path):
    """Load optimizer states from disc. Returns the step number."""
    sd = load_file(opt_path)
    saved_t = int(sd["__t__"].item())
    saved_step = int(sd["__step__"].item())
    optimizer.t = saved_t
    skipped = 0
    restored = 0
    if isinstance(optimizer, CPUAdamW):
        for i in range(len(optimizer.m)):
            if f"m_{i}" in sd:
                if sd[f"m_{i}"].shape == optimizer.m[i].shape:
                    optimizer.m[i].copy_(sd[f"m_{i}"].to(dtype=torch.float32))
                    optimizer.v[i].copy_(sd[f"v_{i}"].to(dtype=torch.float32))
                    restored += 1
                else:
                    skipped += 1
                    tqdm.write(f"    [WARN] CPUAdamW param {i}: shape mismatch "
                               f"saved={sd[f'm_{i}'].shape} vs expected={optimizer.m[i].shape}")
    else:  # ChunkedXPUAdafactor or FusedXPUAdafactor
        for i in range(len(optimizer.vr)):
            p = optimizer.params[i]
            p_rows = p.shape[0]
            p_cols = p.numel() // p_rows if p.dim() >= 2 else 0

            # Load vr and vc independently (they may not always both exist)
            if f"vr_{i}" in sd:
                saved_vr = sd[f"vr_{i}"]
                if saved_vr.shape == (p_rows,):
                    optimizer.vr[i] = saved_vr.to(device="cpu", dtype=torch.float32)
                    restored += 1
                else:
                    skipped += 1
                    if skipped <= 5:
                        tqdm.write(f"    [WARN] vr param {i}: shape mismatch "
                                   f"saved={saved_vr.shape} vs expected=({p_rows},)")
            if f"vc_{i}" in sd:
                saved_vc = sd[f"vc_{i}"]
                if saved_vc.shape == (p_cols,):
                    optimizer.vc[i] = saved_vc.to(device="cpu", dtype=torch.float32)
                    restored += 1
                else:
                    skipped += 1
                    if skipped <= 5:
                        tqdm.write(f"    [WARN] vc param {i}: shape mismatch "
                                   f"saved={saved_vc.shape} vs expected=({p_cols},)")

            if f"vs_{i}" in sd:
                saved_vs = sd[f"vs_{i}"]
                # vs is saved flattened; accept any shape that has the right numel
                if saved_vs.numel() == p.numel():
                    optimizer.vs[i] = saved_vs.float().flatten()
                    restored += 1
                else:
                    skipped += 1
                    if skipped <= 5:
                        tqdm.write(f"    [WARN] vs param {i}: numel mismatch "
                                   f"saved={saved_vs.numel()} vs expected={p.numel()}")
            if f"ea_{i}" in sd:
                if sd[f"ea_{i}"].shape == tuple(p.shape):
                    optimizer.exp_avg[i] = sd[f"ea_{i}"].to(device="cpu", dtype=torch.float32)
                    restored += 1
                else:
                    skipped += 1
                    if skipped <= 5:
                        tqdm.write(f"    [WARN] exp_avg param {i}: shape mismatch "
                                   f"saved={sd[f'ea_{i}'].shape} vs expected={tuple(p.shape)}")
        # Restore tiny-param second moments.
        # FusedXPUAdafactor uses per-index _tiny_vs_map; ChunkedXPUAdafactor uses flat _tiny_vs.
        opt_device = getattr(optimizer, "device", "cpu")
        if isinstance(optimizer, FusedXPUAdafactor):
            tiny_map = {}
            for k in sd:
                if k.startswith("__tiny_vs_") and k.endswith("__"):
                    idx = int(k[len("__tiny_vs_"):-2])
                    tiny_map[idx] = sd[k].float().to(device=opt_device)
            if tiny_map:
                optimizer._tiny_vs_map = tiny_map
                print(f"    Tiny-param second moments restored ({len(tiny_map)} params).")
        elif hasattr(optimizer, "_tiny_vs") and "__tiny_vs__" in sd:
            optimizer._tiny_vs = sd["__tiny_vs__"].float().to(device=opt_device)
            print(f"    Tiny-param second moments restored ({optimizer._tiny_vs.numel()} elements).")
    total_states = len(optimizer.vr) * 4  # vr, vc, vs, ea per param
    tqdm.write(f"    Optimizer load: {restored}/{total_states} states restored, "
               f"{skipped} skipped | t={saved_t}, step={saved_step}")
    if skipped:
        print(f"    Warning: {skipped} optimizer states skipped due to shape mismatch "
              f"(param ordering changed). Momentum will be partially fresh.")
    del sd
    gc.collect()
    if getattr(optimizer, 'device', 'cpu') != 'cpu':
        xpu_empty_cache()
    return saved_step


def save_lora_checkpoint(student: ComfyUNetWrapper, path: str,
                         optimizer=None):
    """Save LoRA adapter weights.

    optimizer: if a FusedXPUAdafactor is passed, its backward hooks are
    temporarily removed during the save to prevent a hook firing on a
    detached/CPU tensor and corrupting second-moment state.
    """
    is_fused = isinstance(optimizer, FusedXPUAdafactor) if optimizer is not None else False
    if is_fused:
        optimizer.remove_hooks()

    student.eval()
    try:
        lora_weights = student.get_lora_weights()
        if not lora_weights:
            print("    [WARN] No LoRA weights found — nothing to save.")
            return
        _atomic_save(lora_weights, path)
        print(f"    LoRA weights saved: {path} ({len(lora_weights)} tensors)")
        del lora_weights
        gc.collect()
    finally:
        student.train()
        if is_fused:
            optimizer.register_hooks()


def load_lora_checkpoint(student: ComfyUNetWrapper, path: str):
    if not student.has_lora():
        print("    [WARN] Model has no LoRA layers injected — cannot load.")
        return
    sd = load_file(path)
    student.load_lora_weights(sd)
    print(f"    LoRA weights loaded: {path} ({len(sd)} tensors)")
    del sd
    gc.collect()
