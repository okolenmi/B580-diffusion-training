"""Optimizer builder — factory for CPUAdamW, ChunkedXPUAdafactor, FusedXPUAdafactor.

Accepts typed CommonSettings instead of argparse.Namespace.
"""

from .config_model import CommonSettings
from .optimizers import CPUAdamW, ChunkedXPUAdafactor, FusedXPUAdafactor, ForeachXPUAdafactor


def build_optimizer(student, config: CommonSettings, device, params=None):
    """Build optimizer from typed config.

    Args:
        student: ComfyUNetWrapper or model with .parameters()
        config: Training CommonSettings (lr, optimizer, lr_strategy, etc.)
        device: torch device string
        params: optional explicit param list (e.g. LoRA-only params).
    """
    base_lr = config.lr

    if params is not None:
        param_groups = None
        _source = params
    elif config.lr_strategy == "radial":
        param_groups = _make_radial_groups(student, base_lr, config)
        _source = None
    else:
        param_groups = None
        _source = student.parameters()

    if config.optimizer == "adamw":
        p = param_groups if param_groups else list(_source)
        opt = CPUAdamW(p, lr=base_lr)
        n_params = len(p)
        print(f"    Optimizer : CPUAdamW  | {n_params} param groups | RAM ~20 GB (FP32)")
    elif config.optimizer == "fused-adafactor":
        p = param_groups if param_groups else list(_source)
        opt = FusedXPUAdafactor(p, lr=base_lr, scale_parameter=config.adafactor_scale_param,
                                weight_decay=1e-2, device=device)
        scale_note = " (scale_param)" if config.adafactor_scale_param else ""
        if config.grad_accum > 1:
            print(f"    WARNING: fused-adafactor ignores grad_accum={config.grad_accum}. "
                  f"Effective grad_accum=1. Switch to xpu-adafactor if accumulation is needed.")
        print(f"    Optimizer : FusedXPUAdafactor (fused backward, low VRAM){scale_note}")
    else:  # xpu-adafactor
        p = param_groups if param_groups else list(_source)
        
        # Optimization: For many small parameters (LoRA), use the vectorized Foreach optimizer.
        # This dramatically reduces CPU bottlenecking by grouping updates.
        if params is not None or len(p) > 200:
            opt = ForeachXPUAdafactor(p, lr=base_lr, scale_parameter=config.adafactor_scale_param,
                                     weight_decay=1e-2, device=device)
            print(f"    Optimizer : ForeachXPUAdafactor (Vectorized XPU)")
        else:
            opt = ChunkedXPUAdafactor(p, lr=base_lr, scale_parameter=config.adafactor_scale_param,
                                      weight_decay=1e-2, device=device)
            print(f"    Optimizer : ChunkedXPUAdafactor (GPU)")
        
        scale_note = " (scale_param)" if config.adafactor_scale_param else ""

    if params is not None:
        p_count = len(params)
        print(f"    Parameters : {p_count} LoRA params ({p_count//2} layers)")
        if config.lr_strategy == "radial":
            print(f"    [WARN] lr_strategy='radial' is configured but has no effect here: "
                  f"an explicit parameter list was passed in (LoRA mode), which always "
                  f"uses a single uniform LR for all LoRA params. Radial per-block "
                  f"multipliers only apply to full/dense fine-tuning. lr={base_lr:.2e} "
                  f"is being used uniformly for all LoRA params.")
    elif config.lr_strategy == "radial":
        n = len(opt.params)
        lr_min = min(opt.param_lr)
        lr_max = max(opt.param_lr)
        print(f"    LR strategy: radial | {n} param groups | "
              f"range {lr_min:.2e}–{lr_max:.2e}")
        print(f"      center×{config.center_mult} | sides×{config.side_mult} | time×{config.time_mult}")
        opt._radial_mults = [lr_i / base_lr for lr_i in opt.param_lr]
    else:
        print(f"    LR strategy: uniform | lr={base_lr:.2e}")

    return opt


def _make_radial_groups(student, base_lr, config: CommonSettings):
    """Build param_groups list with per-block LR multipliers."""
    model_ref = (student.model._orig_mod
                 if hasattr(student.model, "_orig_mod")
                 else student.model)

    groups = []
    n_in = len(model_ref.input_blocks)
    n_out = len(model_ref.output_blocks)

    for i, block in enumerate(model_ref.input_blocks):
        dist = 1.0 - i / max(n_in - 1, 1)
        mult = config.center_mult + (config.side_mult - config.center_mult) * dist
        for p in block.parameters():
            if p.requires_grad:
                groups.append({'params': [p], 'lr': base_lr * mult})

    for p in model_ref.middle_block.parameters():
        if p.requires_grad:
            groups.append({'params': [p], 'lr': base_lr * config.center_mult})

    for i, block in enumerate(model_ref.output_blocks):
        dist = i / max(n_out - 1, 1)
        mult = config.center_mult + (config.side_mult - config.center_mult) * dist
        for p in block.parameters():
            if p.requires_grad:
                groups.append({'params': [p], 'lr': base_lr * mult})

    if hasattr(model_ref, 'time_embed'):
        for p in model_ref.time_embed.parameters():
            if p.requires_grad:
                groups.append({'params': [p], 'lr': base_lr * config.time_mult})

    if hasattr(model_ref, 'label_emb'):
        for p in model_ref.label_emb.parameters():
            if p.requires_grad:
                groups.append({'params': [p], 'lr': base_lr * config.time_mult})

    if hasattr(model_ref, 'out'):
        for p in model_ref.out.parameters():
            if p.requires_grad:
                groups.append({'params': [p], 'lr': base_lr * config.side_mult})

    return groups


def update_lr(optimizer, new_lr):
    """Update optimizer LR, respecting radial multipliers or uniform state."""
    if hasattr(optimizer, "update_lr") and callable(optimizer.update_lr):
        optimizer.update_lr(new_lr)
        return

    mults = getattr(optimizer, "_radial_mults", None)
    if mults is not None:
        for i, m in enumerate(mults):
            optimizer.param_lr[i] = m * new_lr
    elif hasattr(optimizer, "param_lr"):
        for i in range(len(optimizer.param_lr)):
            optimizer.param_lr[i] = new_lr

    optimizer.lr = new_lr
