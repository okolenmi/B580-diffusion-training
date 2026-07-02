"""LR schedule factories — accept typed CommonSettings."""

import math

from .config_model import CommonSettings


def make_cosine_lr(lr: float, total_steps: int):
    """Standard cosine decay: lr → lr×0.05 over total_steps."""
    lr_min = lr * 0.05
    def cosine_lr(step):
        p = step / max(total_steps - 1, 1)
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))
    return cosine_lr


def make_poly_lr(lr_start: float, lr_end: float,
                 total_steps: int, power: float = 2.0):
    """Polynomial decay from lr_start to lr_end over total_steps."""
    def poly_lr(step):
        t = min(step, total_steps) / total_steps
        return lr_end + (lr_start - lr_end) * (1.0 - t) ** power
    return poly_lr


def make_lr_schedule(config: CommonSettings, run_steps: int, start_step: int):
    """
    Build the LR callable for the training loop.
    Returns lr_fn(global_step) → float.
    """
    schedule = config.lr_schedule
    warmup_steps = config.lr_warmup_steps
    # lr_warmup_start defaults to 0.0, which is a legitimate value (ramp up
    # from zero) -- not a sentinel for "unset". Using `or` here used to
    # silently replace an explicit 0.0 with 10% of lr, making a true
    # zero-start warmup impossible to configure.
    warmup_start = config.lr_warmup_start

    decay_steps = max(1, run_steps - warmup_steps)

    if schedule == "poly":
        # Same reasoning as warmup_start: lr_end=0.0 (decay fully to zero) is
        # a valid, intentional setting and must not be overridden.
        lr_end = config.lr_end
        power = config.lr_power
        decay_inner = make_poly_lr(config.lr, lr_end, decay_steps, power)
        decay_label = (f"poly(power={power}) {config.lr:.2e} → {lr_end:.2e}")
    else:
        decay_inner = make_cosine_lr(config.lr, decay_steps)
        decay_label = f"cosine {config.lr:.2e} → {config.lr*0.05:.2e}"

    if warmup_steps > 0:
        def lr_fn_inner(local_step):
            if local_step < warmup_steps:
                return warmup_start + (config.lr - warmup_start) * (local_step / warmup_steps)
            return decay_inner(local_step - warmup_steps)
        print(f"    LR schedule : {decay_label}")
        print(f"    LR warmup   : {warmup_start:.2e} → {config.lr:.2e} "
              f"over {warmup_steps} steps")
    else:
        lr_fn_inner = decay_inner
        print(f"    LR schedule : {decay_label} over {run_steps} steps")

    lr_fn = lambda step, _f=lr_fn_inner, _o=start_step: _f(max(0, step - _o))
    return lr_fn
