"""Activation checkpointing, owned rather than patched into ComfyUI.

Design doc 12, section 7.3, section A. This is the code that was
`gradient_checkpointing.py`'s monkeypatch, lifted out and made the
implementation instead of an override.

ComfyUI's `CheckpointFunction` (in `comfy/ldm/modules/diffusionmodules/util.py`)
is wrong in two independent ways on this hardware, both measured on the
B580 and both reported upstream, neither fixed there:

1. **It raises on frozen parameters.** Its backward calls
   `torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params,
   ...)`, and every parameter in a LoRA-injected block except the adapters
   has `requires_grad=False`. autograd rejects that with "One of the
   differentiated Tensors does not require grad". Here the frozen ones are
   filtered out of the `grad` call and get `None` in the returned tuple --
   which is legal, because autograd matches returned gradients to `forward()`'s
   arguments positionally rather than by name.

2. **Its backward re-enters `torch.cuda.amp.autocast`, unconditionally.**
   On an Intel card that context manager prints "Disabling autocast" and
   enters *disabled*, so a forward pass that ran in fp16 is recomputed in
   fp32 during backward. Measured as a **4.581e-04 relative gradient error**
   against a non-checkpointed reference; **0.0** here. The fix is to
   re-enter the forward's own autocast, on the forward's own device type.

Neither of those was found by comparing against ComfyUI. Both were found by
comparing a checkpointed gradient against a *non-checkpointed* reference,
which is a correctness question that does not involve ComfyUI at all.

**What is kept from ComfyUI, and why:** the checkpointing *strategy* --
run the forward under `no_grad`, recompute it in backward, and take
`grad(allow_unused=True)` -- and the `(run_function, length, *args)` calling
convention, because the callers pass parameters positionally after the
inputs and the returned tuple has to line up with that. The convention is
ComfyUI's and every call site in this project already speaks it.

Provenance: the surrounding structure follows ComfyUI's `checkpoint()`
(Apache-2.0) — same four-argument shape, same `(run_function, length, *args)`
convention underneath, because the callers already speak it. The two fixes
are this project's.
"""

from __future__ import annotations

import contextlib
import threading

import torch

__all__ = ["checkpoint", "make_checkpoint_function", "active_checkpoint_function"]


def _autocast_state(inputs) -> tuple[str | None, dict]:
    """The autocast context to re-enter in backward, or (None, {}).

    Taken from the *forward's* inputs rather than asked of torch, because
    there is no stable API for "which device type is autocast currently
    enabled for", and the inputs' device is the thing that has to match
    anyway.
    """
    device_type = "cuda"
    for tensor in inputs:
        device_type = tensor.device.type
        break
    try:
        available = torch.amp.autocast_mode.is_autocast_available(device_type)
    except Exception:  # noqa: BLE001 -- unknown device type, assume available
        available = True
    if not available:
        return None, {}

    # `get_autocast_dtype` replaced `get_autocast_gpu_dtype`, which torch
    # deprecates and which has no argument at all. The getattr keeps this
    # working on the older torch a ComfyUI venv might pin.
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is None:
        getter = lambda _dt: torch.get_autocast_gpu_dtype()  # noqa: E731
    return device_type, {
        "enabled": torch.is_autocast_enabled(device_type),
        "dtype": getter(device_type),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }


def make_checkpoint_function(recompute_wrapper=None):
    """Build a `CheckpointFunction` with the frozen-param and autocast fixes.

    A factory rather than a module-level class because `recompute_wrapper` is
    per-use: `block_profiler.py` installs its own to measure the real
    recompute, and the two must not share a closure. Each call returns a
    distinct class, so `checkpoint()`'s active one is switched by
    assignment rather than by mutating a shared class -- see
    `set_active_checkpoint_function`.
    """

    class FrozenParamSafeCheckpointFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, run_function, length, *args):
            ctx.run_function = run_function
            ctx.input_tensors = list(args[:length])
            ctx.input_params = list(args[length:])
            # Device-agnostic, where ComfyUI's is CUDA-specific. See
            # _autocast_state() for the measurement and why this is not a
            # cosmetic difference.
            ctx.autocast_device_type, ctx.autocast_kwargs = _autocast_state(
                ctx.input_tensors
            )
            with torch.no_grad():
                return ctx.run_function(*ctx.input_tensors)

        @staticmethod
        def backward(ctx, *output_grads):
            ctx.input_tensors = [x.detach().requires_grad_(True)
                                 for x in ctx.input_tensors]
            # Re-enter the *forward's* autocast, on the forward's device type.
            # ComfyUI's version re-enters torch.cuda.amp.autocast, which on
            # this card silently disables itself -- see the module docstring
            # for the 4.581e-04 gradient error that produced.
            # nullcontext when there was no autocast, or the device type has
            # none, so the recompute is simply un-autocast as before.
            autocast = (
                torch.autocast(device_type=ctx.autocast_device_type,
                                **ctx.autocast_kwargs)
                if ctx.autocast_device_type
                else contextlib.nullcontext()
            )
            with torch.enable_grad(), autocast:
                # Same "first op mutates storage in place" guard as the
                # original -- detach()'d tensors can't be mutated in place.
                shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
                if recompute_wrapper is not None:
                    output_tensors = recompute_wrapper(ctx.run_function,
                                                       shallow_copies)
                else:
                    output_tensors = ctx.run_function(*shallow_copies)

            trainable_params = [p for p in ctx.input_params if p.requires_grad]
            grad_targets = ctx.input_tensors + trainable_params
            computed = torch.autograd.grad(output_tensors, grad_targets,
                                           output_grads, allow_unused=True)

            tensor_grads = computed[:len(ctx.input_tensors)]
            trainable_grads = iter(computed[len(ctx.input_tensors):])
            # One slot per original param, in order -- None for the frozen
            # ones, since autograd matches returned grads to forward()'s
            # *args positionally, not by name.
            param_grads = tuple(next(trainable_grads) if p.requires_grad else None
                                for p in ctx.input_params)

            del ctx.input_tensors, ctx.input_params, output_tensors
            return (None, None) + tuple(tensor_grads) + param_grads

    FrozenParamSafeCheckpointFunction._frozen_param_safe = True
    FrozenParamSafeCheckpointFunction._recompute_wrapper_identity = recompute_wrapper
    return FrozenParamSafeCheckpointFunction


#: The class `checkpoint()` currently dispatches to.
#:
#: It starts as the *fixed* one rather than ComfyUI's, because the fixed
#: version is a strict superset: with every parameter trainable and no
#: autocast in play it computes the same gradients as the original. That is
#: what makes it safe as a default rather than something callers must opt
#: into -- and it is checked, not assumed, by
#: smoke_test_checkpoint.py.
_checkpoint_function = make_checkpoint_function()

_lock = threading.Lock()


def active_checkpoint_function():
    """The class `checkpoint()` will dispatch to."""
    return _checkpoint_function


def set_active_checkpoint_function(factory_result) -> None:
    """Install a class built by `make_checkpoint_function`.

    Idempotent for a given `recompute_wrapper`, because callers apply their
    strategy fresh on every graph run rather than once per process, so
    re-installing the same one has to be free.
    """
    global _checkpoint_function
    current = _checkpoint_function
    if (getattr(current, "_frozen_param_safe", False)
            and getattr(current, "_recompute_wrapper_identity", None)
            is getattr(factory_result, "_recompute_wrapper_identity", None)):
        return
    with _lock:
        _checkpoint_function = factory_result


def checkpoint(func, inputs, params=None, flag: bool = True):
    """Evaluate `func(*inputs)` without caching activations, recomputing in backward.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
        take as explicit arguments -- typically `self.parameters()`. They are
        passed positionally into the autograd Function so that backward has
        something to return gradients *for*; without them a LoRA block's
        adapters would get no gradient at all.
    :param flag: when False, call straight through with no checkpointing and
        no extra compute. This is the knob the caller uses to turn the
        memory/compute trade off, not a detection mechanism -- there is no
        silent fallback in here, because a silent fallback would make the
        memory numbers this project reports describe a different run than the
        one that happened.

    The four-argument shape is ComfyUI's and is kept deliberately: it is what
    `ResBlock.forward` and our own attention-block patch already call, and
    the split between `inputs` and `params` is load-bearing rather than
    cosmetic -- it is what lets the backward return one gradient slot per
    parameter, in order, with `None` for the frozen ones.
    """
    if not flag:
        return func(*inputs)
    args = tuple(inputs) + tuple(params or ())
    return _checkpoint_function.apply(func, len(inputs), *args)