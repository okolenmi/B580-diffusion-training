"""Patches comfy.ldm.modules.diffusionmodules.util.CheckpointFunction so
activation checkpointing works with a frozen base + LoRA model.
See docs/design/03-training-step-orchestration.md section 2.3 for design rationale.

Root cause (confirmed by reading ComfyUI's real source, not guessed): the
stock CheckpointFunction.backward() calls
torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, ...),
where ctx.input_params is a checkpointed block's *entire* parameters()
list, unfiltered. torch.autograd.grad() requires every tensor in its
`inputs=` list to have requires_grad=True, unconditionally --
allow_unused=True (which the stock code does pass) only excuses a tensor
that's unused *in this particular graph*, not one that structurally can
never require grad. A LoRA-injected block almost always has at least one
frozen parameter (a norm weight, a bias, anything target_modules didn't
match) sitting next to the trainable lora_A/lora_B, so the very first
frozen parameter in that list raises "One of the differentiated Tensors
does not require grad" before backward can complete. In a full fine-tune
this never comes up (every parameter requires grad), which is presumably
why comfy's own implementation never needed to handle it.

**ComfyUI's implementation is not the reference for this file.** An
earlier version of this docstring said the opposite -- that everything
except the frozen-param filter was "copied unchanged from comfy's own
implementation -- a filter on top of proven logic, not a
reimplementation of it" -- and that turned out to be the wrong stance
twice over.

1. The frozen-param bug above: fixed here, and **still unfixed in
   ComfyUI**. Nothing upstream depends on this project reporting it, so
   nothing upstream has.
2. The autocast re-entry, below: also a comfyi bug, inherited here by the
   copy, and also still unfixed upstream. Fixed here too.

Two for two is the argument. ComfyUI's model code is a source of *ideas*
and of the SDXL structure; where it is wrong, this project does not match
it, and a fix that has not landed upstream will not arrive here by
updating. So the rule for this file is: each seam is judged on whether it
is correct **for this project on its hardware**, and comfyi's version is
one more input to that judgement rather than the thing being reproduced.

The frozen-param fix changes which of ctx.input_params actually gets
passed to torch.autograd.grad -- the frozen ones are filtered out before
the call and re-inserted as None afterward, at the same positions, since
autograd still needs one gradient slot per original forward() argument
regardless of whether that argument required grad. The shallow-copy
re-run under torch.enable_grad() is kept because it is correct (detach()'d
tensors cannot be mutated in place). The autocast context was rewritten;
see _autocast_state().

ActivationCheckpointingStrategy makes the fix above composable: an
object with an apply() method instead of a global, process-wide
monkeypatch triggered directly by a bare bool. FrozenParamSafeCheckpointing
is the mechanism above, unchanged; NoCheckpointing is the explicit "did
nothing" case for when checkpointing is off.

See nodes/model/block_profiler.py for a third strategy,
ProfilingCheckpointing -- same mechanism, plus per-block recompute
timing/activation-memory instrumentation, composed via
enable_frozen_param_safe_checkpointing()'s optional recompute_wrapper
parameter below rather than a second copy of this delicate autograd
code.

See nodes/model/attention_checkpointing.py for what FrozenParamSafeCheckpointing.apply()
below also installs: ResBlock is not the only block that needs this
frozen-param filter applied to it to actually reach the checkpointed
autograd.Function -- see that module's own docstring for the real,
confirmed second half of what use_checkpoint=True needs to cover SDXL's
actual dominant activation cost (attention blocks, not just ResBlock).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ActivationCheckpointingStrategy(ABC):

    @abstractmethod
    def apply(self) -> None:
        """Install whatever's needed (a monkeypatch, a wrapper) before the
        model is built. Idempotent -- calling twice is a no-op."""


class NoCheckpointing(ActivationCheckpointingStrategy):

    def apply(self) -> None:
        pass  # explicit "did nothing", not "wasn't asked"


class FrozenParamSafeCheckpointing(ActivationCheckpointingStrategy):
    """The fix above, as an object. apply() delegates to
    enable_frozen_param_safe_checkpointing() unchanged rather than
    duplicating that function's body here -- it's delicate autograd code,
    already verified by smoke_test_gradient_checkpointing.py, and
    transcribing it a second place risks the two copies drifting apart
    for no benefit. This class is the interface other code should compose
    with going forward; the free function keeps the one real
    implementation.

    apply() also installs attention_checkpointing.py's
    enable_attention_block_checkpointing() -- both patches together are
    what use_checkpoint=True actually needs to reach the UNet's real
    dominant activation cost, not just ResBlock. See that module's own
    docstring; kept as a second function/file rather than folded into
    enable_frozen_param_safe_checkpointing() itself because it patches a
    different class in a different comfy module (attention.py, not
    diffusionmodules/util.py) for a different, independently-confirmed
    reason -- one delicate-autograd-code file per real seam, matching
    this file's own docstring's reasoning for keeping ProfilingCheckpointing
    a composed second call here rather than a third copy of this backward()."""

    def apply(self) -> None:
        enable_frozen_param_safe_checkpointing()
        from .attention_checkpointing import enable_attention_block_checkpointing
        enable_attention_block_checkpointing()


def _autocast_state(inputs):
    """``(device_type, kwargs)`` describing the *current* autocast context.

    Returned rather than hardcoded because the original captured
    ``torch.is_autocast_enabled()`` / ``torch.get_autocast_gpu_dtype()``
    (both CUDA-flavoured) and re-entered it with
    ``torch.cuda.amp.autocast``, and this project targets Intel Arc
    (ADR 0004). On this machine that call does not fail -- it warns
    "CUDA is not available or torch_xla is imported. Disabling autocast."
    and enters with ``enabled=False``.

    So the effect was: **a forward that ran under fp16 was recomputed in
    fp32**, which is the one thing activation checkpointing must not do.
    Measured, with all parameters trainable so the frozen-param filter is
    not in play:

        no autocast, xpu        relative gradient error  0.000e+00
        autocast('xpu', fp16)   relative gradient error  4.581e-04

    and by instrumenting the checkpointed block directly:

        FORWARD    xpu_autocast=True   out.dtype=float32
        RECOMPUTE  xpu_autocast=False  out.dtype=float32

    **This is currently latent**: nothing in the training path enables
    autocast (measured by grep over ``nodes/`` and ``manager/``), so both
    directions run with ``enabled=False`` and the two passes agree. It
    becomes a wrong-gradients bug the moment anyone turns on mixed
    precision, which for a 12 GB card training SDXL is the obvious next
    step -- so it is fixed now rather than then.

    The device type comes from where the inputs actually are, which is more
    reliable than asking torch what is current: there is no stable API for
    "which device type is autocast currently enabled for", and the inputs'
    device is the thing that has to match anyway.
    """
    import torch

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

    # `get_autocast_dtype(device_type)` replaced `get_autocast_gpu_dtype`,
    # which torch deprecates and which has no argument at all. The getattr
    # keeps this working on the older torch a ComfyUI venv might pin.
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is None:
        getter = lambda _dt: torch.get_autocast_gpu_dtype()  # noqa: E731
    return device_type, {
        "enabled": torch.is_autocast_enabled(device_type),
        "dtype": getter(device_type),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }


def enable_frozen_param_safe_checkpointing(recompute_wrapper=None) -> None:
    """Idempotent per (patched-at-all, recompute_wrapper identity) pair,
    not just "already patched at all" -- calling this twice with the
    same recompute_wrapper (None counts as its own identity) is a
    no-op, matching the original unparameterized behavior exactly when
    recompute_wrapper=None every time (FrozenParamSafeCheckpointing's
    own call site never passes one). Calling it with a *different*
    recompute_wrapper (e.g. switching from FrozenParamSafeCheckpointing
    to nodes/model/block_profiler.py's ProfilingCheckpointing, or back,
    within one process) re-installs the patch with the new wrapper --
    a real, narrow need: ComfyUNetLoRANode.build() calls
    checkpointing_strategy.apply() fresh on every graph run, not once
    per process, so two different runs in the same server process can
    legitimately want different instrumentation.

    recompute_wrapper: optional `(run_function, args) -> output_tensors`,
    called in place of `ctx.run_function(*args)` during backward's own
    recompute -- None (the default) costs nothing extra and is exactly
    the original call. See block_profiler.py's module docstring for why
    this is the one correct place to measure a checkpointed block's real
    recompute time/activation memory: it's the actual, real recompute a
    non-profiled run already pays for, not a separate profiling-only
    forward pass.
    """
    import torch
    from comfy.ldm.modules.diffusionmodules import util as comfy_ckpt_util

    current = comfy_ckpt_util.CheckpointFunction
    if (getattr(current, "_frozen_param_safe", False)
            and getattr(current, "_recompute_wrapper_identity", None) is recompute_wrapper):
        return

    class FrozenParamSafeCheckpointFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, run_function, length, *args):
            ctx.run_function = run_function
            ctx.input_tensors = list(args[:length])
            ctx.input_params = list(args[length:])
            # Device-agnostic, where comfyi's is CUDA-specific. See
            # _autocast_state() for the measurement and why this is not a
            # cosmetic difference.
            ctx.autocast_device_type, ctx.autocast_kwargs = _autocast_state(
                ctx.input_tensors
            )
            with torch.no_grad():
                return ctx.run_function(*ctx.input_tensors)

        @staticmethod
        def backward(ctx, *output_grads):
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
            # Re-enter the *forward's* autocast, on the forward's device
            # type. comfyi's version re-enters torch.cuda.amp.autocast, which
            # on this card silently disables itself -- see _autocast_state().
            # nullcontext when there was no autocast or the device type has
            # none, so the recompute is simply un-autocast as before.
            import contextlib

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
                    output_tensors = recompute_wrapper(ctx.run_function, shallow_copies)
                else:
                    output_tensors = ctx.run_function(*shallow_copies)

            trainable_params = [p for p in ctx.input_params if p.requires_grad]
            grad_targets = ctx.input_tensors + trainable_params
            computed = torch.autograd.grad(output_tensors, grad_targets, output_grads,
                                            allow_unused=True)

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
    comfy_ckpt_util.CheckpointFunction = FrozenParamSafeCheckpointFunction
