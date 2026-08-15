"""Patches comfy.ldm.modules.diffusionmodules.util.CheckpointFunction so
activation checkpointing works with a frozen base + LoRA model.
See docs/training_pipeline_design.md section 2.3 for design rationale.

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

The fix only changes which of ctx.input_params actually gets passed to
torch.autograd.grad -- the frozen ones are filtered out before the call
and re-inserted as None afterward, at the same positions, since autograd
still needs one gradient slot per original forward() argument regardless
of whether that argument required grad. Everything else (forward, the
shallow-copy re-run under torch.enable_grad(), the autocast context) is
copied unchanged from comfy's own implementation -- this is a filter on
top of proven logic, not a reimplementation of it.

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
    implementation."""

    def apply(self) -> None:
        enable_frozen_param_safe_checkpointing()


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
            ctx.gpu_autocast_kwargs = {
                "enabled": torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad():
                return ctx.run_function(*ctx.input_tensors)

        @staticmethod
        def backward(ctx, *output_grads):
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
            with torch.enable_grad(), \
                    torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
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
