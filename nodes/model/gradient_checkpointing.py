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


def enable_frozen_param_safe_checkpointing() -> None:
    """Idempotent -- safe to call every time a LoRA model with
    use_checkpoint=True is built. Only touches comfy's module-global
    CheckpointFunction the first time; every later call is a no-op.
    """
    import torch
    from comfy.ldm.modules.diffusionmodules import util as comfy_ckpt_util

    if getattr(comfy_ckpt_util.CheckpointFunction, "_frozen_param_safe", False):
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
    comfy_ckpt_util.CheckpointFunction = FrozenParamSafeCheckpointFunction
