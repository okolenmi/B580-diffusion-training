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
why ComfyUI's version never needed to handle it.

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

from .checkpoint import (
    make_checkpoint_function,
    set_active_checkpoint_function,
)


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


def enable_frozen_param_safe_checkpointing(recompute_wrapper=None) -> None:
    """Make `checkpoint()` dispatch to the frozen-param-safe function.

    Idempotent per (patched-at-all, recompute_wrapper identity) pair, not
    just "already patched at all": calling this twice with the same
    recompute_wrapper (None counts as its own identity) is a no-op, so
    repeated application costs nothing. Calling it with a *different*
    recompute_wrapper -- switching from FrozenParamSafeCheckpointing to
    block_profiler.py's ProfilingCheckpointing, or back, within one process
    -- installs the new one, which is a real need: ComfyUNetLoRANode.build()
    calls checkpointing_strategy.apply() fresh on every graph run rather than
    once per process, so two runs in the same server process can legitimately
    want different instrumentation.

    recompute_wrapper: optional `(run_function, args) -> output_tensors`,
    called in place of `ctx.run_function(*args)` during backward's own
    recompute -- None (the default) costs nothing extra and is exactly the
    original call. See block_profiler.py's module docstring for why this is
    the one correct place to measure a checkpointed block's real recompute
    time/activation memory: it's the actual, real recompute a non-profiled
    run already pays for, not a separate profiling-only forward pass.

    The class itself lives in `checkpoint.py`. It used to be built here and
    assigned onto `comfy.ldm.modules.diffusionmodules.util`, which made this
    a patch of someone else's module; it is now our implementation, and this
    function only chooses which one is active.
    """
    set_active_checkpoint_function(make_checkpoint_function(recompute_wrapper))
