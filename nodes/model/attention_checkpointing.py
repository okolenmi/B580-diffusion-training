"""Closes the gap this project already found and deliberately deferred:
`comfy.ldm.modules.attention.BasicTransformerBlock.forward()` never calls
`checkpoint()` at all, in any ComfyUI version checked so far -- its
`checkpoint=True` constructor parameter is accepted and then silently
dropped (never even assigned to `self`), so `use_checkpoint=True`
anywhere in this project has only ever actually checkpointed `ResBlock`
instances. See `docs/known-issues/deferred.md` ("Only ResBlock instances
ever route through this project's checkpoint patch") and
`docs/design/03-training-step-orchestration.md` section 2.3 for the
original finding -- both confirmed there by cloning
comfyanonymous/ComfyUI directly. Re-confirmed here the same way, against
a fresh checkout of comfyanonymous/ComfyUI's current `master`
(comfy/ldm/modules/attention.py's `BasicTransformerBlock`/
`SpatialTransformer`): `SpatialTransformer.forward()`'s own
`transformer_blocks` loop (`x = block(x, context=context[i],
transformer_options=transformer_options)`) is a plain call, and
`BasicTransformerBlock.__init__` never assigns its `checkpoint` argument
to `self` at all -- so there was never a flag for a call to `checkpoint()`
to have consulted even if one existed. Not a new bug found in this
codebase; a real gap in ComfyUI's own reference implementation that this
project's `use_checkpoint` was silently unable to reach. Exact line
numbers weren't recorded against a pinned commit (this project doesn't
vendor or pin a ComfyUI checkout -- see docs/setup.md), so, same caveat
`gradient_checkpointing.py`'s own tests carry: this confirms the
mechanism, not that today's exact ComfyUI-on-disk source layout still
matches -- worth a quick re-check on a real ComfyUI install before
relying on this in a real run.

Why this matters more than it might look like: SDXL's UNet spends most
of its parameters (and, going by the SpatialTransformer stacks at
`transformer_depth` up to 10 per level, most of its activation memory
too) inside `BasicTransformerBlock` attention + feed-forward, not inside
`ResBlock`'s two convolutions. Checkpointing only `ResBlock` -- which is
all `use_checkpoint=True` has ever actually done in this project --
leaves the dominant activation cost fully resident regardless. This is
almost certainly why `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`'s
third addendum found activation memory still ~half of real reserved
VRAM in both reported OOM cases: `use_checkpoint=True` was on in both
(it defaults `True` all the way through `build_lora_injected_unet()`,
and nothing overrides it for either trainer route -- checked directly,
not assumed, while investigating this), it just wasn't doing what its
own doc string promised.

`enable_attention_block_checkpointing()` fixes this the same way
`gradient_checkpointing.py` fixes `ResBlock`'s frozen-LoRA-parameter
problem: a monkeypatch on the exact seam, not a reimplementation of
SDXL's attention math. It does NOT reimplement `BasicTransformerBlock`'s
own `forward()` body -- it captures the original, real, unmodified
method once (`_ORIGINAL_FORWARD` at patch time) and wraps it in a
closure passed to `checkpoint()`, `comfy.ldm.modules.diffusionmodules.
util`'s own free function (unchanged, the same one `ResBlock.forward()`
calls) -- so it automatically reuses whatever `CheckpointFunction`
that module currently has installed: the real, frozen-param-safe one
this project's `enable_frozen_param_safe_checkpointing()` installs
(LoRA freezes most of a `BasicTransformerBlock`'s own parameters too --
`to_q`/`to_k`/`to_v`/`to_out`/`ff` -- the exact same "one frozen
parameter in ctx.input_params" problem `gradient_checkpointing.py`'s own
module docstring describes for `ResBlock`, unfixed here would raise the
same "One of the differentiated Tensors does not require grad" the
moment any `use_checkpoint=True` LoRA run reached a checkpointed
attention block), or `ProfilingCheckpointing`'s instrumented one, with
zero duplicated autograd code either way.

Not gated by a rediscovered `self.checkpoint`/instance-level flag --
deliberately. This project has no mechanism anywhere for a *per-block*
checkpoint choice (`SDXL_CONFIG`'s `use_checkpoint` is one flag for the
whole UNet, and `GreedyRatioPlacement`, the only thing that could someday
make a real per-block choice, isn't wired into real construction yet --
`nodes/model/checkpoint_placement.py`'s own docstring). So whether this
patch's wrapped `forward()` actually checkpoints is entirely gated by
whether this function was ever called at all (i.e. by
`FrozenParamSafeCheckpointing`/`ProfilingCheckpointing` vs
`NoCheckpointing` -- the existing `ActivationCheckpointingStrategy`
choice), exactly mirroring `EveryBlockPlacement`'s own documented
"today's actual, only behavior... checkpoint everything, unconditionally"
for `ResBlock`. Simpler and lower-risk than also patching `__init__` to
resurrect the dropped constructor argument for a distinction nothing in
this codebase can act on yet.

`context` (the cross-attention conditioning tensor) is passed to
`checkpoint()` as a second real tensor input, exactly the way
`ResBlock.forward()` passes its own `emb` (timestep embedding) alongside
`x` -- not just closed over -- so a recompute rebuilds a real
differentiable graph through it too, not only through `x`. This project
always freezes the text encoder (`core/clip_encode.py`'s
`SDXLClipEncoder`, unconditional `p.requires_grad_(False)` -- checked
directly), so `context.requires_grad` is `False` in every real run
today and this makes no observable difference now -- included anyway
because it costs nothing and avoids a real, if currently unreachable,
correctness trap (silently dropping gradient flow into the text encoder
the day something in this project ever makes it trainable), the same
reasoning `checkpoint()`'s own reference implementation apparently
already applied to `emb`. `context` can legitimately be `None` (a
`BasicTransformerBlock` with no cross-attention at all is structurally
possible, per `SpatialTransformer.__init__`'s own signature, though SDXL
as this project builds it always supplies real conditioning) -- handled
as a separate branch rather than crashing on `None.detach()`.
`transformer_options` is a plain, possibly-mutated-by-patches dict, not
a tensor -- closed over by reference (same object, not copied), matching
how `ctx.run_function` being a bound method already closes over `self`
in the existing `ResBlock` case.
"""

from __future__ import annotations


def enable_attention_block_checkpointing() -> None:
    """Idempotent per process (a sentinel on the class itself, same
    style as `gradient_checkpointing.enable_frozen_param_safe_checkpointing`'s
    own `_frozen_param_safe` flag) -- calling this more than once, even
    across different `ActivationCheckpointingStrategy` instances in the
    same process, re-wraps nothing the second time. Not parameterized by
    a `recompute_wrapper` the way the `ResBlock` patch is: this function
    doesn't reimplement backward, it only makes `BasicTransformerBlock`
    route through the *existing*, already-patchable `checkpoint()`/
    `CheckpointFunction` seam -- whichever `CheckpointFunction` variant
    (plain frozen-param-safe, or `ProfilingCheckpointing`'s instrumented
    one) is installed there at actual call time is the one that runs,
    with no separate copy of that choice to keep in sync here.
    """
    from comfy.ldm.modules import attention as comfy_attn

    current = comfy_attn.BasicTransformerBlock
    if getattr(current, "_attention_block_checkpointing_enabled", False):
        return

    original_forward = current.forward

    def patched_forward(self, x, context=None, transformer_options={}):
        from comfy.ldm.modules.diffusionmodules.util import checkpoint as comfy_checkpoint

        if context is None:
            # No cross-attention tensor to recompute a real graph through --
            # closed over as a plain constant instead (see module docstring).
            def run(x_):
                return original_forward(self, x_, context=None,
                                         transformer_options=transformer_options)
            inputs = (x,)
        else:
            def run(x_, context_):
                return original_forward(self, x_, context=context_,
                                         transformer_options=transformer_options)
            inputs = (x, context)

        # label_for() in block_profiler.py reads run_function.__self__ to
        # build a real "BasicTransformerBlock#N" label, the same way it
        # already does for ResBlock's own bound-method `self._forward` --
        # `run` is a plain closure, not a bound method, so it wouldn't have
        # one naturally. Set directly rather than reworking label_for()
        # itself to also accept a plain callable with no real attached
        # instance -- ResBlock's own case is not changing, only this one
        # needs it added.
        run.__self__ = self

        return comfy_checkpoint(run, inputs, tuple(self.parameters()), True)

    current.forward = patched_forward
    current._attention_block_checkpointing_enabled = True
