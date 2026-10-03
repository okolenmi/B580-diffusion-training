"""Correctness check for nodes/model/attention_checkpointing.py's patch,
against the block it actually patches.

**This file used to test a hand-written stand-in for the block, not the
block.** The patch targeted ComfyUI's `BasicTransformerBlock`, which lives
in another process's file, so the only way to test the patch in isolation was
to fabricate a `comfy.ldm.modules.attention` module holding a fake class named
`BasicTransformerBlock` with a `forward` that took a `transformer_options`
dict. That fake is gone: `nodes/model/attention.py` is ours, the UNet builds
it, and `enable_attention_block_checkpointing(block_cls=...)` lets a check
patch a throwaway subclass of the real one.

Which matters beyond tidiness. The fake exercised the patch's control flow --
fraction, density, the single-input branch, idempotency -- against a forward
that shared no arithmetic with the real one. A patch that dropped the
`attn2` residual, or checkpointed the wrong sub-expression, passed. This
version runs the real forward, so the "matches a non-checkpointed reference"
check now covers the actual transformer.

What is checked:

- **Gradients are the point.** The block is given a frozen parameter and a
  trainable one, because that is the LoRA shape and because ComfyUI's stock
  `CheckpointFunction` raises on it. The patch must produce gradients equal
  to a non-checkpointed reference, must not fabricate one for the frozen
  parameter, and must pass a gradient into `context`.
- **The stock crash still reproduces**, as a control. It runs a verbatim copy
  of ComfyUI's `CheckpointFunction`, so "our version does not raise" is a
  statement about the fix rather than about the test.
- **The fraction knob**: 0.0 leaves the class untouched *and* does not burn
  the idempotency sentinel, so a later real call can still patch; 0.5 and
  0.75 checkpoint the right counts.
- **Idempotency**, and that a fresh subclass is patchable independently --
  which is what makes per-check subclasses safe here.

CPU only. Run: `python nodes/smoke_tests/smoke_test_attention_checkpointing.py`
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.attention import BasicTransformerBlock  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _frozen_plus_trainable(cls):
    """A subclass of the real block with one parameter frozen.

    Returned as a class, not an instance, because the patch sets a sentinel on
    the class: each check needs its own.
    """
    class _Block(cls):
        def __init__(self):
            super().__init__(dim=16, n_heads=2, d_head=8, context_dim=12)
            # The LoRA shape: one projection frozen, the rest trainable.
            # Only one -- freezing several leaves nothing to check a
            # gradient *on*, which is how the first version of this file
            # ended up asserting a gradient existed on a frozen tensor.
            self.attn1.to_q.weight.requires_grad_(False)

    return _Block


def _install_stub_comfy_checkpoint_module():
    """Register comfy's util module with a verbatim copy of its
    `CheckpointFunction`, for the control that proves the crash is real.

    Exec'd into the module's own __dict__ rather than closed over, so
    `checkpoint` resolves `CheckpointFunction` as a module global the way
    ComfyUI's does.
    """
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules",
                 "comfy.ldm.modules.diffusionmodules"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    util = types.ModuleType("comfy.ldm.modules.diffusionmodules.util")
    util.__dict__["torch"] = torch
    exec(
        "import torch\n"
        "\n"
        "class CheckpointFunction(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, run_function, length, *args):\n"
        "        ctx.run_function = run_function\n"
        "        ctx.input_tensors = list(args[:length])\n"
        "        ctx.input_params = list(args[length:])\n"
        "        ctx.gpu_autocast_kwargs = {\n"
        "            'enabled': torch.is_autocast_enabled(),\n"
        "            'dtype': torch.get_autocast_gpu_dtype(),\n"
        "            'cache_enabled': torch.is_autocast_cache_enabled(),\n"
        "        }\n"
        "        with torch.no_grad():\n"
        "            return ctx.run_function(*ctx.input_tensors)\n"
        "\n"
        "    @staticmethod\n"
        "    def backward(ctx, *output_grads):\n"
        "        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]\n"
        "        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):\n"
        "            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]\n"
        "            output_tensors = ctx.run_function(*shallow_copies)\n"
        "        input_grads = torch.autograd.grad(\n"
        "            output_tensors, ctx.input_tensors + ctx.input_params, output_grads,\n"
        "            allow_unused=True,\n"
        "        )\n"
        "        del ctx.input_tensors, ctx.input_params, output_tensors\n"
        "        return (None, None) + input_grads\n"
        "\n"
        "def checkpoint(func, inputs, params, flag):\n"
        "    if flag:\n"
        "        args = tuple(inputs) + tuple(params)\n"
        "        return CheckpointFunction.apply(func, len(inputs), *args)\n"
        "    return func(*inputs)\n",
        util.__dict__,
    )
    sys.modules["comfy.ldm.modules.diffusionmodules.util"] = util
    sys.modules["comfy.ldm.modules.diffusionmodules"].util = util
    return util


def _install_ours():
    """The two patches production installs together."""
    from nodes.model.attention_checkpointing import (
        enable_attention_block_checkpointing,
    )
    from nodes.model.gradient_checkpointing import (
        enable_frozen_param_safe_checkpointing,
    )
    enable_frozen_param_safe_checkpointing()
    return enable_attention_block_checkpointing


def check_stock_crash_is_real():
    print("\n=== control: comfy's stock CheckpointFunction still raises ===")
    util = _install_stub_comfy_checkpoint_module()
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    block = block_cls().eval()
    x = torch.randn(3, 5, 16, requires_grad=True)
    context = torch.randn(3, 4, 12)

    out = util.checkpoint(block.forward, (x, context),
                          tuple(block.parameters()), True)
    try:
        out.sum().backward()
    except RuntimeError as exc:
        check("comfy's own function raises on a frozen+trainable block",
              "does not require grad" in str(exc), str(exc)[:80])
    else:
        check("comfy's own function raises on a frozen+trainable block",
              False, "it did not raise, so the control proves nothing")
        return
    check("so the fixed version not raising is a statement about the fix",
          True)


def check_gradients_match_reference():
    print("\n=== patched forward: gradients match a non-checkpointed reference ===")
    patch = _install_ours()
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    patch(block_cls=block_cls)

    torch.manual_seed(0)
    block = block_cls().eval()
    x = torch.randn(3, 5, 16)
    context = torch.randn(3, 4, 12)

    x_ckpt = x.clone().requires_grad_(True)
    ctx_ckpt = context.clone().requires_grad_(True)
    block(x_ckpt, context=ctx_ckpt).sum().backward()

    check("the frozen projection's .grad stayed None, not a fabricated zero",
          block.attn1.to_q.weight.grad is None)
    check("the trainable projection got a real gradient",
          block.attn2.to_v.weight.grad is not None
          and bool(torch.isfinite(block.attn2.to_v.weight.grad).all()))
    check("x's own gradient is finite", bool(torch.isfinite(x_ckpt.grad).all()))
    check("and so is context's -- a real second input, not closed over",
          bool(torch.isfinite(ctx_ckpt.grad).all()))

    # Reference: the same weights, never checkpointed.
    torch.manual_seed(0)
    ref = block_cls().eval()
    ref.load_state_dict(block.state_dict())
    x_ref = x.clone().requires_grad_(True)
    ctx_ref = context.clone().requires_grad_(True)
    BasicTransformerBlock.forward(ref, x_ref, context=ctx_ref).sum().backward()

    def same(a, b):
        return torch.equal(a, b) or torch.allclose(a, b, rtol=1e-5, atol=1e-6)

    check("x's gradient matches the non-checkpointed reference",
          same(x_ckpt.grad, x_ref.grad),
          f"max diff {(x_ckpt.grad - x_ref.grad).abs().max().item():.3e}")
    check("context's gradient matches too",
          same(ctx_ckpt.grad, ctx_ref.grad),
          f"max diff {(ctx_ckpt.grad - ctx_ref.grad).abs().max().item():.3e}")
    check("the trainable projection's gradient matches",
          same(block.attn2.to_v.weight.grad, ref.attn2.to_v.weight.grad),
          f"max diff "
          f"{(block.attn2.to_v.weight.grad - ref.attn2.to_v.weight.grad).abs().max().item():.3e}")


def _self_attention_only(cls):
    """A block whose attn2 is a second self-attention, i.e. context_dim=None.

    The single-input branch only makes sense for one of these. A block with a
    real cross-attention context cannot take context=None at all -- its to_k
    expects the context width, and substituting x is a shape error. That is
    ComfyUI's behaviour too, so it is not something to work around; the
    check below pins the clear error instead.
    """
    class _Block(cls):
        def __init__(self):
            super().__init__(dim=16, n_heads=2, d_head=8, context_dim=None)
            self.attn1.to_q.weight.requires_grad_(False)
    return _Block


def check_context_none():
    print("\n=== context=None: self-attention block works, cross-attention says so ===")
    patch = _install_ours()
    block_cls = _self_attention_only(BasicTransformerBlock)
    patch(block_cls=block_cls)

    torch.manual_seed(1)
    block = block_cls().eval()
    x = torch.randn(3, 5, 16, requires_grad=True)
    block(x, context=None).sum().backward()
    check("a self-attention block takes the single-input branch cleanly", True)
    check("and the frozen projection is still untouched",
          block.attn1.to_q.weight.grad is None)
    check("while a trainable one still has a gradient",
          block.attn2.to_v.weight.grad is not None)

    # And the cross-attention case refuses clearly instead of failing on a
    # matmul about shapes.
    cross_cls = _frozen_plus_trainable(BasicTransformerBlock)
    x2 = torch.randn(3, 5, 16)
    try:
        cross_cls()(x2, context=None)
    except ValueError as exc:
        check("a cross-attention block given context=None raises a "
              "ValueError naming the widths", "cannot attend to itself"
              in str(exc), str(exc)[:90])
    else:
        check("a cross-attention block given context=None raises a "
              "ValueError naming the widths", False, "it did not raise")


def check_idempotent():
    print("\n=== idempotency, per class ===")
    patch = _install_ours()
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    patch(block_cls=block_cls)
    first = block_cls.forward
    patch(block_cls=block_cls)
    check("patching the same class twice does not double-wrap",
          block_cls.forward is first)

    other_cls = _frozen_plus_trainable(BasicTransformerBlock)
    patch(block_cls=other_cls)
    check("a fresh subclass is independently patchable, which is what makes "
          "per-check subclasses safe",
          other_cls.forward is not BasicTransformerBlock.forward)


def check_fraction_knob():
    print("\n=== fraction knob ===")
    patch = _install_ours()

    # 0.0: equivalent to never calling, and must not burn the sentinel.
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    original_forward = block_cls.forward
    patch(fraction=0.0, block_cls=block_cls)
    check("fraction=0.0 leaves forward untouched",
          block_cls.forward is original_forward)
    check("and sets no sentinel, so a later real call can still patch",
          not getattr(block_cls, "_attention_block_checkpointing_enabled",
                      False))

    # 0.5 over 4 blocks: indices 0 and 2.
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    patch(fraction=0.5, block_cls=block_cls)
    check("fraction=0.5 does patch",
          getattr(block_cls, "_attention_block_checkpointing_enabled", False))

    import nodes.model.checkpoint as ckpt_module
    real_checkpoint = ckpt_module.checkpoint
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real_checkpoint(*a, **k)

    ckpt_module.checkpoint = counting   # patched_forward imports it per call
    try:
        blocks = [block_cls().eval() for _ in range(4)]
        for b in blocks:
            b(torch.randn(3, 5, 16), context=torch.randn(3, 4, 12))
    finally:
        ckpt_module.checkpoint = real_checkpoint

    check("blocks get sequential traversal indices",
          [b._ac_seq_idx for b in blocks] == [0, 1, 2, 3],
          f"{[b._ac_seq_idx for b in blocks]}")
    check("density 0.5 checkpoints exactly 2 of 4", calls["n"] == 2,
          f"got {calls['n']}")

    # 0.75 over 8: 6 of 8. The earlier stride form silently mapped 0.75 to
    # density 0.5, which this catches.
    block_cls = _frozen_plus_trainable(BasicTransformerBlock)
    patch(fraction=0.75, block_cls=block_cls)
    calls["n"] = 0
    ckpt_module.checkpoint = counting
    try:
        for _ in range(8):
            block_cls().eval()(torch.randn(3, 5, 16),
                               context=torch.randn(3, 4, 12))
    finally:
        ckpt_module.checkpoint = real_checkpoint
    check("density 0.75 checkpoints 6 of 8, not 4", calls["n"] == 6,
          f"got {calls['n']}")


def check_fraction_one_and_the_real_class():
    print("\n=== the real class is still unpatched ===")
    check("BasicTransformerBlock.forward has not been replaced in this "
          "process, because every check patched a subclass",
          "_attention_block_checkpointing_enabled"
          not in vars(BasicTransformerBlock))


def main() -> int:
    check_stock_crash_is_real()
    check_gradients_match_reference()
    check_context_none()
    check_idempotent()
    check_fraction_knob()
    check_fraction_one_and_the_real_class()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())