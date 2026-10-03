"""Correctness checks for nodes/model/checkpoint.py -- the activation
checkpointing this project owns.

The implementation used to be a monkeypatch onto
`comfy.ldm.modules.diffusionmodules.util` and these checks drove it through
a stand-in module registered at that exact import path, holding a verbatim
copy of ComfyUI's `CheckpointFunction`/`checkpoint()`. `checkpoint.py` is
now the implementation (design doc 12 section 7.3, section A), so the real
checks run against the real code and nothing here depends on the import
path.

The stand-in is still here, and still load-bearing, but only for the
**control**: the first check runs ComfyUI's stock code and requires it to
raise on a frozen+trainable block, and the autocast check requires the
stock recompute to drop the autocast context. Those assertions are the
evidence that the two upstream bugs were real and are still real -- without
them the fixed version's passing would only show that our code agrees with
itself. They are also self-defeating if upstream changes, deliberately: a
control that stops failing tells you the control is no longer proving
anything, rather than quietly passing.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn


def _install_stub_comfy_checkpoint_module():
    """Registers comfy.ldm.modules.diffusionmodules.util in sys.modules
    with the stock (unpatched) CheckpointFunction/checkpoint(), verbatim
    from ComfyUI's real source. Built via exec() into the module's own
    __dict__ -- not nested Python closures -- specifically so checkpoint()
    resolves CheckpointFunction the same way the real file does (a
    module-level global lookup against comfy_ckpt_util's own namespace,
    which is what makes enable_frozen_param_safe_checkpointing()'s
    reassignment of that name actually take effect for later calls). A
    closure-based stub would capture the original class at definition
    time and never see the patch -- this bit the first version of this
    test, which is exactly why it's called out here.

    Also registers a minimal comfy.ldm.modules.attention stub (just
    enough of a BasicTransformerBlock -- a class with a forward
    attribute -- for FrozenParamSafeCheckpointing.apply()/
    ProfilingCheckpointing.apply() to successfully also call
    nodes/model/attention_checkpointing.py's
    enable_attention_block_checkpointing() without crashing on a
    missing import, now that both patches are always installed
    together). Every caller of this function exercises .apply(), not
    just enable_frozen_param_safe_checkpointing() directly, so this
    needs to be here rather than added separately per caller -- see
    nodes/smoke_tests/smoke_test_attention_checkpointing.py for the
    real, faithful verification of that patch's own logic; this stub
    only needs to exist, not be correct, for every *other* test in this
    project that merely needs the overall pipeline to not crash. Fresh
    class each call, same reason `util`'s own CheckpointFunction is
    rebuilt fresh each call above: so one check's patched `forward`/
    idempotency sentinel can never leak into the next.
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
        "            output_tensors = ctx.run_function(*ctx.input_tensors)\n"
        "        return output_tensors\n"
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
        "        del ctx.input_tensors\n"
        "        del ctx.input_params\n"
        "        del output_tensors\n"
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

    attention = types.ModuleType("comfy.ldm.modules.attention")

    class BasicTransformerBlock:
        def forward(self, x, context=None, transformer_options={}):
            return x

    attention.BasicTransformerBlock = BasicTransformerBlock
    sys.modules["comfy.ldm.modules.attention"] = attention
    sys.modules["comfy.ldm.modules"].attention = attention

    return util


class _FrozenPlusTrainableBlock(nn.Module):
    """The exact shape that breaks the stock implementation: a
    checkpointed block containing both a frozen parameter (norm.weight,
    requires_grad=False -- standing in for the base model's frozen
    weights) and a trainable one (adapter, standing in for lora_A/lora_B)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.randn(4))
        self.norm.requires_grad_(False)
        self.adapter = nn.Parameter(torch.randn(4) * 0.1)

    def _forward(self, x):
        return x * self.norm + x * self.adapter

    def forward(self, x, use_checkpoint, ckpt):
        # `ckpt` is a callable, not a module: it used to be
        # `util_module.checkpoint`, because the implementation *was* a
        # monkeypatch onto comfy's util namespace. It is now this project's
        # own (nodes/model/checkpoint.py), and the stock-Crash check below
        # still passes the stub to show upstream really does fail.
        return ckpt(self._forward, (x,), tuple(self.parameters()), use_checkpoint)


def check_stock_version_reproduces_the_documented_crash():
    print("[stock CheckpointFunction really does crash on a frozen+trainable block]")
    util = _install_stub_comfy_checkpoint_module()
    block = _FrozenPlusTrainableBlock()
    x = torch.randn(4, requires_grad=True)
    out = block(x, True, util.checkpoint)
    try:
        out.sum().backward()
        raise AssertionError("expected the stock implementation to raise")
    except RuntimeError as e:
        assert "does not require grad" in str(e)
        print(f"    PASS: reproduces the documented crash exactly: {e}")


def check_patched_version_matches_unchecked_reference():
    print("[patched CheckpointFunction: real gradients, matching a non-checkpointed reference]")
    _install_stub_comfy_checkpoint_module()
    from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing
    from nodes.model.checkpoint import checkpoint
    enable_frozen_param_safe_checkpointing()

    torch.manual_seed(0)
    block = _FrozenPlusTrainableBlock()
    x = torch.randn(4, requires_grad=True)

    x_ckpt = x.detach().clone().requires_grad_(True)
    out_ckpt = block(x_ckpt, True, checkpoint)
    out_ckpt.sum().backward()
    adapter_grad_ckpt = block.adapter.grad.clone()
    assert block.norm.grad is None, "frozen param must not get a fabricated gradient"
    block.adapter.grad = None

    # Independent reference: same block, same input, no checkpointing at all.
    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = block._forward(x_ref)
    out_ref.sum().backward()
    adapter_grad_ref = block.adapter.grad.clone()

    torch.testing.assert_close(out_ckpt, out_ref)
    torch.testing.assert_close(adapter_grad_ckpt, adapter_grad_ref)
    torch.testing.assert_close(x_ckpt.grad, x_ref.grad)
    print("    PASS: checkpointed forward/backward exactly matches the non-checkpointed reference")
    print("    PASS: frozen param's .grad stayed None -- no fabricated gradient for it")


class _AutocastProbeBlock(nn.Module):
    """Records the autocast state and dtype of every call to the block.

    All-trainable on purpose: the frozen-param filter is covered elsewhere,
    and a block with a frozen parameter makes the stock implementation
    raise before the recompute is ever reached -- which would hide the
    thing this probe is looking at.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(8, 4) / 8 ** 0.5)
        self.seen = []

    def forward(self, x):
        out = torch.nn.functional.silu(x @ self.w)
        self.seen.append({
            "autocast": torch.is_autocast_enabled("cpu"),
            "dtype": out.dtype,
        })
        return out


def _autocast_states(ckpt, block, x):
    """One checkpointed pass: forward under autocast, backward *outside* it.

    The shape matters and getting it wrong hides the bug. ComfyUI captures
    the autocast state in ``forward`` precisely because it expects
    ``backward`` to run later, after the surrounding context has exited --
    so the recompute has to put the context back itself. A first version of
    this check called ``backward`` inside the ``with`` block, and the
    control then reported the recompute as correctly autocast, because the
    outer context was still open and hiding a backward that re-enters
    nothing.

    So: forward inside, backward outside. The returned list is every state
    the block saw, forward first and the recompute last.
    """
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = ckpt(block, (x,), tuple(block.parameters()), True)
    out.backward(torch.ones_like(out))
    return block.seen


def check_recompute_reenters_the_forward_autocast():
    """The recompute must run in the same autocast context as the forward.

    This is the check that was missing, and its absence is why a
    CUDA-only autocast call survived in an XPU-targeted project: every other
    check here runs *without* autocast, where a forward and its recompute
    agree no matter what the recompute re-enters.

    The bug, on CPU so it needs no accelerator to demonstrate: comfyi's
    backward re-enters `torch.cuda.amp.autocast`, which on a machine
    without CUDA prints "Disabling autocast" and enters with
    `enabled=False`. So a forward that ran in bfloat16 was recomputed in
    float32.

    Run on CPU deliberately. The same defect is present on the XPU card --
    measured there at 4.6e-04 relative gradient error, against 0.0 for the
    fixed version -- but asserting a tolerance on GPU autocast would be a
    weaker and flakier test than asserting the context itself, and the
    context is the thing that was wrong.
    """
    print("[recompute re-enters the forward's autocast context, not a CUDA one]")
    import torch as _torch

    # -- control: the stock stub really does drop the autocast ----------
    util = _install_stub_comfy_checkpoint_module()
    stock_block = _AutocastProbeBlock()
    x = _torch.randn(4, 8, requires_grad=True)
    stock_seen = _autocast_states(util.checkpoint, stock_block, x)
    assert len(stock_seen) >= 2, f"expected a forward and a recompute, got {len(stock_seen)}"
    assert stock_seen[0]["autocast"] is True, "the control's forward was not autocast"
    assert stock_seen[-1]["autocast"] is False, (
        "expected the stock implementation to drop autocast in the recompute; "
        f"it did not, so this control proves nothing any more: {stock_seen}"
    )
    print(f"    PASS: control -- stock drops it: forward {stock_seen[0]} "
          f"-> recompute {stock_seen[-1]}")

    # -- and the patched version keeps it --------------------------------
    _install_stub_comfy_checkpoint_module()
    from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing
    from nodes.model.checkpoint import checkpoint
    enable_frozen_param_safe_checkpointing()
    block = _AutocastProbeBlock()
    x = _torch.randn(4, 8, requires_grad=True)
    seen = _autocast_states(checkpoint, block, x)
    assert len(seen) >= 2, f"expected a forward and a recompute, got {len(seen)}"
    assert seen[0]["autocast"] is True, "the patched forward was not autocast"
    assert seen[-1]["autocast"] is True, (
        "the patched recompute dropped autocast -- it must re-enter the "
        f"forward's context: {seen}"
    )
    print(f"    PASS: patched keeps it: forward {seen[0]} -> recompute {seen[-1]}")

    # -- and the two passes agree numerically ---------------------------
    # Written through the same helper rather than with an inline
    # checkpoint(...) call. The context assertions above are the
    # load-bearing ones; this adds a numeric check, and it is deliberately
    # built on the one call shape already exercised twice above rather than
    # a third spelling of it.
    torch.manual_seed(0)
    ckpt_block = _AutocastProbeBlock()
    ref_block = _AutocastProbeBlock()
    ref_block.load_state_dict(ckpt_block.state_dict())

    probe = torch.randn(4, 8, requires_grad=True)
    _autocast_states(checkpoint, ckpt_block, probe)
    ckpt_grad = ckpt_block.w.grad.clone()

    ref_block.zero_grad()
    plain = probe.detach().clone().requires_grad_(True)
    with _torch.autocast(device_type="cpu", dtype=_torch.bfloat16):
        out = ref_block(plain)
    out.backward(_torch.ones_like(out))

    torch.testing.assert_close(ckpt_grad, ref_block.w.grad)
    print("    PASS: and the checkpointed gradient matches a plain one under autocast")


def check_idempotent():
    print("[idempotency: patching twice doesn't double-wrap]")
    from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing
    from nodes.model.checkpoint import active_checkpoint_function
    enable_frozen_param_safe_checkpointing()
    first = active_checkpoint_function()
    enable_frozen_param_safe_checkpointing()
    assert active_checkpoint_function() is first
    print("    PASS: second call is a no-op")


def main():
    check_stock_version_reproduces_the_documented_crash()
    check_patched_version_matches_unchecked_reference()
    check_recompute_reenters_the_forward_autocast()
    check_idempotent()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED "
          "(the control ran against a verbatim copy of comfy's "
          "CheckpointFunction; the fixed paths ran against nodes/model/checkpoint.py)")


if __name__ == "__main__":
    main()
