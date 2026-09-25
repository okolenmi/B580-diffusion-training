"""Verifies nodes/model/attention_checkpointing.py's patch logic.

Same constraint and same approach as smoke_test_gradient_checkpointing.py
(see that file's own docstring): no real `comfy` package in this
sandbox, so this registers stand-in modules at the exact two import
paths enable_attention_block_checkpointing() touches --
comfy.ldm.modules.diffusionmodules.util (the real, verbatim
CheckpointFunction/checkpoint(), same technique/content as the other
smoke test's own helper) and comfy.ldm.modules.attention (a stand-in
BasicTransformerBlock whose forward(x, context=None,
transformer_options={}) signature matches the real one exactly --
confirmed directly against a fresh comfyanonymous/ComfyUI clone while
building the patch itself, not reconstructed from memory) -- then runs
the REAL patch function against them and checks real gradients,
including the one thing that's easy to get subtly wrong here and that
ResBlock's own case never has to worry about: that `context` is
threaded through checkpoint() as a real second input (matching how
ResBlock's own `emb` is), not merely closed over, so a gradient into
whatever produced `context` doesn't silently go missing. This verifies
the patch's actual logic exactly; it does not verify the import path
itself still matches ComfyUI's current source layout, which needs
confirming on a machine with ComfyUI installed (same disclosed limit as
the other smoke test).

Each check gets its own fresh stub `BasicTransformerBlock` class (a
distinct dynamic subclass per _install_stub_comfy_attention_module()
call, not one shared class reused across checks) specifically so
enable_attention_block_checkpointing()'s own class-level monkeypatch
(forward reassignment + a sentinel attribute) from one check can't leak
into the next -- the same isolation
smoke_test_gradient_checkpointing.py's own per-check fresh util module
already gives that test for free, needed here explicitly since a class
being patched, not a module-level name being reassigned, is what this
patch actually does.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn


def _install_stub_comfy_checkpoint_module():
    """Identical technique/content to smoke_test_gradient_checkpointing.py's
    own helper of the same purpose -- not imported from there, this
    project's smoke tests are each self-contained (no shared fixture
    module exists under nodes/smoke_tests/ to import this from instead).
    See that file's own docstring for why exec() into the module's own
    __dict__ matters here (a closure-based stub would never see the
    patch)."""
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
    return util


class _FrozenPlusTrainableAttentionBlock(nn.Module):
    """Stand-in for BasicTransformerBlock, never itself patched -- used
    directly (not through the stub module) as the "independent
    reference" computation in check_patched_version_matches_unchecked_reference,
    and as the base class _install_stub_comfy_attention_module() derives
    a fresh, independently-patchable subclass from for each check.

    A frozen `base` parameter next to a trainable `lora` one mirrors
    attn1/attn2/ff each being exactly this mix once LoRA is injected
    into the real block (the same shape
    smoke_test_gradient_checkpointing.py's own stand-in uses for
    ResBlock); a real dependence on `context` lets a test tell whether
    context's own upstream gradient survives; reading
    `transformer_options` lets a test confirm the same dict object, not
    a copy, still reaches the real computation once patched. forward's
    own signature matches comfy's real BasicTransformerBlock.forward()
    exactly: `(self, x, context=None, transformer_options={})`."""

    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.randn(4))
        self.base.requires_grad_(False)
        self.lora = nn.Parameter(torch.randn(4) * 0.1)

    def forward(self, x, context=None, transformer_options={}):
        scale = transformer_options.get("scale", 1.0)
        out = (x * self.base + x * self.lora) * scale
        if context is not None:
            out = out + context
        return out


def _install_stub_comfy_attention_module():
    """A fresh dynamic subclass of _FrozenPlusTrainableAttentionBlock
    each call -- see module docstring for why a shared class would let
    one check's patch leak into the next."""
    block_cls = type("BasicTransformerBlock", (_FrozenPlusTrainableAttentionBlock,), {})
    module = types.ModuleType("comfy.ldm.modules.attention")
    module.BasicTransformerBlock = block_cls
    sys.modules["comfy.ldm.modules.attention"] = module
    sys.modules["comfy.ldm.modules"].attention = module
    return module


def check_stock_version_reproduces_the_documented_crash():
    print("[stock checkpoint() really does crash a frozen+trainable attention block]")
    _install_stub_comfy_checkpoint_module()
    attn = _install_stub_comfy_attention_module()
    block = attn.BasicTransformerBlock()
    x = torch.randn(4, requires_grad=True)

    def run(x_):
        return block(x_, context=None, transformer_options={})

    from comfy.ldm.modules.diffusionmodules.util import checkpoint
    out = checkpoint(run, (x,), tuple(block.parameters()), True)
    try:
        out.sum().backward()
        raise AssertionError("expected the stock implementation to raise")
    except RuntimeError as e:
        assert "does not require grad" in str(e)
        print(f"    PASS: reproduces the documented crash exactly: {e}")


def check_patched_version_matches_unchecked_reference():
    print("[patched BasicTransformerBlock.forward: matches a non-checkpointed reference, "
          "including gradient into a trainable context]")
    _install_stub_comfy_checkpoint_module()
    attn = _install_stub_comfy_attention_module()
    from nodes.model.attention_checkpointing import enable_attention_block_checkpointing
    enable_attention_block_checkpointing()

    torch.manual_seed(0)
    block = attn.BasicTransformerBlock()
    x = torch.randn(4, requires_grad=False)
    context_param = torch.randn(4, requires_grad=False)
    transformer_options = {"scale": 2.0}

    x_ckpt = x.clone().requires_grad_(True)
    context_ckpt = context_param.clone().requires_grad_(True)
    out_ckpt = block(x_ckpt, context=context_ckpt, transformer_options=transformer_options)
    out_ckpt.sum().backward()
    lora_grad_ckpt = block.lora.grad.clone()
    context_grad_ckpt = context_ckpt.grad.clone()
    assert block.base.grad is None, "frozen param must not get a fabricated gradient"

    # Independent reference: the plain, never-patched base class, same
    # weights, same inputs, real (non-checkpointed) autograd throughout.
    ref_block = _FrozenPlusTrainableAttentionBlock.__new__(_FrozenPlusTrainableAttentionBlock)
    nn.Module.__init__(ref_block)
    ref_block.base = nn.Parameter(block.base.detach().clone())
    ref_block.base.requires_grad_(False)
    ref_block.lora = nn.Parameter(block.lora.detach().clone())
    x_ref = x.clone().requires_grad_(True)
    context_ref = context_param.clone().requires_grad_(True)
    out_ref = _FrozenPlusTrainableAttentionBlock.forward(
        ref_block, x_ref, context=context_ref, transformer_options=transformer_options)
    out_ref.sum().backward()

    torch.testing.assert_close(out_ckpt, out_ref)
    torch.testing.assert_close(lora_grad_ckpt, ref_block.lora.grad)
    torch.testing.assert_close(x_ckpt.grad, x_ref.grad)
    torch.testing.assert_close(context_grad_ckpt, context_ref.grad)
    print("    PASS: checkpointed forward/backward exactly matches the non-checkpointed reference")
    print("    PASS: frozen param's .grad stayed None -- no fabricated gradient for it")
    print("    PASS: context's own gradient survived -- real second input, not just closed over")


def check_context_none_does_not_crash():
    print("[context=None (self-attention-only block) takes the single-input branch cleanly]")
    _install_stub_comfy_checkpoint_module()
    attn = _install_stub_comfy_attention_module()
    from nodes.model.attention_checkpointing import enable_attention_block_checkpointing
    enable_attention_block_checkpointing()

    block = attn.BasicTransformerBlock()
    x = torch.randn(4, requires_grad=True)
    out = block(x, context=None, transformer_options={})
    out.sum().backward()
    assert block.lora.grad is not None
    assert block.base.grad is None
    print("    PASS: no crash, real gradients, frozen param still untouched")


def check_idempotent():
    print("[idempotency: patching twice doesn't double-wrap]")
    _install_stub_comfy_checkpoint_module()
    attn = _install_stub_comfy_attention_module()
    from nodes.model.attention_checkpointing import enable_attention_block_checkpointing
    enable_attention_block_checkpointing()
    first = attn.BasicTransformerBlock.forward
    enable_attention_block_checkpointing()
    assert attn.BasicTransformerBlock.forward is first
    print("    PASS: second call is a no-op")


def main():
    check_stock_version_reproduces_the_documented_crash()
    check_patched_version_matches_unchecked_reference()
    check_context_none_does_not_crash()
    check_idempotent()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED "
          "(against a faithful stand-in for comfy's attention module -- see module docstring)")


if __name__ == "__main__":
    main()
