"""Verifies nodes/model/gradient_checkpointing.py's patch logic.

ComfyUI itself isn't installed in this sandbox (no `comfy` package, no
COMFY_DIR) -- true for everything ComfyUI-dependent in this project. What this
test CAN do, and does: register a stand-in module at the exact import
path enable_frozen_param_safe_checkpointing() patches
(comfy.ldm.modules.diffusionmodules.util), containing a faithful,
verbatim reproduction of the real CheckpointFunction/checkpoint() (fetched
directly from github.com/comfyanonymous/ComfyUI, not reconstructed from
memory) -- then run the REAL patch function against it and check real
gradients, not just "doesn't crash." This verifies the patch's actual
logic exactly; it does not verify the import path itself still matches
ComfyUI's current source layout, which needs confirming on a machine with
ComfyUI installed.
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

    def forward(self, x, use_checkpoint, util_module):
        return util_module.checkpoint(self._forward, (x,), tuple(self.parameters()), use_checkpoint)


def check_stock_version_reproduces_the_documented_crash():
    print("[stock CheckpointFunction really does crash on a frozen+trainable block]")
    util = _install_stub_comfy_checkpoint_module()
    block = _FrozenPlusTrainableBlock()
    x = torch.randn(4, requires_grad=True)
    out = block(x, True, util)
    try:
        out.sum().backward()
        raise AssertionError("expected the stock implementation to raise")
    except RuntimeError as e:
        assert "does not require grad" in str(e)
        print(f"    PASS: reproduces the documented crash exactly: {e}")


def check_patched_version_matches_unchecked_reference():
    print("[patched CheckpointFunction: real gradients, matching a non-checkpointed reference]")
    util = _install_stub_comfy_checkpoint_module()
    from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing
    enable_frozen_param_safe_checkpointing()

    torch.manual_seed(0)
    block = _FrozenPlusTrainableBlock()
    x = torch.randn(4, requires_grad=True)

    x_ckpt = x.detach().clone().requires_grad_(True)
    out_ckpt = block(x_ckpt, True, util)
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


def check_idempotent():
    print("[idempotency: patching twice doesn't double-wrap]")
    util = _install_stub_comfy_checkpoint_module()
    from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing
    enable_frozen_param_safe_checkpointing()
    first = util.CheckpointFunction
    enable_frozen_param_safe_checkpointing()
    assert util.CheckpointFunction is first
    print("    PASS: second call is a no-op")


def main():
    check_stock_version_reproduces_the_documented_crash()
    check_patched_version_matches_unchecked_reference()
    check_idempotent()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED "
          "(against a faithful stand-in for comfy's util module -- see module docstring)")


if __name__ == "__main__":
    main()
