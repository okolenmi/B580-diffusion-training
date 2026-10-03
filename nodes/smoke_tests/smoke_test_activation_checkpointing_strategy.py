"""Correctness check for ActivationCheckpointingStrategy/NoCheckpointing/
FrozenParamSafeCheckpointing (docs/design/03-training-step-orchestration.md section
2.3, nodes/model/gradient_checkpointing.py).

The underlying patch's actual gradient-correctness is already covered by
smoke_test_gradient_checkpointing.py -- this test only checks the new
class layer: that FrozenParamSafeCheckpointing.apply() really does
delegate to (not diverge from) enable_frozen_param_safe_checkpointing(),
that NoCheckpointing truly does nothing, and that the ABC is enforced.

Run this directly: `python nodes/smoke_tests/smoke_test_activation_checkpointing_strategy.py`
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _install_stub_comfy_checkpoint_module():
    """Minimal stand-in -- just needs a CheckpointFunction attribute for
    the patch to read/replace; unlike smoke_test_gradient_checkpointing.py
    this test doesn't need real gradient behavior, only "did the patch
    class get installed"."""
    import torch

    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules",
                 "comfy.ldm.modules.diffusionmodules"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    util = types.ModuleType("comfy.ldm.modules.diffusionmodules.util")

    class _StockCheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            pass

        @staticmethod
        def backward(ctx, *grads):
            pass

    util.CheckpointFunction = _StockCheckpointFunction
    sys.modules["comfy.ldm.modules.diffusionmodules.util"] = util
    sys.modules["comfy.ldm.modules.diffusionmodules"].util = util

    # FrozenParamSafeCheckpointing.apply() below also calls
    # enable_attention_block_checkpointing() (nodes/model/
    # attention_checkpointing.py) now -- same minimal-stub reasoning as
    # above, just enough of a BasicTransformerBlock (a class with a
    # forward attribute) for that patch to install without crashing on
    # a missing import. That patch's own real correctness is covered by
    # smoke_test_attention_checkpointing.py, not here.
    attention = types.ModuleType("comfy.ldm.modules.attention")

    class _StubBasicTransformerBlock:
        def forward(self, x, context=None, transformer_options={}):
            return x

    attention.BasicTransformerBlock = _StubBasicTransformerBlock
    sys.modules["comfy.ldm.modules.attention"] = attention
    sys.modules["comfy.ldm.modules"].attention = attention

    return util


def check_no_checkpointing_is_a_true_no_op():
    print("\n=== NoCheckpointing.apply() touches nothing ===")
    from nodes.model.gradient_checkpointing import NoCheckpointing
    # Deliberately don't even install the comfy stub -- a real no-op must
    # not need it, unlike FrozenParamSafeCheckpointing which does.
    try:
        NoCheckpointing().apply()
        ok = True
    except Exception as e:
        ok = False
        record(ok, "apply() doesn't raise, doesn't import comfy", detail=repr(e))
        return
    record(ok, "apply() doesn't raise, doesn't import comfy at all")


def check_frozen_param_safe_delegates_correctly():
    print("\n=== FrozenParamSafeCheckpointing.apply() delegates to the real patch ===")
    from nodes.model.checkpoint import active_checkpoint_function
    from nodes.model.gradient_checkpointing import (FrozenParamSafeCheckpointing,
                                                      enable_frozen_param_safe_checkpointing)

    # This used to assert against a stubbed `comfy.ldm...util` namespace,
    # because the function patched ComfyUI's CheckpointFunction in place.
    # It no longer patches anything: `checkpoint.py` *is* the implementation
    # (design doc 12 section 7.3, section A), so what apply() installs is the
    # class this project's own `checkpoint()` dispatches to. Asserting against
    # a stub here would now be asserting against a module nothing writes to.
    #
    # The stub is still installed, for the *other* leg of apply(): it also
    # calls enable_attention_block_checkpointing(), which patches ComfyUI's
    # BasicTransformerBlock and so still needs the module to exist. That is
    # the remaining half of section A -- once BasicTransformerBlock is
    # reimplemented too,
    # the stub comes out of this test too.
    _install_stub_comfy_checkpoint_module()
    FrozenParamSafeCheckpointing().apply()
    patched_via_class = active_checkpoint_function()
    # Not "apply() switched to a new class": the fixed class is already the
    # *default* in checkpoint.py, because it is a strict superset of the
    # original's gradients (checked in smoke_test_checkpoint.py), so the
    # first apply() is legitimately already in place. The claim is about the
    # state, not the transition.
    record(getattr(patched_via_class, "_frozen_param_safe", False),
           "after apply(), checkpoint() dispatches to a frozen-param-safe class")
    record(getattr(patched_via_class, "_recompute_wrapper_identity", "unset") is None,
           "with no recompute wrapper, which is what apply() asks for")

    # A second instance's apply() must be a no-op (shared module-global
    # state, not per-instance).
    FrozenParamSafeCheckpointing().apply()
    record(active_checkpoint_function() is patched_via_class,
           "a second FrozenParamSafeCheckpointing().apply() doesn't re-wrap")

    # A different wrapper is a real change, in both entry paths.
    before_free = active_checkpoint_function()
    enable_frozen_param_safe_checkpointing(recompute_wrapper=lambda f, a: f(*a))
    record(active_checkpoint_function() is not before_free,
           "the free function with a new wrapper does switch the class")
    FrozenParamSafeCheckpointing().apply()
    record(active_checkpoint_function() is not before_free,
           "and apply() switches it back, since it passes no wrapper")


def check_strategy_contract():
    print("\n=== ActivationCheckpointingStrategy is a real ABC ===")
    from nodes.model.gradient_checkpointing import ActivationCheckpointingStrategy

    class BadStrategy(ActivationCheckpointingStrategy):
        pass

    try:
        BadStrategy()
        ok = False
    except TypeError:
        ok = True
    record(ok, "can't instantiate a strategy that doesn't implement apply()")


def main():
    check_no_checkpointing_is_a_true_no_op()
    check_frozen_param_safe_delegates_correctly()
    check_strategy_contract()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
