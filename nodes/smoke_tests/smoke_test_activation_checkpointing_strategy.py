"""Correctness check for ActivationCheckpointingStrategy/NoCheckpointing/
FrozenParamSafeCheckpointing (docs/training_pipeline_design.md section
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
    from nodes.model.gradient_checkpointing import (FrozenParamSafeCheckpointing,
                                                      enable_frozen_param_safe_checkpointing)
    util = _install_stub_comfy_checkpoint_module()
    original = util.CheckpointFunction

    FrozenParamSafeCheckpointing().apply()
    patched_via_class = util.CheckpointFunction
    record(patched_via_class is not original,
           "apply() actually installs the patched CheckpointFunction")
    record(getattr(patched_via_class, "_frozen_param_safe", False),
           "installed class carries the _frozen_param_safe idempotency marker")

    # A second instance's apply() must be a no-op (same guard the free
    # function always had -- shared module-global state, not per-instance).
    FrozenParamSafeCheckpointing().apply()
    record(util.CheckpointFunction is patched_via_class,
           "a second FrozenParamSafeCheckpointing().apply() doesn't re-wrap")

    # Cross-check against calling the free function directly on a fresh
    # stub -- both paths must install bit-for-bit the same kind of object.
    util2 = _install_stub_comfy_checkpoint_module()
    enable_frozen_param_safe_checkpointing()
    record(util2.CheckpointFunction.__name__ == patched_via_class.__name__,
           "class path and free-function path install the same patched class",
           detail=f"{util2.CheckpointFunction.__name__} vs {patched_via_class.__name__}")


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
