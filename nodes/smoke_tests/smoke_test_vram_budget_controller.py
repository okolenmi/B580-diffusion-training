"""Checks nodes/memory/vram_budget_controller.py's VRAMBudgetControllerNode.

No existing smoke test exercised this node at all before this session
(checked directly: grep for VRAMBudgetControllerNode across
nodes/smoke_tests/ turned up nothing) -- a real gap, alongside the new
total_memory_mb() sanity check this session added to build() itself
(DeviceContext.total_memory_mb()'s own docstring has the full
reasoning). DeviceContext.for_device() only ever returns a real
XPU/CUDA/null context based on a device string, so this monkeypatches
the staticmethod itself to inject a scripted total -- build() calls it
fresh every time rather than holding one DeviceContext across calls, so
patching the staticmethod directly (restored unconditionally after) is
enough here, no constructor-injection seam needed.
"""

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.components.device import DeviceContext
from nodes.memory.control_handle import BudgetedResourceControlHandle
from nodes.memory.vram_budget_controller import VRAMBudgetControllerNode

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _FixedTotal:
    def __init__(self, total_mb):
        self._total_mb = total_mb

    def total_memory_mb(self):
        return self._total_mb


def _build_with_scripted_total(total_mb, **build_kwargs):
    """Swap DeviceContext.for_device() for exactly one build() call,
    capturing whatever it prints to stdout -- restores the real
    staticmethod unconditionally, so a failure here can't leave it
    patched for whatever smoke test runs next in the same process."""
    original = DeviceContext.for_device
    DeviceContext.for_device = staticmethod(lambda device: _FixedTotal(total_mb))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = VRAMBudgetControllerNode().build(**build_kwargs)
    finally:
        DeviceContext.for_device = original
    return result, buf.getvalue()


def check_builds_a_real_handle():
    print("[build() returns a real BudgetedResourceControlHandle, unconditionally]")
    result = VRAMBudgetControllerNode().build(vram_budget_mb=8000.0)
    record(isinstance(result["control"], BudgetedResourceControlHandle),
           "output is a BudgetedResourceControlHandle", detail=repr(result["control"]))


def check_no_warning_when_under_the_real_total():
    print("[budget under the device's own real total: no warning printed]")
    _, out = _build_with_scripted_total(16000.0, vram_budget_mb=8000.0)
    record("WARNING" not in out, "no WARNING line", detail=out)


def check_warns_when_over_the_real_total():
    print("[budget over the device's own real total: a clear WARNING, "
          "build() still succeeds -- not auto-corrected]")
    result, out = _build_with_scripted_total(8000.0, vram_budget_mb=12500.0)
    record("WARNING" in out and "12500" in out and "8000" in out,
           "WARNING line names both the stated budget and the real total", detail=out)
    record(isinstance(result["control"], BudgetedResourceControlHandle),
           "still returns a real handle -- a warning, not a refusal", detail=repr(result["control"]))


def check_no_crash_when_total_memory_mb_is_none():
    print("[total_memory_mb() unavailable (this sandbox's real, CPU-only case): "
          "no crash, no warning, nothing to check against]")
    result = VRAMBudgetControllerNode().build(vram_budget_mb=8000.0)  # real for_device(), no patch
    record(isinstance(result["control"], BudgetedResourceControlHandle),
           "still returns a real handle", detail=repr(result["control"]))


def main():
    check_builds_a_real_handle()
    check_no_warning_when_under_the_real_total()
    check_warns_when_over_the_real_total()
    check_no_crash_when_total_memory_mb_is_none()

    print()
    print("=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
