"""Behavioral equivalence check: nodes/components/device.py's DeviceContext
vs. the legacy core.comfy_setup.xpu_empty_cache/xpu_synchronize/
xpu_memory_stats it's a fresh reimplementation of.

These aren't numerical comparisons (there's no XPU hardware in this sandbox)
-- what's checked is that the guard behavior is identical: same
hasattr(torch, "xpu")/is_available() gating, same no-op-when-unavailable
result, moved from three free functions into DeviceContext's three methods
without changing what they do. On a machine with real XPU/CUDA hardware
this same script exercises the non-no-op branches too.

Run this directly: `python nodes/smoke_tests/smoke_test_device_context_equivalence.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.comfy_setup import xpu_empty_cache, xpu_memory_stats, xpu_synchronize
from nodes.components.device import (DeviceContext, _CUDADeviceContext,
                                      _NullDeviceContext, _XPUDeviceContext)

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_for_device_dispatch():
    print("\n=== DeviceContext.for_device() dispatch ===")
    cases = [
        ("cpu", _NullDeviceContext),
        ("xpu", _XPUDeviceContext),
        ("xpu:0", _XPUDeviceContext),
        ("cuda", _CUDADeviceContext),
        ("cuda:0", _CUDADeviceContext),
        (torch.device("cpu"), _NullDeviceContext),
    ]
    for device, expected_type in cases:
        ctx = DeviceContext.for_device(device)
        record(isinstance(ctx, expected_type),
               f"for_device({device!r}) -> {expected_type.__name__}",
               detail=f"got {type(ctx).__name__}")


def check_null_context():
    print("\n=== _NullDeviceContext: no-op by construction ===")
    ctx = _NullDeviceContext()
    try:
        ctx.empty_cache()
        ctx.synchronize()
        ok = True
    except Exception as e:
        ok = False
        record(ok, "empty_cache()/synchronize() don't raise", detail=str(e))
    else:
        record(ok, "empty_cache()/synchronize() don't raise")
    record(ctx.memory_stats() is None, "memory_stats() is None")


def check_xpu_context_matches_legacy():
    print("\n=== _XPUDeviceContext vs core.comfy_setup.xpu_* "
          "(same hasattr/is_available guard) ===")
    xpu_actually_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"  (xpu_actually_available = {xpu_actually_available} in this sandbox)")
    ctx = _XPUDeviceContext()

    try:
        ctx.empty_cache()
        xpu_empty_cache()
        ok = True
    except Exception as e:
        ok = False
        record(ok, "empty_cache() raises the same as legacy xpu_empty_cache()", detail=str(e))
    else:
        record(ok, "empty_cache() behaves the same as legacy xpu_empty_cache()")

    try:
        ctx.synchronize()
        xpu_synchronize()
        ok = True
    except Exception as e:
        ok = False
        record(ok, "synchronize() raises the same as legacy xpu_synchronize()", detail=str(e))
    else:
        record(ok, "synchronize() behaves the same as legacy xpu_synchronize()")

    new_stats = ctx.memory_stats()
    ref_stats = xpu_memory_stats()
    record(new_stats == ref_stats, "memory_stats() == xpu_memory_stats()",
           detail=f"new={new_stats!r} ref={ref_stats!r}")
    if not xpu_actually_available:
        record(new_stats is None, "memory_stats() is None when XPU unavailable")


def check_cuda_context_no_legacy_equivalent():
    print("\n=== _CUDADeviceContext: guarded the same way, no legacy equivalent ===")
    ctx = _CUDADeviceContext()
    cuda_available = torch.cuda.is_available()
    print(f"  (torch.cuda.is_available() = {cuda_available} in this sandbox)")
    try:
        ctx.empty_cache()
        ctx.synchronize()
        ok = True
    except Exception as e:
        ok = False
        record(ok, "empty_cache()/synchronize() don't raise", detail=str(e))
    else:
        record(ok, "empty_cache()/synchronize() don't raise")
    if not cuda_available:
        record(ctx.memory_stats() is None, "memory_stats() is None when CUDA unavailable")


def main():
    print("Device: cpu (equivalence check -- pure behavioral comparison, "
          "real XPU/CUDA hardware not required)")
    check_for_device_dispatch()
    check_null_context()
    check_xpu_context_matches_legacy()
    check_cuda_context_no_legacy_equivalent()

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
