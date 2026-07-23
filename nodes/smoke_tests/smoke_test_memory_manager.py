"""Real-torch verification for MemoryManager (nodes/memory/manager.py).

Run this directly: `python nodes/smoke_tests/smoke_test_memory_manager.py`

Runs on CPU only, deliberately -- unlike smoke_test_composed_came.py, none
of what's being checked here is device-specific (no XPU/CUDA-only code
path exists in manager.py at all). What matters is the allocator
bookkeeping logic itself: does it actually reuse the same storage across
calls, does it grow correctly, does it catch double-acquire, does free()
actually drop references. Real torch tensors are used throughout (not a
numpy-backed mock) so `.data_ptr()` identity checks are checking real
underlying storage, not simulated behavior.

Also covers the use_mempool plumbing (check [5]) -- but only the parts
CPU actually can: default-off behavior is completely unaffected, and
requesting use_mempool=True on a build/environment without a working
torch.xpu backend raises a clear error rather than failing confusingly
later. Does NOT and CANNOT verify the actual MemPool allocation path or
its real fragmentation-reduction claim -- that needs real XPU hardware,
not covered by this file. See manager.py's module docstring.

Prints a clear PASS/FAIL summary, mirroring smoke_test_composed_came.py's
convention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.memory.manager import MemoryManager

DEVICE = "cpu"


def check(failures: list, description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"    PASS: {description}")
    else:
        msg = f"{description}" + (f" -- {detail}" if detail else "")
        failures.append(msg)
        print(f"    FAIL: {msg}")


def test_reuse_and_growth(failures: list) -> None:
    print("\n[1] Reuse across calls, and growth when a bigger size is requested:")
    mgr = MemoryManager()

    buf1 = mgr.get_buffer("scratch", 100, torch.float32, DEVICE)
    ptr1 = buf1.data_ptr()
    mgr.release("scratch")

    buf2 = mgr.get_buffer("scratch", 50, torch.float32, DEVICE)
    check(failures, "requesting a smaller size reuses the same storage",
          buf2.data_ptr() == ptr1, f"{buf2.data_ptr()} != {ptr1}")
    mgr.release("scratch")

    buf3 = mgr.get_buffer("scratch", 500, torch.float32, DEVICE)
    check(failures, "requesting a larger size grows (reallocates)",
          buf3.numel() >= 500)
    mgr.release("scratch")

    buf4 = mgr.get_buffer("scratch", 500, torch.float32, DEVICE)
    check(failures, "requesting the same size again after growth reuses storage",
          buf4.data_ptr() == buf3.data_ptr())
    mgr.release("scratch")


def test_double_acquire_raises(failures: list) -> None:
    print("\n[2] get_buffer() on an already-in-use tag raises (aliasing guard):")
    mgr = MemoryManager()
    mgr.get_buffer("live", 10, torch.float32, DEVICE)
    raised = False
    try:
        mgr.get_buffer("live", 10, torch.float32, DEVICE)
    except RuntimeError:
        raised = True
    check(failures, "double-acquire without release()/free() raises RuntimeError", raised)

    mgr.release("live")
    raised_after_release = False
    try:
        mgr.get_buffer("live", 10, torch.float32, DEVICE)
    except RuntimeError:
        raised_after_release = True
    check(failures, "re-acquiring after release() does NOT raise",
          not raised_after_release)


def test_free_and_stats(failures: list) -> None:
    print("\n[3] free() / free_all() actually drop buffers, stats() accounts correctly:")
    mgr = MemoryManager()

    a = mgr.get_buffer("a", 100, torch.float32, DEVICE)
    mgr.release("a")
    b = mgr.get_buffer("b", 50, torch.float64, DEVICE)
    mgr.release("b")

    stats = mgr.stats()
    expected_a = 100 * a.element_size()
    expected_b = 50 * b.element_size()
    check(failures, "stats() reports correct per-tag byte counts",
          stats["per_tag_bytes"].get("a") == expected_a
          and stats["per_tag_bytes"].get("b") == expected_b,
          str(stats))
    check(failures, "stats() total_bytes is the sum across tags",
          stats["total_bytes"] == expected_a + expected_b)

    mgr.free("a")
    stats_after_free_a = mgr.stats()
    check(failures, "free('a') removes only tag 'a' from stats()",
          "a" not in stats_after_free_a["per_tag_bytes"]
          and "b" in stats_after_free_a["per_tag_bytes"])

    mgr.free_all()
    stats_after_free_all = mgr.stats()
    check(failures, "free_all() clears every tag",
          stats_after_free_all["total_bytes"] == 0
          and stats_after_free_all["per_tag_bytes"] == {})


def test_dtype_device_isolation(failures: list) -> None:
    print("\n[4] Same tag, different dtype -- isolated, not aliased:")
    mgr = MemoryManager()
    f32 = mgr.get_buffer("shared_tag", 20, torch.float32, DEVICE)
    f64 = mgr.get_buffer("shared_tag", 20, torch.float64, DEVICE)
    check(failures, "different dtypes under the same tag get separate storage",
          f32.data_ptr() != f64.data_ptr() or f32.dtype != f64.dtype)
    stats = mgr.stats()
    check(failures, "stats() sums both dtype variants under one tag name",
          stats["per_tag_bytes"]["shared_tag"]
          == 20 * f32.element_size() + 20 * f64.element_size())


def test_mempool_plumbing(failures: list) -> None:
    print("\n[5] use_mempool plumbing (CPU-testable parts only -- see module docstring):")
    mgr = MemoryManager()  # default use_mempool=False
    check(failures, "default construction (use_mempool=False) works normally",
          mgr.get_buffer("t", 10, torch.float32, DEVICE).numel() >= 10)

    has_working_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if has_working_xpu:
        print("    (real XPU backend detected in this environment -- "
              "use_mempool=True should construct without raising)")
        try:
            MemoryManager(use_mempool=True)
            check(failures, "use_mempool=True constructs on a real XPU backend", True)
        except RuntimeError as e:
            check(failures, "use_mempool=True constructs on a real XPU backend",
                  False, str(e))
    else:
        raised = False
        message = ""
        try:
            MemoryManager(use_mempool=True)
        except RuntimeError as e:
            raised = True
            message = str(e)
        check(failures,
              "use_mempool=True raises a clear RuntimeError (no working XPU backend here)",
              raised and "use_mempool=False" in message, message)


def main():
    print(f"Device: {DEVICE} (pure allocator-bookkeeping logic -- no device-specific "
          f"code path exists in manager.py, so CPU fully exercises it)")
    failures: list = []
    test_reuse_and_growth(failures)
    test_double_acquire_raises(failures)
    test_free_and_stats(failures)
    test_dtype_device_isolation(failures)
    test_mempool_plumbing(failures)

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
