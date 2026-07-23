"""Real-XPU-hardware-only tests for MemoryManager's torch.xpu.MemPool
integration (use_mempool=True).

NOT run or verified by Claude -- this needs real XPU hardware, which
wasn't available in the sandbox that wrote this. Every API call here was
confirmed against PyTorch's actual source and against what this
sandbox's (CUDA-only) torch build returns for the no-device case
(all-zero/empty results, no exceptions) -- but the numbers this file
actually prints, and whether the MemPool integration behaves correctly
under real allocation pressure, needs a human reading real output on
real hardware. Please read each check's printed output yourself rather
than trusting only the final PASS/FAIL line, especially check [4].

Run this directly on your XPU machine:

    python nodes/smoke_tests/xpu_mempool_hardware_check.py
    python nodes/smoke_tests/xpu_mempool_hardware_check.py --stress   # adds check [5], see its docstring first

Deliberately named so run_all.py's `smoke_test_*.py` glob does NOT pick
this up automatically -- it needs real hardware (would exit immediately
on CPU/CI), is heavier than the rest of the suite, and check [4]'s
output needs a human judgment call, not an automatic pass/fail.

What each check covers, and why it matters:

  [1] Basic construction and allocation: does MemoryManager(use_mempool=
      True) actually construct and route a real allocation through
      torch.xpu.MemPool on this hardware at all? Gate before anything
      else here means anything.

  [2] Correctness: turning MemPool on should change WHERE memory comes
      from, never WHAT ends up in it. Runs the same
      ComposedOptimizerHandle training loop (CAME + chunked strategy)
      twice, identical seeds, once with use_mempool=False and once True,
      and diffs the resulting parameters. Expected: bit-exact (0.0 max
      diff). Any difference would mean the integration is doing
      something to values, not just allocation routing -- a real bug,
      not a rounding-noise question like some of this project's other
      real-vs-legacy comparisons.

  [3] free_all() actually releases real device memory, not just this
      manager's own internal bookkeeping -- uses
      torch.xpu.memory_allocated() before/after (the actual device
      driver's view), not manager.stats() (which only reports what the
      manager itself thinks it's holding and would trivially "pass" even
      if the underlying MemPool never gave anything back).

  [4] Fragmentation comparison -- the actual claimed benefit of MemPool
      integration, not confirmed anywhere yet. Repeatedly allocates and
      frees varying, awkward sizes (deliberately unfriendly to a naive
      allocator) under both configurations and prints
      torch.xpu.memory_stats()'s reserved/peak-reserved bytes side by
      side. Deliberately NOT scored pass/fail -- there's no threshold to
      assert against without a first real reading of these numbers on
      real hardware. Lower reserved/peak under use_mempool=True would
      support the fragmentation-reduction claim from
      docs/nodes_package_design.md; no meaningful difference (or worse)
      would argue against enabling it here.

  [5] (--stress only, off by default) A closer look at the documented
      OOM-retry tradeoff (pytorch/pytorch#159674): allocations inside
      use_mem_pool skip the default caching allocator's retry-after-
      defragmentation behavior. This check pushes allocation size up
      toward a fraction of currently-free device memory to see whether
      use_mempool=True fails at a point use_mempool=False would have
      recovered from. Off by default because deliberately approaching an
      OOM boundary can affect whatever else is running on the same
      device -- read this function's own docstring before passing
      --stress on a shared or production machine.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.memory.manager import MemoryManager
from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.chunked import ChunkedScratchBufferStrategy

DEVICE = "xpu"


def require_real_xpu() -> None:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print(
            "No working torch.xpu backend detected in this environment "
            "(torch.xpu.is_available() is False). This file needs real XPU "
            "hardware to produce meaningful output -- on a build without it, "
            "torch.xpu.memory_allocated()/memory_stats() silently return "
            "0/empty rather than raising, which would make every check below "
            "look trivially fine for the wrong reason. Exiting rather than "
            "printing misleading results."
        )
        sys.exit(1)


def check1_basic_construction_and_allocation() -> bool:
    print("\n[1] Basic construction and allocation on real XPU:")
    mgr = MemoryManager(use_mempool=True)
    buf = mgr.get_buffer("t", 1000, torch.float32, DEVICE)
    ok = buf.numel() >= 1000 and buf.device.type == "xpu"
    print(f"    {'PASS' if ok else 'FAIL'}: allocated buffer shape={tuple(buf.shape)}, "
          f"device={buf.device}")
    mgr.free_all()
    return ok


def check2_correctness_vs_no_mempool(n_steps: int = 60) -> bool:
    print("\n[2] Correctness: use_mempool=True must not change computed VALUES:")
    results = {}
    for use_mempool in (False, True):
        torch.manual_seed(42)
        W_init = torch.randn(64, 64, device=DEVICE) * 0.1
        W = W_init.clone().requires_grad_(True)
        algo = CAMEAlgorithm()
        mem = MemoryManager(use_mempool=use_mempool)
        strategy = ChunkedScratchBufferStrategy(memory=mem)
        handle = ComposedOptimizerHandle(algorithm=algo, strategy=strategy,
                                          params=[W], lr=0.01, device=DEVICE)
        for step in range(n_steps):
            torch.manual_seed(1000 + step)
            g = torch.randn(64, 64, device=DEVICE) * 0.05
            W.grad = g.clone()
            handle.step()
            handle.zero_grad()
        results[use_mempool] = W.detach().clone()
        mem.free_all()

    diff = (results[False] - results[True]).abs().max().item()
    ok = diff == 0.0
    print(f"    {'PASS' if ok else 'FAIL'}: max abs diff between "
          f"use_mempool=False and use_mempool=True over {n_steps} steps = {diff:.3e} "
          f"(expected exactly 0.0)")
    return ok


def check3_memory_actually_released() -> bool:
    print("\n[3] free_all() actually releases real device memory:")
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    before = torch.xpu.memory_allocated(DEVICE)

    mgr = MemoryManager(use_mempool=True)
    buf = mgr.get_buffer("big", 50_000_000, torch.float32, DEVICE)  # ~200MB
    torch.xpu.synchronize()
    during = torch.xpu.memory_allocated(DEVICE)

    del buf
    mgr.free_all()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    after = torch.xpu.memory_allocated(DEVICE)

    grew = during > before
    # Small tolerance -- other allocator bookkeeping can leave a little
    # reserved even after a real free; this isn't meant to be exact to
    # the byte, just confirm the ~200MB buffer's memory came back.
    tolerance = 5_000_000
    released = after <= before + tolerance

    print(f"    before={before:,} bytes, during={during:,} bytes, after={after:,} bytes")
    print(f"    {'PASS' if grew else 'FAIL'}: allocation actually consumed real device memory")
    print(f"    {'PASS' if released else 'FAIL'}: memory returned close to baseline after free_all() "
          f"(within {tolerance:,} byte tolerance)")
    return grew and released


def check4_fragmentation_comparison(n_allocs: int = 200) -> None:
    print("\n[4] Fragmentation comparison -- DIAGNOSTIC ONLY, read the numbers yourself:")
    import random
    random.seed(0)
    sizes = [random.randint(1_000, 5_000_000) for _ in range(n_allocs)]

    for use_mempool in (False, True):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats(DEVICE)
        mgr = MemoryManager(use_mempool=use_mempool)
        # 10 rotating tags, occasionally force-freed -- deliberately
        # awkward/churning allocation pattern, not the smooth
        # monotonic-growth case get_buffer() already handles trivially well.
        for i, size in enumerate(sizes):
            tag = f"buf{i % 10}"
            mgr.get_buffer(tag, size, torch.float32, DEVICE)
            mgr.release(tag)
            if i % 20 == 0:
                mgr.free(tag)
        torch.xpu.synchronize()
        stats = torch.xpu.memory_stats(DEVICE)
        reserved = stats.get("reserved_bytes.all.current", "n/a")
        peak_reserved = stats.get("reserved_bytes.all.peak", "n/a")
        allocated = stats.get("allocated_bytes.all.current", "n/a")
        print(f"    use_mempool={use_mempool!s:5s}: allocated={allocated}, "
              f"reserved={reserved}, peak_reserved={peak_reserved}")
        mgr.free_all()

    print("    No automatic pass/fail here. Lower reserved/peak_reserved under "
          "use_mempool=True would support the fragmentation-reduction claim in "
          "docs/nodes_package_design.md; no meaningful difference (or worse) "
          "numbers would argue against enabling it for this workload.")


def check5_oom_retry_stress(budget_fraction: float = 0.6) -> None:
    """Off by default -- read before enabling with --stress.

    Pushes allocation size up toward `budget_fraction` of currently-free
    device memory under both configurations, to see whether
    use_mempool=True fails at a point use_mempool=False recovers from
    (the documented pytorch/pytorch#159674 tradeoff). Deliberately stays
    under 100% of free memory rather than trying to force a real OOM
    (which could affect other processes on a shared device) -- this is
    a *closer look*, not a guaranteed reproduction. If it doesn't show a
    difference, that's inconclusive, not a clean "no problem here."
    """
    print("\n[5] OOM-retry tradeoff, closer look (--stress only):")
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    free_bytes, _total = torch.xpu.mem_get_info(DEVICE) if hasattr(torch.xpu, "mem_get_info") \
        else (None, None)
    if free_bytes is None:
        print("    torch.xpu.mem_get_info not available in this torch version -- skipping. "
              "(Present in recent PyTorch releases; check your version if this matters to you.)")
        return

    target_bytes = int(free_bytes * budget_fraction)
    target_numel = target_bytes // 4  # float32
    print(f"    free device memory: {free_bytes:,} bytes; targeting {target_bytes:,} bytes "
          f"({budget_fraction:.0%} of free)")

    for use_mempool in (False, True):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        mgr = MemoryManager(use_mempool=use_mempool)
        try:
            mgr.get_buffer("stress", target_numel, torch.float32, DEVICE)
            print(f"    use_mempool={use_mempool}: succeeded allocating ~{target_bytes:,} bytes")
        except RuntimeError as e:
            print(f"    use_mempool={use_mempool}: FAILED -- {type(e).__name__}: {str(e)[:150]}")
        finally:
            mgr.free_all()
            torch.xpu.synchronize()
            torch.xpu.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", action="store_true",
                         help="Also run check [5] (OOM-retry tradeoff) -- read its docstring first")
    args = parser.parse_args()

    require_real_xpu()
    device_name = torch.xpu.get_device_name(0)
    print(f"Device: xpu ({device_name})")

    results = [
        ("basic construction/allocation", check1_basic_construction_and_allocation()),
        ("correctness vs no-mempool", check2_correctness_vs_no_mempool()),
        ("memory actually released", check3_memory_actually_released()),
    ]
    check4_fragmentation_comparison()  # diagnostic, no pass/fail
    if args.stress:
        check5_oom_retry_stress()  # diagnostic, no pass/fail

    print("\n" + "=" * 60)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All pass/fail checks [1]-[3] passed. Read check [4]'s (and [5]'s, if run) "
              "printed numbers yourself -- no automatic verdict there.")


if __name__ == "__main__":
    main()
