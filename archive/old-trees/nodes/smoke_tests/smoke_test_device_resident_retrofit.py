"""Correctness check for DeviceResident.footprint_bytes() on the
optimizer handles that remain -- this is new behavior with no legacy
equivalent to compare against (not an equivalence test), so what's
checked is: sane values, plus the release()-then-footprint_bytes() round
trip, since some of these classes drop their state attributes entirely
rather than clearing them, and a footprint_bytes() that raised (or
reported a stale non-zero) after release() would be a real leak-shaped
bug in the VRAM accounting.

The wrapped-legacy-optimizer handles this used to cover have all been
retired as each was proven equivalent to its Composed* replacement, or
-- in AdafactorOptimizerHandle's case (2026-10-02) -- retired the other
way round, by deleting it once it turned out its only unreplicated
behavior was the cross-parameter batching that
smoke_test_adafactor_tiny_parameter_gap.py Part C measured to be
*contamination* rather than a pure optimization. See
docs/known-issues/open.md. So the classes checked now are the composed
handles, which own their state directly and need no such allowance.

Run this directly: `python nodes/smoke_tests/smoke_test_device_resident_retrofit.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.memory.handle import DeviceResident
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.strategies.simple import SimpleLoopStrategy

DEVICE = "cpu"
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _params():
    torch.manual_seed(0)
    return [torch.randn(32, 32, requires_grad=True), torch.randn(16, requires_grad=True)]


def _step(handle, params):
    for p in params:
        p.grad = torch.randn_like(p)
    handle.step()


def check_eager_family():
    """Composed: state exists from construction -- footprint_bytes()
    should be exactly right immediately, no step needed."""
    print("\n=== Eagerly-allocated state: exact byte count from construction ===")

    params = _params()
    expected = sum(p.numel() * p.element_size() for p in params) * 2  # m + v, same dtype/shape
    algorithm = AdamWAlgorithm(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    strategy = SimpleLoopStrategy()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=strategy,
                                      params=params, lr=1e-3, device=DEVICE)
    record(isinstance(handle, DeviceResident), "ComposedOptimizerHandle is a DeviceResident")
    fp_before = handle.footprint_bytes()
    record(fp_before == expected, "ComposedOptimizerHandle.footprint_bytes() exact",
           detail=f"got {fp_before}, expected {expected}")
    handle.release()
    record(handle.footprint_bytes() == 0,
           "ComposedOptimizerHandle.footprint_bytes() == 0 after release()")


def check_release_family():
    """release() drops state outright rather than moving it to cpu, so
    footprint_bytes() has to report 0 afterwards without raising.

    Retargeted 2026-10-02 from the AdafactorOptimizerHandle case, which
    went away with nodes/optimizer/adafactor.py. That case was here for a
    real reason and the reason did not expire with the class:
    ChunkedXPUAdafactor routed every parameter under 10,000 elements
    through a separate batched fast path with its own single shared state
    tensor, which that Handle's footprint_bytes() missed -- it silently
    reported 0 for an all-small-parameters optimizer, which is how this
    test found it. The invariant is the under-reporting, not the class it
    was found in, so it is checked here against the handle that remains,
    on the same all-small-parameter shapes that caught it (a 32x32 and a
    16-element parameter, both under the threshold, with the threshold
    actually set so the tiny path is the one exercised).
    """
    print("\n=== release() drops state: footprint_bytes() == 0, no AttributeError ===")

    name = "ComposedOptimizerHandle(AdafactorAlgorithm, all-small params)"
    params = _params()
    handle = ComposedOptimizerHandle(
        algorithm=AdafactorAlgorithm(tiny_parameter_threshold=10_000),
        strategy=SimpleLoopStrategy(), params=params, lr=1e-3, device=DEVICE)
    record(isinstance(handle, DeviceResident), f"{name} is a DeviceResident")
    _step(handle, params)
    fp_after_step = handle.footprint_bytes()
    record(fp_after_step > 0, f"{name}.footprint_bytes() > 0 after step()",
           detail=f"got {fp_after_step}")
    try:
        handle.release()
        fp_after_release = handle.footprint_bytes()
        ok = fp_after_release == 0
    except AttributeError as e:
        ok = False
        fp_after_release = f"raised {e!r}"
    record(ok, f"{name}.footprint_bytes() == 0 after release() (no AttributeError)",
           detail=str(fp_after_release))


def check_offload_reload_alias_delegates():
    """offload()/reload() are new aliases onto offload_states_to_cpu()/
    reload_states_to_device() -- confirm they actually call through (no
    real cross-device move is checkable in this CPU-only sandbox, but the
    delegation itself, and that it doesn't raise, is)."""
    print("\n=== offload()/reload() alias delegation doesn't raise, footprint unchanged (cpu->cpu) ===")
    params = _params()
    algorithm = AdamWAlgorithm(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    strategy = SimpleLoopStrategy()
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=strategy,
                                      params=params, lr=1e-3, device=DEVICE)
    fp_before = handle.footprint_bytes()
    try:
        handle.offload()
        handle.reload()
        ok = handle.footprint_bytes() == fp_before
    except Exception as e:
        ok = False
        record(ok, "offload()/reload() round trip", detail=repr(e))
        return
    record(ok, "offload()/reload() round trip leaves footprint_bytes() unchanged")


def main():
    print("Device: cpu")
    check_eager_family()
    check_release_family()
    check_offload_reload_alias_delegates()

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
