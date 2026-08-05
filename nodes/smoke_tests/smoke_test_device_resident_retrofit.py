"""Correctness check for the new DeviceResident.footprint_bytes() on every
concrete OptimizerHandle -- this is new behavior with no legacy equivalent
to compare against (not an equivalence test), so what's checked is: sane
values, and specifically the release()-then-footprint_bytes() round trip,
since several wrapped legacy classes (ChunkedXPUAdafactor/ChunkedXPUCAME/
ForeachXPUAdafactor/ForeachXPUCAME/FusedXPUAdafactor/CPUAdamW) `del` their
state attributes entirely in free_states() rather than clearing them --
confirmed by reading core/optimizers.py directly, not assumed.

Run this directly: `python nodes/smoke_tests/smoke_test_device_resident_retrofit.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import (ChunkedXPUAdafactor, ChunkedXPUCAME, CPUAdamW,
                              ForeachXPUAdafactor, ForeachXPUCAME, FusedXPUAdafactor)
from nodes.memory.handle import DeviceResident
from nodes.optimizer.adamw import AdamWOptimizerHandle, SimpleAdamWOptimizerHandle
from nodes.optimizer.adafactor import AdafactorOptimizerHandle
from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.came import CAMEOptimizerHandle
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.foreach_adafactor import ForeachAdafactorOptimizerHandle
from nodes.optimizer.foreach_came import ForeachCAMEOptimizerHandle
from nodes.optimizer.fused_adafactor import FusedAdafactorOptimizerHandle
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


def _fused_legacy(params):
    # FusedAdafactorOptimizerNode.build() calls register_hooks() as its
    # last step (see that module's docstring) -- done manually here since
    # this test constructs the legacy class directly, bypassing the Node.
    legacy = FusedXPUAdafactor(params, lr=1e-3, device=DEVICE)
    legacy.register_hooks()
    return legacy


def _step(handle, params):
    if isinstance(handle, FusedAdafactorOptimizerHandle):
        # This family applies updates from backward-pass hooks, not a
        # separate step() call (step() is a real no-op here -- see
        # FusedAdafactorOptimizerHandle.step()'s own comment) -- so
        # triggering it means an actual backward(), not p.grad + step().
        loss = sum(p.sum() for p in params)
        loss.backward()
        return
    for p in params:
        p.grad = torch.randn_like(p)
    handle.step()


def check_eager_family():
    """Composed/AdamW/SimpleAdamW: state exists from construction --
    footprint_bytes() should be exactly right immediately, no step needed."""
    print("\n=== Eagerly-allocated state: exact byte count from construction ===")

    params = _params()
    expected = sum(p.numel() * p.element_size() for p in params) * 2  # m + v, same dtype/shape
    legacy = CPUAdamW(params, lr=1e-3)  # CPU-resident by design, no device kwarg
    handle = AdamWOptimizerHandle(legacy)
    record(isinstance(handle, DeviceResident), "AdamWOptimizerHandle is a DeviceResident")
    record(handle.footprint_bytes() == expected, "AdamWOptimizerHandle.footprint_bytes() exact",
           detail=f"got {handle.footprint_bytes()}, expected {expected}")

    params = _params()
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


def check_lazy_family():
    """Adafactor/CAME (+ foreach/fused variants): state is None until the
    first step() lazily allocates it, and free_states() `del`s the
    attributes entirely -- footprint_bytes() must handle both."""
    print("\n=== Lazily-allocated state: 0 before step(), >0 after, 0 again after release() ===")

    cases = [
        ("AdafactorOptimizerHandle",
         lambda p: AdafactorOptimizerHandle(ChunkedXPUAdafactor(p, lr=1e-3, device=DEVICE))),
        ("CAMEOptimizerHandle",
         lambda p: CAMEOptimizerHandle(ChunkedXPUCAME(p, lr=1e-3, device=DEVICE))),
        ("ForeachAdafactorOptimizerHandle",
         lambda p: ForeachAdafactorOptimizerHandle(
             ForeachXPUAdafactor(p, lr=1e-3, device=DEVICE))),
        ("ForeachCAMEOptimizerHandle",
         lambda p: ForeachCAMEOptimizerHandle(ForeachXPUCAME(p, lr=1e-3, device=DEVICE))),
        ("FusedAdafactorOptimizerHandle",
         lambda p: FusedAdafactorOptimizerHandle(_fused_legacy(p))),
    ]
    for name, build in cases:
        params = _params()
        handle = build(params)
        record(isinstance(handle, DeviceResident), f"{name} is a DeviceResident")
        record(handle.footprint_bytes() == 0, f"{name}.footprint_bytes() == 0 before any step()",
               detail=f"got {handle.footprint_bytes()}")
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


def check_simple_adamw():
    print("\n=== SimpleAdamWOptimizerHandle (torch.optim.AdamW-backed) ===")
    params = _params()
    legacy = torch.optim.AdamW(params, lr=1e-3)
    handle = SimpleAdamWOptimizerHandle(legacy, DEVICE)
    record(isinstance(handle, DeviceResident), "SimpleAdamWOptimizerHandle is a DeviceResident")
    record(handle.footprint_bytes() == 0, "footprint_bytes() == 0 before any step() "
           "(torch.optim.AdamW's state dict is empty until the first step())",
           detail=f"got {handle.footprint_bytes()}")
    _step(handle, params)
    fp = handle.footprint_bytes()
    record(fp > 0, "footprint_bytes() > 0 after step()", detail=f"got {fp}")
    handle.release()
    record(handle.footprint_bytes() == 0, "footprint_bytes() == 0 after release()")


def check_offload_reload_alias_delegates():
    """offload()/reload() are new aliases onto offload_states_to_cpu()/
    reload_states_to_device() -- confirm they actually call through (no
    real cross-device move is checkable in this CPU-only sandbox, but the
    delegation itself, and that it doesn't raise, is)."""
    print("\n=== offload()/reload() alias delegation doesn't raise, footprint unchanged (cpu->cpu) ===")
    params = _params()
    legacy = CPUAdamW(params, lr=1e-3)  # CPU-resident by design, no device kwarg
    handle = AdamWOptimizerHandle(legacy)
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
    check_lazy_family()
    check_simple_adamw()
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
