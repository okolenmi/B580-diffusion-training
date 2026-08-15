"""Correctness check for nodes/memory/profile.py's ResourceProfile
(backlog item 1, docs/training_pipeline_design.md section 5.5).

Run this directly: `python nodes/smoke_tests/smoke_test_resource_profile.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.components.device import DeviceContext
from nodes.memory.coordinator import ResourceCoordinator
from nodes.memory.handle import DeviceResident
from nodes.memory.manager import MemoryManager
from nodes.memory.profile import ResourceProfile

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _FakeResident(DeviceResident):
    def __init__(self, footprint: int):
        self._footprint = footprint

    def footprint_bytes(self):
        return self._footprint

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _FixedDeviceContext(DeviceContext):
    """Real production code only ever gets a _NullDeviceContext (None)
    on CPU (see smoke_test_device_context_equivalence.py) -- this fakes
    what a real XPU/CUDA memory_stats() dict looks like, so capture()'s
    "pass the dict straight through unchanged" behavior is exercised
    for the non-None case too, without needing real GPU hardware here."""

    def __init__(self, stats: dict):
        self._stats = stats

    def empty_cache(self):
        pass

    def synchronize(self):
        pass

    def memory_stats(self):
        return self._stats


def check_capture_builds_per_resident_breakdown():
    print("\n=== capture(): per_resident_bytes matches coordinator's real breakdown ===")
    coord = ResourceCoordinator()
    coord.register("model", _FakeResident(1000))
    coord.register("optimizer", _FakeResident(2000))
    profile = ResourceProfile.capture(coord, None, DeviceContext.for_device("cpu"))
    record(profile.per_resident_bytes == {"model": 1000, "optimizer": 2000},
           "per_resident_bytes is exactly coordinator.per_resident_footprint_bytes()'s output",
           detail=str(profile.per_resident_bytes))


def check_capture_with_no_memory_manager():
    print("\n=== capture(): memory=None -> memory_manager_stats is None, not an error or a 0 ===")
    coord = ResourceCoordinator()
    profile = ResourceProfile.capture(coord, None, DeviceContext.for_device("cpu"))
    record(profile.memory_manager_stats is None,
           "memory_manager_stats is None when no MemoryManager was given",
           detail=str(profile.memory_manager_stats))


def check_capture_with_a_real_memory_manager():
    print("\n=== capture(): a real MemoryManager's stats() is captured, not stubbed ===")
    coord = ResourceCoordinator()
    memory = MemoryManager()
    memory.get_buffer("scratch", 64, torch.float32, "cpu")
    profile = ResourceProfile.capture(coord, memory, DeviceContext.for_device("cpu"))
    expected = memory.stats()
    record(profile.memory_manager_stats == expected,
           "memory_manager_stats matches memory.stats() exactly",
           detail=f"got={profile.memory_manager_stats} expected={expected}")
    record(profile.memory_manager_stats["total_bytes"] > 0,
           "a real allocated buffer shows up as nonzero total_bytes",
           detail=str(profile.memory_manager_stats))


def check_capture_passes_allocator_stats_through_unchanged():
    print("\n=== capture(): allocator_stats is device_ctx.memory_stats(), untouched ===")
    coord = ResourceCoordinator()

    fake_stats = {"allocated_mb": 512.0, "reserved_mb": 640.0, "num_alloc_retries": 0.0}
    profile = ResourceProfile.capture(coord, None, _FixedDeviceContext(fake_stats))
    record(profile.allocator_stats == fake_stats,
           "a real (faked) non-None memory_stats() dict passes through unchanged",
           detail=str(profile.allocator_stats))

    profile_cpu = ResourceProfile.capture(coord, None, DeviceContext.for_device("cpu"))
    record(profile_cpu.allocator_stats is None,
           "CPU (_NullDeviceContext) has no allocator concept -- None, matching "
           "DeviceContext.memory_stats()'s own documented CPU behavior",
           detail=str(profile_cpu.allocator_stats))


def check_profile_is_a_frozen_snapshot():
    print("\n=== ResourceProfile is frozen -- a snapshot, not a live view ===")
    coord = ResourceCoordinator()
    coord.register("model", _FakeResident(100))
    profile = ResourceProfile.capture(coord, None, DeviceContext.for_device("cpu"))
    try:
        profile.per_resident_bytes = {}
        record(False, "assigning to a field should raise (frozen dataclass)")
    except Exception:
        record(True, "assigning to a field raises, as a frozen dataclass should")
    coord.register("optimizer", _FakeResident(999))
    record(profile.per_resident_bytes == {"model": 100},
           "registering a new resident after capture() doesn't retroactively change the "
           "already-captured snapshot", detail=str(profile.per_resident_bytes))


def main():
    check_capture_builds_per_resident_breakdown()
    check_capture_with_no_memory_manager()
    check_capture_with_a_real_memory_manager()
    check_capture_passes_allocator_stats_through_unchanged()
    check_profile_is_a_frozen_snapshot()

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
