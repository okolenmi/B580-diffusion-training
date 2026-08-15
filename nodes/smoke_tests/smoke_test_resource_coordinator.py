"""Correctness check for nodes/memory/coordinator.py's ResourceCoordinator
and OffloadOrchestrator (backlog item 12,
docs/training_pipeline_design.md sections 5.1, 5.2).

Run this directly: `python nodes/smoke_tests/smoke_test_resource_coordinator.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.memory.handle import DeviceResident
from nodes.memory.coordinator import (CacheRebuildStarting, CheckpointSaveStarting,
                                       OffloadOrchestrator, PreviewGenerationStarting,
                                       ResourceCoordinator, TrainingLifecycleEvent)

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
        self.offloaded = False
        self.reloaded_with = "not called"
        self.released = False

    def footprint_bytes(self):
        return self._footprint

    def offload(self):
        self.offloaded = True

    def reload(self, device=None):
        self.reloaded_with = device

    def release(self):
        self.released = True


def check_register_and_total_footprint():
    print("\n=== register()/per_resident_footprint_bytes()/total_footprint_bytes() ===")
    coord = ResourceCoordinator()
    coord.register("a", _FakeResident(100))
    coord.register("b", _FakeResident(250))
    record(coord.total_footprint_bytes() == 350, "sums every registered resident's footprint")
    per_resident = coord.per_resident_footprint_bytes()
    record(per_resident == {"a": 100, "b": 250},
           "per_resident_footprint_bytes() returns the real per-name breakdown",
           detail=str(per_resident))
    record(sum(per_resident.values()) == coord.total_footprint_bytes(),
           "total_footprint_bytes() is exactly the sum of per_resident_footprint_bytes() "
           "(single source of truth, not two separately-maintained numbers)")


def check_offload_all_except():
    print("\n=== offload_all_except() ===")
    coord = ResourceCoordinator()
    a, b, c = _FakeResident(1), _FakeResident(2), _FakeResident(3)
    coord.register("a", a)
    coord.register("b", b)
    coord.register("c", c)
    coord.offload_all_except({"b"})
    record(a.offloaded and c.offloaded and not b.offloaded,
           "offloads everything except the kept set",
           detail=f"a={a.offloaded} b={b.offloaded} c={c.offloaded}")


def check_reload_delegates_to_the_named_resident():
    print("\n=== reload() delegates to exactly the named resident ===")
    coord = ResourceCoordinator()
    a, b = _FakeResident(1), _FakeResident(2)
    coord.register("a", a)
    coord.register("b", b)
    coord.reload("a", "xpu:0")
    record(a.reloaded_with == "xpu:0" and b.reloaded_with == "not called",
           "only the named resident's reload() was called, with the right device",
           detail=f"a.reloaded_with={a.reloaded_with!r} b.reloaded_with={b.reloaded_with!r}")


def check_orchestrator_dispatches_only_the_matching_event_type():
    print("\n=== OffloadOrchestrator.publish() dispatches only to matching handlers ===")
    coord = ResourceCoordinator()
    device_ctx = object()  # opaque -- just checking it's passed through unchanged
    orch = OffloadOrchestrator(coord, device_ctx)

    calls = []
    orch.on(CacheRebuildStarting, lambda event, c, dctx: calls.append(("cache", event, c, dctx)))
    orch.on(CheckpointSaveStarting, lambda event, c, dctx: calls.append(("checkpoint", event, c, dctx)))

    orch.publish(CacheRebuildStarting(cache_name="renoise"))
    record(len(calls) == 1 and calls[0][0] == "cache",
           "only the CacheRebuildStarting handler fired", detail=str(calls))
    record(calls[0][1].cache_name == "renoise", "the actual event object reached the handler")
    record(calls[0][2] is coord and calls[0][3] is device_ctx,
           "handler received the same coordinator/device_ctx instances, unchanged")

    orch.publish(PreviewGenerationStarting())
    record(len(calls) == 1, "an event type with no registered handler is a silent no-op, not an error")


def check_multiple_handlers_for_the_same_event_all_fire():
    print("\n=== Multiple handlers for the same event type all fire, in registration order ===")
    coord = ResourceCoordinator()
    orch = OffloadOrchestrator(coord, device_ctx=None)
    order = []
    orch.on(CheckpointSaveStarting, lambda e, c, d: order.append("first"))
    orch.on(CheckpointSaveStarting, lambda e, c, d: order.append("second"))
    orch.publish(CheckpointSaveStarting(path="/tmp/x.safetensors"))
    record(order == ["first", "second"], "both handlers fired, in registration order",
           detail=str(order))


def check_events_are_real_typed_values():
    print("\n=== TrainingLifecycleEvent subclasses are real, distinguishable values ===")
    e1 = CacheRebuildStarting(cache_name="renoise")
    e2 = CheckpointSaveStarting(path="/tmp/x.safetensors")
    record(isinstance(e1, TrainingLifecycleEvent) and isinstance(e2, TrainingLifecycleEvent),
           "both are TrainingLifecycleEvents")
    record(type(e1) is not type(e2), "different event types are actually distinguishable")


def main():
    check_register_and_total_footprint()
    check_offload_all_except()
    check_reload_delegates_to_the_named_resident()
    check_orchestrator_dispatches_only_the_matching_event_type()
    check_multiple_handlers_for_the_same_event_all_fire()
    check_events_are_real_typed_values()

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
