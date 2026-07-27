"""Checks monitor_bus.py and its wiring through nodes.core.ExecutionContext
into nodes/monitor/ -- pure asyncio + stdlib, no torch needed. Covers the
things that actually broke while building this: cross-thread delivery
(graph execution runs in a FastAPI worker thread, not the event loop),
history replay for a late subscriber, and that the bus reaching a node
is real dependency injection (the same instance flows through), not a
global.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monitor_bus import MonitorBus
from nodes.core import ExecutionContext
from nodes.monitor.training_progress import TrainingProgressMonitorNode


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_no_module_level_singleton():
    import monitor_bus as mb
    check(not hasattr(mb, "bus"), "monitor_bus module must not expose a module-level instance")


def test_context_injection_reaches_handle():
    bus = MonitorBus()
    ctx = ExecutionContext(monitor_bus=bus)
    node = TrainingProgressMonitorNode(ctx)
    handle = node.build(monitor_id="t1")["monitor"]
    check(handle._bus is bus, "the injected bus instance should reach the handle unchanged")

    handle.report({"step": 1})
    check(list(bus._history["t1"]) == [{"step": 1}], "report() should land in bus history")


def test_two_contexts_do_not_share_state():
    bus_a, bus_b = MonitorBus(), MonitorBus()
    node_a = TrainingProgressMonitorNode(ExecutionContext(monitor_bus=bus_a))
    node_b = TrainingProgressMonitorNode(ExecutionContext(monitor_bus=bus_b))
    node_a.build(monitor_id="x")["monitor"].report({"v": 1})
    node_b.build(monitor_id="x")["monitor"].report({"v": 2})
    check(list(bus_a._history["x"]) == [{"v": 1}], "bus_a should only see its own report")
    check(list(bus_b._history["x"]) == [{"v": 2}], "bus_b should only see its own report")


async def _async_checks():
    bus = MonitorBus()
    bus.report("m1", {"step": 1})  # before any subscriber -- must not crash, must buffer

    q = bus.subscribe("m1")
    check(q.get_nowait() == 'data: {"step": 1}\n\n', "late subscriber must get history replay")

    def worker():
        time.sleep(0.05)
        bus.report("m1", {"step": 2})  # from a different thread, like graph execution does

    t = threading.Thread(target=worker)
    t.start()
    item = await asyncio.wait_for(q.get(), timeout=2)
    t.join()
    check(item == 'data: {"step": 2}\n\n', "cross-thread report must be delivered")

    bus.unsubscribe("m1", q)
    bus.report("m1", {"step": 3})
    check(q.empty(), "unsubscribed queue must not receive further reports")


def test_async_bus_behavior():
    asyncio.run(_async_checks())


def main():
    test_no_module_level_singleton()
    test_context_injection_reaches_handle()
    test_two_contexts_do_not_share_state()
    test_async_bus_behavior()
    print("All monitor_bus checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
