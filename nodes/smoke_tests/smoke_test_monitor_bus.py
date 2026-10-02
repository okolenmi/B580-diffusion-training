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


def test_history_cap_matches_dashboard_and_evicts_oldest():
    """The two caps must agree: the bus replays on (re)connect, the
    dashboard keeps MAX_RECORDS -- a smaller bus cap truncates the graph
    AND the hero's elapsed readout (elapsed spans the oldest report seen)
    on every reload, which is exactly what happened at 500."""
    import re

    import monitor_bus as mb

    # The live dashboard is frontend/js/monitor.js (the archived
    # server/static/monitor_dashboard.js it replaced declares the same
    # 100000); the invariant -- replay cap == displayed records -- is
    # unchanged, only its target moved with M9.
    src = (Path(__file__).resolve().parents[2] / "frontend" / "js"
           / "monitor.js").read_text()
    m = re.search(r"MAX_RECORDS = (\d+)", src)
    check(m is not None, "the dashboard must declare MAX_RECORDS")
    check(mb.HISTORY_LIMIT == int(m.group(1)),
          "bus replay cap must equal the live dashboard's MAX_RECORDS")

    # Finite and oldest-first once over the cap (patch the module global
    # BEFORE the deque is created -- the defaultdict factory reads it then).
    bus = MonitorBus()
    original = mb.HISTORY_LIMIT
    mb.HISTORY_LIMIT = 3
    try:
        for i in range(5):
            bus.report("cap", {"step": i})
    finally:
        mb.HISTORY_LIMIT = original
    check([r["step"] for r in bus._history["cap"]] == [2, 3, 4],
          "reports past the cap must evict the oldest first")
    check(mb.HISTORY_LIMIT == original, "the patched cap must not leak past the test")


def test_clear_removes_history():
    bus = MonitorBus()
    bus.report("m", {"step": 1})
    bus.report("m", {"step": 2})
    check(len(bus._history["m"]) == 2, "sanity: history should have both reports before clear()")
    bus.clear("m")
    check(len(bus._history["m"]) == 0, "clear() must empty the history deque")


def test_a_second_build_against_the_same_monitor_id_clears_the_first_runs_history():
    """The actual reported bug: re-running training against the same
    monitor_id overlaid the new run's line on the old one, because
    MonitorBus never cleared between runs -- a late subscriber (or the
    same still-open dashboard reconnecting) replayed both runs' data
    concatenated on one step axis. Fixed by having
    TrainingProgressMonitorNode.build() -- called exactly once per node
    per graph run -- clear its own monitor_id's history first."""
    bus = MonitorBus()
    ctx = ExecutionContext(monitor_bus=bus)

    first_run_handle = TrainingProgressMonitorNode(ctx).build(monitor_id="run-id")["monitor"]
    first_run_handle.report({"step": 0, "loss": 1.0})
    first_run_handle.report({"step": 1, "loss": 0.9})
    check(len(bus._history["run-id"]) == 2, "sanity: first run's own reports should be there")

    # A second graph run, same monitor_id -- exactly what happens re-running training
    # without changing the monitor node's id.
    second_run_handle = TrainingProgressMonitorNode(ctx).build(monitor_id="run-id")["monitor"]
    check(len(bus._history["run-id"]) == 0,
          "the first run's history must be gone the moment the second run's monitor "
          "node is built, before it's reported anything of its own")

    second_run_handle.report({"step": 0, "loss": 0.5})
    check(list(bus._history["run-id"]) == [{"step": 0, "loss": 0.5}],
          "only the second run's own data should remain -- no overlap with the first")


async def _clear_broadcasts_to_a_live_subscriber():
    bus = MonitorBus()
    bus.report("m", {"step": 1})
    q = bus.subscribe("m")
    check(q.get_nowait() == 'data: {"step": 1}\n\n', "sanity: history replay on subscribe")

    bus.clear("m")
    item = await asyncio.wait_for(q.get(), timeout=2)
    check(item == 'data: {"type": "clear"}\n\n',
          "a currently-connected dashboard must be told to reset live, not just leave "
          "stale history for the next fresh subscriber to (not) see")


def test_clear_broadcasts_to_a_live_subscriber():
    asyncio.run(_clear_broadcasts_to_a_live_subscriber())


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


def test_report_never_raises_into_the_caller():
    """docs 08 N-08: report() runs json.dumps on the *calling* thread,
    which is the training step. An unserializable value (a tensor, a
    Path, a set) would raise straight into a training loop -- and only
    while a dashboard happened to be connected, so it would look like a
    monitor-dependent training crash."""
    bus = MonitorBus()
    node = TrainingProgressMonitorNode(ExecutionContext(monitor_bus=bus))
    handle = node.build(monitor_id="t")["monitor"]

    class Unserializable:
        pass

    handle.report({"step": 1})
    # The call must simply not raise.
    handle.report({"step": 2, "tensor": Unserializable()})
    handle.report({"step": 3})

    kept = list(bus._history["t"])
    check([f.get("step") for f in kept] == [1, 3],
          f"the bad frame is dropped, the good ones around it survive "
          f"(got {[f.get('step') for f in kept]})")

    # And it must not poison the replay: a later subscriber replays
    # history, so a stored unserializable frame would fail there too.
    async def replay():
        q = bus.subscribe("t")
        frames = []
        while not q.empty():
            frames.append(q.get_nowait())
        return frames

    frames = asyncio.run(replay())
    check(len(frames) == 2,
          f"replay returns only the serializable frames (got {len(frames)})")
    check(all("data:" in f for f in frames), "and each is a well-formed frame")


def test_a_live_subscriber_sees_the_frames_after_a_bad_one():
    """The guard must not take the bus down: a subscriber attached
    across the bad frame still gets what comes after it."""
    bus = MonitorBus()
    seen = []

    async def scenario():
        q = bus.subscribe("t")
        node = TrainingProgressMonitorNode(ExecutionContext(monitor_bus=bus))
        handle = node.build(monitor_id="t")["monitor"]
        # build() broadcasts a clear frame to live subscribers; drain it.
        await asyncio.sleep(0)
        while not q.empty():
            seen.append(q.get_nowait())

        handle.report({"step": 1})
        handle.report({"bad": object()})
        handle.report({"step": 2})
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        while not q.empty():
            seen.append(q.get_nowait())

    asyncio.run(scenario())
    payloads = [f for f in seen if "step" in f]
    check(any('"step": 1' in f for f in payloads), "step 1 reached the subscriber")
    check(any('"step": 2' in f for f in payloads),
          f"and step 2 after the bad frame too (got {payloads})")


def test_unserializable_frames_warn_once_not_per_frame():
    """A stream that is broken in a loop must say so once, not once per
    frame, or the log is all warning and nothing else."""
    bus = MonitorBus()
    node = TrainingProgressMonitorNode(ExecutionContext(monitor_bus=bus))
    handle = node.build(monitor_id="t")["monitor"]

    class Unserializable:
        pass

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        for i in range(20):
            handle.report({"n": i, "bad": Unserializable()})
    lines = [ln for ln in buf.getvalue().splitlines() if "unserializable" in ln]
    check(len(lines) == 1,
          f"20 bad frames produce exactly one warning (got {len(lines)})")
    check("TypeError" in lines[0], f"and it names the cause (got {lines[0]!r})")


def main():
    test_no_module_level_singleton()
    test_context_injection_reaches_handle()
    test_two_contexts_do_not_share_state()
    test_history_cap_matches_dashboard_and_evicts_oldest()
    test_clear_removes_history()
    test_a_second_build_against_the_same_monitor_id_clears_the_first_runs_history()
    test_clear_broadcasts_to_a_live_subscriber()
    test_async_bus_behavior()
    test_report_never_raises_into_the_caller()
    test_a_live_subscriber_sees_the_frames_after_a_bad_one()
    test_unserializable_frames_warn_once_not_per_frame()
    print("All monitor_bus checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
