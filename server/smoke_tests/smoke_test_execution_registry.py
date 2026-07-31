"""Real threading, real GraphExecutor, a trivial one-node graph (no torch
needed -- IntConstantNode is pure Python). Targets server/routes_nodegraph.py's
_ExecutionRegistry directly: new logic (background thread + polling +
stop), real risk of a race (an earlier version of this had one -- the
worker thread could start running before the real, cancel_event-wired
context was attached, caught before shipping by re-reading the code, not
by this test, but this test guards against it recurring).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.graph_executor import EdgeSpec, NodeSpec
from server.routes_nodegraph import _ExecutionRegistry


def _wait_for(registry, execution_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        execution = registry.get(execution_id)
        if execution.status != "running":
            return execution
        time.sleep(0.01)
    raise TimeoutError(f"execution {execution_id} still running after {timeout}s")


def check_normal_run_finishes_with_results():
    print("[a normal run finishes and reports results]")
    registry = _ExecutionRegistry()
    nodes = [NodeSpec(id="a", class_name="IntConstantNode", params={"value": 7})]
    execution_id = registry.start(nodes, [], monitor_bus=None)
    execution = _wait_for(registry, execution_id)
    assert execution.status == "finished", execution.status
    assert execution.results[0]["outputs"]["value"] == 7
    print("    PASS")


def check_unknown_class_reports_via_status_poll():
    print("[a bad class name reports as an error status, not a synchronous raise]")
    registry = _ExecutionRegistry()
    nodes = [NodeSpec(id="a", class_name="NotARealNodeClass", params={})]
    execution_id = registry.start(nodes, [], monitor_bus=None)  # does not raise
    execution = _wait_for(registry, execution_id)
    assert execution.status == "error", execution.status
    assert "NotARealNodeClass" in execution.error
    print("    PASS")


def check_unknown_execution_id():
    print("[polling/stopping an unknown id]")
    registry = _ExecutionRegistry()
    assert registry.get("does-not-exist") is None
    assert registry.stop("does-not-exist") is False
    print("    PASS")


def check_stop_sets_the_cancel_event():
    print("[stop() sets the specific run's cancel_event]")
    registry = _ExecutionRegistry()
    nodes = [NodeSpec(id="a", class_name="IntConstantNode", params={"value": 1})]
    execution_id = registry.start(nodes, [], monitor_bus=None)
    execution = registry.get(execution_id)
    assert not execution.cancel_event.is_set()
    assert registry.stop(execution_id) is True
    assert execution.cancel_event.is_set()
    _wait_for(registry, execution_id)  # let the (instant) node finish either way
    print("    PASS")


def check_two_concurrent_runs_have_independent_cancel_events():
    print("[two runs started together don't share state]")
    registry = _ExecutionRegistry()
    nodes = [NodeSpec(id="a", class_name="IntConstantNode", params={"value": 1})]
    id1 = registry.start(nodes, [], monitor_bus=None)
    id2 = registry.start(nodes, [], monitor_bus=None)
    assert id1 != id2
    registry.stop(id1)
    assert registry.get(id1).cancel_event.is_set()
    assert not registry.get(id2).cancel_event.is_set()
    _wait_for(registry, id1)
    _wait_for(registry, id2)
    print("    PASS")


def main():
    check_normal_run_finishes_with_results()
    check_unknown_class_reports_via_status_poll()
    check_unknown_execution_id()
    check_stop_sets_the_cancel_event()
    check_two_concurrent_runs_have_independent_cancel_events()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
